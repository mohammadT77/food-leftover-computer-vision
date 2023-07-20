#include "segmentation.hpp"
#include "bow.hpp"

using namespace std;
using namespace cv;

vector<Mat> extractPlates(const Mat& img, vector<Vec3f>& circles, vector<pair<Point,int>>& circlesData) {
    vector<Mat> extractedRegions;   // vector to store the images of each plate

    // Find the circles
    circles = detectCircles(img);

    // Create a mask (for each plate) to fill non-plate regions with black (the region corresponding to plates are white)     
    for (size_t i = 0; i < circles.size(); i++)
    {
        // Save the circles information in circlesData
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        circlesData.push_back(make_pair(center, radius));

        // Create the mask (circle image)
        Mat circle_image = Mat::zeros(img.size(), CV_8UC1);
        circle(circle_image, center, radius, Scalar(255), -1);   

        // Create a new image containing only the region of interest using the mask
        Mat extractedRegion;
        img.copyTo(extractedRegion, circle_image);

        // Add the extracted region to the vector
        extractedRegions.push_back(extractedRegion);
    }

    return extractedRegions;
}

Mat extractPlateFoodMask(const Mat& img, const Mat& plateRegion, const pair<Point,int>& circleData) {

        Mat segmentedImage = performSegmentation(plateRegion);

        vector<vector<Point>> contours;
        findContours(segmentedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Draw filled contours on the image
        Mat filledContours = drawFilledContours(segmentedImage, contours);
        
  
        removeCircle(filledContours, filledContours, circleData); //in the case the detected circle is not perfectly equivalent to the plate border

        return filledContours;
}

Mat getSegmentedFoodImage(const Mat& img, const Mat& mask) {

    // Create the image for the segmented food (original image in white pixels of the filledContours image)
    Mat segmentedFoodFilled = createSegmentedFoodImage(mask, img);

    Mat color_mask = segmentedFoodFilled.clone();
    Mat NewSegmentedFood = segmentedFoodFilled.clone();

    Mat blurredSegmentedFood;
    GaussianBlur(segmentedFoodFilled, blurredSegmentedFood , Size(3, 3), 0);

    // Obtain a color mask to highlight plate parts

    colorFilter(&blurredSegmentedFood, &color_mask);

    // Use the color mask to remove part of dish (parts that are both highlighted by the color mask and non present (black) in the segmented image are removed)
    applyColorMask(mask, color_mask, NewSegmentedFood);

    return NewSegmentedFood;
}

Rect getSegmentedFoodRect(const Mat& img, const Mat& mask){
    // Find contours in the binary image
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Rect unionRect = boundingRect(contours[0]);
    // float biggestRectArea = -1;
    // Iterate through each contour and draw a rectangle around it
    for (const auto& contour : contours) {
        // Find the bounding rectangle for the contour
        Rect boundingBox = boundingRect(contour);
        unionRect |= boundingBox;
        // if (boundingBox.area() > biggestRectArea) unionRect = boundingBox;
    }    
    return unionRect;
}


Mat getCroppedSegmentedFoodImage(const Mat& img, const Mat& mask) {
    return img(getSegmentedFoodRect(img, mask));
}

Mat getGrabCutSegmentedFoodImage(const Mat& croppedImage) {
    // Define the bounding box for the foreground object
    Rect boundingBox(1, 1, croppedImage.cols-1, croppedImage.rows-1);
    // Perform bounds checking for the ROI
    boundingBox.x = max(0, boundingBox.x);
    boundingBox.y = max(0, boundingBox.y);
    boundingBox.width = min(boundingBox.width, croppedImage.cols - boundingBox.x);
    boundingBox.height = min(boundingBox.height, croppedImage.rows - boundingBox.y);

    // Create a mask to indicate the foreground and background regions
    Mat mask(croppedImage.size(), CV_8UC1, Scalar(GC_BGD));
    mask(boundingBox).setTo(Scalar(GC_PR_FGD));  // Set the bounding box region as probable foreground

    // Run the GrabCut algorithm
    Mat result;
    grabCut(croppedImage, mask, boundingBox, Mat(), Mat(), 5, GC_INIT_WITH_RECT);

    // Extract the foreground region based on the final mask
    Mat foregroundMask = (mask == GC_PR_FGD) | (mask == GC_FGD);
    Mat foregroundImage;
    croppedImage.copyTo(foregroundImage, foregroundMask);

    // // Display the original croppedImage, mask, and extracted foreground
    // namedWindow("Original Image", WINDOW_NORMAL);
    // namedWindow("Mask", WINDOW_NORMAL);
    // namedWindow("Foreground", WINDOW_NORMAL);
    // imshow("Original Image", croppedImage);
    // imshow("Mask", mask * 64);  // Scale the mask for better visualization
    // imshow("Foreground", foregroundImage);
    // waitKey(0);
    return result;
}

vector<pair<Mat,BOWResult>> findPlatesSuperTypes(const vector<Mat>& croppedPlates, vector<BOWResult> bowResults, const vector<vector<DistanceIndexPair>>& trainDists) {
    map<int,int> resultIdx;
    vector<pair<Mat,BOWResult>> result;
    double dists[croppedPlates.size()][bowResults.size()] = {-1.0f};
    for (int i=0; i<croppedPlates.size(); i++){
        for (int j=0; j<bowResults.size(); j++){
            vector<DistanceIndexPair> resultDists = process_image(croppedPlates[i], bowResults[j]);
            // double sumDists = 0;
            // for (const auto& dist: resultDists)
            //     sumDists+=dist.distance;
            dists[i][j] = resultDists[0].distance;//sumDists/resultDists.size();
        }
    }
    vector<pair<int,int>> sortedDists = {{0,0}};
    for (int j=0; j<bowResults.size(); j++){
        for (int i=0; i<croppedPlates.size(); i++){
            if (i==0 && j==0) continue;
            double d = dists[i][j];
            bool inserted=false;
            for (int x=0; x < sortedDists.size(); x++){
                pair<int,int> p = sortedDists[x];
                
                if (d < dists[p.first][p.second]) {
                    sortedDists.insert(sortedDists.begin()+x,{i,j});
                    inserted=true;
                    break;
                }
            }
            if (!inserted) {
                sortedDists.push_back({i,j});
            }
            
        }
    }
    for (const auto& e:sortedDists) cout << e.first << e.second << ':'<< dists[e.first][e.second] <<endl;
    for (int i=0; i<croppedPlates.size(); i++){
        int selected_j;
        for (const auto& e:sortedDists){
            if(e.first==i){
                selected_j = e.second;
                result.push_back({croppedPlates[i], bowResults[e.second]});
                break;
            }
        }
        for (int x=0;x<sortedDists.size();x++){
            if (sortedDists[x].second==selected_j)
                sortedDists.erase(sortedDists.begin()+x);
        }
    }
    return result;
}


int main(int argc, char* argv[]) {
 
    Mat img1 = imread(argv[1], IMREAD_COLOR);

    if (img1.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    namedWindow("Original image", WINDOW_NORMAL);
    namedWindow("Segmented Food1", WINDOW_NORMAL);
    namedWindow("Segmented Food2", WINDOW_NORMAL);

    imshow("Original image", img1);

    
    // Train BOVWs
    BOWResult general_bowResult; general_bowResult.label = "general";
    BOWResult breadsalad_bowResult; breadsalad_bowResult.label = "breadsalad";
    BOWResult primi_bowResult; primi_bowResult.label = "primi";
    BOWResult secondi_bowResult; secondi_bowResult.label = "secondi";

    // vector<DistanceIndexPair> general_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/refined", 2000}, general_bowResult);
    vector<DistanceIndexPair> breadsalad_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/12_13", 200}, breadsalad_bowResult);
    vector<DistanceIndexPair> primi_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/primi", 200}, primi_bowResult);
    vector<DistanceIndexPair> secondi_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/secondi", 200}, secondi_bowResult);
    
    // cout << "general_trainDists: ";
    // for (const auto& e:general_trainDists) {
    //     cout << e << ',';
    // } cout << endl;

    cout << "breadsalad_trainDists: ";
    for (const auto& e:breadsalad_trainDists) {
        cout << e << ',';
    } cout << endl;
    cout << "primi_bowResult: ";
    for (const auto& e:primi_trainDists) {
        cout << e << ',';
    } cout << endl;
    cout << "secondi_bowResult: ";
    for (const auto& e:secondi_trainDists) {
        cout << e << ',';
    } cout << endl;


    vector<Vec3f> circles;
    vector<pair<Point,int>> circlesData;

    vector<Mat> extractedRegions = extractPlates(img1, circles, circlesData);

    vector<Mat> croppedImages;
    // iterate over each circular region (corresponding to the two plates)
    for (int j = 0; j < extractedRegions.size(); j++) {
        Mat segFoodMask = extractPlateFoodMask(img1, extractedRegions[j], circlesData[j]);
        // imshow("Segmented Food1", segFoodMask);
        Mat segFood = getSegmentedFoodImage(img1, segFoodMask);

        Mat croppedImage = getCroppedSegmentedFoodImage(img1, segFoodMask);
        // imshow("Segmented Food2", croppedImage);
        Mat gcResult = getGrabCutSegmentedFoodImage(croppedImage);
        croppedImages.push_back(gcResult);
        
        // vector<DistanceIndexPair> result = process_image(croppedImage, breadsalad_bowResult);
        // for (const auto& e:result) {
        //     cout << e << ',';
        // } cout << endl;

        // result = process_image(croppedImage, primi_bowResult);
        // for (const auto& e:result) {
        //     cout << e << ',';
        // } cout << endl;

        // result = process_image(croppedImage, secondi_bowResult);
        // for (const auto& e:result) {
        //     cout << e << ',';
        // } cout << endl;
        
        
        // waitKey(0);
        

        // imshow("Plate", extractedRegions[j]);
        // waitKey(0);
        
    }

    // for (const auto& img: croppedImages){
    //     vector<DistanceIndexPair> generalDists = process_image(img, general_bowResult);
    //     imshow("Segmented Food2", img);
    //     cout << "general result: ";
    //     for (const auto& e:generalDists) {
    //         cout << e << ',';
    //     } cout << endl;
    //     waitKey(0);
    // }
    

    vector<BOWResult> bowResults = {breadsalad_bowResult,primi_bowResult,secondi_bowResult};
    vector<pair<Mat,BOWResult>> plateSuperTypes = findPlatesSuperTypes(croppedImages, bowResults, {breadsalad_trainDists,primi_trainDists,secondi_trainDists});
    for(const auto& entry: plateSuperTypes){
        cout << entry.second.label << endl;
        imshow("Segmented Food2", entry.first);
        waitKey(0);
    }

    return 0;
}
