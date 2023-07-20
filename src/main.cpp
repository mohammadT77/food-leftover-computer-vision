#include "segmentation.hpp"
#include "bow.hpp"

using namespace std;
using namespace cv;

vector<Mat> extractPlates(const Mat& img, vector<Vec3f>& circles, vector<pair<Point,int>>& circlesData) {
    vector<Mat> extractedRegions;   // vector to store the images of each plate

    // Find the circles
    circles = detectCircles(img);
    // detectObject(img, circles);
    // namedWindow("X", WINDOW_NORMAL);

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

Mat getCroppedSegmentedFoodImage(const Mat& img, const Rect& rect) {
    return img(rect);
}

Mat getCroppedSegmentedFoodImage(const Mat& img, const Mat& mask) {
    return getCroppedSegmentedFoodImage(img, getSegmentedFoodRect(img, mask));
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
    return foregroundImage;
}

struct PlateDistSuperType {
    int plateIdx;
    int bowIdx;
    double dist;

    friend std::ostream& operator<<(std::ostream& os, const PlateDistSuperType& p) {
        os << '(' << p.plateIdx << ","  << p.bowIdx << "):" << p.dist;
        return os;
    }

    bool operator==(const PlateDistSuperType& other) const {
        // Reverse the comparison to maintain a min-heap
        return plateIdx == other.plateIdx || bowIdx == other.bowIdx;
    }
};


bool comparePlates_dist(const PlateDistSuperType& p1, const PlateDistSuperType& p2) {
    return p1.dist < p2.dist;
}

vector<PlateDistSuperType> findPlatesSuperTypes(const vector<Mat>& croppedPlates, const vector<BOWResult>& bowResults, const vector<vector<DistanceIndexPair>>& trainDists) {
    map<int,int> result;
    for (int i=0; i < croppedPlates.size(); i++) result[i] = -1;

    double dists[croppedPlates.size()][bowResults.size()] = {-1.0f};
    for (int i=0; i<croppedPlates.size(); i++){
        for (int j=0; j<bowResults.size(); j++){
            vector<DistanceIndexPair> resultDists = process_image(croppedPlates[i], trainDists[j], bowResults[j]);
            double sumDists = 0;
            for (const auto& dist: resultDists)
                sumDists+=dist.distance;
            dists[i][j] = sumDists/resultDists.size();
        }
    }
    vector<PlateDistSuperType> plateDists;
    for (int j=0; j<bowResults.size(); j++){
        for (int i=0; i<croppedPlates.size(); i++){
            double d = dists[i][j];
            if (d!=0)
                plateDists.push_back({i,j, d});
        }
    }

    // sort(plateDists.begin(), plateDists.end(), comparePlates_plate);
    sort(plateDists.begin(), plateDists.end(), comparePlates_dist);
    cout << "Plates1: ";
    for (const auto& p: plateDists) cout << p << ", ";
    cout << endl;
    // sort(plateDists.begin(), plateDists.end(), comparePlates_bow);
    // cout << "Plates2: ";
    // for (const auto& p: plateDists) cout << p << ", ";
    // cout << endl;
    // sort(plateDists.begin(), plateDists.end(), comparePlates_plate);
    // cout << "Plates3: ";
    // for (const auto& p: plateDists) cout << p << ", ";
    // cout << endl;
    // cout << endl;
    vector<PlateDistSuperType> res;
    res.push_back(plateDists[0]);
    for (const auto& p: plateDists) {

        if (find(res.begin(), res.end(), p) == res.end()) {
            res.push_back(p);
            result[p.plateIdx] = p.bowIdx;
        }
    }
    cout << "PlatesRes: ";
    for (const auto& p: res) cout << p << ", ";
    cout << endl;

    return res;
}


int calculateAmount(const Mat& gcImage) {
    Mat grayImg;
    cvtColor(gcImage, grayImg, COLOR_BGR2GRAY);
    return countNonZero(grayImg);
}

vector<DistanceIndexPair> evaluateFoodType(const Mat& foodImage, const BOWResult& bowResult){
    return process_image(foodImage, bowResult);
}


map<int, pair<Mat, int>> processTray(Mat& trayImage, const vector<BOWResult>& bowResults, const vector<vector<DistanceIndexPair>>& trainDists) {
    vector<Vec3f> circles;
    vector<pair<Point,int>> circlesData;

    vector<Mat> extractedRegions = extractPlates(trayImage, circles, circlesData);
    // for (const auto& m: extractedRegions){imshow("X", m); waitKey(0);}
    vector<Mat> croppedImages;
    vector<Mat> masks;
    vector<Rect> rects;
    // iterate over each circular region (corresponding to the two plates)
    for (int j = 0; j < extractedRegions.size(); j++) {
        Mat segFoodMask = extractPlateFoodMask(trayImage, extractedRegions[j], circlesData[j]);
        masks.push_back(segFoodMask);
        Mat segFood = getSegmentedFoodImage(trayImage, segFoodMask);
        Rect foodRect = getSegmentedFoodRect(segFood, segFoodMask);
        rects.push_back(foodRect);
        Mat croppedImage = getCroppedSegmentedFoodImage(trayImage, foodRect);
        // imshow("Segmented Food2", croppedImage);
        Mat gcResult = getGrabCutSegmentedFoodImage(croppedImage);
        croppedImages.push_back(gcResult);
        
        
    }
    map<int, pair<Mat, int>> result;
    vector<PlateDistSuperType> plateSuperTypes = findPlatesSuperTypes(croppedImages, bowResults, trainDists);
    for(const PlateDistSuperType& entry: plateSuperTypes){
        const Mat& croppedImage = croppedImages[entry.plateIdx];
        const BOWResult& bowResult = bowResults[entry.bowIdx];
        const Mat& mask = masks[entry.plateIdx];
        const Rect& rect = rects[entry.plateIdx];
        const Point centerOfRect = (rect.br()+rect.tl()) / 2;
        
        Mat _mask(trayImage.size(), trayImage.type(), Scalar(0,0,0));
        int amount = calculateAmount(croppedImage);
        string type =  "(" + to_string(entry.plateIdx) + "," + to_string(entry.bowIdx) + ")" + bowResult.label + ": ";
        if (bowResult.label == "breadsalad" || bowResult.label == "primi"){
            vector<DistanceIndexPair> foodDists = evaluateFoodType(croppedImage, bowResult);
            type += to_string(foodDists[0].classIndex);
            if (bowResult.label == "breadsalad") _mask.setTo(Scalar(255,0,0), mask);
            else _mask.setTo(Scalar(0,255,0), mask);
            result[foodDists[0].classIndex] = {croppedImage, amount};
        }
        else if(bowResult.label == "secondi") {
            vector<DistanceIndexPair> foodDists = evaluateFoodType(croppedImage, bowResult);
            type += to_string(foodDists[0].classIndex) + ", " + to_string(foodDists[1].classIndex) + ", " +to_string(foodDists[2].classIndex);
            _mask.setTo(Scalar(0,0,255), mask);
            result[foodDists[0].classIndex] = {croppedImage, amount};
        }

        
        
        trayImage -= _mask * 0.7;
        putText(trayImage, type,centerOfRect,FONT_HERSHEY_DUPLEX,1,Scalar(255,255,255),2,false);
    }

    return result;
}



int main(int argc, char* argv[]) {
 
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);

    if (img1.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    if (img2.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }


    BOWResult breadsalad_bowResult; breadsalad_bowResult.label = "breadsalad";
    BOWResult primi_bowResult; primi_bowResult.label = "primi";
    BOWResult secondi_bowResult; secondi_bowResult.label = "secondi";

    // vector<DistanceIndexPair> general_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/refined", 2000}, general_bowResult);
    vector<DistanceIndexPair> breadsalad_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/12_13", 100}, breadsalad_bowResult);
    vector<DistanceIndexPair> primi_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/primi", 150}, primi_bowResult);
    vector<DistanceIndexPair> secondi_trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/secondi", 200}, secondi_bowResult);


    map<int, pair<Mat, int>> tray1Result = processTray(img1,{breadsalad_bowResult,primi_bowResult,secondi_bowResult},{breadsalad_trainDists,primi_trainDists,secondi_trainDists});

    map<int, pair<Mat, int>> tray2Result = processTray(img2,{breadsalad_bowResult,primi_bowResult,secondi_bowResult},{breadsalad_trainDists,primi_trainDists,secondi_trainDists});

    namedWindow("Tray1", WINDOW_NORMAL);
    namedWindow("Tray2", WINDOW_NORMAL);
    imshow("Tray1", img1);
    imshow("Tray2", img2);
    waitKey(0);
    return 0;
}
