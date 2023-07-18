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

// void evaluateBOW(const Mat& img) {
    

    
// }


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

    // Train breadSalad bow
    map<int, vector<Mat>> breadsalad_dictionary;
    map<int, vector<Mat>> breadsalad_feat_vectors;
    Mat breadsalad_words;
    map<int, vector<Mat>> breadsalad_bow;

    vector<DistanceIndexPair> trainDists = prepareEvaluatedBOW({"../data/bow_dictionary/primi", 1000},breadsalad_dictionary,breadsalad_feat_vectors,breadsalad_words,breadsalad_bow );
    
    cout << "Train dists: ";
    for (const auto& e:trainDists) {
        cout << e << ',';
    } cout << endl;


    vector<Vec3f> circles;
    vector<pair<Point,int>> circlesData;

    vector<Mat> extractedRegions = extractPlates(img1, circles, circlesData);


    // iterate over each circular region (corresponding to the two plates)
    for (int j = 0; j < extractedRegions.size(); j++) {
        Mat segFoodMask = extractPlateFoodMask(img1, extractedRegions[j], circlesData[j]);
        imshow("Segmented Food1", segFoodMask);
        Mat segFood = getSegmentedFoodImage(img1, segFoodMask);

        // Find contours in the binary image
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(segFoodMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Rect biggestRect;
        float biggestRectArea = -1;
        // Iterate through each contour and draw a rectangle around it
        for (const auto& contour : contours) {
            // Find the bounding rectangle for the contour
            Rect boundingBox = boundingRect(contour);
            
            if (boundingBox.area() > biggestRectArea) biggestRect = boundingBox;
            // // Draw the rectangle on the original image
            // rectangle(img1, boundingBox, Scalar(0, 255, 0), 2);
        }

        Mat croppedImage = img1(biggestRect);
        imshow("Segmented Food2", croppedImage);
        

        vector<DistanceIndexPair> result = process_image(croppedImage, breadsalad_words, breadsalad_bow);
        for (const auto& e:result) {
            cout << e << ',';
        } cout << endl;
        
        waitKey(0);
        

        // imshow("Plate", extractedRegions[j]);
        // waitKey(0);
        
        
    }

    return 0;
}
