#include "segmentation.hpp"

using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {
 
    Mat img1 = imread(argv[1], IMREAD_COLOR);

    if (img1.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    imshow("Original image", img1);
    waitKey(0);

    // Find the circles

    vector<Vec3f> circles = detectCircles(img1);

    // Create a mask (for each plate) to fill non-plate regions with black (the region corresponding to plates are white)

    vector<pair<Point,int>> circlesData;     
    vector<Mat> extractedRegions;            //vector to store the images of each plate

    for (size_t i = 0; i < circles.size(); i++)
    {
        // Save the circles information in circlesData
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        circlesData.push_back(make_pair(center, radius));

        // Create the mask (circle image)
        Mat circle_image = Mat::zeros(img1.size(), CV_8UC1);
        circle(circle_image, center, radius, Scalar(255), -1);   

        // Create a new image containing only the region of interest using the mask
        Mat extractedRegion;
        img1.copyTo(extractedRegion, circle_image);

        // Add the extracted region to the vector
        extractedRegions.push_back(extractedRegion);
    }

    // iterate over each circular region (corresponding to the two plates)
    for (int j = 0; j < extractedRegions.size(); j++) {

        imshow("Plate", extractedRegions[j]);
        waitKey(0);

        Mat segmentedImage = performSegmentation(extractedRegions[j]);

        //imshow("Segmentation mask", segmentedImage);
        //waitKey(0);

        vector<vector<Point>> contours;
        findContours(segmentedImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Draw filled contours on the image

        Mat filledContours = drawFilledContours(segmentedImage, contours);

        //imshow("Filled Contours", filledContours);
        //waitKey(0);
  
        removeCircle(filledContours, filledContours, circlesData[j]); //in the case the detected circle is not perfectly equivalent to the plate border

        // Create the image for the segmented food (original image in white pixels of the filledContours image)

        Mat segmentedFoodFilled = createSegmentedFoodImage(filledContours, img1);

        imshow("Segmented Food", segmentedFoodFilled);
        waitKey(0);

        Mat color_mask = segmentedFoodFilled.clone();
        Mat NewSegmentedFood = segmentedFoodFilled.clone();


        Mat blurredSegmentedFood;
        GaussianBlur(segmentedFoodFilled,blurredSegmentedFood , Size(3, 3), 0);

        // Obtain a color mask to highlight plate parts

        colorFilter(&blurredSegmentedFood, &color_mask);

        //imshow("Color tresholded image", color_mask);
        //waitKey(0);

        // Use the color mask to remove part of dish (parts that are both highlighted by the color mask and non present (black) in the segmented image are removed)
        applyColorMask(segmentedImage, color_mask, NewSegmentedFood);
        
        imshow("Segmented food", NewSegmentedFood);
        waitKey(0);
    }

    return 0;
}
