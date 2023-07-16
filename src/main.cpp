
#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;

vector<Vec3f> detectCircles(const Mat& inputImage);
Mat performSegmentation(const Mat& extractedRegion);
Mat createSegmentedFoodImage(const Mat& filledContours, const Mat& img1);
Mat drawFilledContours(const Mat& segmentedImage, const vector<vector<Point>>& contours);
void applyColorMask(const Mat& segmentedImage, const Mat& color_mask, Mat& newSegmentedFood);
void colorFilter(cv::Mat* userdata, cv::Mat* new_image_ptr);
void removeCircle(cv::Mat& imgLaplacian, cv::Mat& new_laplacian, const std::pair<cv::Point, int> circleData);

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

vector<Vec3f> detectCircles(const Mat& inputImage) {
    Mat gray;
    cvtColor(inputImage, gray, COLOR_BGR2GRAY);

    Mat blurredImage;
    GaussianBlur(gray, blurredImage, Size(5, 5), 0);

    vector<Vec3f> circles;
    HoughCircles(blurredImage, circles, HOUGH_GRADIENT_ALT, 1, 100, 0.7, 0.9, 150, 600);

    return circles;
}

Mat performSegmentation(const Mat& extractedRegion) {
    Mat blurredExtractedRegions;
    GaussianBlur(extractedRegion, blurredExtractedRegions, Size(3, 3), 0);

    Mat segmentation = Mat::zeros(blurredExtractedRegions.size(), CV_8UC1);

    for (int i = 0; i < blurredExtractedRegions.cols; i++) {
        for (int j = 0; j < blurredExtractedRegions.rows; j++) {
            float distance1 = abs(blurredExtractedRegions.at<Vec3b>(j, i)[0] - blurredExtractedRegions.at<Vec3b>(j, i)[1]);
            float distance2 = abs(blurredExtractedRegions.at<Vec3b>(j, i)[1] - blurredExtractedRegions.at<Vec3b>(j, i)[2]);
            float distance3 = abs(blurredExtractedRegions.at<Vec3b>(j, i)[2] - blurredExtractedRegions.at<Vec3b>(j, i)[0]);
            if (abs(distance1 - distance2) > 25 || abs(distance2 - distance3) > 25 || abs(distance3 - distance1) > 25) {
                //tresholds: 50 for pasta, 40 for vegetable, 25 for other things
                segmentation.at<uchar>(j, i) = 255;
            }
        }
    }

    int dilationSize = 5;
    int erosionSize = 4;
    Mat elementDilation = getStructuringElement(MORPH_RECT, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
    Mat elementErosion = getStructuringElement(MORPH_RECT, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point(erosionSize, erosionSize));

    dilate(segmentation, segmentation, elementDilation);
    erode(segmentation, segmentation, elementErosion);

    return segmentation;
}

Mat createSegmentedFoodImage(const Mat& filledContours, const Mat& img1) {
    Mat segmentedFood(img1.size(), img1.type(), Vec3b(0, 0, 0));

    // Iterate over each pixel in the mask image
    for (int y = 0; y < filledContours.rows; ++y) {
        for (int x = 0; x < filledContours.cols; ++x) {
            // Check if the pixel in the mask image is white
            if (filledContours.at<uchar>(y, x) == 255) {
                // Replace the pixel with the corresponding one in the original image
                segmentedFood.at<Vec3b>(y, x) = img1.at<Vec3b>(y, x);
            }
        }
    }

    return segmentedFood;
}

Mat drawFilledContours(const Mat& segmentedImage, const vector<vector<Point>>& contours) {
    Mat filledContours = Mat::zeros(segmentedImage.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); ++i) {
        // Calculate the area of the current contour
        double area = contourArea(contours[i]);

        // Check if the current contour meets the area threshold
        if (area > 300)
            drawContours(filledContours, contours, static_cast<int>(i), Scalar(255), FILLED);
    }

    return filledContours;
}

void applyColorMask(const Mat& segmentedImage, const Mat& color_mask, Mat& newSegmentedFood) {
    for (int y = 0; y < segmentedImage.rows; ++y) {
        for (int x = 0; x < segmentedImage.cols; ++x) {
            // Check if the pixel in the color mask is the color (92, 37, 201) and the corresponding pixel in the segmentedImage is black (0)
            if (color_mask.at<Vec3b>(y, x) == Vec3b(92, 37, 201) && segmentedImage.at<uchar>(y, x) == 0) {
                // Set the corresponding pixel in the newSegmentedFood to black (0, 0, 0)
                newSegmentedFood.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
        }
    }
}

void colorFilter(Mat* userdata, Mat* new_image_ptr) {

    Mat& image = *userdata;
    Mat& new_image = *new_image_ptr;
    Mat mask_image = image.clone();
    int T = 8;
    int T1 = 8;
    int T2 = 25;
    int T4 = 20;

    Scalar mean(54, 81, 92);
    Scalar mean1(156.5, 163.0, 174.0);

    for (int v = 0; v < image.cols; v++)
        for (int u = 0; u < image.rows; u++) {

            Vec3b pixel = image.at<Vec3b>(u, v);

            if ((((abs(pixel[2] - pixel[0]) < T2 || abs(pixel[0] - pixel[1]) < T1) || abs(pixel[2] - pixel[1]) < T) &&
                !(pixel[2] < 20 && pixel[1] < 20 && pixel[0] < 20)) || (abs(mean[0] - pixel[0]) < T4 && abs(mean[1] - pixel[1]) < T4 && abs(mean[2] - pixel[2]) < T4)) {
                mask_image.at<Vec3b>(u, v)[0] = 255;
                mask_image.at<Vec3b>(u, v)[1] = 255;
                mask_image.at<Vec3b>(u, v)[2] = 255;
            }
            else {
                mask_image.at<Vec3b>(u, v)[0] = 0;
                mask_image.at<Vec3b>(u, v)[1] = 0;
                mask_image.at<Vec3b>(u, v)[2] = 0;
            }
        }

    for (int v = 0; v < image.cols; v++)
        for (int u = 0; u < image.rows; u++) {
            if (mask_image.at<Vec3b>(u, v)[0] == 0 && mask_image.at<Vec3b>(u, v)[1] == 0 && mask_image.at<Vec3b>(u, v)[2] == 0) {
                new_image.at<Vec3b>(u, v)[0] = image.at<Vec3b>(u, v)[0];
                new_image.at<Vec3b>(u, v)[1] = image.at<Vec3b>(u, v)[1];
                new_image.at<Vec3b>(u, v)[2] = image.at<Vec3b>(u, v)[2];
            }
            else {
                new_image.at<Vec3b>(u, v)[0] = 92; //92
                new_image.at<Vec3b>(u, v)[1] = 37; //37
                new_image.at<Vec3b>(u, v)[2] = 201; //201
            }
        }
}

void removeCircle(Mat& imgLaplacian, Mat& new_laplacian, const std::pair<Point, int> circleData)
{
    new_laplacian = imgLaplacian.clone();

    Point center = circleData.first;
    int radius = circleData.second;

    // Iterate over the pixels within the circle region
    for (int y = center.y - (radius + 50); y <= center.y + (radius + 50); ++y)
    {
        for (int x = center.x - (radius + 50); x <= center.x + (radius + 50); ++x)
        {
            // Check if the pixel is within the circle using the equation of a circle
            if (abs(pow(x - center.x, 2) + pow(y - center.y, 2) - pow(radius + 2, 2)) <= 200)
            {
                // Create a black square with centre (x,y)
                int squareSize = 30;
                Point topLeft(x - squareSize / 2, y - squareSize / 2);
                Point bottomRight(x + squareSize / 2, y + squareSize / 2);

                // Draw the black square on the result image
                rectangle(new_laplacian, topLeft, bottomRight, Scalar(0, 0, 0), -1);
            }
        }
    }
}
