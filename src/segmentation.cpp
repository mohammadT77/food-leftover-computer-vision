#include "segmentation.hpp"

using namespace std;
using namespace cv;


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


void detectObject(const Mat& inputImage, vector<Vec3f>& circles) {
    
    Mat non_object = inputImage.clone();
    for(const auto& circle : circles){
        Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        cv::circle(non_object, center, radius, Scalar(0,0,0), -1);
    }

    Mat segmentation = Mat::zeros(non_object.size(), CV_8UC1);
    for (int i = 0; i < non_object.cols; i++){
        for (int j = 0; j < non_object.rows; j++){
            int Blue = (int)non_object.at<Vec3b>(j,i)[0];
            int Green = (int)non_object.at<Vec3b>(j,i)[1];
            int Red = (int)non_object.at<Vec3b>(j,i)[2];
            int distance1 = abs(Blue - Green); 
            int distance2 = abs(Green - Red);
            int distance3 = abs(Red - Blue);
            if (((Blue - Red > 20) && (Blue - Green > 20))||((Red > Blue) && (Green > Blue) && (distance2 <= 21)) ){
                continue;
            }
            else if (abs(distance1 - distance2) > 35 || abs(distance2 - distance3) > 37 || abs(distance3 - distance1) > 35){
                segmentation.at<uchar>(j,i) = 255;    
            }
                        
        }
    }

    Mat elementDilation = getStructuringElement(MORPH_ELLIPSE, Size(13,13));

    Mat elementErosion = getStructuringElement(MORPH_RECT, Size(3,3));

    // imshow("segmentation", segmentation);
    // waitKey();

    erode(segmentation, segmentation, elementErosion);
    dilate(segmentation, segmentation, elementDilation);
    dilate(segmentation, segmentation, elementDilation);
    dilate(segmentation, segmentation, elementDilation);


    // Find contours
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(segmentation, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<Point>> filteredContours;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);

        if (area < 4500) {
            continue;
        }

        bool isInsideAnotherContour = false;

        for (int j = 0; j >= 0; j = hierarchy[j][0]) {
            if (j != static_cast<int>(i) && pointPolygonTest(contours[j], contours[i][0], false) >= 0) {
                isInsideAnotherContour = true;
                break;
            }
        }

        if (!isInsideAnotherContour) {
            filteredContours.push_back(contours[i]);
        }
    }

    Mat contour_image = Mat::zeros(segmentation.size(), CV_8UC3);
    for (size_t i = 0; i < filteredContours.size(); i++) {
        drawContours(contour_image, filteredContours, static_cast<int>(i), Scalar(0, 0, 255), 1);
    }


    for (size_t i = 0; i < filteredContours.size(); i++) {
        // Find the minimum enclosing circle for the contour
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(filteredContours[i], center, radius);
        radius = radius - 5;
        Vec3f new_circle(center.x, center.y, radius);
        circles.push_back(new_circle);

    }
}