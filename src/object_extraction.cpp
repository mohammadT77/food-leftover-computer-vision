#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;


int computeOutput(int x, int r1, int s1, int r2, int s2)
{
    float result;
    if(0 <= x && x <= r1){
        result = s1/r1 * x;
    }else if(r1 < x && x <= r2){
        result = ((s2 - s1)/(r2 - r1)) * (x - r1) + s1;
    }else if(r2 < x && x <= 255){
        result = ((255 - s2)/(255 - r2)) * (x - r2) + s2;
    }
    return (int)result;
}


int main( int argc, char** argv )
{

    try {
        if (argc < 2) {
            throw invalid_argument("Not enough arguments!");
        }    
    } catch (const invalid_argument& e) {
        cerr << "Invalid argument: " << e.what() << endl;
        return -1;
    }
    Mat src = imread(argv[1], IMREAD_COLOR);

    if( src.empty() )
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    blur( src, src, Size(3,3) );

    


    imshow( "original image", src );
    waitKey();
    Mat segmentation = Mat::zeros(src.size(), CV_8UC1);

    for (int i = 0; i < src.cols; i++){
        for (int j = 0; j < src.rows; j++){
            float distance1 = abs(src.at<Vec3b>(j,i)[0] - src.at<Vec3b>(j,i)[1]); 
            float distance2 = abs(src.at<Vec3b>(j,i)[1] - src.at<Vec3b>(j,i)[2]);
            float distance3 = abs(src.at<Vec3b>(j,i)[2] - src.at<Vec3b>(j,i)[0]);
            if (abs(distance1 - distance2) > 35 || abs(distance2 - distance3) > 35 || abs(distance3 - distance1) > 35){
                segmentation.at<uchar>(j,i) = 255;    
            }            
        }
    }

    int dilationSize = 5;
    int erosionSize = 4;
    Mat elementDilation = getStructuringElement(MORPH_RECT, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
    Mat elementErosion = getStructuringElement(MORPH_RECT, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point(erosionSize, erosionSize));

    dilate(segmentation, segmentation, elementDilation);
    erode(segmentation, segmentation, elementErosion);
    // erode(segmentation, segmentation, elementErosion);

    imshow("segmentation", segmentation);
    waitKey();

    // Find contours
   std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(segmentation, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<Point>> filteredContours;

    for (size_t i = 0; i < contours.size(); i++) {
    // Calculate the area of the current contour
    double area = cv::contourArea(contours[i]);

    // Check if the area is less than 100
    if (area < 5000) {
        // Skip contours with area less than 100
        continue;
    }

    bool isInsideAnotherContour = false;

    // Iterate over the hierarchy to find if the contour is inside another contour
    for (int j = 0; j >= 0; j = hierarchy[j][0]) {
        if (j != static_cast<int>(i) && pointPolygonTest(contours[j], contours[i][0], false) >= 0) {
            // Contour i is inside another contour
            isInsideAnotherContour = true;
            break;
        }
    }

    if (!isInsideAnotherContour) {
        // Contour i is not inside any other contour, add it to filteredContours
        filteredContours.push_back(contours[i]);
    }
    }

    // Print the remaining number of contours
    std::cout << "Number of remaining contours: " << filteredContours.size() << std::endl;


    for (size_t i = 0; i < filteredContours.size(); i++) {
        Rect boundingRect = cv::boundingRect(filteredContours[i]);
        Mat region = src(boundingRect);
        imshow("region", region);
        waitKey();
        string filename = "../detection/region_" + to_string(i) + ".jpg";
        imwrite(filename, region);
    }

    // Draw contours
    Mat contour_image = Mat::zeros(segmentation.size(), CV_8UC3);
    for (size_t i = 0; i < filteredContours.size(); i++) {
    drawContours(contour_image, filteredContours, static_cast<int>(i), Scalar(0, 0, 255), 1);
    }

    // Show the contour image
    imshow("Contours", contour_image);
    waitKey(0);

    
   return 0;
}