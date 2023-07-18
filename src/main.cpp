#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;


// void segmentation_reduction(Mat& segmentation, int threshold){

//     std::vector<std::vector<Point>> contours;
//     std::vector<Vec4i> hierarchy;
//     findContours(segmentation, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

//     Mat contour_image = Mat::zeros(segmentation.size(), CV_8UC3);
//     for (size_t i = 0; i < contours.size(); i++) {
//     drawContours(contour_image, contours, static_cast<int>(i), Scalar(0, 0, 255), 1);
//     }

//     // Show the contour image
//     imshow("segmentation", segmentation);
//     imshow("Contours", contour_image);
//     waitKey(0);

//     std::vector<std::vector<Point>> filteredContours;

//     for (size_t i = 0; i < contours.size(); i++) {
//         double area = cv::contourArea(contours[i]);

//         if (area < threshold) {
//             for (const auto& point : contours[i]) {
//                 segmentation.at<uchar>(point) = 0;
//             }
//             continue;
//         }
//         else{
//             filteredContours.push_back(contours[i]);
//         }
//     }

//     // Draw contours
//     Mat contour_image2 = Mat::zeros(segmentation.size(), CV_8UC3);
//     for (size_t i = 0; i < filteredContours.size(); i++) {
//     drawContours(contour_image2, filteredContours, static_cast<int>(i), Scalar(0, 0, 255), 1);
//     }

//     // Show the contour image
//     imshow("segmentation", segmentation);
//     imshow("Contours", contour_image2);
//     waitKey(0);

// }

bool insideCircle(int center_x, int center_y, int x, int y, int radius){
    float distance = sqrt(pow(x - center_x , 2) + pow(y - center_y , 2));
    if (distance < radius){
        return true;
    }
    else return false;
}

bool hasIntersections(const vector<Vec3f>& circles, const Vec3f& circle)
{
    int intersections = 0;
    for (const auto& otherCircle : circles)
    {
        if (otherCircle != circle)
        {
            float distance = sqrt(pow(circle[0] - otherCircle[0], 2) + pow(circle[1] - otherCircle[1], 2));
            
            if (distance < (circle[2] + otherCircle[2]))
            {
                intersections++;
            }
        }
    }
    if(intersections < 3){
        return false;
    }
    return true;
}

void extract_plates(Mat& src, Mat& object, Mat& non_object){
    Mat src_gray;
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    blur( src_gray, src_gray, Size(7,7) );

    vector<Vec3f> circles;
    HoughCircles(src_gray, circles, HOUGH_GRADIENT_ALT, 1,100, 0.7, 0.9, 150, 800);
    
    vector<Vec3f> filteredCircles;
    for (const auto& circle : circles)
    {
        if (!hasIntersections(circles, circle))
            filteredCircles.push_back(circle);
    }
    int j = 0;

    for( size_t i = 0; i < filteredCircles.size(); i++ )
    {
        Vec3i c = filteredCircles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];

        int left = c[0] - radius >= 0 ? radius - 1 : c[0] - 1;
        int right = src.cols - c[0] >= radius ? radius - 1 : src.cols - c[0] - 1;
        int up = c[1] - radius >= 0 ? radius - 1 : c[1] - 1;
        int down = src.rows - c[1] >= radius ? radius - 1 : src.rows - c[1] - 1;

        for(int row = c[1] - up; row< c[1] + down; row++){
            for(int col=c[0] - left; col < c[0] + right; col++){
                if (insideCircle(c[0],c[1],col,row, radius-10)){
                    Vec3b pixel = src.at<Vec3b>(row,col);
                    object.at<Vec3b>(row,col) = pixel;
                }
                if (insideCircle(c[0],c[1],col,row, radius + 5)){
                    non_object.at<Vec3b>(row,col)[0] = 0;
                    non_object.at<Vec3b>(row,col)[1] = 0;
                    non_object.at<Vec3b>(row,col)[2] = 0;
                }
            }
        }


        j++;
    }
    
}

void extract_objects(Mat& object, Mat& non_object){

    Mat segmentation = Mat::zeros(non_object.size(), CV_8UC1);
    for (int i = 0; i < non_object.cols; i++){
        for (int j = 0; j < non_object.rows; j++){
            int Blue = (int)non_object.at<Vec3b>(j,i)[0];
            int Green = (int)non_object.at<Vec3b>(j,i)[1];
            int Red = (int)non_object.at<Vec3b>(j,i)[2];
            int distance1 = abs(Blue - Green); 
            int distance2 = abs(Green - Red);
            int distance3 = abs(Red - Blue);
            if (((Blue - Red > 50) && (Blue - Green > 50)) || ((Red > Blue) && (Green > Blue) && (distance2 <= 21)) ){
                continue;
            }
            else if (abs(distance1 - distance2) > 35 || abs(distance2 - distance3) > 35 || abs(distance3 - distance1) > 35){
                segmentation.at<uchar>(j,i) = 255;    
            }
                        
        }
    }


    int dilationSize = 6;

    int erosionSize1 = 2;

    Mat elementDilation = getStructuringElement(MORPH_RECT, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
    Mat elementDilation2 = getStructuringElement(MORPH_RECT, Size(5,5), Point(2,2));

    Mat elementErosion = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));

    imshow("segmentation", segmentation);
    waitKey();

    erode(segmentation, segmentation, elementErosion);
    dilate(segmentation, segmentation, elementDilation);
    dilate(segmentation, segmentation, elementDilation);
    dilate(segmentation, segmentation, elementDilation2);


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


    for (size_t i = 0; i < filteredContours.size(); i++) {
        Rect boundingRect = cv::boundingRect(filteredContours[i]);
        // object_regions.push_back(boundingRect);
        for(int col = 0; col < boundingRect.width; col++){
            for(int row = 0; row < boundingRect.height; row++){
                if(object.at<Vec3b>(boundingRect.y + row, boundingRect.x + col) == Vec3b(0, 0, 0)){
                    object.at<Vec3b>(boundingRect.y + row, boundingRect.x + col) = non_object.at<Vec3b>(boundingRect.y + row, boundingRect.x + col);

                }
            }
        }

        cv::Mat roi = non_object(boundingRect);
        roi.setTo(cv::Scalar(0, 0, 0));        
    }

    // Draw contours
    Mat contour_image = Mat::zeros(segmentation.size(), CV_8UC3);
    for (size_t i = 0; i < filteredContours.size(); i++) {
    drawContours(contour_image, filteredContours, static_cast<int>(i), Scalar(0, 0, 255), 1);
    drawContours(contour_image, contours, static_cast<int>(i), Scalar(0, 255, 255), 1);

    }

    // Show the contour image
    imshow("Contours", contour_image);
    waitKey(0);

}

void get_objects(Mat& objects, vector<cv::Rect>& regions){
    Mat segmentation = Mat::zeros(objects.size(), CV_8UC1);
    
    for(int i = 0 ; i < objects.rows; i++){
        for(int j = 0; j< objects.cols; j++){
            if(objects.at<Vec3b>(i,j) != Vec3b(0,0,0)){
                segmentation.at<uchar>(i,j) = 255;
            }
        }
    }
    Mat elementDilation = getStructuringElement(MORPH_RECT, Size(4,4), Point(2,2));
    dilate(segmentation, segmentation, elementDilation);
    Mat elementErosion = getStructuringElement(MORPH_RECT, Size(4,4), Point(2,2));
    erode(segmentation, segmentation, elementErosion);

    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(segmentation, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundingRect = cv::boundingRect(contours[i]);
        regions.push_back(boundingRect);        
    }
    // Draw contours
    Mat contour_image = Mat::zeros(segmentation.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
    drawContours(contour_image, contours, static_cast<int>(i), Scalar(0, 0, 255), 1);
    }
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

// applying laplacian filter
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for (int i = 0; i < channels.size(); ++i)
    {
        cv::Mat sharpenedChannel;
        cv::Laplacian(channels[i], sharpenedChannel, CV_8U);
        channels[i] = sharpenedChannel;
    }

    cv::Mat sharpenedImage;
    cv::merge(channels, sharpenedImage);
    cv::Mat image_sharp = src - sharpenedImage;

// variable declaration of images 
    Mat objects = Mat::zeros(image_sharp.size(), CV_8UC3);
    Mat non_objects = image_sharp.clone();    

// extract plates and objects
    vector<cv::Rect> object_regions;
    extract_plates(image_sharp, objects, non_objects);
    extract_objects(objects, non_objects);

    get_objects(objects, object_regions);
    for(const auto& region : object_regions){
        cout<<region<<endl;
    }
// displaying objects and non_objects images
    imshow("detected", objects);
    imshow("non detected", non_objects);    
    waitKey();


    return 0;
}