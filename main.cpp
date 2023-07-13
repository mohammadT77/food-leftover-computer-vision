#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;



bool insideCircle(int center_x, int center_y, int x, int y, int radius){
    float distance = sqrt(pow(x - center_x , 2) + pow(y - center_y , 2));
    if (distance < radius){
        return true;
    }
    else return false;
}


// to check if the detected circles has intersection with eachother
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
            if (((Blue - Red > 50) && (Blue - Green > 50)) || ((Red > Blue) && (Green > Blue) && (distance2 <= 21))){
    
                continue;
            }
            else if (abs(distance1 - distance2) > 35 || abs(distance2 - distance3) > 35 || abs(distance3 - distance1) > 35){
                segmentation.at<uchar>(j,i) = 255;    
            }
                        
        }
    }

    int dilationSize = 4;
    int erosionSize = 2;

    Mat elementDilation = getStructuringElement(MORPH_RECT, Size(2 * dilationSize + 1, 2 * dilationSize + 1), Point(dilationSize, dilationSize));
    Mat elementErosion = getStructuringElement(MORPH_RECT, Size(2 * erosionSize + 1, 2 * erosionSize + 1), Point(erosionSize, erosionSize));

    dilate(segmentation, segmentation, elementDilation);
    erode(segmentation, segmentation, elementErosion);
    dilate(segmentation, segmentation, elementDilation);

    imshow("segmentation", segmentation);
    waitKey();

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
        for(int col = 0; col < boundingRect.width; col++){
            for(int row = 0; row < boundingRect.height; row++){
                if(object.at<Vec3b>(boundingRect.y + row, boundingRect.x + col) == Vec3b(0, 0, 0)){
                    object.at<Vec3b>(boundingRect.y + row, boundingRect.x + col) = non_object.at<Vec3b>(boundingRect.y + row, boundingRect.x + col);

                }
            }
        }
        // Mat region = non_object(boundingRect);
        // region.copyTo(object(boundingRect));
        cv::Mat roi = non_object(boundingRect);
        roi.setTo(cv::Scalar(0, 0, 0));        
    }

    // Draw contours
    Mat contour_image = Mat::zeros(segmentation.size(), CV_8UC3);
    for (size_t i = 0; i < filteredContours.size(); i++) {
    drawContours(contour_image, filteredContours, static_cast<int>(i), Scalar(0, 0, 255), 1);
    }

    // Show the contour image
    imshow("Contours", contour_image);
    waitKey(0);

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

    Mat objects = Mat::zeros(src.size(), CV_8UC3);
    Mat non_objects = src.clone();    

    extract_plates(src, objects, non_objects);
    extract_objects(objects, non_objects);

    imshow("detected", objects);
    imshow("non detected", non_objects);    
    waitKey();
    
   return 0;
}