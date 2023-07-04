#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat detected_edges;
const char* window_name2 = "original image";

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
    cout<<intersections<<endl;
    if(intersections < 2){
        return false;
    }
    return true;
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



    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    imshow( window_name2, src );
    waitKey();
    blur( src_gray, src_gray, Size(7,7) );

    vector<Vec3f> circles;
    HoughCircles(src_gray, circles, HOUGH_GRADIENT_ALT, 1,200, 0.7, 0.9, 150, 600);
    
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

        int x = c[0] - c[2];
        int y = c[1] - c[2];
        int width = 2 * radius;
        int height = 2 * radius;
        
        if(x < 2){
            x = 2;    
        }
        else if(x + width > src.cols){
            width = src.cols - x - 2;
        }
        if(y < 2){
            y = 2;    
        }
        else if(y + height > src.rows){
            height = src.rows - y - 2;
        }

        cv::Rect cropRect(x, y, width, height);
        cv::Mat croppedImage = src(cropRect);
        imwrite("../detection/"+to_string(j)+".jpg", croppedImage);

        imshow("cropped part", croppedImage);

        j++;
        waitKey();
    }

   return 0;
}