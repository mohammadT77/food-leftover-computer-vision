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


    Mat src_gray;
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    imshow( "original image", src );
    waitKey();
    blur( src_gray, src_gray, Size(7,7) );

    vector<Vec3f> circles;
    HoughCircles(src_gray, circles, HOUGH_GRADIENT_ALT, 1,100, 0.7, 0.9, 150, 600);
    
    vector<Vec3f> filteredCircles;
    for (const auto& circle : circles)
    {
        if (!hasIntersections(circles, circle))
            filteredCircles.push_back(circle);
    }
    int j = 0;
    Mat output_image = Mat::zeros(src.size(), src.type());
    Mat nonObject_output_image = src.clone();

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
                    output_image.at<Vec3b>(row,col) = pixel;
                }
                if (insideCircle(c[0],c[1],col,row, radius + 5)){
                    // Vec3b pixel = src.at<Vec3b>(row,col);
                    nonObject_output_image.at<Vec3b>(row,col)[0] = 0;
                    nonObject_output_image.at<Vec3b>(row,col)[1] = 0;
                    nonObject_output_image.at<Vec3b>(row,col)[2] = 0;
                }
            }
        }


        j++;
    }

    imwrite("../detection/object"+to_string(j)+".jpg", output_image);
    imwrite("../detection/nonObject"+to_string(j)+".jpg", nonObject_output_image);

    imshow("detected plate", output_image);
    imshow("not detected objects", nonObject_output_image);

    waitKey();
    
   return 0;
}