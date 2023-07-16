#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/core.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <filesystem>
using namespace std;
using namespace cv;


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



    vector<Mat> descriptors_vec;

    Mat descriptor_total;

    for(int id = 1; id <= 13; id++){

        vector<cv::String> image_paths;

        cv::utils::fs::glob("../../data/objects/","*.jpg", image_paths, true, true);
        
        vector<cv::String> filtered_image_paths;

        // int size = ;
        // cout<< size<<endl;
        for (const auto& mypath : image_paths) {
            std::filesystem::path path(mypath);
            std::string filename = path.filename().string();
            if (filename.substr(0, to_string(id).size() + 1) == to_string(id)+"_") {
                filtered_image_paths.push_back(mypath);
            }
        }

        Mat descriptor_class;
        for(const auto& path : filtered_image_paths){
            Mat img = imread(path);
            cout<<id<<endl;
            try {
                if (img.empty()) {
                    throw invalid_argument("missing file, improper permissions, unsupported or invalid format!");
                }
            } catch (const invalid_argument& e) {
                cerr << "Invalid argument: " << e.what() << endl;
            }

            // Ptr<SIFT> sift1 = cv::SIFT::create(0,3,0.04,10,1.6);
            // vector<KeyPoint> keypoints1;
            // Mat descriptors1;
            // sift1->detectAndCompute(img, noArray(), keypoints1, descriptors1);

            Ptr<ORB> orb1 = cv::ORB::create();
            vector<KeyPoint> keypoints1;
            Mat descriptors1;
            orb1->detectAndCompute(img, noArray(), keypoints1, descriptors1);

            Mat new_descriptors1;
            // cout<<new_descriptors1.row(0)<<endl;
            for(int i=0; i<descriptors1.rows; i++){
                int x = keypoints1[i].pt.x;
                int y = keypoints1[i].pt.y;

                Vec3b intensity = img.at<Vec3b>(static_cast<int>(y), static_cast<int>(x));
                int r = intensity[2];
                int g = intensity[1];
                int b = intensity[0];


                Mat rgbFeature = (Mat_<int>(1 ,3) << r, g, b);
                rgbFeature.convertTo(rgbFeature, CV_8U);

                // cout<<descriptors1.row(i)<<endl<<rgbFeature<<endl;
                // cout<<descriptors1.type()<<endl<<rgbFeature.type()<<endl;


                Mat descriptorWithRGB;
                hconcat(descriptors1.row(i), rgbFeature, descriptorWithRGB);
                // normalize(descriptorWithRGB, descriptorWithRGB, NORM_L2);
                descriptorWithRGB.convertTo(descriptorWithRGB, CV_32F);


                new_descriptors1.push_back(descriptorWithRGB);


            }

            if (descriptor_class.empty()){
                descriptor_class = descriptors1.clone();                
            }
            else{
                cv::vconcat(descriptors1, descriptor_class, descriptor_class);
            }

            if (descriptor_total.empty()){
                descriptor_total = new_descriptors1.clone();                
            }
            else{
                cv::vconcat(new_descriptors1, descriptor_total, descriptor_total);
            }
        }


    }
    cout<<descriptor_total.size()<<endl;

    int k = 13;

    // Set termination criteria for k-means algorithm
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10000, 0.0001);

    // Set the flags and attempts for k-means algorithm
    int flags = KMEANS_RANDOM_CENTERS;
    int attempts = 5;

    cout<<descriptor_total.row(0)<<endl;
    // Perform k-means clustering
    Mat labels, centroids;
    kmeans(descriptor_total, k, labels, criteria, attempts, flags, centroids);

    cout<<centroids.size()<<endl;

    cout<<"labels: "<<labels.size()<<endl;

    // implement the recognition part on the patches from src image
    int patch_size = 100;
    Mat mask = Mat(src.size(), CV_8UC1);
    for(int i = 0; i <= src.cols - patch_size; i = i + patch_size){
        for(int j = 0; j <= src.rows - patch_size; j = j + patch_size){
            Rect cropRect(i, j, patch_size, patch_size);
            Mat croppedImage = src(cropRect);

            // Ptr<SIFT> sift = SIFT::create(0,3,0.04,10,1.6);
            // vector<KeyPoint> src_keypoints;
            // Mat src_descriptors;
            // sift->detectAndCompute(croppedImage, noArray(), src_keypoints, src_descriptors);
            
            Ptr<ORB> orb1 = cv::ORB::create();
            vector<KeyPoint> src_keypoints;
            Mat src_descriptors;
            orb1->detectAndCompute(croppedImage, noArray(), src_keypoints, src_descriptors);

            Mat new_src_descriptor;

            for(int i=0; i<src_descriptors.rows; i++){
                int x = src_keypoints[i].pt.x;
                int y = src_keypoints[i].pt.y;

                Vec3b intensity = croppedImage.at<Vec3b>(static_cast<int>(y), static_cast<int>(x));
                int r = intensity[2];
                int g = intensity[1];
                int b = intensity[0];


                Mat rgbFeature = (Mat_<int>(1 ,3) << r, g, b);
                rgbFeature.convertTo(rgbFeature, CV_8U);

                Mat descriptorWithRGB;
                hconcat(src_descriptors.row(i), rgbFeature, descriptorWithRGB);
                descriptorWithRGB.convertTo(descriptorWithRGB, CV_32F);

                // normalize(descriptorWithRGB, descriptorWithRGB, NORM_L2);

                new_src_descriptor.push_back(descriptorWithRGB);
            }

            vector<int> histogram = {0,0,0,0,0,0,0,0,0,0,0,0,0};

            for(int k=0; k < new_src_descriptor.rows; k++){
                Mat distances;
                for (int i = 0; i < centroids.rows; ++i) {
                    float distance = norm(centroids.row(i), new_src_descriptor.row(k), NORM_L2);
                    distances.push_back(distance);
                }

                // Find the index of the closest centroid
                double minVal, maxVal;
                Point minLoc, maxLoc;
                minMaxLoc(distances, &minVal, &maxVal, &minLoc, &maxLoc);
                histogram[minLoc.y] += 1; 
            }

            for(const auto& hist : histogram){
                cout<<hist<<", ";
            }
            cout<<"]"<<endl;



            // cout<<src_keypoints.size()<<"====>"<<src_descriptors<<endl;
            Mat outputKeypoints;
            drawKeypoints(croppedImage, src_keypoints, outputKeypoints);

            imshow("src", outputKeypoints);
            waitKey();

        }
    }




    return 0;
}