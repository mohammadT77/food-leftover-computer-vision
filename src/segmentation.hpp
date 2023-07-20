# ifndef SEGMENTATION_H
# define SEGMENTATION_H

#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

std::vector<cv::Vec3f> detectCircles(const cv::Mat& inputImage);
cv::Mat performSegmentation(const cv::Mat& extractedRegion);
cv::Mat createSegmentedFoodImage(const cv::Mat& filledContours, const cv::Mat& img1);
cv::Mat drawFilledContours(const cv::Mat& segmentedImage, const std::vector<std::vector<cv::Point>>& contours);
void applyColorMask(const cv::Mat& segmentedImage, const cv::Mat& color_mask, cv::Mat& newSegmentedFood);
void colorFilter(cv::Mat* userdata, cv::Mat* new_image_ptr);
void removeCircle(cv::Mat& imgLaplacian, cv::Mat& new_laplacian, const std::pair<cv::Point, int> circleData);
void detectObject(const cv::Mat& inputImage, std::vector<cv::Vec3f>& circles);

# endif