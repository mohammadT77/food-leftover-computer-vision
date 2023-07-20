#ifndef EVAL_H
#define EVAL_H

#include <iostream>
#include <vector>
#include <string>
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <fstream>
#include <sstream>
#include <unordered_set>

// Enum to represent the type of image
enum class ImageType
{
    BeforeMeal,
    Leftover1,
    Leftover2,
    Leftover3
};

// Data structure to store information for each food item
struct FoodItemInfo
{
    int categoryID;  // ID representing the food category
    int x, y, width, height;  // Bounding box coordinates
    cv::Mat segmentationMask;
};

// Data structure to store information for each image
struct ImageInfo
{
    ImageType imageType;  // Type of the image (BeforeMeal, Leftover1, Leftover2, Leftover3)
    std::vector<FoodItemInfo> foodItems;  // Vector to store information of each food item in the image
};

struct TrayInfo {
    int trayID;
    std::vector<ImageInfo> images;
};

// Function to calculate IoU between two bounding boxes
double calculateIoU(const FoodItemInfo& groundTruth, const FoodItemInfo& prediction);

// Function to compute the IoU for binary segmentation masks
double calculateBinaryIoU(const cv::Mat & mask1, const cv::Mat & mask2);

int findTotalGroundTruth(const std::vector<TrayInfo>& groundTruth, int categoryID);

double calculateAP(const std::vector<TrayInfo>& groundTruth, const std::vector<TrayInfo>& predictions, int categoryID);

// Function to calculate the mean Average Precision (mAP) for food localization
double calculateMAP(const std::vector<TrayInfo>& groundTruth, const std::vector<TrayInfo>& predictions);

// Function to calculate the mean Intersection over Union (mIoU) for food segmentation
double calculateMIoU(const std::vector<TrayInfo>& groundTruth, const std::vector<TrayInfo>& predictions);

// Function to estimate the quantity of food leftover and calculate the difference
double calculateLeftover(const ImageInfo & beforeImage, const ImageInfo & afterImage);

// Function to split food masks in the input image and store them as pairs of mask and ID
std::vector<std::pair<cv::Mat, int>> splitFoodMasks(const cv::Mat& inputImage);
void readGroundTruthFile(const std::string& filename,std::vector<TrayInfo>& groundTruth);

int eval(std::vector<TrayInfo> predictions);
#endif