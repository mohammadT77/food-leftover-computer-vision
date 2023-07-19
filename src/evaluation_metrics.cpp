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

using namespace std;
using namespace cv;

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
    Mat segmentationMask;
};

// Data structure to store information for each image
struct ImageInfo
{
    ImageType imageType;  // Type of the image (BeforeMeal, Leftover1, Leftover2, Leftover3)
    vector<FoodItemInfo> foodItems;  // Vector to store information of each food item in the image
};

struct TrayInfo {
    vector<ImageInfo> images;
};

// Function to calculate IoU between two bounding boxes
double calculateIoU(const FoodItemInfo& groundTruth, const FoodItemInfo& prediction) {
    // Coordinates of the area of intersection
    int ix1 = max(groundTruth.x, prediction.x);
    int iy1 = max(groundTruth.y, prediction.y);
    int ix2 = min(groundTruth.x + groundTruth.width, prediction.x + prediction.width);
    int iy2 = min(groundTruth.y + groundTruth.height, prediction.y + prediction.height);

    // Intersection height and width
    int i_height = max(iy2 - iy1 + 1, 0);
    int i_width = max(ix2 - ix1 + 1, 0);

    int area_of_intersection = i_height * i_width;

    int area_of_union = groundTruth.height * groundTruth.width + prediction.height * prediction.width - area_of_intersection;

    double iou = static_cast<double>(area_of_intersection) / static_cast<double>(area_of_union);

    return iou;
}

// Function to compute the IoU for binary segmentation masks
double calculateBinaryIoU(const Mat & mask1, const Mat & mask2) {
    CV_Assert(mask1.size() == mask2.size());

    int intersectionArea = 0;
    int unionArea = 0;

    for (int i = 0; i < mask1.rows; i++) {
        for (int j = 0; j < mask1.cols; j++) {
            if (mask1.at<uchar>(i, j) == 255 && mask2.at<uchar>(i, j) == 255) {
                intersectionArea++;
                unionArea++;
            }
            else if (mask1.at<uchar>(i, j) == 255 || mask2.at<uchar>(i, j) == 255) {
                unionArea++;
            }
        }
    }

    if (unionArea == 0) {
        return 0.0; 
    }

    return static_cast<double>(intersectionArea) / static_cast<double>(unionArea);
}

int findTotalGroundTruth(const vector<TrayInfo>& groundTruth, int categoryID) {
    int totalGroundTruths = 0;

    for (const auto& tray : groundTruth) {
        for (const auto& image : tray.images) {
            for (const auto& foodItem : image.foodItems) {
                if (foodItem.categoryID == categoryID) {
                    totalGroundTruths++;
                }
            }
        }
    }

    return totalGroundTruths;
}

double calculateAP(const vector<TrayInfo>& groundTruth, const vector<TrayInfo>& predictions, int categoryID) {
    vector<double> precisions;
    vector<double> recalls;

    int cumulativeTruePositive = 0;
    int cumulativeFalsePositive = 0;

    // Find the number of foods with the given ID in the whole set
    int totalGroundTruths = findTotalGroundTruth(groundTruth, categoryID);

    // Iterate over each tray and images in the tray (for gt and predictions)
    for (size_t i = 0; i < groundTruth.size(); ++i) {
        for (size_t j = 0; j < groundTruth[i].images.size(); ++j) {
            // Find the specific class (food category) in the tray and compute TP and FP
            // Iterate over each gt food, looking for foods with the given ID
            for (const FoodItemInfo& gtFood : groundTruth[i].images[j].foodItems) {  
                
                if (gtFood.categoryID == categoryID) {

                    //iteration over the food of the prediction image correspondent to the ground truth -note: prediction and ground truth have to have been ordered in the same way
                    for (const FoodItemInfo& predFood : predictions[i].images[j].foodItems) {
                        if (predFood.categoryID == categoryID) {
                            double iou = calculateIoU(gtFood, predFood);

                            if (iou >= 0.5) 
                                cumulativeTruePositive++;
                            else
                                cumulativeFalsePositive++;
                            break;
                        }     
 
                    }
                    // Whenever a food with the correct ID is found, the precisions and recalls vectors are updated
                    double precision = static_cast<double>(cumulativeTruePositive) / static_cast<double>(cumulativeTruePositive + cumulativeFalsePositive);
                    double recall = static_cast<double>(cumulativeTruePositive) / static_cast<double>(totalGroundTruths); 

                    precisions.push_back(precision);
                    recalls.push_back(recall);
                }
            }
        }
    }

    // Calculate AP using PASCAL VOC 11 Point Interpolation Method
    double ap = 0.0;
    double minRecallStep = 0.1;

    for (double recallStep = 0.0; recallStep <= 1.0; recallStep += minRecallStep) {
        double maxPrecision = 0.0;

        for (size_t i = 0; i < recalls.size(); ++i) {
            if (recalls[i] >= recallStep) {
                maxPrecision = max(maxPrecision, precisions[i]);
            }
        }

        ap += maxPrecision;
    }

    ap /= 11.0;

    return ap;
}

// Function to calculate the mean Average Precision (mAP) for food localization
double calculateMAP(const vector<TrayInfo>& groundTruth, const vector<TrayInfo>& predictions)
{
    double totalAP = 0.0;
    int totalCategories = 0;
    const int MAX_CATEGORY_ID = 13;                

    // Loop through each food category ID
    for (int categoryID = 1; categoryID <= MAX_CATEGORY_ID; ++categoryID) {
        // Find total number of ground truths for this category
        int totalGroundTruths = findTotalGroundTruth(groundTruth, categoryID);

        // If there are ground truths for this category, calculate AP
        if (totalGroundTruths > 0) {
            double ap = calculateAP(groundTruth, predictions, categoryID);
            totalAP += ap;
            totalCategories++;
        }
    }

    // Calculate mean Average Precision (mAP)
    double mAP = totalAP / totalCategories;
    return mAP;
}

// Function to calculate the mean Intersection over Union (mIoU) for food segmentation
double calculateMIoU(const vector<TrayInfo>& groundTruth, const vector<TrayInfo>& predictions)
{
    double totalIoU = 0.0;
    int totalFoodItems = 0;

    for (size_t i = 0; i < groundTruth.size(); ++i) {
        for (size_t j = 0; j < groundTruth[i].images.size(); ++j) {
            const ImageInfo& gtImageInfo = groundTruth[i].images[j];
            const ImageInfo& predImageInfo = predictions[i].images[j];

            for (size_t k = 0; k < gtImageInfo.foodItems.size(); ++k) {
                const FoodItemInfo& gtFoodInfo = gtImageInfo.foodItems[k];
                const FoodItemInfo& predFoodInfo = predImageInfo.foodItems[k];

                double iou = calculateBinaryIoU(gtFoodInfo.segmentationMask, predFoodInfo.segmentationMask);
                totalIoU += iou;
                totalFoodItems++;
            }
        }
    }
    double mIoU = totalIoU / totalFoodItems;
    return mIoU;
}

// Function to estimate the quantity of food leftover and calculate the difference
double calculateLeftover(const ImageInfo & beforeImage, const ImageInfo & afterImage)
{
    double totalPixelsBefore = 0.0;
    double totalPixelsAfter = 0.0;

    for (const auto& foodBefore : beforeImage.foodItems)
        totalPixelsBefore += countNonZero(foodBefore.segmentationMask);

    for (const auto& foodAfter : afterImage.foodItems)
        totalPixelsAfter += countNonZero(foodAfter.segmentationMask);

    // Calculate the ratio of pixels between "after" and "before" images
    double leftoverRatio = totalPixelsAfter / totalPixelsBefore;

    return leftoverRatio;
}

int main()
{
    vector<TrayInfo> groundTruth; // Load ground truth data 
    vector<TrayInfo> predictions; // Load predicted data 

    // Calculate and display the mean Average Precision (mAP) for food localization
    double mAP = calculateMAP(groundTruth, predictions);
    cout << "Mean Average Precision (mAP) for food localization: " << mAP << endl;

    // Calculate and display the mean Intersection over Union (mIoU) for food segmentation
    double mIoU = calculateMIoU(groundTruth, predictions);
    cout << "Mean Intersection over Union (mIoU) for food segmentation: " << mIoU << endl;

    // Estimate food leftover 

     for (size_t i = 0; i < groundTruth.size(); ++i) // iterate over each tray
     {
        const TrayInfo& gtTray = groundTruth[i];
        const TrayInfo& predTray = predictions[i];

        // Variables to store the image information for each image type
        const ImageInfo* beforeMealPredImage = nullptr;  // predictions
        const ImageInfo* leftover1PredImage = nullptr;
        const ImageInfo* leftover2PredImage = nullptr;
        const ImageInfo* leftover3PredImage = nullptr;

        const ImageInfo* beforeMealgtImage = nullptr;    // ground truths
        const ImageInfo* leftover1gtImage = nullptr;
        const ImageInfo* leftover2gtImage = nullptr;
        const ImageInfo* leftover3gtImage = nullptr;

        // Iterate over each image in the current tray
        for (size_t j = 0; j < gtTray.images.size(); ++j)
        {
            const ImageInfo& gtImage = gtTray.images[j];
            const ImageInfo& predImage = predTray.images[j];

            // Assign the image information based on their image types
            if (predImage.imageType == ImageType::BeforeMeal)
                beforeMealPredImage = &predImage;
            else if (predImage.imageType == ImageType::Leftover1)
                leftover1PredImage = &predImage;
            else if (predImage.imageType == ImageType::Leftover2)
                leftover2PredImage = &predImage;
            else if (predImage.imageType == ImageType::Leftover3)
                leftover3PredImage = &predImage;

            if (gtImage.imageType == ImageType::BeforeMeal)
                beforeMealgtImage = &gtImage;
            else if (gtImage.imageType == ImageType::Leftover1)
                leftover1gtImage = &gtImage;
            else if (gtImage.imageType == ImageType::Leftover2)
                leftover2gtImage = &gtImage;
            else if (gtImage.imageType == ImageType::Leftover3)
                leftover3gtImage = &gtImage;
        }

        double PredictedLeftoverRatio = calculateLeftover(*beforeMealPredImage, *leftover1PredImage);        
        cout << "Tray: " << i << " Image: Prediction leftover 1  Leftover Ratio: " << PredictedLeftoverRatio << endl;   
        double GroundTruthLeftoverRatio = calculateLeftover(*beforeMealgtImage, *leftover1gtImage);
        cout << "Tray: " << i << " Image: Ground truth leftover 1  Leftover Ratio: " << PredictedLeftoverRatio << endl;

        double PredictedLeftoverRatio2 = calculateLeftover(*beforeMealPredImage, *leftover2PredImage);
        cout << "Tray: " << i << " Image: Prediction leftover 2  Leftover Ratio: " << PredictedLeftoverRatio2 << endl;
        double GroundTruthLeftoverRatio2 = calculateLeftover(*beforeMealgtImage, *leftover2gtImage);
        cout << "Tray: " << i << " Image: Ground truth leftover 2  Leftover Ratio: " << GroundTruthLeftoverRatio2 << endl;

        double PredictedLeftoverRatio3 = calculateLeftover(*beforeMealPredImage, *leftover3PredImage);
        cout << "Tray: " << i << " Image: Prediction leftover 3  Leftover Ratio: " << PredictedLeftoverRatio3 << endl;
        double GroundTruthLeftoverRatio3 = calculateLeftover(*beforeMealgtImage, *leftover3gtImage);
        cout << "Tray: " << i << " Image: Ground truth leftover 3  Leftover Ratio: " << GroundTruthLeftoverRatio3 << endl << endl;
     }

    return 0;
}
