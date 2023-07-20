
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
    int trayID;
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
            if (image.imageType == ImageType::Leftover3)
                continue;
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
            if (groundTruth[i].images[j].imageType == ImageType::Leftover3)
                continue;

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
            if (groundTruth[i].images[j].imageType == ImageType::Leftover3)
                continue;

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

// Function to split food masks in the input image and store them as pairs of mask and ID
vector<pair<Mat, int>> splitFoodMasks(const Mat& inputImage)
{
    vector<pair<Mat, int>> foodMasks;

    // Create a set to keep track of unique pixel values (food IDs)
    unordered_set<int> foodIDs;

    for (int y = 0; y < inputImage.rows; y++)
        for (int x = 0; x < inputImage.cols; x++)
        {
            // Get the pixel value (food ID) at the current position
            int pixelValue = inputImage.at<uchar>(y, x);

            if (pixelValue != 0 && foodIDs.find(pixelValue) == foodIDs.end())
                foodIDs.insert(pixelValue);
            
        }
    
    // Create an image for each unique food ID and copy the corresponding mask into it
    for (int foodID : foodIDs)
    {
        Mat foodMask = Mat::zeros(inputImage.size(), CV_8UC1);

        for (int y = 0; y < inputImage.rows; y++)
        {
            for (int x = 0; x < inputImage.cols; x++)
            {
                int pixelValue = inputImage.at<uchar>(y, x);

                if (pixelValue == foodID)
                    foodMask.at<uchar>(y, x) = 255;     
            }
        }

        foodMasks.push_back(make_pair(foodMask, foodID));
    }

    return foodMasks;
}

void readGroundTruthFile(const string& filename,vector<TrayInfo>& groundTruth)
{
    ifstream file(filename);
    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    getline(file, line); 

    TrayInfo currentTray;
    int prevTrayID = -1;
    int cnt = 0;
    Mat foodMask;

    while (getline(file, line))
    {
        istringstream iss(line);
        int trayID, categoryID, x, y, width, height;
        string imagePath;
        string prevImagePath;
        int imageTypeInt;


        if (iss >> trayID >> imagePath >> imageTypeInt >> categoryID >> x >> y >> width >> height)
        {
            ImageType imageType = static_cast<ImageType>(imageTypeInt);

            if (trayID != prevTrayID)
            {
                // Start a new tray
                if (!currentTray.images.empty())
                    groundTruth.push_back(currentTray);
                
                currentTray.trayID = trayID;
                currentTray.images.clear();
                prevTrayID = trayID;
            }

            FoodItemInfo foodItem;
            foodItem.categoryID = categoryID;
            foodItem.x = x;
            foodItem.y = y;
            foodItem.width = width;
            foodItem.height = height;

            // Add the gt segmentation masks of the food related to the current line to groundTruth 
            if (imagePath != prevImagePath || cnt == 0) {
                Mat foodMask = imread(imagePath);

                if (foodMask.empty())
                    cerr << "Error: Could not load the image: " << foodMask << endl;

                vector<pair<Mat, int>> foodMasksWithID = splitFoodMasks(foodMask);


                for (const auto& image : currentTray.images) {
                    if (image.imageType == imageType) {
                        for (const auto& food : image.foodItems) {
                            for (const auto& maskWithID : foodMasksWithID) {
                                if (foodItem.categoryID == maskWithID.second) {
                                    foodItem.segmentationMask = maskWithID.first;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            ImageInfo imageInfo;
            imageInfo.imageType = imageType;
            imageInfo.foodItems.push_back(foodItem);

            currentTray.images.push_back(imageInfo);
        }
        cnt++;
    }

    // Add the last tray to the groundTruth vector
    groundTruth.push_back(currentTray);


    file.close();
}



int main()
{
    vector<TrayInfo> groundTruth; // Load ground truth data 
    vector<TrayInfo> predictions; // Load predicted data 

    string fileName = "ground_truth.txt"; // Replace with the actual file path

    // Read the ground truth file and populate the groundTruth vector
    readGroundTruthFile(fileName, groundTruth);

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

        if (beforeMealPredImage != nullptr && leftover1PredImage != nullptr)
        {
            double PredictedLeftoverRatio = calculateLeftover(*beforeMealPredImage, *leftover1PredImage);
            cout << "Tray: " << i << " Image: Prediction leftover 1  Leftover Ratio: " << PredictedLeftoverRatio << endl;
        }
        if (beforeMealgtImage != nullptr && leftover1gtImage != nullptr)
        {
            double GroundTruthLeftoverRatio = calculateLeftover(*beforeMealgtImage, *leftover1gtImage);
            cout << "Tray: " << i << " Image: Ground truth leftover 1  Leftover Ratio: " << GroundTruthLeftoverRatio << endl;
        }

        if (beforeMealPredImage != nullptr && leftover2PredImage != nullptr)
        {
            double PredictedLeftoverRatio2 = calculateLeftover(*beforeMealPredImage, *leftover2PredImage);
            cout << "Tray: " << i << " Image: Prediction leftover 2  Leftover Ratio: " << PredictedLeftoverRatio2 << endl;
        }
        if (beforeMealgtImage != nullptr && leftover2gtImage != nullptr)
        {
            double GroundTruthLeftoverRatio2 = calculateLeftover(*beforeMealgtImage, *leftover2gtImage);
            cout << "Tray: " << i << " Image: Ground truth leftover 2  Leftover Ratio: " << GroundTruthLeftoverRatio2 << endl;
        }

        if (beforeMealPredImage != nullptr && leftover3PredImage != nullptr)
        {
            double PredictedLeftoverRatio3 = calculateLeftover(*beforeMealPredImage, *leftover3PredImage);
            cout << "Tray: " << i << " Image: Prediction leftover 3  Leftover Ratio: " << PredictedLeftoverRatio3 << endl;
        }
        if (beforeMealgtImage != nullptr && leftover3gtImage != nullptr)
        {
            double GroundTruthLeftoverRatio3 = calculateLeftover(*beforeMealgtImage, *leftover3gtImage);
            cout << "Tray: " << i << " Image: Ground truth leftover 3  Leftover Ratio: " << GroundTruthLeftoverRatio3 << endl;
        }
     }

    return 0;
}
