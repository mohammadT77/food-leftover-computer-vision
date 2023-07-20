#ifndef BOW_H
#define BOW_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <queue>
#include <numeric>
#include <cmath>
#include <iomanip>


struct KMeanConfig {
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.01);
    int attempts = 20;
    int flags = cv::KMEANS_RANDOM_CENTERS;
};

extern KMeanConfig default_kmean_config;

struct BOWConfig {
    std::string dir_path;
    int k;  // Number of clusters
    KMeanConfig kmean_config = default_kmean_config;
};

struct BOWResult {
    std::string label = "";
    std::map<int, std::vector<cv::Mat>> images;
    std::map<int, std::vector<cv::Mat>> features;
    cv::Mat visualWords; 
    std::map<int, std::vector<cv::Mat>> bow;
};

struct DistanceIndexPair {
    double distance;
    int classIndex;

    bool operator<(const DistanceIndexPair& other) const {
        // Reverse the comparison to maintain a min-heap
        return distance > other.distance;
    }

    bool operator==(const DistanceIndexPair& other) const {
        // Reverse the comparison to maintain a min-heap
        return classIndex == other.classIndex;
    }

    friend std::ostream& operator<<(std::ostream& os, const DistanceIndexPair& pair) {
        os << pair.classIndex << " (" << std::fixed << std::setprecision(2) << pair.distance << ")";
        return os;
    }
};


int getImageCategoryId(const std::string& filename);
std::map<int, std::vector<cv::Mat>> load_dictionary(const std::string& folder);
void calculate_features(const cv::Mat& image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptors );
void calculate_features(const cv::Mat& image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptors );
std::map<int, std::vector<cv::Mat>> extract_features(const std::map<int, std::vector<cv::Mat>>& images, std::vector<cv::Mat>& descriptor_list);
cv::Mat kmeans(int k, const std::vector<cv::Mat>& descriptor_list, KMeanConfig config = KMeanConfig());
std::vector<cv::Mat> image_class(const std::vector<cv::Mat>& words, cv::Mat& centers);
std::map<int, std::vector<cv::Mat>> image_class(const std::map<int, std::vector<cv::Mat>>& all_bow, cv::Mat& centers);
std::vector<int> knn(const std::map<int, std::vector<cv::Mat>>& images, const std::map<int, std::vector<cv::Mat>>& tests);
void assignDescriptorsToVisualWordsBF(const cv::Mat& image, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptors, const cv::Mat& clusters, std::map<int, cv::Point2f>& visualWords);
void createHistogram(const std::map<int, cv::Point2f>& visualWords, int numVisualWords, cv::Mat& histogram);
std::vector<DistanceIndexPair> compareHistogramsAndClassify(const cv::Mat& histogram, const std::map<int, std::vector<cv::Mat>>& bow);
std::vector<DistanceIndexPair> process_image(const cv::Mat& image, const cv::Mat& words, const std::map<int, std::vector<cv::Mat>>& bow);
std::vector<DistanceIndexPair> process_image(const cv::Mat& image, const BOWResult& result);
void prepareBOW(BOWConfig config, std::map<int, std::vector<cv::Mat>>& images, std::map<int, std::vector<cv::Mat>>& features, cv::Mat& visualWords, std::map<int, std::vector<cv::Mat>>& bow);
void prepareBOW(BOWConfig config, BOWResult& result);
std::vector<DistanceIndexPair> prepareEvaluatedBOW(BOWConfig config, std::map<int, std::vector<cv::Mat>>& images, std::map<int, std::vector<cv::Mat>>& features, cv::Mat& visualWords, std::map<int, std::vector<cv::Mat>>& bow);
std::vector<DistanceIndexPair> prepareEvaluatedBOW(BOWConfig config, BOWResult& result);
std::vector<DistanceIndexPair> process_image(const cv::Mat& image, const std::vector<DistanceIndexPair>& trainDists, const cv::Mat& words, const std::map<int, std::vector<cv::Mat>>& bow);
std::vector<DistanceIndexPair> process_image(const cv::Mat& image, const std::vector<DistanceIndexPair>& trainDists, const BOWResult& bowResult);

#endif