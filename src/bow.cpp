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

using namespace std;
using namespace cv;


int getCategoryId(const string& filename) {
    string id_str = filename.substr(0, filename.find("_"));
    return stoi(id_str);
}

// Takes all images and converts them to grayscale.
// Returns a dictionary that holds all images category by category.
map<int, vector<Mat>> load_dictionary(const string& folder) {
    map<int, vector<Mat>> images;
    for (const auto& image : std::filesystem::directory_iterator(folder)) {
        string filename = image.path().filename().string();
        int category = getCategoryId(filename);
        Mat img = imread(folder+"/"+filename);
        images[category].push_back(img);
    }
    return images;
}

map<int, vector<Mat>> sift_features(const map<int, vector<Mat>>& images, vector<Mat>& descriptor_list) {
    map<int, vector<Mat>> sift_vectors;
    shared_ptr<SIFT> sift = SIFT::create();

    for (const auto& entry : images) {
        const int key = entry.first;
        const vector<Mat>& value = entry.second;
        vector<Mat> features;
        
        for (const auto& img : value) {
            vector<KeyPoint> kp;
            Mat des;
            sift->detectAndCompute(img, noArray(), kp, des);
            
            descriptor_list.push_back(des);
            features.push_back(des);
        }
        sift_vectors[key] = features;
    }
    return sift_vectors;
}


Mat kmeans(int k, const vector<Mat>& descriptor_list) {
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
    int attempts = 10;
    int flags = KMEANS_RANDOM_CENTERS;
    
    cv::BOWKMeansTrainer trainer(k, criteria, attempts, flags);

    for (const auto& descriptor: descriptor_list) {
        trainer.add(descriptor);
    }
    
    return trainer.cluster();
}
// Find the index of a feature in the centers array
int find_index(const Mat& feature, const Mat& centers) {
    float minDist = numeric_limits<float>::max();
    int index = -1;

    for (int i = 0; i < centers.rows; i++) {
        float dist = norm(feature, centers.row(i));
        if (dist < minDist) {
            minDist = dist;
            index = i;
        }
    }

    return index;
}

// Create histograms for each image class
map<int, vector<Mat>> image_class(const map<int, vector<Mat>>& all_bow, Mat& centers) {
    map<int, vector<Mat>> dict_feature;
    for (const auto& entry : all_bow) {
        const int key = entry.first;
        const vector<Mat>& value = entry.second;
        vector<Mat> category;
        for (const auto& img : value) {
            Mat histogram = Mat::zeros(1, centers.rows, CV_32F);
            for (int i = 0; i < img.rows; i++) {
                int ind = find_index(img.row(i), centers);
                histogram.at<float>(0, ind) += 1;
            }
            category.push_back(histogram);
        }
        dict_feature[key] = category;
    }
    return dict_feature;
}

vector<int> knn(const map<int, vector<Mat>>& images, const map<int, vector<Mat>>& tests) {
    int num_test = 0;
    int correct_predict = 0;
    map<int, vector<int>> class_based;
    
    for (const auto& test_entry : tests) {
        const int& test_key = test_entry.first;
        const vector<Mat>& test_val = test_entry.second;
        class_based[test_key] = {0, 0}; // [correct, all]
        
        for (const auto& tst : test_val) {
            int predict_start = 0;
            double minimum = 0;
            int key = 0; // predicted
            
            for (const auto& train_entry : images) {
                const int& train_key = train_entry.first;
                const vector<Mat>& train_val = train_entry.second;
                
                for (const auto& train : train_val) {
                    if (predict_start == 0) {
                        minimum = norm(tst, train);
                        key = train_key;
                        predict_start += 1;
                    }
                    else {
                        double dist = norm(tst, train);
                        if (dist < minimum) {
                            minimum = dist;
                            key = train_key;
                        }
                    }
                }
            }
            
            if (test_key == key) {
                correct_predict += 1;
                class_based[test_key][0] += 1;
            }
            
            num_test += 1;
            class_based[test_key][1] += 1;
        }
    }
    
    return {num_test, correct_predict};
}

int main() {
    map<int, vector<Mat>> image_dictionary = load_dictionary("../data/bow_dictionary");
    // map<int, vector<Mat>> test_dictionary = load_dictionary("../data/bow_dictionary/test");
    
    vector<Mat> descriptor_list = {};
    // vector<Mat> _ = {};
    map<int, vector<Mat>> sift_vectors = sift_features(image_dictionary, descriptor_list);
    // map<int, vector<Mat>> test_sift_features = sift_features(test_dictionary, _);


    Mat words = kmeans(150, descriptor_list);
    
    map<int, vector<Mat>> bow = image_class(sift_vectors, words);
    // map<int, vector<Mat>> bow_test = image_class(test_sift_features, words);

    // vector<int> result = knn(bow, bow_test);
    // cout << result[0] << ',' << result[1] << endl;

    return 0;
}