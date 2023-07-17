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

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


int getCategoryId(const string& filename) {
    string id_str = filename.substr(0, filename.find("_"));
    return stoi(id_str);
}


vector<Mat> load_dictionary(const string& folder, int category) {
    vector<Mat> images;
    for (const auto& image : std::filesystem::directory_iterator(folder)) {
        string filename = image.path().filename().string();
        int img_category = getCategoryId(filename);
        
        if (img_category != category) continue;

        Mat img = imread(folder+"/"+filename);
        images.push_back(img);
    }
    return images;
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

void calculate_features(const Mat& image, vector<KeyPoint>& keyPoint, Mat& descriptors ) {
    shared_ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(image, noArray(), keyPoint, descriptors);
    descriptors.convertTo(descriptors, CV_32F);
}

map<int, vector<Mat>> extract_features(const map<int, vector<Mat>>& images, vector<Mat>& descriptor_list) {
    map<int, vector<Mat>> feat_vectors;
    shared_ptr<ORB> orb = ORB::create();

    for (const auto& entry : images) {
        const int key = entry.first;
        const vector<Mat>& value = entry.second;
        vector<Mat> features;
        
        for (const auto& img : value) {
            vector<KeyPoint> kp;
            Mat des;
            calculate_features(img, kp, des);
            
            // // <==DEBUG==>
            // Mat _img = img.clone();
            // for (const auto &_kp: kp)
            //     circle(_img, _kp.pt, 1, Scalar(0, 255, 0), 1, 8, 0);

            // imshow("s", _img);
            // waitKey(0);
            // // <==DEBUG==>
            
            descriptor_list.push_back(des);
            features.push_back(des);
        }
        feat_vectors[key] = features;
    }
    return feat_vectors;
}


Mat kmeans(int k, const vector<Mat>& descriptor_list) {
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.01);
    int attempts = 20;
    int flags = KMEANS_RANDOM_CENTERS;
    
    cv::BOWKMeansTrainer trainer(k, criteria, attempts, flags);

    for (const auto& descriptor: descriptor_list) {
        trainer.add(descriptor);
    }
    
    Mat clusters = trainer.cluster();
    // cout << clusters.type() << clusters.size << endl;
    // clusters.convertTo(clusters, CV_8U);
    // cout << clusters.type() << clusters.size << endl;
    return clusters;
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

vector<Mat> image_class(const vector<Mat>& words, Mat& centers) {
    vector<Mat> result;
    for (const auto& img : words) {
        Mat histogram = Mat::zeros(1, centers.rows, CV_32F);
        for (int i = 0; i < img.rows; i++) {
            int ind = find_index(img.row(i), centers);
            histogram.at<float>(0, ind) += 1;
        }
        float sum = cv::sum(histogram)[0];
        histogram /= sum;
        result.push_back(histogram);
    }
    return result;
}

// Create histograms for each image class
map<int, vector<Mat>> image_class(const map<int, vector<Mat>>& all_bow, Mat& centers) {
    map<int, vector<Mat>> dict_feature;
    for (const auto& entry : all_bow) {
        const int key = entry.first;
        vector<Mat> category = image_class(entry.second, centers);
        // const vector<Mat>& value = entry.second;
        // for (const auto& img : value) {
        //     Mat histogram = Mat::zeros(1, centers.rows, CV_32F);
        //     for (int i = 0; i < img.rows; i++) {
        //         int ind = find_index(img.row(i), centers);
        //         histogram.at<float>(0, ind) += 1;
        //     }
        //     category.push_back(histogram);
        // }
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


void assignDescriptorsToVisualWordsBF(const vector<KeyPoint>& keyPoints, const Mat& descriptors, const Mat& clusters, vector<int>& visualWords) {
    cv::BFMatcher matcher(cv::NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors, clusters, matches, 1);
    // cout << "desc size: " << descriptors.size() << endl;
    for (const vector<DMatch>& match : matches) {
        int visualWordIndex = match[0].trainIdx;
        // cout << visualWordIndex << ',';
        visualWords.push_back(visualWordIndex);
    }
    // cout << endl;
}

void assignDescriptorsToVisualWords(const Mat& descriptors, const Mat& clusters, vector<int>& visualWords) {
    cv::flann::Index flannIndex(clusters, cv::flann::KDTreeIndexParams(1)); // Build KD-Tree for efficient search

    for (int i = 0; i < descriptors.rows; i++) {
        Mat descriptor = descriptors.row(i);

        Mat indices, distances;
        flannIndex.knnSearch(descriptor, indices, distances, 1, cv::flann::SearchParams());

        visualWords.push_back(indices.at<int>(0));
    }
}

// TODO: Create histogram for the tray image
void createHistogram(const vector<int>& visualWords, int numVisualWords, Mat& histogram) {
    histogram = Mat::zeros(1, numVisualWords, CV_32F);
    // cout << "Num VW: " << numVisualWords << endl;
    for (int visualWordIndex : visualWords) {
        histogram.at<float>(0, visualWordIndex) += 1;
    }
}

// TODO: Normalize the histogram
void normalizeHistogram(Mat& histogram) {
    float sum = cv::sum(histogram)[0];
    histogram /= sum;
}

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

    friend ostream& operator<<(ostream& os, const DistanceIndexPair& pair) {
        os << pair.classIndex << " (" << fixed << setprecision(2) << pair.distance << ")";
        return os;
    }
};

vector<DistanceIndexPair> compareHistogramsAndClassify(const Mat& histogram, const map<int, vector<Mat>>& bow) {
    priority_queue<DistanceIndexPair> closestDistances;
    map<int, vector<float>> distMap;

    for (const auto& entry : bow) {
        int classIndex = entry.first;
        const vector<Mat>& classHistograms = entry.second;

        for (const Mat& classHistogram : classHistograms) {
            double distance = cv::compareHist(histogram, classHistogram, cv::HISTCMP_CHISQR);
            
            distMap[classIndex].push_back(distance);
            // closestDistances.push({distance, classIndex});

        }
    }

    for (const auto& distPair: distMap){
        vector<float> distances_vec = distPair.second;
        float sum_dists = 0;
        for (const auto& d: distances_vec) sum_dists+=d;
        closestDistances.push({sum_dists/distances_vec.size(), distPair.first});

    }


    vector<DistanceIndexPair> closestClasses;
    while (!closestDistances.empty()) {
        DistanceIndexPair new_item = closestDistances.top();
        // if (std::find(closestClasses.begin(), closestClasses.end(), new_item) != closestClasses.end()){  // Repetitive item
        //     closestDistances.pop();
        //     cout << "Item " << new_item << " poped!" << endl;
        //     continue;
        // }
        // cout << "Item " << new_item << " added!" << endl;
        closestClasses.push_back(new_item);
        closestDistances.pop();
    }

    return closestClasses;
}



void process_image(const Mat& image, int k, const Mat& words, const map<int, vector<Mat>>& bow) {
    // Extracting orb features
    vector<KeyPoint> keyPoint;
    Mat descriptors;
    calculate_features(image, keyPoint, descriptors);

    // // <==DEBUG==>
    // Mat _img = image.clone();
    // for (const auto &_kp: keyPoint)
    //     circle(_img, _kp.pt, 1, Scalar(0, 255, 0), 2, 8, 0);

    // imshow("s", _img);
    // waitKey(0);
    // // <==DEBUG==>

    // // // Clustering features
    // TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
    // int attempts = 10;
    // int flags = KMEANS_RANDOM_CENTERS;
    
    // cv::BOWKMeansTrainer trainer(k, criteria, attempts, flags);

    // trainer.add(descriptor);
    
    // descriptor = trainer.cluster();    

    // Assign descriptors to visual words
    vector<int> visualWords;
    assignDescriptorsToVisualWordsBF(keyPoint, descriptors, words, visualWords);

    // Create histogram for the tray image
    Mat histogram;
    createHistogram(visualWords, words.rows, histogram);

    // Normalize the histogram
    normalizeHistogram(histogram);

    // Compare histograms and perform classification
    vector<DistanceIndexPair> res = compareHistogramsAndClassify(histogram, bow);
    
    for (const auto& d: res) {
        cout << d << ',';
    }
    cout << endl;
    

}

int main() {
    map<int, vector<Mat>> breadsalad_dictionary = load_dictionary("../data/bow_dictionary/12_13");
    map<int, vector<Mat>> primi_dictionary = load_dictionary("../data/bow_dictionary/primi");
    map<int, vector<Mat>> secondi_dictionary = load_dictionary("../data/bow_dictionary/secondi");
    // map<int, vector<Mat>> test_dictionary = load_dictionary("../data/bow_dictionary/test");
    
    vector<Mat> breadsalad_descriptor_list = {};
    vector<Mat> primi_descriptor_list = {};
    vector<Mat> secondi_descriptor_list = {};
    map<int, vector<Mat>> breadsalad_feat_vectors = extract_features(breadsalad_dictionary, breadsalad_descriptor_list);
    map<int, vector<Mat>> primi_feat_vectors = extract_features(primi_dictionary, primi_descriptor_list);
    map<int, vector<Mat>> secondi_feat_vectors = extract_features(secondi_dictionary, secondi_descriptor_list);


    Mat breadsalad_words = kmeans(200, breadsalad_descriptor_list);
    Mat primi_words = kmeans(1000, primi_descriptor_list);
    Mat secondi_words = kmeans(1300, secondi_descriptor_list);
    
    map<int, vector<Mat>> breadsalad_bow = image_class(breadsalad_feat_vectors, breadsalad_words);  // Bag of Visual words
    map<int, vector<Mat>> primi_bow = image_class(primi_feat_vectors, primi_words);  // Bag of Visual words
    map<int, vector<Mat>> secondi_bow = image_class(secondi_feat_vectors, secondi_words);  // Bag of Visual words

    // TODO: complete
    // process_image(imread("../data/bow_dictionary/refined/1_tray1.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/2_tray2.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/3_tray5.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/4_tray6.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/5_tray4.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/6_tray6.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/7_tray7.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/8_tray5.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/9_tray8.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/10_tray5.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/11_tray7.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/12_tray8.jpg"),30, words, bow);
    // process_image(imread("../data/bow_dictionary/refined/13_tray1.jpg"),30, words, bow);
    cout << endl;
    process_image(imread("../data/bow_dictionary/13_tray1.jpg"),30, breadsalad_words, breadsalad_bow);
    process_image(imread("../data/bow_dictionary/12_tray7.jpg"),30, breadsalad_words, breadsalad_bow);

    process_image(imread("../data/tray1/food_image.jpg"),30, primi_words, primi_bow);
    process_image(imread("../data/tray2/food_image.jpg"),30, primi_words, primi_bow);

    process_image(imread("../data/tray1/food_image.jpg"),30, secondi_words, secondi_bow);
    process_image(imread("../data/tray2/food_image.jpg"),30, secondi_words, secondi_bow);
    // process_image(imread("../data/bow_dictionary/3_tray5.jpg"),30, words, bow);

    return 0;
}