#include "bow.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

KMeanConfig default_kmean_config = KMeanConfig();

int getImageCategoryId(const string& filename) {
    string id_str = filename.substr(0, filename.find("_"));
    return stoi(id_str);
}


// Takes all images and converts them to grayscale.
// Returns a dictionary that holds all images category by category.
map<int, vector<Mat>> load_dictionary(const string& folder) {
    map<int, vector<Mat>> images;
    for (const auto& image : std::filesystem::directory_iterator(folder)) {
        string filename = image.path().filename().string();
        int category = getImageCategoryId(filename);
        Mat img = imread(folder+"/"+filename);
        images[category].push_back(img);
    }
    return images;
}

void calculate_features(const Mat& image, vector<KeyPoint>& keyPoint, Mat& descriptors ) {
    shared_ptr<SIFT> algo1 = SIFT::create();
    shared_ptr<DAISY> algo2 = DAISY::create();
    algo1->detect(image, keyPoint);
    // algo->detectAndCompute(image, noArray(), keyPoint, descriptors);
    algo2->compute(image, keyPoint, descriptors);
    // descriptors.convertTo(descriptors, CV_32F);
}

map<int, vector<Mat>> extract_features(const map<int, vector<Mat>>& images, vector<Mat>& descriptor_list) {
    map<int, vector<Mat>> feat_vectors;

    for (const auto& entry : images) {
        const int key = entry.first;
        const vector<Mat>& value = entry.second;
        vector<Mat> features;
        
        for (const auto& img : value) {
            vector<KeyPoint> kp;
            Mat des;
            calculate_features(img, kp, des);
            
            descriptor_list.push_back(des);
            features.push_back(des);
        }
        feat_vectors[key] = features;
    }
    return feat_vectors;
}


Mat kmeans(int k, const vector<Mat>& descriptor_list, KMeanConfig config) {
    cv::BOWKMeansTrainer trainer(k, config.criteria, config.attempts, config.flags);

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

void assignDescriptorsToVisualWordsBF(const Mat& image, const vector<KeyPoint>& keyPoints, const Mat& descriptors, const Mat& clusters, map<int, Point2f>& visualWords) {
    cv::BFMatcher matcher(cv::NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors, clusters, matches, 2);
    
    const float ratioThreshold = 0.9f; // Ratio threshold for discarding ambiguous matches

    for (const vector<DMatch>& match : matches) {
        if (match[0].distance < ratioThreshold * match[1].distance) {
            int visualWordIndex = match[0].trainIdx;
            visualWords[visualWordIndex] = keyPoints[match[0].queryIdx].pt;
        }
    }

}

void createHistogram(const map<int, Point2f>& visualWords, int numVisualWords, Mat& histogram) {
    histogram = Mat::zeros(1, numVisualWords, CV_32F);
    for (const auto& visualWord : visualWords) {
        histogram.at<float>(0, visualWord.first) += 1;
    }
    float sum = cv::sum(histogram)[0];
    histogram /= sum;
}


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
        closestClasses.push_back(new_item);
        closestDistances.pop();
    }

    return closestClasses;
}



vector<DistanceIndexPair> process_image(const Mat& image, const Mat& words, const map<int, vector<Mat>>& bow) {
    // Extracting orb features
    vector<KeyPoint> keyPoint;
    Mat descriptors;
    calculate_features(image, keyPoint, descriptors); 

    // Assign descriptors to visual words
    map<int, Point2f> visualWords;
    assignDescriptorsToVisualWordsBF(image, keyPoint, descriptors, words, visualWords);

    // Create histogram for the tray image
    Mat histogram;
    createHistogram(visualWords, words.rows, histogram);

    // Compare histograms and perform classification
    vector<DistanceIndexPair> result = compareHistogramsAndClassify(histogram, bow);

    return result;
}

vector<DistanceIndexPair> process_image(const Mat& image, const BOWResult& result) {
    return process_image(image, result.visualWords, result.bow);
}


void prepareBOW(BOWConfig config, map<int, vector<Mat>>& images, map<int, vector<Mat>>& features, Mat& visualWords, map<int, vector<Mat>>& bow) {
    vector<Mat> descriptors_list = {};

    images = load_dictionary(config.dir_path);
    features = extract_features(images, descriptors_list);
    visualWords = kmeans(config.k, descriptors_list, config.kmean_config);
    bow = image_class(features, visualWords);    
}

void prepareBOW(BOWConfig config, BOWResult& result) {
    prepareBOW(config, result.images, result.features, result.visualWords, result.bow);
}

vector<DistanceIndexPair> prepareEvaluatedBOW(BOWConfig config, map<int, vector<Mat>>& images, map<int, vector<Mat>>& features, Mat& visualWords, map<int, vector<Mat>>& bow) {
    prepareBOW(config, images, features, visualWords, bow);
    
    vector<DistanceIndexPair> result;
    for (const auto& entry: images) {
        const vector<Mat>& categoryImages = entry.second;
        for (const auto& pair: process_image(categoryImages[0], visualWords, bow)){
            if (pair.classIndex == entry.first){
                result.push_back(pair);
                break;
            }
            
        }
    }
    return result;
}

vector<DistanceIndexPair> prepareEvaluatedBOW(BOWConfig config, BOWResult& result) {
    return prepareEvaluatedBOW(config, result.images, result.features, result.visualWords, result.bow);
}

vector<DistanceIndexPair> process_image(const Mat& image, const vector<DistanceIndexPair>& trainDists, const Mat& words, const map<int, vector<Mat>>& bow) {
    vector<DistanceIndexPair> imageDists = process_image(image, words,bow);
    vector<DistanceIndexPair> result;

    for (const auto& trainDist: trainDists) {
        // cout << trainDist <<endl;
        for (const auto& imgDist: imageDists) {
            
            if (trainDist.classIndex == imgDist.classIndex) {
                // cout << imgDist;
                DistanceIndexPair dip;
                dip.distance = abs(imgDist.distance-trainDist.distance)/(imgDist.distance+trainDist.distance);
                dip.classIndex = imgDist.classIndex;
                result.push_back(dip);
                // cout << ": " << dip;
            }
        }  
        // cout << endl;
    }
    return result;
}

vector<DistanceIndexPair> process_image(const Mat& image, const vector<DistanceIndexPair>& trainDists, const BOWResult& bowResult) {
    return process_image(image, trainDists, bowResult.visualWords, bowResult.bow);
}

