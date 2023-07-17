#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "bow.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;


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
    // process_image(imread("../data/bow_dictionary/refined/1_tray1.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/2_tray2.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/3_tray5.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/4_tray6.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/5_tray4.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/6_tray6.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/7_tray7.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/8_tray5.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/9_tray8.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/10_tray5.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/11_tray7.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/12_tray8.jpg"), words, bow);
    // process_image(imread("../data/bow_dictionary/refined/13_tray1.jpg"), words, bow);
    cout << endl;
    process_image(imread("../data/bow_dictionary/13_tray1.jpg"), breadsalad_words, breadsalad_bow);
    process_image(imread("../data/bow_dictionary/12_tray7.jpg"), breadsalad_words, breadsalad_bow);

    process_image(imread("../data/tray1/food_image.jpg"), primi_words, primi_bow);
    process_image(imread("../data/tray2/food_image.jpg"), primi_words, primi_bow);

    process_image(imread("../data/tray1/food_image.jpg"), secondi_words, secondi_bow);
    process_image(imread("../data/tray2/food_image.jpg"), secondi_words, secondi_bow);
    // process_image(imread("../data/bow_dictionary/3_tray5.jpg"), words, bow);

    return 0;
}