# Food Recognition and Leftover Estimation

This project aims to automatically recognize food types in a tray and estimate the amount of leftovers by comparing images taken before and after a meal. 

For full details, refer to the [Project Report](https://docs.google.com/document/d/1ceZdJ_YGlh9bx3gU3cZAe3r7uLgBtuAp5II5ySzbRxA/edit?usp=sharing) and the included [Report PDF](./CV_finalProject_report.pdf).

## Features & Methodology

Our proposed system analyzes tray images ("before" and "after" eating) to estimate the consumed portions. The architecture consists of four main modules:

1. **Pre-processing:** Sharpens the image using a Laplacian filter and identifies potential circular objects (like plates and bowls) using the Hough Transform (`cv::HoughCircles`).
2. **Food Segmentation:** Uses intensity slicing based on general color specifications of foods versus their surrounding regions (plates). The bounding regions map out potential areas of interest.
3. **Food Recognition:** Employs a combination of SIFT or DAISY features to train a Bag of Visual Words (BoVW) model. This is used to recognize different types of foods, typically grouped into three categories: "primi" (pasta/risotto), "secondi" (main courses), and "breadsalad" (bread or salad bowls).
4. **Leftover Estimation:** Directly compares the segmented components in the "before" and "after" images to determine the amount of pixels belonging to each category, evaluating the leftovers.

## Project Structure

*   `src/`: Contains all the C++ source files implementing pre-processing, segmentation, BoVW logic, and the core detection loop.
    *   `main.cpp`: The entry point script evaluating tray images.
    *   `bow.cpp`: Bag of Words implementation logic.
    *   `segmentation.cpp`: Food plate and object segmentation methods.
    *   `evaluation_metrics.cpp`: Logic evaluating leftover estimations against ground truth.
*   `data/`: Contains food image datasets, training samples for BoVW dictionaries, and "tray" images (before meal and after meal).
*   `build/`: Used for holding compilation output binaries.

## Prerequisites

*   **CMake** (minimum version 3.1)
*   **OpenCV** (configured with the following modules: `core`, `highgui`, `imgproc`, `imgcodecs`, `ml`, `objdetect`, `xfeatures2d`, `video`). Note that `xfeatures2d` generally requires installing `opencv_contrib`.

## Building and Running

1. **Build the executable** using CMake:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```

2. **Run the program** by providing paths to the before and after tray images, or by executing the `main` output to run automated tests on all internal dataset trays.
   ```bash
   ./main
   ```
   *Note: If testing custom images, arguments may need to be adjusted in the runtime setup as configured in `main.cpp`.*
