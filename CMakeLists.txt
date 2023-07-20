cmake_minimum_required(VERSION 3.1)
project( CV_FinalProject )
find_package( OpenCV REQUIRED core highgui imgproc imgcodecs ml objdetect imgcodecs xfeatures2d video) 
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_executable( main src/main.cpp src/bow.cpp src/segmentation.cpp src/evaluation_metrics.cpp)
# add_executable( eval )

target_link_libraries( main ${OpenCV_LIBS})
# target_link_libraries( eval ${OpenCV_LIBS})
