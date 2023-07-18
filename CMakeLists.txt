cmake_minimum_required(VERSION 3.1)
project( CV_FinalProject )
find_package( OpenCV REQUIRED core highgui imgproc imgcodecs )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable( utils src/utils.cpp)

target_link_libraries( utils ${OpenCV_LIBS})