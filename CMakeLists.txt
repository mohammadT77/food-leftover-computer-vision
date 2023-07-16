cmake_minimum_required(VERSION 3.1)
project( CV_FinalProject )
find_package( OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable( main src/main.cpp)
add_executable( edge_detection src/edge_detection.cpp)
add_executable( Hough src/generalized_Hough_transform.cpp)
add_executable( BoW src/BoW.cpp)

target_link_libraries( main ${OpenCV_LIBS})
target_link_libraries( edge_detection ${OpenCV_LIBS})
target_link_libraries( Hough ${OpenCV_LIBS})
target_link_libraries( BoW ${OpenCV_LIBS})