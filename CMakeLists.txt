cmake_minimum_required(VERSION 3.8)
project(ImageAnalisis)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")




set(OpenCV_DIR "/Users/kikepieraserra/GitHub/ImageAnalisis/opencv-3.3.0/build/CMakeFiles/Export/share/OpenCV")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)





set(SOURCE_FILES main.cpp)
add_executable(ImageAnalisis ${SOURCE_FILES})

# linking
target_link_libraries( ImageAnalisis ${OpenCV_LIBS} )

