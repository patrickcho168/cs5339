cmake_minimum_required(VERSION 2.8)

project(EdgeBox)
set(CMAKE_BUILD_TYPE Release)

set(EdgeBox_VERSION_MAJOR 1)
set(EdgeBox_VERSION_MINOR 0)

set(GCC_COVERAGE_COMPILE_FLAGS "-static-libstdc++ -g -Wall -Wextra -pedantic -ansi -fPIC -O3 -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS ${GCC_COVERAGE_COMPILE_FLAGS})

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(EdgeBox SHARED
	src/EdgeBox/wrappers.cpp
	src/EdgeBox/pugixml.cpp
	src/EdgeBox/edgesNms.cpp
	src/EdgeBox/edgesDetect.cpp
	src/EdgeBox/gradient.cpp
	src/EdgeBox/findEdges.cpp
	src/EdgeBox/edgeBoxes.cpp
	src/EdgeBox/evaluation.cpp
	src/sample/jsoncpp.cpp
)
target_link_libraries(EdgeBox ${OpenCV_LIBS})

add_executable(main src/sample/main.cpp)
target_link_libraries(main EdgeBox ${OpenCV_LIBS})

add_executable(evaluate src/sample/evaluate.cpp)
target_link_libraries(evaluate EdgeBox ${OpenCV_LIBS})

add_executable(fish src/sample/fish.cpp)
target_link_libraries(fish EdgeBox ${OpenCV_LIBS})