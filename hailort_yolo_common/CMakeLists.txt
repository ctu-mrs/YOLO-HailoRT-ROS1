cmake_minimum_required(VERSION 3.0.2)
project(hailort_yolo_common)

## Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  # Add other ROS packages you need here
)

## System dependencies are found with CMake's conventions
find_package(Threads REQUIRED)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp
  DEPENDS OpenCV HailoRT
)

## Specify include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
  src/yolo_hailortpp.cpp
  src/detection_inference.cpp
)

## Specify libraries to link the library target against
target_link_libraries(${PROJECT_NAME}
  Threads::Threads
  HailoRT::libhailort
  ${OpenCV_LIBS}
  stdc++fs
)

## Declare a C++ executable
add_executable(yolo_tiny example/yolox_tiny.cpp)

## Specify libraries to link the executable target against
target_link_libraries(yolo_tiny
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

## Add cmake target dependencies of the executable
add_dependencies(yolo_tiny ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## If testing is enabled, find testing dependencies
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  # Add any testing-related commands here
endif()
