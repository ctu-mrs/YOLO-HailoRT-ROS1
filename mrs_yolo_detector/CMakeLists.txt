cmake_minimum_required(VERSION 3.15.0)
project(mrs_yolo_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_MODULE_PATH               "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# set(CMAKE_CXX_FLAGS                 "${CMAKE_CXX_FLAGS} -std=c++17 -DIL_STD -Werror -DNDEBUG -O3 -flto")
# set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -flto")
# set(LINKER_OPTIONS                  -flto -Wl,--no-as-needed)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  nodelet
  nav_msgs
  mrs_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  std_msgs
  cv_bridge
  image_transport
  image_geometry
  hailort_yolo_common
  )

find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

set(LIBRARIES
  MrsYoloDetector
  )

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_DEPENDENCIES}
  )

catkin_package(
  LIBRARIES ${LIBRARIES}
  CATKIN_DEPENDS ${CATKIN_DEPENDENCIES}
  )

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${HailoRT_INCLUDE_DIRS}
  )

## Declare a C++ library
add_library(MrsYoloDetector
  src/mrs_yolo_detector.cpp
)

add_dependencies(MrsYoloDetector
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )

## Specify libraries to link a library or executable target against
target_link_libraries(MrsYoloDetector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${HailoRT_LIBRARIES}
  )

## --------------------------------------------------------------
## |                           Install                          |
## --------------------------------------------------------------

install(TARGETS ${LIBRARIES}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
