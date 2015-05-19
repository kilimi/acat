cmake_minimum_required(VERSION 2.8.3)
project(pose_estimation)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs image_transport dynamic_reconfigure cv_bridge sensor_msgs tf pcl_ros)

## Declare ROS messages and services
add_service_files(FILES PoseEstimation.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs sensor_msgs geometry_msgs)

## Declare a catkin package
catkin_package(
 CATKIN_DEPENDS std_msgs roscpp 
)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


find_package(CoViS REQUIRED HINTS $ENV{HOME}/CoViS $ENV{HOME}/Code/CoViS $ENV{HOME}/workspace/CoViS ../../../CoViS)
include_directories( ${COVIS_INCLUDE_DIR} )
link_directories( ${COVIS_LIBRARY_DIRS} )


add_executable(inSceneDetector src/inSceneDetector.cpp)
target_link_libraries(inSceneDetector ${catkin_LIBRARIES} ${COVIS_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(inSceneDetector inSceneDetector_cpp)


