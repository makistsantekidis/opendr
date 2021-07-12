cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  vision_msgs
  std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package()

###########
## Build ##
###########

include_directories(
    ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
   scripts/pose_estimation.py
<<<<<<< HEAD
   scripts/detr.py
=======
>>>>>>> 12bd0b0c1e483e68f0e5e1748a42b2407ada7e37
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
