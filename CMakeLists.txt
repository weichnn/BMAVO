cmake_minimum_required(VERSION 2.8.3)
project(bamvo)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  eigen_conversions
  geometry_msgs
  image_transport
  roscpp
  std_msgs
  tf
  tf_conversions
)

find_package(Boost COMPONENTS system regex timer thread )
find_package(TBB REQUIRED)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES bamvo
#  CATKIN_DEPENDS cmake_modules cv_bridge eigen_conversions geometry_msgs image_transport roscpp std_msgs tf tf_conversions
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
)


add_executable(bamvo_node src/bamvo_node.cpp)
set_target_properties(bamvo_node PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS} -std=c++11 -Ofast -msse -msse2 -msse3 -mavx -march=native -ffast-math -m128bit-long-double")
add_dependencies(bamvo_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(bamvo_node ${catkin_LIBRARIES} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
