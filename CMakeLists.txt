cmake_minimum_required(VERSION 3.10)
project(PeopleCounter)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(PeopleCounter src/main.cpp src/camera/CameraStream.cpp)
target_link_libraries( PeopleCounter ${OpenCV_LIBS} )


