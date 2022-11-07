cmake_minimum_required(VERSION 3.2)

# set(CMAKE_CXX_STANDARD 17)
project(tracker)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -pthread")
# set(CMAKE_GENERATOR Ninja)
set(CMAKE_BUILD_TYPE)

find_package(CUDA REQUIRED)
find_package(jetson-utils)
find_package(jetson-inference)
find_package (Eigen3 3.3 NO_MODULE)
# Find all executables
file(GLOB project_SRCS src/main.cpp src/peopleDetector/*cpp src/peopleDetector/*cu)

include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/local/include/jetson-utils)
# add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# Add project executable
cuda_add_executable(tracker ${project_SRCS})
target_link_libraries (tracker Eigen3::Eigen)
target_link_libraries( tracker jetson-inference)
target_link_libraries( tracker jetson-utils)
