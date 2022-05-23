cmake_minimum_required(VERSION 3.1)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -msse4.1")
endif()

set(CMAKE_BUILD_TYPE Release)

# OpenCV
set(OpenCV_DIR ./opencv-4.0.0/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_BUILD_TYPE "Debug")
file(GLOB srcs ./*.cpp ./*.h*)
add_executable(stixel ${srcs})
target_link_libraries(stixel ${OpenCV_LIBS})