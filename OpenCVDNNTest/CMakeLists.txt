cmake_minimum_required (VERSION 3.0)
project (c++inference)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mavx")

# for OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
message("using OpenMP")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3 -march=native")

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

#For MKL
#add_definitions(-DEIGEN_USE_MKL_ALL -DMKL_DIRECT_CALL)
#include_directories(${MKLROOT}/include)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_LP64 -m64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")

add_executable (${PROJECT_NAME} caffeinference.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#Eigen
#target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)
