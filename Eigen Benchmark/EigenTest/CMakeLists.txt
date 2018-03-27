cmake_minimum_required (VERSION 3.0)
project (c++_benchmark)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Optimizations 
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mavx")

# for OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
message("using OpenMP")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -DNDEBUG  -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message("The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#For MKL 
add_definitions(-DEIGEN_USE_MKL_ALL -DMKL_DIRECT_CALL)
include_directories(${MKLROOT}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_LP64 -m64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")

#Eigen 
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

add_executable (${PROJECT_NAME} c++_benchmark.cpp)


