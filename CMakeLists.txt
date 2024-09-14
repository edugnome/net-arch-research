cmake_minimum_required(VERSION 3.29)
project(MatrixMultiplication)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(MatrixMultiplication main.cpp)
target_link_libraries(MatrixMultiplication ${CUDA_LIBRARIES})