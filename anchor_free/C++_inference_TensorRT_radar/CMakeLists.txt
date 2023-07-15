cmake_minimum_required(VERSION 3.1)
project(radar_infer)
add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/model)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.0") #cuda位置

find_package(CUDA REQUIRED)
find_package(OpenCV)
find_package(yaml-cpp REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_DIR}/include
        ${TENSORRT_DIR}/include
        ${CUDNN_DIR}/include
)

link_directories(
        ${TENSORRT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDNN_DIR}/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O0 -Wfatal-errors -pthread -w -g")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -O3 -Xcompiler -fPIC -gencode arch=compute_86,code=sm_86")
file(GLOB cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(cucodes SHARED ${cuda_srcs})
target_link_libraries(cucodes nvinfer nvinfer_plugin)
target_link_libraries(cucodes cuda cublas cudart cudnn)
target_link_libraries(cucodes ${OpenCV_LIBS})
add_executable(radar_infer ${cpp_srcs})

# 手动链接插件
target_link_libraries(cucodes nvinfer nvonnxparser)
target_link_libraries(radar_infer ${OpenCV_LIBS} yaml-cpp)
target_link_libraries(radar_infer cucodes)