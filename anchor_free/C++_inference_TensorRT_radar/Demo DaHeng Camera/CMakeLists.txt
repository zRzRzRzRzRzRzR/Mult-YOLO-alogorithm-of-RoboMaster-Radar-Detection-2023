cmake_minimum_required(VERSION 3.1)
project(DaHeng_Camera_TensorRT_Radar)

add_compile_options(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.0")
set(CMAKE_EXE_LINKER_FLAGS
        -I$(GENICAM_ROOT_V2_3)/library/CPP/include
        -I$(DAHENG_ROOT)/sdk/include
        )

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
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
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(cucodes SHARED ${cuda_srcs})
target_link_libraries(cucodes nvinfer nvinfer_plugin)
target_link_libraries(cucodes cuda cublas cudart cudnn)
target_link_libraries(cucodes ${OpenCV_LIBS})
add_executable(
        DaHeng_Camera_TensorRT_Radar
        ${sourcefile}
        src/CamWrapper.cpp
        include/CamWrapper.h
        include/CamWrapperDH.h
        include/DxImageProc.h
        include/GxIAPI.h
        include/DH_CAM.h
        src/DH_CAM.cpp
        src/yolov8_radar.cpp
        include/yolov8_radar.h
        src/yolov8_radar.cu)

target_link_libraries(
        DaHeng_Camera_TensorRT_Radar
        ${OpenCV_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        -lglog
        -lgxiapi
)
target_link_libraries(cucodes nvinfer nvonnxparser)
target_link_libraries(DaHeng_Camera_TensorRT_Radar cucodes)




