cmake_minimum_required(VERSION 3.10)
project(ModelDLL LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 libtorch 和 CUDA 工具路径
list(APPEND CMAKE_PREFIX_PATH "/DLL/include/libtorch")
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# 手动指定 GPU 架构
set(TORCH_CUDA_ARCH_LIST "6.1")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")

# 添加线程库和 Python 支持
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Threads REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories("/DLL/include")

# 添加动态库 model_so
add_library(model_so SHARED scr_so.cpp)
set_target_properties(model_so PROPERTIES PREFIX "") # 确保输出为 libmodel_so.so 而不是 liblibmodel_so.so

# 链接库
target_link_libraries(model_so
    "${TORCH_LIBRARIES}"
    "${PYTHON_LIBRARIES}"
    Threads::Threads
)

# 添加可执行文件 B
add_executable(use use_dll.cpp)

# 链接 model_so 到可执行文件 B
target_link_libraries(use PRIVATE model_so)

# 设置可执行文件的 CXX 标准
set_property(TARGET use PROPERTY CXX_STANDARD 17)
