cmake_minimum_required(VERSION 3.16)
project(vdb_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

add_library(vdb_engine STATIC
    cpp/search.cpp
    cpp/ivf_index.cpp
)
target_include_directories(vdb_engine PUBLIC include)

option(VDB_BUILD_PYTHON_BINDINGS "Build the pybind11 Python module" OFF)

if(VDB_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
