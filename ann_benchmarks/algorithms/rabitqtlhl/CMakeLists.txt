cmake_minimum_required(VERSION 3.16)

project(rabitqtlhl_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(rabitqtlhl SHARED src/rabitqtlhl_c_api.cpp)

target_include_directories(rabitqtlhl PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rabitqtlhl PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(rabitqtlhl PRIVATE
        /EHsc /O2 /arch:AVX2 /utf-8 /DNOMINMAX
    )
else()
    target_compile_options(rabitqtlhl PRIVATE
        -O3 -march=native -fPIC -Wall -Wextra -ftree-vectorize -fexceptions
    )
endif()

set_target_properties(rabitqtlhl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}"
)
