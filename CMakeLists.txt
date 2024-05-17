cmake_minimum_required(VERSION 3.11)
project(ftvp)

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
set(CMAKE_INSTALL_LIBDIR $ENV{HOME}/lib)
set(CMAKE_CXX_STANDARD 17)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found")
    enable_language(CUDA)
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()
find_package(CUDAToolkit 11.3)

include_directories(libftvp PRIVATE ${CMAKE_SOURCE_DIR}/src)
include_directories(libftvp PRIVATE ${CMAKE_SOURCE_DIR}/src/kernels)


IF(TARGET CUDA::cudart)
    set(CUDA_ARCHITECTURES_NUM "35;70")
    include_directories(CUDAToolkit_INCLUDE_DIRS)

    SET(FTVP_SOURCES ${CMAKE_SOURCE_DIR}/src/ftvp.cu ${CMAKE_SOURCE_DIR}/src/memory.cu ${CMAKE_SOURCE_DIR}/src/kernels/kernels-2d.cu)
    SET(FTVP_COLOR_SOURCES ${CMAKE_SOURCE_DIR}/src/ftvp-color.cu ${CMAKE_SOURCE_DIR}/src/memory-color.cu ${CMAKE_SOURCE_DIR}/src/kernels/kernels-2d-color.cu)
    add_library(ftvp SHARED ${FTVP_SOURCES})
    add_library(ftvp-color SHARED ${FTVP_COLOR_SOURCES})

    set_target_properties(ftvp 
        PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES_NUM}"
    )
    set_target_properties(ftvp-color
        PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES_NUM}"
    )

install(TARGETS ftvp
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ftvp-color
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

else()
    message(FATAL_ERROR "CUDAToolkit not found")
endif()
