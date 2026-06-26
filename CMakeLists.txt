cmake_minimum_required (VERSION 3.18)
project (PRPA)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)


pkg_check_modules(GStreamer REQUIRED gstreamer-1.0 IMPORTED_TARGET)
pkg_check_modules(GStreamerVideo REQUIRED gstreamer-video-1.0 IMPORTED_TARGET)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")


add_library(stb_image src/stb_image.h src/stb_image.c)
add_library(Compute
    src/Compute.cpp
    src/Compute.cu
    src/apply.cpp
    src/apply.cu
    src/background.cpp
    src/background.cu
    src/cie.cpp
    src/cie.cu
    src/hysteresis.cpp
    src/hysteresis.cu
    src/logo.c
    src/opening.cpp
    src/opening.cu
    src/utils.cpp
)
target_link_libraries(Compute PRIVATE CUDA::cudart)
set_target_properties(Compute PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(Compute PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(stream src/stream.cpp src/gstfilter.c src/gstfilter.h)


target_link_libraries(stream PUBLIC PkgConfig::GStreamerVideo cudart)

## La beauté de NIX....
target_link_libraries(stream PUBLIC $<TARGET_FILE:Compute>)
target_link_directories(stream PUBLIC "${CUDAToolkit_LIBRARY_DIR}")
set_target_properties(stream PROPERTIES
  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)

