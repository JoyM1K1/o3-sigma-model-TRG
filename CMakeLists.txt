cmake_minimum_required(VERSION 3.16)
project(o3_sigma_model)

add_subdirectory(./libraries)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-fopenmp -Wl,-stack_size -Wl,100000000")

include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib)

add_executable(spherical_harmonics spherical_harmonics.cpp)
add_executable(gauss_quadrature gauss_quadrature.cpp)
add_executable(gauss_quadrature_HOTRG gauss_quadrature_HOTRG.cpp)
add_executable(2point 2point.cpp)
add_executable(gauss_legendre_test gauss_legendre_test.cpp libraries/include/legendre_zero_point.hpp)
add_executable(test test.cpp)

target_link_libraries(spherical_harmonics mkl_intel_ilp64 mkl_sequential mkl_core TRG frac CG)
target_link_libraries(gauss_quadrature mkl_intel_ilp64 mkl_sequential mkl_core TRG legendre_zero_point)
target_link_libraries(gauss_quadrature_HOTRG mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor legendre_zero_point)
target_link_libraries(2point mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor legendre_zero_point)
target_link_libraries(test tensor)

