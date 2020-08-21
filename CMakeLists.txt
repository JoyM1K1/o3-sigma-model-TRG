cmake_minimum_required(VERSION 2.8)
project(o3_sigma_model)

add_subdirectory(./libraries)

set(CMAKE_CXX_EXTENSIONS OFF)

if (APPLE)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -Wl,-stack_size -Wl,100000000 -lgsl -lgslcblas")

    include_directories(/opt/intel/mkl/include)
    link_directories(/opt/intel/mkl/lib)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)

    add_executable(gauss_quadrature_TRG gauss_quadrature_TRG.cpp)
    target_link_libraries(gauss_quadrature_TRG mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature TRG tensor time_counter)

    add_executable(gauss_quadrature_HOTRG gauss_quadrature_HOTRG.cpp)
    target_link_libraries(gauss_quadrature_HOTRG mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_1point gauss_quadrature_HOTRG_1point.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_1point mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point gauss_quadrature_HOTRG_2point.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_alt gauss_quadrature_HOTRG_2point_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_alt mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_manual gauss_quadrature_HOTRG_2point_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_manual mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass gauss_quadrature_HOTRG_mass.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_manual gauss_quadrature_HOTRG_mass_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_manual mkl_intel_ilp64 mkl_sequential mkl_core gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_TRG spherical_harmonics_TRG.cpp)
    target_link_libraries(spherical_harmonics_TRG mkl_intel_ilp64 mkl_sequential mkl_core spherical_harmonics TRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG spherical_harmonics_HOTRG.cpp)
    target_link_libraries(spherical_harmonics_HOTRG mkl_intel_ilp64 mkl_sequential mkl_core spherical_harmonics HOTRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point spherical_harmonics_HOTRG_2point.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point mkl_intel_ilp64 mkl_sequential mkl_core spherical_harmonics HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_alt spherical_harmonics_HOTRG_2point_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_alt mkl_intel_ilp64 mkl_sequential mkl_core spherical_harmonics HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_manual spherical_harmonics_HOTRG_2point_manual.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_manual mkl_intel_ilp64 mkl_sequential mkl_core spherical_harmonics HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_legendre_test gauss_legendre_test.cpp)
    target_link_libraries(gauss_legendre_test legendre_zero_point)
else ()
    set(CMAKE_CXX_COMPILER icpc)
    set(CMAKE_CXX_FLAGS "-std=c++11 -mkl -axMIC-AVX512 -qopenmp -lgsl -lgslcblas")

    add_executable(gauss_quadrature_TRG gauss_quadrature_TRG.cpp)
    target_link_libraries(gauss_quadrature_TRG gauss_quadrature TRG tensor time_counter)

    add_executable(gauss_quadrature_HOTRG gauss_quadrature_HOTRG.cpp)
    target_link_libraries(gauss_quadrature_HOTRG gauss_quadrature HOTRG tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_1point gauss_quadrature_HOTRG_1point.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_1point gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point gauss_quadrature_HOTRG_2point.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_alt gauss_quadrature_HOTRG_2point_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_alt gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_manual gauss_quadrature_HOTRG_2point_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_manual gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass gauss_quadrature_HOTRG_mass.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_manual gauss_quadrature_HOTRG_mass_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_manual gauss_quadrature HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_TRG spherical_harmonics_TRG.cpp)
    target_link_libraries(spherical_harmonics_TRG spherical_harmonics TRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG spherical_harmonics_HOTRG.cpp)
    target_link_libraries(spherical_harmonics_HOTRG spherical_harmonics HOTRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point spherical_harmonics_HOTRG_2point.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point spherical_harmonics HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_alt spherical_harmonics_HOTRG_2point_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_alt spherical_harmonics HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_manual spherical_harmonics_HOTRG_2point_manual.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_manual spherical_harmonics HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_legendre_test gauss_legendre_test.cpp)
    target_link_libraries(gauss_legendre_test legendre_zero_point)
endif ()