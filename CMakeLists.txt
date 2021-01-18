cmake_minimum_required(VERSION 2.8)
project(o3_sigma_model)

add_subdirectory(./libraries)

set(CMAKE_CXX_EXTENSIONS OFF)

if (APPLE)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -Wl,-stack_size -Wl,100000000 -lgsl -lgslcblas")

    include_directories($ENV{MKL_DIR}/include)
    link_directories($ENV{MKL_DIR}/lib)
    include_directories($ENV{GSL_DIR}/include)
    link_directories($ENV{GSL_DIR}/lib)

    add_executable(gauss_quadrature_HOTRG_1point_alt gauss_quadrature_HOTRG_1point_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_1point_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_alt gauss_quadrature_HOTRG_2point_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_manual gauss_quadrature_HOTRG_2point_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_manual mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_alt gauss_quadrature_HOTRG_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass gauss_quadrature_HOTRG_mass.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_alt gauss_quadrature_HOTRG_mass_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_manual gauss_quadrature_HOTRG_mass_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_manual mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_v1 gauss_quadrature_HOTRG_mass_v1.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_v1 mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_TRG gauss_quadrature_TRG.cpp)
    target_link_libraries(gauss_quadrature_TRG mkl_intel_ilp64 mkl_sequential mkl_core TRG tensor time_counter)

    add_executable(gauss_quadrature_TRG_2point gauss_quadrature_TRG_2point.cpp)
    target_link_libraries(gauss_quadrature_TRG_2point mkl_intel_ilp64 mkl_sequential mkl_core TRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_1point_alt spherical_harmonics_HOTRG_1point_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_1point_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_alt spherical_harmonics_HOTRG_2point_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_manual spherical_harmonics_HOTRG_2point_manual.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_manual mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_alt spherical_harmonics_HOTRG_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_mass_alt spherical_harmonics_HOTRG_mass_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_mass_alt mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_mass_v1 spherical_harmonics_HOTRG_mass_v1.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_mass_v1 mkl_intel_ilp64 mkl_sequential mkl_core HOTRG tensor time_counter)

    add_executable(spherical_harmonics_TRG spherical_harmonics_TRG.cpp)
    target_link_libraries(spherical_harmonics_TRG mkl_intel_ilp64 mkl_sequential mkl_core TRG tensor time_counter)
else ()
    set(CMAKE_CXX_COMPILER icpc)
    set(CMAKE_CXX_FLAGS "-std=c++11 -mkl -axMIC-AVX512 -qopenmp -lgsl -lgslcblas")

    add_executable(gauss_quadrature_HOTRG_1point_alt gauss_quadrature_HOTRG_1point_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_1point_alt HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_alt gauss_quadrature_HOTRG_2point_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_alt HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_2point_manual gauss_quadrature_HOTRG_2point_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_2point_manual HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_alt gauss_quadrature_HOTRG_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_alt HOTRG tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass gauss_quadrature_HOTRG_mass.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_alt gauss_quadrature_HOTRG_mass_alt.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_alt HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_manual gauss_quadrature_HOTRG_mass_manual.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_manual HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_HOTRG_mass_v1 gauss_quadrature_HOTRG_mass_v1.cpp)
    target_link_libraries(gauss_quadrature_HOTRG_mass_v1 HOTRG tensor impure_tensor time_counter)

    add_executable(gauss_quadrature_TRG gauss_quadrature_TRG.cpp)
    target_link_libraries(gauss_quadrature_TRG TRG tensor time_counter)

    add_executable(gauss_quadrature_TRG_2point gauss_quadrature_TRG_2point.cpp)
    target_link_libraries(gauss_quadrature_TRG_2point TRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_1point_alt spherical_harmonics_HOTRG_1point_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_1point_alt HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_alt spherical_harmonics_HOTRG_2point_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_alt HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_2point_manual spherical_harmonics_HOTRG_2point_manual.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_2point_manual HOTRG tensor impure_tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_alt spherical_harmonics_HOTRG_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_alt HOTRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_mass_alt spherical_harmonics_HOTRG_mass_alt.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_mass_alt HOTRG tensor time_counter)

    add_executable(spherical_harmonics_HOTRG_mass_v1 spherical_harmonics_HOTRG_mass_v1.cpp)
    target_link_libraries(spherical_harmonics_HOTRG_mass_v1 HOTRG tensor time_counter)

    add_executable(spherical_harmonics_TRG spherical_harmonics_TRG.cpp)
    target_link_libraries(spherical_harmonics_TRG TRG tensor time_counter)

    add_executable(gauss_legendre_test gauss_legendre_test.cpp)
    target_link_libraries(gauss_legendre_test legendre_zero_point)
endif ()