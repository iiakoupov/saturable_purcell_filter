cmake_minimum_required(VERSION 3.5)
project(saturable_purcell_filter)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

option(USE_INTEL_MKL "Use Intel Math Kernel Library" OFF)
option(USE_ROCM "Use ROCm" OFF)

#The default CMAKE_CXX_FLAGS_RELEASE is "-O3 -DNDEBUG".
#We set only -DNDEBUG, while "-O3" is added to all
#build types via CMAKE_CXX_FLAGS.
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
set(CMAKE_CXX_FLAGS "-std=c++17 -march=native -O3 -fopenmp")

if(USE_INTEL_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIR})
    add_definitions(-DEIGEN_USE_MKL_ALL)
    add_definitions(-DMKL_DIRECT_CALL)
endif(USE_INTEL_MKL)

if (USE_ROCM)
    # Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
    # (Copied from rocsparse cmake files)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm)

    find_package(rocsparse REQUIRED CONFIG PATHS /opt/rocm/rocsparse)
    add_definitions(-DUSE_ROCM)
    add_subdirectory(qroc)
endif (USE_ROCM)

find_package(NLopt REQUIRED)
include_directories(${NLOPT_INCLUDE_DIR})

add_subdirectory(quadrature)

set(decay_with_jqf_plots_SRCS
    decay_with_jqf_plots.cpp
)

set(Faddeeva_SRCS
    Faddeeva.cc
)

set(jqf_SRCS
    jqf.cpp
)

set(jqf_adrk4_SRCS
    jqf_adrk4.cpp
)

set(jqf_superoperator_SRCS
    jqf_superoperator.cpp
)

set(measure_with_jqf_SRCS
    measure_with_jqf.cpp
)

set(measure_with_jqf_plots_SRCS
    measure_with_jqf_plots.cpp
)

set(io_SRCS
    io.cpp
)

set(operator_SRCS
    operator.cpp
)

set(control_with_jqf_plots_SRCS
    control_with_jqf_plots.cpp
)

#Libraries

add_library(Faddeeva ${Faddeeva_SRCS})

add_library(io ${io_SRCS})

add_library(operator ${operator_SRCS})
if(USE_INTEL_MKL)
    target_link_libraries(operator ${MKL_LIBRARIES})
endif(USE_INTEL_MKL)

add_library(jqf_adrk4 ${jqf_adrk4_SRCS})
target_link_libraries(jqf_adrk4 Faddeeva jqf_superoperator quad ${NLOPT_LIBRARIES})

add_library(jqf_superoperator ${jqf_superoperator_SRCS})
target_link_libraries(jqf_superoperator operator)
if(USE_INTEL_MKL)
    target_link_libraries(jqf_superoperator ${MKL_LIBRARIES})
endif(USE_INTEL_MKL)

add_library(jqf ${jqf_SRCS})
target_link_libraries(jqf jqf_adrk4 jqf_superoperator operator ${NLOPT_LIBRARIES})
if(USE_INTEL_MKL)
    target_link_libraries(jqf ${MKL_LIBRARIES})
endif(USE_INTEL_MKL)
if (USE_ROCM)
    target_link_libraries(jqf master_equation_roc)
endif (USE_ROCM)

add_library(measure_with_jqf ${measure_with_jqf_SRCS})
target_link_libraries(measure_with_jqf jqf_superoperator operator)
if(USE_INTEL_MKL)
    target_link_libraries(measure_with_jqf ${MKL_LIBRARIES})
endif(USE_INTEL_MKL)
if (USE_ROCM)
    target_link_libraries(measure_with_jqf master_equation_roc)
endif (USE_ROCM)

#Executables

add_executable(decay_with_jqf_plots ${decay_with_jqf_plots_SRCS})
target_link_libraries(decay_with_jqf_plots jqf io ${NLOPT_LIBRARIES})

add_executable(control_with_jqf_plots ${control_with_jqf_plots_SRCS})
target_link_libraries(control_with_jqf_plots jqf io ${NLOPT_LIBRARIES})

add_executable(measure_with_jqf_plots ${measure_with_jqf_plots_SRCS})
target_link_libraries(measure_with_jqf_plots measure_with_jqf io ${NLOPT_LIBRARIES})
