cmake_minimum_required(VERSION 3.10)

# Prevent in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed. Please create a build directory and run CMake from there. You may need to remove CMakeCache.txt and CMakeFiles/ first.")
endif()

project(bartMachine_port)

# Repository locations
# Original Repository: /Users/mark/Documents/Cline/bartMachine
# Port Repository: /Users/mark/Documents/Cline/bartMachine_port
# Original Java Source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine
# Original R Source: /Users/mark/Documents/Cline/bartMachine/bartMachine

# Set C++ standard`
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for debugging and warnings
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Default to Debug build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Include directories
include_directories(src/cpp/include)
include_directories(.)  # Include the root directory for tests

# Source files
set(MERSENNE_SOURCES
    src/cpp/exact_port_mersenne_twister.cpp
)

set(STAT_TOOLBOX_SOURCES
    src/cpp/stat_toolbox.cpp
)

set(STAT_UTIL_SOURCES
    src/cpp/stat_util.cpp
)

set(BARTMACHINE_BASE_SOURCES
    src/cpp/bartmachine_a_base.cpp
)

set(BARTMACHINE_HYPERPARAMS_SOURCES
    src/cpp/bartmachine_b_hyperparams.cpp
)

set(BARTMACHINE_TREE_NODE_SOURCES
    src/cpp/bartmachine_tree_node.cpp
)

set(BARTMACHINE_C_DEBUG_SOURCES
    src/cpp/bartmachine_c_debug.cpp
)

set(BARTMACHINE_D_INIT_SOURCES
    src/cpp/bartmachine_d_init.cpp
)

set(BARTMACHINE_E_GIBBS_BASE_SOURCES
    src/cpp/bartmachine_e_gibbs_base.cpp
)

set(BARTMACHINE_F_GIBBS_INTERNAL_SOURCES
    src/cpp/bartmachine_f_gibbs_internal.cpp
)

set(BARTMACHINE_G_MH_SOURCES
    src/cpp/bartmachine_g_mh.cpp
)

set(BARTMACHINE_H_EVAL_SOURCES
    src/cpp/bartmachine_h_eval.cpp
)

set(BARTMACHINE_I_PRIOR_COV_SPEC_SOURCES
    src/cpp/bartmachine_i_prior_cov_spec.cpp
)

set(BARTMACHINE_REGRESSION_SOURCES
    src/cpp/bartmachine_regression.cpp
)

set(BARTMACHINE_CLASSIFICATION_SOURCES
    src/cpp/bartmachine_classification.cpp
)

# Create library for MersenneTwister
add_library(exact_port_mersenne_twister ${MERSENNE_SOURCES})

# Create libraries
add_library(bartmachine_a_base ${BARTMACHINE_BASE_SOURCES})
add_library(bartmachine_b_hyperparams ${BARTMACHINE_HYPERPARAMS_SOURCES})
add_library(stat_toolbox ${STAT_TOOLBOX_SOURCES})
add_library(stat_util ${STAT_UTIL_SOURCES})
add_library(bartmachine_tree_node ${BARTMACHINE_TREE_NODE_SOURCES})
add_library(bartmachine_c_debug ${BARTMACHINE_C_DEBUG_SOURCES})
add_library(bartmachine_d_init ${BARTMACHINE_D_INIT_SOURCES})
add_library(bartmachine_e_gibbs_base ${BARTMACHINE_E_GIBBS_BASE_SOURCES})
add_library(bartmachine_f_gibbs_internal ${BARTMACHINE_F_GIBBS_INTERNAL_SOURCES})
add_library(bartmachine_g_mh ${BARTMACHINE_G_MH_SOURCES})
add_library(bartmachine_h_eval ${BARTMACHINE_H_EVAL_SOURCES})
add_library(bartmachine_i_prior_cov_spec ${BARTMACHINE_I_PRIOR_COV_SPEC_SOURCES})
add_library(bartmachine_regression ${BARTMACHINE_REGRESSION_SOURCES})
add_library(bartmachine_classification ${BARTMACHINE_CLASSIFICATION_SOURCES})

# Set up dependencies
target_link_libraries(bartmachine_b_hyperparams bartmachine_a_base)
target_link_libraries(stat_toolbox exact_port_mersenne_twister bartmachine_b_hyperparams)
target_link_libraries(bartmachine_tree_node stat_toolbox bartmachine_b_hyperparams)
target_link_libraries(bartmachine_a_base bartmachine_tree_node)
target_link_libraries(bartmachine_c_debug bartmachine_b_hyperparams)
target_link_libraries(bartmachine_d_init bartmachine_c_debug stat_toolbox)
target_link_libraries(bartmachine_e_gibbs_base bartmachine_d_init)
target_link_libraries(bartmachine_f_gibbs_internal bartmachine_e_gibbs_base stat_toolbox)
target_link_libraries(bartmachine_g_mh bartmachine_f_gibbs_internal)
target_link_libraries(bartmachine_h_eval bartmachine_g_mh)
target_link_libraries(bartmachine_i_prior_cov_spec bartmachine_h_eval)
target_link_libraries(bartmachine_regression bartmachine_i_prior_cov_spec)
target_link_libraries(bartmachine_classification bartmachine_i_prior_cov_spec stat_util)

# Include tests directory
include_directories(tests)

# Create test executables
add_executable(test_basic_compilation
    test_basic_compilation.cpp
)

# Task 1 tests
add_executable(test_task_1_2
    tests/test_task_1_2.cpp
)

add_executable(test_task_1_3
    tests/test_task_1_3.cpp
)

add_executable(test_task_1_4
    tests/test_task_1_4.cpp
)

# Task 2 tests
add_executable(test_task_2_1
    tests/test_task_2_1.cpp
)

add_executable(test_task_2_2
    tests/test_task_2_2.cpp
)

add_executable(test_task_2_3
    tests/test_task_2_3.cpp
)

add_executable(test_task_2_4
    tests/test_task_2_4.cpp
)

add_executable(test_task_2_5
    tests/test_task_2_5.cpp
)

add_executable(test_task_2_6
    tests/test_task_2_6.cpp
)

# Task 3 tests
add_executable(test_task_3_1
    tests/test_task_3_1.cpp
)

add_executable(test_task_3_2
    tests/test_task_3_2.cpp
)

add_executable(test_task_3_3
    tests/test_task_3_3.cpp
)

# Task 4 tests
add_executable(test_task_4_1
    tests/test_task_4_1.cpp
)

add_executable(test_task_4_2
    tests/test_task_4_2.cpp
)

add_executable(test_task_4_3
    tests/test_task_4_3.cpp
)

# Task 5 tests
add_executable(test_task_5_1
    tests/test_task_5_1.cpp
)

add_executable(test_task_5_2
    tests/test_task_5_2.cpp
)

add_executable(test_task_5_3
    tests/test_task_5_3.cpp
)

add_executable(test_task_5_4
    tests/test_task_5_4.cpp
)

add_executable(test_task_5_5
    tests/test_task_5_5.cpp
)

add_executable(test_task_5_6
    tests/test_task_5_6.cpp
)

# Task 6 tests
add_executable(test_task_6_1
    tests/test_task_6_1.cpp
)

add_executable(test_task_6_2
    tests/test_task_6_2.cpp
)

# Link the libraries to the tests
target_link_libraries(test_basic_compilation exact_port_mersenne_twister)
target_link_libraries(test_task_1_2 exact_port_mersenne_twister)
target_link_libraries(test_task_1_3 exact_port_mersenne_twister)
target_link_libraries(test_task_1_4 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_2_1 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_2_2 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_2_3 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_2_4 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_2_5 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_2_6 stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_3_1 stat_toolbox exact_port_mersenne_twister bartmachine_a_base bartmachine_b_hyperparams)
target_link_libraries(test_task_3_2 stat_toolbox exact_port_mersenne_twister bartmachine_a_base bartmachine_b_hyperparams)
target_link_libraries(test_task_3_3 stat_toolbox exact_port_mersenne_twister bartmachine_a_base bartmachine_b_hyperparams)
target_link_libraries(test_task_4_1 bartmachine_tree_node stat_toolbox exact_port_mersenne_twister bartmachine_a_base bartmachine_b_hyperparams)
target_link_libraries(test_task_4_2 bartmachine_tree_node stat_toolbox exact_port_mersenne_twister bartmachine_a_base bartmachine_b_hyperparams)
target_link_libraries(test_task_4_3 bartmachine_tree_node stat_toolbox exact_port_mersenne_twister bartmachine_a_base bartmachine_b_hyperparams)
target_link_libraries(test_task_5_1 bartmachine_a_base bartmachine_b_hyperparams stat_toolbox exact_port_mersenne_twister)
target_link_libraries(test_task_5_2
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    exact_port_mersenne_twister
)

target_link_libraries(test_task_5_3
    bartmachine_g_mh
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    exact_port_mersenne_twister
)

target_link_libraries(test_task_5_4
    bartmachine_g_mh
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    exact_port_mersenne_twister
)

target_link_libraries(test_task_5_5
    bartmachine_g_mh
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    exact_port_mersenne_twister
)

target_link_libraries(test_task_5_6
    bartmachine_g_mh
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    exact_port_mersenne_twister
)

target_link_libraries(test_task_6_1
    bartmachine_i_prior_cov_spec
    bartmachine_h_eval
    bartmachine_g_mh
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    exact_port_mersenne_twister
)

target_link_libraries(test_task_6_2
    bartmachine_regression
    bartmachine_classification
    bartmachine_i_prior_cov_spec
    bartmachine_h_eval
    bartmachine_g_mh
    bartmachine_f_gibbs_internal
    bartmachine_e_gibbs_base
    bartmachine_d_init
    bartmachine_c_debug
    bartmachine_b_hyperparams
    bartmachine_a_base
    bartmachine_tree_node
    stat_toolbox
    stat_util
    exact_port_mersenne_twister
)

# Enable testing
enable_testing()

# Add basic compilation test
add_test(NAME BasicCompilation COMMAND test_basic_compilation)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
