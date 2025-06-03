cmake_minimum_required(VERSION 3.10)
project(bartMachine_port)

# Repository locations
# Original Repository: /Users/mark/Documents/Cline/bartMachine
# Port Repository: /Users/mark/Documents/Cline/bartMachine_port
# Original Java Source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine
# Original R Source: /Users/mark/Documents/Cline/bartMachine/bartMachine

# Set C++ standard
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

# Source files
set(MERSENNE_SOURCES
    src/cpp/exact_port_mersenne_twister.cpp
)

set(STAT_TOOLBOX_SOURCES
    src/cpp/stat_toolbox.cpp
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

# Create library for MersenneTwister
add_library(exact_port_mersenne_twister ${MERSENNE_SOURCES})

# Create libraries
add_library(bartmachine_a_base ${BARTMACHINE_BASE_SOURCES})
add_library(bartmachine_b_hyperparams ${BARTMACHINE_HYPERPARAMS_SOURCES})
add_library(stat_toolbox ${STAT_TOOLBOX_SOURCES})
add_library(bartmachine_tree_node ${BARTMACHINE_TREE_NODE_SOURCES})

# Set up dependencies
target_link_libraries(bartmachine_b_hyperparams bartmachine_a_base)
target_link_libraries(stat_toolbox exact_port_mersenne_twister bartmachine_b_hyperparams)
target_link_libraries(bartmachine_tree_node stat_toolbox bartmachine_b_hyperparams)
target_link_libraries(bartmachine_a_base bartmachine_tree_node)

# Create test executables
add_executable(test_basic_compilation
    test_basic_compilation.cpp
)

add_executable(test_task_1_2
    test_task_1_2.cpp
)

add_executable(test_task_1_3
    test_task_1_3.cpp
)

add_executable(test_task_1_4
    test_task_1_4.cpp
)

add_executable(test_task_2_1
    test_task_2_1.cpp
)

add_executable(test_task_2_2
    test_task_2_2.cpp
)

add_executable(test_task_2_3
    test_task_2_3.cpp
)

add_executable(test_task_2_4
    test_task_2_4.cpp
)

add_executable(test_task_2_5
    test_task_2_5.cpp
)

add_executable(test_task_2_6
    test_task_2_6.cpp
)

add_executable(test_task_3_1
    test_task_3_1.cpp
)

add_executable(test_task_3_2
    test_task_3_2.cpp
)

add_executable(test_task_3_3
    test_task_3_3.cpp
)

add_executable(test_task_4_1
    test_task_4_1.cpp
)

add_executable(test_task_4_2
    test_task_4_2.cpp
)

add_executable(test_task_4_3
    test_task_4_3.cpp
)

add_executable(test_task_5_1
    test_task_5_1.cpp
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

# Enable testing
enable_testing()

# Add basic compilation test
add_test(NAME BasicCompilation COMMAND test_basic_compilation)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
