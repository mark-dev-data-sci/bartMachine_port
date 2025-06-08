# Task 7.1: Random Number Generation Review - File Locations

## Overview
This document provides the file locations for the random number generation components in both the original Java implementation and the C++ port. These files will be the focus of the review in Task 7.1.

## Primary Files for Comparison

### ExactPortMersenneTwister Implementation
- **Java File**: `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/ExactPortMersenneTwister.java`
- **C++ File**: `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/exact_port_mersenne_twister.cpp`
- **C++ Header**: `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/exact_port_mersenne_twister.h`

### StatToolbox RNG Interface
- **Java File**: `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/StatToolbox.java`
- **C++ File**: `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/stat_toolbox.cpp`
- **C++ Header**: `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/stat_toolbox.h`

## Files with RNG-Dependent Code

### bartMachine Base Classes
- **Java Files**:
  - `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine.java`
  - `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachineTreeNode.java`
- **C++ Files**:
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/bartmachine_a_base.cpp`
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/bartmachine_tree_node.cpp`
- **C++ Headers**:
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/bartmachine_a_base.h`
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/bartmachine_tree_node.h`

### MCMC Engine
- **Java Files**:
  - `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachineRegressionGibbs.java`
  - `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachineClassificationGibbs.java`
- **C++ Files**:
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/bartmachine_e_gibbs_base.cpp`
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/bartmachine_f_gibbs_internal.cpp`
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/bartmachine_g_mh.cpp`
- **C++ Headers**:
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/bartmachine_e_gibbs_base.h`
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/bartmachine_f_gibbs_internal.h`
  - `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/bartmachine_g_mh.h`

## Pre-computed Random Arrays

### Initialization in Java
- **Java File**: `/Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachineRegressionGibbs.java`
- **Method**: `initializeWithSeed(long seed)`

### Initialization in C++
- **C++ File**: `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/bartmachine_d_init.cpp`
- **C++ Header**: `/Users/mark/Documents/Cline/bartMachine_port/src/cpp/include/bartmachine_d_init.h`
- **Method**: `initializeWithSeed(long seed)`

## Test Files

### Java Tests
- `/Users/mark/Documents/Cline/bartMachine/test/bartMachine/ExactPortMersenneTwisterTest.java`
- `/Users/mark/Documents/Cline/bartMachine/test/bartMachine/StatToolboxTest.java`

### C++ Tests
- `/Users/mark/Documents/Cline/bartMachine_port/tests/test_task_1_2.cpp` (MersenneTwister tests)
- `/Users/mark/Documents/Cline/bartMachine_port/tests/test_task_1_3.cpp` (StatToolbox RNG tests)

## Note on File Paths
The file paths provided are based on the repository locations specified in CONSTRAINTS.md. If the repositories are cloned to different locations, the paths will need to be adjusted accordingly.
