# Project Plan: Investigating Prediction Discrepancies in bartMachine Python Port

## Overview

Our testing has revealed a consistent negative correlation (around -0.3) between predictions from the R and Python implementations of bartMachine, despite convergence in variable importance with longer MCMC chains. This project plan outlines a systematic approach to identify and resolve these discrepancies.

## Source Code Locations

- **Python Implementation**: Current project directory (`/Users/mark/Documents/Cline/bartMachine_port/bartmachine_py`)
  - Key files: `bartmachine/bartMachine.py`, `bartmachine/zzz.py` (Java interface), `bartmachine/bart_package_predicts.py`
  - Note: Previous files like java_bridge.py, arrays.py, and bart_arrays.py are now in the redundant_files directory
- **Original R Implementation**: Local clone of the original bartMachine repository (`/Users/mark/Documents/Cline/bartMachine`)
  - Key files: `R/bart_package_builders.R`, `R/bart_package_predicts.R`, `R/rJava_interface.R`

## Phase 1: Data Flow Analysis

### 1.1 Input Data Handling
- Compare how input data is processed in R vs. Python
- Verify matrix orientation (row vs. column major)
- Check for any transposition issues in data passing to Java
- Confirm data types and scaling are consistent

### 1.2 Parameter Passing
- Trace how model parameters are passed to Java in both implementations
- Verify parameter types and values at Java interface
- Check for any parameter transformation differences

### 1.3 Random Number Generation
- Compare random number generation and seeding in both implementations
- Verify that the same seed produces the same sequence in both environments
- Check how random numbers are used in the tree building process

## Phase 2: Algorithm Implementation Analysis

### 2.1 Tree Building Process
- Trace the tree building process step by step in both implementations
- Compare splitting criteria and how splits are determined
- Analyze how leaf node values are calculated

### 2.2 Prediction Generation
- Compare how predictions are generated from trees
- Verify how predictions are aggregated across trees
- Check for any sign inversions or scaling differences

### 2.3 MCMC Chain Analysis
- Compare how MCMC chains are processed
- Verify burn-in handling and posterior sampling
- Check for differences in how trees are updated during MCMC

## Phase 3: Debugging and Verification

### 3.1 Instrumented Runs
- Add detailed logging to both implementations
- Compare intermediate values at each step
- Identify the exact point where divergence occurs

### 3.2 Single Tree Analysis
- Build models with a single tree in both implementations
- Compare the structure and predictions from this single tree
- Isolate whether the issue is in tree building or aggregation

### 3.3 Controlled Experiments
- Create synthetic datasets with known patterns
- Test with simplified models (e.g., constant leaf values)
- Verify behavior with edge cases

## Phase 4: Solution Implementation

### 4.1 Fix Implementation
- Implement fixes based on identified issues
- Ensure compatibility with existing code
- Maintain Pythonic interface while ensuring numerical equivalence

### 4.2 Comprehensive Testing
- Test with multiple datasets
- Verify numerical equivalence across different parameter settings
- Ensure performance is maintained

### 4.3 Documentation
- Document the identified issues and solutions
- Update API documentation to reflect any changes
- Add notes about R-Python equivalence

## Timeline

- Phase 1: 1-2 days
- Phase 2: 2-3 days
- Phase 3: 2-3 days
- Phase 4: 2-3 days

Total estimated time: 7-11 days

## Resources Required

- Access to both R and Python environments
- Source code for both implementations (already available at the locations specified above)
- Test datasets
- Debugging tools for Java, R, and Python
