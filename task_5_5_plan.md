# Task 5.5 Implementation Plan: Metropolis-Hastings - Change Operation

## Overview
Task 5.5 focuses on implementing the Metropolis-Hastings change operation for the BART model. This operation modifies the splitting rule of an internal node without changing the tree structure. The change operation is one of the three possible steps (grow, prune, change) that can be taken during the Metropolis-Hastings tree proposal.

## Methods to Implement

### 1. doMHChangeAndCalcLnR()
This method performs the change step on a tree and returns the log Metropolis-Hastings ratio. It:
- Selects a node suitable for changing (using pickPruneNodeOrChangeNode())
- Changes the splitting rule of the node
- Calculates the log likelihood ratio for the change
- Returns the log Metropolis-Hastings ratio

### 2. calcLnLikRatioChange()
This method calculates the log likelihood ratio for a change step. It compares the likelihood of the data under the original tree and the proposal tree with the changed splitting rule.

### 3. Related Helper Methods
Any additional helper methods needed for the change operation.

## Implementation Strategy

1. **Understand the Java Implementation**:
   - Study the original Java code for doMHChangeAndCalcLnR() and calcLnLikRatioChange()
   - Identify any dependencies or helper methods

2. **Implement doMHChangeAndCalcLnR()**:
   - Port the method line-by-line from Java to C++
   - Ensure proper handling of tree node selection and modification
   - Calculate the log Metropolis-Hastings ratio correctly

3. **Implement calcLnLikRatioChange()**:
   - Port the method line-by-line from Java to C++
   - Ensure correct calculation of the log likelihood ratio

4. **Implement Any Helper Methods**:
   - Identify and implement any additional helper methods needed

5. **Testing**:
   - Create test cases for each method
   - Verify that the C++ implementation produces the same results as the Java implementation
   - Test edge cases and error handling

## Validation Approach

1. **Unit Tests**:
   - Test each method individually with controlled inputs
   - Verify outputs against expected values

2. **Integration Tests**:
   - Test the change operation as part of the full Metropolis-Hastings algorithm
   - Verify that the change operation works correctly in conjunction with grow and prune operations

3. **Numerical Equivalence**:
   - Verify that the C++ implementation produces numerically equivalent results to the Java implementation
   - Use the same random seed to ensure deterministic behavior

## Dependencies

- Task 5.3 (Grow Operation) and Task 5.4 (Prune Operation) must be completed first
- The pickPruneNodeOrChangeNode() method from Task 5.4 will be reused for selecting nodes to change

## Expected Challenges

1. **Tree Manipulation**:
   - Ensuring that the tree structure is correctly maintained during the change operation
   - Properly propagating data through the tree after changing a splitting rule

2. **Numerical Precision**:
   - Ensuring that the log likelihood ratio calculations match the Java implementation exactly
   - Handling potential numerical precision issues

3. **RNG Consistency**:
   - Ensuring that random number generation is consistent with the Java implementation
   - Maintaining deterministic behavior for testing

## Timeline

1. Study and understand the Java implementation: 1 hour
2. Implement doMHChangeAndCalcLnR(): 2 hours
3. Implement calcLnLikRatioChange(): 2 hours
4. Implement helper methods: 1 hour
5. Testing and validation: 2 hours
6. Documentation and cleanup: 1 hour

Total estimated time: 9 hours
