# Task 5.5: Metropolis-Hastings - Change Operation

## Objective
Implement the Metropolis-Hastings change operation for the BART model in C++, ensuring exact equivalence with the Java implementation.

## Background
The Metropolis-Hastings algorithm is used to sample from the posterior distribution of trees in the BART model. The change operation is one of the three possible steps (grow, prune, change) that can be taken during the Metropolis-Hastings tree proposal. The change operation modifies the splitting rule of an internal node without changing the tree structure.

## Requirements

### Methods to Implement

1. **doMHChangeAndCalcLnR()**
   - This method performs the change step on a tree and returns the log Metropolis-Hastings ratio.
   - It should select a node suitable for changing, change the splitting rule, and calculate the log Metropolis-Hastings ratio.
   - The method signature should match the Java implementation:
     ```java
     protected double doMHChangeAndCalcLnR(bartMachineTreeNode T_i, bartMachineTreeNode T_star)
     ```

2. **calcLnLikRatioChange()**
   - This method calculates the log likelihood ratio for a change step.
   - It compares the likelihood of the data under the original tree and the proposal tree with the changed splitting rule.
   - The method signature should match the Java implementation:
     ```java
     protected double calcLnLikRatioChange(bartMachineTreeNode eta, bartMachineTreeNode eta_star)
     ```

3. **Any Related Helper Methods**
   - Implement any additional helper methods needed for the change operation.

### Implementation Guidelines

1. **Exact Port**
   - The C++ implementation should be an exact port of the Java implementation.
   - Maintain the same logic, control flow, and variable names where possible.
   - Ensure numerical equivalence with the Java implementation.

2. **RNG Consistency**
   - Ensure that random number generation is consistent with the Java implementation.
   - Use the StatToolbox.rand() method for random number generation.

3. **Error Handling**
   - Implement the same error handling as the Java implementation.
   - Use appropriate C++ error handling mechanisms (e.g., exceptions, error codes).

4. **Documentation**
   - Include comments explaining the purpose and behavior of each method.
   - Document any differences between the Java and C++ implementations (if any).

## Validation

1. **Unit Tests**
   - Create unit tests for each method to verify correct behavior.
   - Test edge cases and error handling.

2. **Integration Tests**
   - Test the change operation as part of the full Metropolis-Hastings algorithm.
   - Verify that the change operation works correctly in conjunction with grow and prune operations.

3. **Numerical Equivalence**
   - Verify that the C++ implementation produces numerically equivalent results to the Java implementation.
   - Use the same random seed to ensure deterministic behavior.

## Deliverables

1. C++ implementation of doMHChangeAndCalcLnR() in bartmachine_g_mh.cpp
2. C++ implementation of calcLnLikRatioChange() in bartmachine_g_mh.cpp
3. Implementation of any related helper methods
4. Unit tests for all implemented methods
5. Documentation of the implementation

## References

1. Original Java implementation in bartMachine_g_mh.java
2. BART paper: Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. The Annals of Applied Statistics, 4(1), 266-298.
3. Metropolis-Hastings algorithm: Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1), 97-109.
