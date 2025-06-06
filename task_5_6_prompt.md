# Task 5.6: Metropolis-Hastings - Integration

## Objective
Complete the Metropolis-Hastings algorithm implementation for the BART model in C++, ensuring exact equivalence with the Java implementation.

## Background
The Metropolis-Hastings algorithm is used to sample from the posterior distribution of trees in the BART model. We have already implemented the grow, prune, and change operations. Now we need to ensure that the overall Metropolis-Hastings algorithm is correctly implemented, including the step selection and integration of the three operations.

## Requirements

### Methods to Review and Refine

1. **metroHastingsPosteriorTreeSpaceIteration()**
   - This method performs one Metropolis-Hastings step for one tree.
   - It selects a proposal step (grow, prune, or change), performs the step, and decides whether to accept or reject the proposal.
   - The method signature should match the Java implementation:
     ```java
     protected bartMachineTreeNode metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode T_i, int tree_num, boolean[][] accept_reject_mh, char[][] accept_reject_mh_steps)
     ```

2. **randomlyPickAmongTheProposalSteps()**
   - This method randomly chooses among the valid tree proposal steps from a multinomial distribution.
   - The method signature should match the Java implementation:
     ```java
     protected Steps randomlyPickAmongTheProposalSteps()
     ```

3. **Any Other Related Methods**
   - Implement or refine any additional methods needed for the complete Metropolis-Hastings algorithm.

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
   - Test the complete Metropolis-Hastings algorithm with all three operations (grow, prune, change).
   - Verify that the algorithm produces the expected results.

3. **Numerical Equivalence**
   - Verify that the C++ implementation produces numerically equivalent results to the Java implementation.
   - Use the same random seed to ensure deterministic behavior.

## Deliverables

1. Refined C++ implementation of metroHastingsPosteriorTreeSpaceIteration() in bartmachine_g_mh.cpp
2. Refined C++ implementation of randomlyPickAmongTheProposalSteps() in bartmachine_g_mh.cpp
3. Implementation of any other related methods
4. Unit tests for all implemented methods
5. Documentation of the implementation

## References

1. Original Java implementation in bartMachine_g_mh.java
2. BART paper: Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. The Annals of Applied Statistics, 4(1), 266-298.
3. Metropolis-Hastings algorithm: Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1), 97-109.
