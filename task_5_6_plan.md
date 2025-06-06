# Task 5.6 Implementation Plan: Metropolis-Hastings - Integration

## Overview
Task 5.6 focuses on completing the Metropolis-Hastings algorithm implementation for the BART model. We have already implemented the grow, prune, and change operations. Now we need to ensure that the overall Metropolis-Hastings algorithm is correctly implemented, including the step selection and integration of the three operations.

## Methods to Review and Refine

### 1. metroHastingsPosteriorTreeSpaceIteration()
This method performs one Metropolis-Hastings step for one tree. It:
- Selects a proposal step (grow, prune, or change)
- Performs the step
- Decides whether to accept or reject the proposal
- Returns the next tree (either the proposal tree if accepted or the original tree if rejected)

### 2. randomlyPickAmongTheProposalSteps()
This method randomly chooses among the valid tree proposal steps from a multinomial distribution. It:
- Draws a random number
- Compares it to the probabilities of each step
- Returns the selected step

### 3. Any Other Related Methods
Any additional methods needed for the complete Metropolis-Hastings algorithm.

## Implementation Strategy

1. **Review the Current Implementation**:
   - Examine the current C++ implementation of metroHastingsPosteriorTreeSpaceIteration() and randomlyPickAmongTheProposalSteps()
   - Compare with the original Java implementation
   - Identify any discrepancies or missing functionality

2. **Refine metroHastingsPosteriorTreeSpaceIteration()**:
   - Ensure that the method correctly selects a proposal step
   - Verify that the method correctly performs the step
   - Confirm that the method correctly decides whether to accept or reject the proposal
   - Make any necessary adjustments to match the Java implementation exactly

3. **Refine randomlyPickAmongTheProposalSteps()**:
   - Ensure that the method correctly selects a step based on the probabilities
   - Verify that the method uses the same random number generation as the Java implementation
   - Make any necessary adjustments to match the Java implementation exactly

4. **Implement Any Missing Methods**:
   - Identify and implement any additional methods needed for the complete Metropolis-Hastings algorithm

5. **Testing**:
   - Create test cases for each method
   - Verify that the C++ implementation produces the same results as the Java implementation
   - Test edge cases and error handling

## Validation Approach

1. **Unit Tests**:
   - Test each method individually with controlled inputs
   - Verify outputs against expected values

2. **Integration Tests**:
   - Test the complete Metropolis-Hastings algorithm with all three operations (grow, prune, change)
   - Verify that the algorithm produces the expected results

3. **Numerical Equivalence**:
   - Verify that the C++ implementation produces numerically equivalent results to the Java implementation
   - Use the same random seed to ensure deterministic behavior

## Dependencies

- Task 5.3 (Grow Operation), Task 5.4 (Prune Operation), and Task 5.5 (Change Operation) must be completed first
- The doMHGrowAndCalcLnR(), doMHPruneAndCalcLnR(), and doMHChangeAndCalcLnR() methods from previous tasks will be used

## Expected Challenges

1. **Integration of Operations**:
   - Ensuring that the three operations (grow, prune, change) work together correctly
   - Maintaining the correct probabilities for each operation

2. **RNG Consistency**:
   - Ensuring that random number generation is consistent with the Java implementation
   - Maintaining deterministic behavior for testing

3. **Edge Cases**:
   - Handling edge cases such as stumps (trees with only a root node)
   - Ensuring that the algorithm behaves correctly in all scenarios

## Timeline

1. Review the current implementation: 1 hour
2. Refine metroHastingsPosteriorTreeSpaceIteration(): 2 hours
3. Refine randomlyPickAmongTheProposalSteps(): 1 hour
4. Implement any missing methods: 1 hour
5. Testing and validation: 2 hours
6. Documentation and cleanup: 1 hour

Total estimated time: 8 hours
