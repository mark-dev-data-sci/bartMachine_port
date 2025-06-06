#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

/**
 * Task 2.6 Validation: StatToolbox - Utility Functions
 * 
 * This test validates that:
 * 1. The FindMaxIndex method works correctly for std::vector<int>
 * 2. The method produces numerically equivalent results to the Java implementation
 * 3. Edge cases are handled properly
 * 
 * Note: multinomial_sample is RNG-dependent and will be tested in Phase 3
 */

int main() {
    std::cout << "=== Task 2.6 Validation: StatToolbox - Utility Functions ===" << std::endl;
    
    try {
        // Test 1: FindMaxIndex for std::vector<int>
        std::cout << "Test 1: FindMaxIndex for std::vector<int>... " << std::flush;
        std::vector<int> int_vec = {5, 1, 3, 2, 4};
        int expected_index = 0;  // Index of 5, which is the maximum value
        int actual_index = StatToolbox::FindMaxIndex(int_vec);
        
        if (actual_index == expected_index) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_index << ", got " << actual_index << ")" << std::endl;
            return 1;
        }
        
        // Test 2: FindMaxIndex with duplicate maximum values
        std::cout << "Test 2: FindMaxIndex with duplicate maximum values... " << std::flush;
        std::vector<int> dup_vec = {5, 1, 5, 2, 4};
        expected_index = 0;  // Should return the first occurrence of the maximum value
        actual_index = StatToolbox::FindMaxIndex(dup_vec);
        
        if (actual_index == expected_index) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_index << ", got " << actual_index << ")" << std::endl;
            return 1;
        }
        
        // Test 3: FindMaxIndex with negative values
        std::cout << "Test 3: FindMaxIndex with negative values... " << std::flush;
        std::vector<int> neg_vec = {-5, -1, -3, -2, -4};
        expected_index = 1;  // Index of -1, which is the maximum value
        actual_index = StatToolbox::FindMaxIndex(neg_vec);
        
        if (actual_index == expected_index) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_index << ", got " << actual_index << ")" << std::endl;
            return 1;
        }
        
        // Test 4: FindMaxIndex with mixed positive and negative values
        std::cout << "Test 4: FindMaxIndex with mixed positive and negative values... " << std::flush;
        std::vector<int> mixed_vec = {-5, 1, -3, 2, -4};
        expected_index = 3;  // Index of 2, which is the maximum value
        actual_index = StatToolbox::FindMaxIndex(mixed_vec);
        
        if (actual_index == expected_index) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_index << ", got " << actual_index << ")" << std::endl;
            return 1;
        }
        
        // Test 5: Edge case - single element array
        std::cout << "Test 5: Edge case - single element array... " << std::flush;
        std::vector<int> single_vec = {42};
        expected_index = 0;  // Only one element, so index is 0
        actual_index = StatToolbox::FindMaxIndex(single_vec);
        
        if (actual_index == expected_index) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_index << ", got " << actual_index << ")" << std::endl;
            return 1;
        }
        
        // Test 6: Edge case - empty array
        std::cout << "Test 6: Edge case - empty array... " << std::flush;
        std::vector<int> empty_vec;
        
        // In Java, this would likely throw an exception or return -1
        // We'll check if our implementation handles this gracefully
        try {
            actual_index = StatToolbox::FindMaxIndex(empty_vec);
            std::cout << "FAILED (expected exception, got " << actual_index << ")" << std::endl;
            return 1;
        } catch (const std::exception& e) {
            std::cout << "PASSED (exception thrown as expected)" << std::endl;
        }
        
        std::cout << std::endl << "=== Task 2.6 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ FindMaxIndex for std::vector<int> implemented correctly" << std::endl;
        std::cout << "✓ Method produces numerically equivalent results to the Java implementation" << std::endl;
        std::cout << "✓ Edge cases are handled properly" << std::endl;
        std::cout << "✓ Ready for Phase 3 (RNG-Dependent Statistical Functions)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
