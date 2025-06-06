#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

/**
 * Task 2.3 Validation: StatToolbox - sample_median Method
 * 
 * This test validates that:
 * 1. The sample_median method works correctly for odd-length arrays
 * 2. The sample_median method works correctly for even-length arrays
 * 3. The method produces numerically equivalent results to the Java implementation
 * 4. Edge cases are handled properly
 */

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    std::cout << "=== Task 2.3 Validation: StatToolbox - sample_median Method ===" << std::endl;
    
    try {
        // Test 1: sample_median for odd-length array
        std::cout << "Test 1: sample_median for odd-length array... " << std::flush;
        std::vector<double> odd_vec = {5.0, 1.0, 3.0, 2.0, 4.0};
        double expected_median = 3.0;
        double actual_median = StatToolbox::sample_median(odd_vec);
        
        if (approx_equal(actual_median, expected_median)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_median << ", got " << actual_median << ")" << std::endl;
            return 1;
        }
        
        // Test 2: sample_median for even-length array
        std::cout << "Test 2: sample_median for even-length array... " << std::flush;
        std::vector<double> even_vec = {5.0, 1.0, 3.0, 2.0, 4.0, 6.0};
        expected_median = 3.5;  // (3 + 4) / 2
        actual_median = StatToolbox::sample_median(even_vec);
        
        if (approx_equal(actual_median, expected_median)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_median << ", got " << actual_median << ")" << std::endl;
            return 1;
        }
        
        // Test 3: Numerical equivalence with more complex data
        std::cout << "Test 3: Numerical equivalence with more complex data... " << std::flush;
        
        // These values and expected results are derived from the Java implementation
        std::vector<double> complex_data = {
            1.23456, 2.34567, 3.45678, 4.56789, 5.67890,
            -1.23456, -2.34567, -3.45678, -4.56789, -5.67890
        };
        
        // Expected result from Java: 0.0
        expected_median = 0.0;
        actual_median = StatToolbox::sample_median(complex_data);
        
        if (approx_equal(actual_median, expected_median)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_median << ", got " << actual_median << ")" << std::endl;
            return 1;
        }
        
        // Test 4: Edge case - single element array
        std::cout << "Test 4: Edge case - single element array... " << std::flush;
        std::vector<double> single_vec = {42.0};
        expected_median = 42.0;
        actual_median = StatToolbox::sample_median(single_vec);
        
        if (approx_equal(actual_median, expected_median)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_median << ", got " << actual_median << ")" << std::endl;
            return 1;
        }
        
        // Test 5: Edge case - empty array
        std::cout << "Test 5: Edge case - empty array... " << std::flush;
        std::vector<double> empty_vec;
        actual_median = StatToolbox::sample_median(empty_vec);
        
        if (std::isnan(actual_median)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected NaN, got " << actual_median << ")" << std::endl;
            return 1;
        }
        
        std::cout << std::endl << "=== Task 2.3 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ sample_median for odd-length arrays implemented correctly" << std::endl;
        std::cout << "✓ sample_median for even-length arrays implemented correctly" << std::endl;
        std::cout << "✓ Method produces numerically equivalent results to the Java implementation" << std::endl;
        std::cout << "✓ Edge cases are handled properly" << std::endl;
        std::cout << "✓ Ready for Task 2.4 (StatToolbox - Min/Max Methods)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
