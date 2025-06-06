#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

/**
 * Task 2.4 Validation: StatToolbox - Min/Max Methods
 * 
 * This test validates that:
 * 1. The sample_minimum method works correctly for std::vector<double>
 * 2. The sample_minimum method works correctly for double* (TDoubleArrayList equivalent)
 * 3. The sample_minimum method works correctly for std::vector<int>
 * 4. The sample_maximum method works correctly for std::vector<double>
 * 5. The sample_maximum method works correctly for double* (TDoubleArrayList equivalent)
 * 6. The sample_maximum method works correctly for std::vector<int>
 * 7. All methods produce numerically equivalent results to the Java implementation
 * 8. Edge cases are handled properly
 */

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    std::cout << "=== Task 2.4 Validation: StatToolbox - Min/Max Methods ===" << std::endl;
    
    try {
        // Test 1: sample_minimum for std::vector<double>
        std::cout << "Test 1: sample_minimum for std::vector<double>... " << std::flush;
        std::vector<double> double_vec = {5.0, 1.0, 3.0, 2.0, 4.0};
        double expected_min = 1.0;
        double actual_min = StatToolbox::sample_minimum(double_vec);
        
        if (approx_equal(actual_min, expected_min)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_min << ", got " << actual_min << ")" << std::endl;
            return 1;
        }
        
        // Test 2: sample_minimum for double* (TDoubleArrayList equivalent)
        std::cout << "Test 2: sample_minimum for double* (TDoubleArrayList equivalent)... " << std::flush;
        double double_arr[] = {5.0, 1.0, 3.0, 2.0, 4.0};
        expected_min = 1.0;
        actual_min = StatToolbox::sample_minimum(double_arr, 5);
        
        if (approx_equal(actual_min, expected_min)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_min << ", got " << actual_min << ")" << std::endl;
            return 1;
        }
        
        // Test 3: sample_minimum for std::vector<int>
        std::cout << "Test 3: sample_minimum for std::vector<int>... " << std::flush;
        std::vector<int> int_vec = {5, 1, 3, 2, 4};
        int expected_int_min = 1;
        double actual_int_min = StatToolbox::sample_minimum(int_vec);
        
        if (approx_equal(actual_int_min, expected_int_min)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_int_min << ", got " << actual_int_min << ")" << std::endl;
            return 1;
        }
        
        // Test 4: sample_maximum for std::vector<double>
        std::cout << "Test 4: sample_maximum for std::vector<double>... " << std::flush;
        expected_min = 5.0;
        actual_min = StatToolbox::sample_maximum(double_vec);
        
        if (approx_equal(actual_min, expected_min)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_min << ", got " << actual_min << ")" << std::endl;
            return 1;
        }
        
        // Test 5: sample_maximum for double* (TDoubleArrayList equivalent)
        std::cout << "Test 5: sample_maximum for double* (TDoubleArrayList equivalent)... " << std::flush;
        expected_min = 5.0;
        actual_min = StatToolbox::sample_maximum(double_arr, 5);
        
        if (approx_equal(actual_min, expected_min)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_min << ", got " << actual_min << ")" << std::endl;
            return 1;
        }
        
        // Test 6: sample_maximum for std::vector<int>
        std::cout << "Test 6: sample_maximum for std::vector<int>... " << std::flush;
        int expected_int_max = 5;
        double actual_int_max = StatToolbox::sample_maximum(int_vec);
        
        if (approx_equal(actual_int_max, expected_int_max)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_int_max << ", got " << actual_int_max << ")" << std::endl;
            return 1;
        }
        
        // Test 7: Numerical equivalence with more complex data
        std::cout << "Test 7: Numerical equivalence with more complex data... " << std::flush;
        
        // These values and expected results are derived from the Java implementation
        std::vector<double> complex_data = {
            1.23456, 2.34567, 3.45678, 4.56789, 5.67890,
            -1.23456, -2.34567, -3.45678, -4.56789, -5.67890
        };
        
        // Expected results from Java
        double expected_complex_min = -5.67890;
        double expected_complex_max = 5.67890;
        
        double actual_complex_min = StatToolbox::sample_minimum(complex_data);
        double actual_complex_max = StatToolbox::sample_maximum(complex_data);
        
        if (approx_equal(actual_complex_min, expected_complex_min) && 
            approx_equal(actual_complex_max, expected_complex_max)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            if (!approx_equal(actual_complex_min, expected_complex_min)) {
                std::cout << "  Min: expected " << expected_complex_min << ", got " << actual_complex_min << std::endl;
            }
            if (!approx_equal(actual_complex_max, expected_complex_max)) {
                std::cout << "  Max: expected " << expected_complex_max << ", got " << actual_complex_max << std::endl;
            }
            return 1;
        }
        
        // Test 8: Edge case - empty vector
        std::cout << "Test 8: Edge case - empty vector... " << std::flush;
        std::vector<double> empty_vec;
        
        // In Java, this would result in ILLEGAL_FLAG
        double actual_empty_min = StatToolbox::sample_minimum(empty_vec);
        double actual_empty_max = StatToolbox::sample_maximum(empty_vec);
        
        if (approx_equal(actual_empty_min, StatToolbox::ILLEGAL_FLAG) && 
            approx_equal(actual_empty_max, StatToolbox::ILLEGAL_FLAG)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            if (!approx_equal(actual_empty_min, StatToolbox::ILLEGAL_FLAG)) {
                std::cout << "  Min: expected " << StatToolbox::ILLEGAL_FLAG << ", got " << actual_empty_min << std::endl;
            }
            if (!approx_equal(actual_empty_max, StatToolbox::ILLEGAL_FLAG)) {
                std::cout << "  Max: expected " << StatToolbox::ILLEGAL_FLAG << ", got " << actual_empty_max << std::endl;
            }
            return 1;
        }
        
        // Test 9: Edge case - single element array
        std::cout << "Test 9: Edge case - single element array... " << std::flush;
        std::vector<double> single_vec = {42.0};
        double expected_single = 42.0;
        double actual_single_min = StatToolbox::sample_minimum(single_vec);
        double actual_single_max = StatToolbox::sample_maximum(single_vec);
        
        if (approx_equal(actual_single_min, expected_single) && 
            approx_equal(actual_single_max, expected_single)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            if (!approx_equal(actual_single_min, expected_single)) {
                std::cout << "  Min: expected " << expected_single << ", got " << actual_single_min << std::endl;
            }
            if (!approx_equal(actual_single_max, expected_single)) {
                std::cout << "  Max: expected " << expected_single << ", got " << actual_single_max << std::endl;
            }
            return 1;
        }
        
        std::cout << std::endl << "=== Task 2.4 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ sample_minimum for std::vector<double> implemented correctly" << std::endl;
        std::cout << "✓ sample_minimum for double* (TDoubleArrayList equivalent) implemented correctly" << std::endl;
        std::cout << "✓ sample_minimum for std::vector<int> implemented correctly" << std::endl;
        std::cout << "✓ sample_maximum for std::vector<double> implemented correctly" << std::endl;
        std::cout << "✓ sample_maximum for double* (TDoubleArrayList equivalent) implemented correctly" << std::endl;
        std::cout << "✓ sample_maximum for std::vector<int> implemented correctly" << std::endl;
        std::cout << "✓ All methods produce numerically equivalent results to the Java implementation" << std::endl;
        std::cout << "✓ Edge cases are handled properly" << std::endl;
        std::cout << "✓ Ready for Task 2.5 (StatToolbox - Variance Methods)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
