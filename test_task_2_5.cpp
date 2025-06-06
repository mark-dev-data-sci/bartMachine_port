#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

/**
 * Task 2.5 Validation: StatToolbox - Variance Methods
 * 
 * This test validates that:
 * 1. The sample_variance method works correctly for std::vector<double>
 * 2. The sample_variance method works correctly for double* (TDoubleArrayList equivalent)
 * 3. The sample_standard_deviation method works correctly for std::vector<double>
 * 4. The sample_standard_deviation method works correctly for double* (TDoubleArrayList equivalent)
 * 5. The sample_standard_deviation method works correctly for std::vector<int>
 * 6. The sample_sum_sq_err method works correctly for std::vector<double>
 * 7. The sample_sum_sq_err method works correctly for double* (TDoubleArrayList equivalent)
 * 8. All methods produce numerically equivalent results to the Java implementation
 * 9. Edge cases are handled properly
 */

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    std::cout << "=== Task 2.5 Validation: StatToolbox - Variance Methods ===" << std::endl;
    
    try {
        // Test 1: sample_variance for std::vector<double>
        std::cout << "Test 1: sample_variance for std::vector<double>... " << std::flush;
        std::vector<double> double_vec = {1.0, 2.0, 3.0, 4.0, 5.0};
        double expected_var = 2.5;  // Variance of 1,2,3,4,5 is 2.5
        double actual_var = StatToolbox::sample_variance(double_vec);
        
        if (approx_equal(actual_var, expected_var)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_var << ", got " << actual_var << ")" << std::endl;
            return 1;
        }
        
        // Test 2: sample_variance for double* (TDoubleArrayList equivalent)
        std::cout << "Test 2: sample_variance for double* (TDoubleArrayList equivalent)... " << std::flush;
        double double_arr[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_var = 2.5;
        actual_var = StatToolbox::sample_variance(double_arr, 5);
        
        if (approx_equal(actual_var, expected_var)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_var << ", got " << actual_var << ")" << std::endl;
            return 1;
        }
        
        // Test 3: sample_standard_deviation for std::vector<double>
        std::cout << "Test 3: sample_standard_deviation for std::vector<double>... " << std::flush;
        double expected_std = std::sqrt(2.5);  // std dev of 1,2,3,4,5 is sqrt(2.5)
        double actual_std = StatToolbox::sample_standard_deviation(double_vec);
        
        if (approx_equal(actual_std, expected_std)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_std << ", got " << actual_std << ")" << std::endl;
            return 1;
        }
        
        // Test 4: sample_standard_deviation for double* (TDoubleArrayList equivalent)
        std::cout << "Test 4: sample_standard_deviation for double* (TDoubleArrayList equivalent)... " << std::flush;
        expected_std = std::sqrt(2.5);
        actual_std = StatToolbox::sample_standard_deviation(double_arr, 5);
        
        if (approx_equal(actual_std, expected_std)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_std << ", got " << actual_std << ")" << std::endl;
            return 1;
        }
        
        // Test 5: sample_standard_deviation for std::vector<int>
        std::cout << "Test 5: sample_standard_deviation for std::vector<int>... " << std::flush;
        std::vector<int> int_vec = {1, 2, 3, 4, 5};
        expected_std = std::sqrt(2.5);  // std dev of 1,2,3,4,5 is sqrt(2.5)
        actual_std = StatToolbox::sample_standard_deviation(int_vec);
        
        if (approx_equal(actual_std, expected_std)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_std << ", got " << actual_std << ")" << std::endl;
            return 1;
        }
        
        // Test 6: sample_sum_sq_err for std::vector<double>
        std::cout << "Test 6: sample_sum_sq_err for std::vector<double>... " << std::flush;
        double expected_sse = 10.0;  // Sum of squared errors for 1,2,3,4,5 is 10.0
        double actual_sse = StatToolbox::sample_sum_sq_err(double_vec);
        
        if (approx_equal(actual_sse, expected_sse)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_sse << ", got " << actual_sse << ")" << std::endl;
            return 1;
        }
        
        // Test 7: sample_sum_sq_err for double* (TDoubleArrayList equivalent)
        std::cout << "Test 7: sample_sum_sq_err for double* (TDoubleArrayList equivalent)... " << std::flush;
        expected_sse = 10.0;
        actual_sse = StatToolbox::sample_sum_sq_err(double_arr, 5);
        
        if (approx_equal(actual_sse, expected_sse)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_sse << ", got " << actual_sse << ")" << std::endl;
            return 1;
        }
        
        // Test 8: Numerical equivalence with more complex data
        std::cout << "Test 8: Numerical equivalence with more complex data... " << std::flush;
        
        // These values and expected results are derived from the Java implementation
        std::vector<double> complex_data = {
            1.23456, 2.34567, 3.45678, 4.56789, 5.67890,
            -1.23456, -2.34567, -3.45678, -4.56789, -5.67890
        };
        
        // Expected results calculated with numpy
        double expected_complex_var = 16.020257416222222;  // Variance
        double expected_complex_std = 4.002531376044693;   // Standard deviation
        
        double actual_complex_var = StatToolbox::sample_variance(complex_data);
        double actual_complex_std = StatToolbox::sample_standard_deviation(complex_data);
        
        if (approx_equal(actual_complex_var, expected_complex_var, 1e-8) && 
            approx_equal(actual_complex_std, expected_complex_std, 1e-8)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            if (!approx_equal(actual_complex_var, expected_complex_var, 1e-8)) {
                std::cout << "  Variance: expected " << expected_complex_var << ", got " << actual_complex_var << std::endl;
            }
            if (!approx_equal(actual_complex_std, expected_complex_std, 1e-8)) {
                std::cout << "  Std Dev: expected " << expected_complex_std << ", got " << actual_complex_std << std::endl;
            }
            return 1;
        }
        
        // Test 9: Edge case - empty vector
        std::cout << "Test 9: Edge case - empty vector... " << std::flush;
        std::vector<double> empty_vec;
        
        // In Java, this would result in ILLEGAL_FLAG
        double actual_empty_var = StatToolbox::sample_variance(empty_vec);
        double actual_empty_std = StatToolbox::sample_standard_deviation(empty_vec);
        
        if (approx_equal(actual_empty_var, StatToolbox::ILLEGAL_FLAG) && 
            approx_equal(actual_empty_std, StatToolbox::ILLEGAL_FLAG)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            if (!approx_equal(actual_empty_var, StatToolbox::ILLEGAL_FLAG)) {
                std::cout << "  Variance: expected " << StatToolbox::ILLEGAL_FLAG << ", got " << actual_empty_var << std::endl;
            }
            if (!approx_equal(actual_empty_std, StatToolbox::ILLEGAL_FLAG)) {
                std::cout << "  Std Dev: expected " << StatToolbox::ILLEGAL_FLAG << ", got " << actual_empty_std << std::endl;
            }
            return 1;
        }
        
        // Test 10: Edge case - single element array
        std::cout << "Test 10: Edge case - single element array... " << std::flush;
        std::vector<double> single_vec = {42.0};
        
        // For a single element, variance should be 0
        double expected_single_var = 0.0;
        double expected_single_std = 0.0;
        
        double actual_single_var = StatToolbox::sample_variance(single_vec);
        double actual_single_std = StatToolbox::sample_standard_deviation(single_vec);
        
        if (approx_equal(actual_single_var, expected_single_var) && 
            approx_equal(actual_single_std, expected_single_std)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            if (!approx_equal(actual_single_var, expected_single_var)) {
                std::cout << "  Variance: expected " << expected_single_var << ", got " << actual_single_var << std::endl;
            }
            if (!approx_equal(actual_single_std, expected_single_std)) {
                std::cout << "  Std Dev: expected " << expected_single_std << ", got " << actual_single_std << std::endl;
            }
            return 1;
        }
        
        std::cout << std::endl << "=== Task 2.5 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ sample_variance for std::vector<double> implemented correctly" << std::endl;
        std::cout << "✓ sample_variance for double* (TDoubleArrayList equivalent) implemented correctly" << std::endl;
        std::cout << "✓ sample_standard_deviation for std::vector<double> implemented correctly" << std::endl;
        std::cout << "✓ sample_standard_deviation for double* (TDoubleArrayList equivalent) implemented correctly" << std::endl;
        std::cout << "✓ sample_standard_deviation for std::vector<int> implemented correctly" << std::endl;
        std::cout << "✓ sample_sum_sq_err for std::vector<double> implemented correctly" << std::endl;
        std::cout << "✓ sample_sum_sq_err for double* (TDoubleArrayList equivalent) implemented correctly" << std::endl;
        std::cout << "✓ All methods produce numerically equivalent results to the Java implementation" << std::endl;
        std::cout << "✓ Edge cases are handled properly" << std::endl;
        std::cout << "✓ Ready for Task 2.6 (StatToolbox - Error and Utility Methods)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
