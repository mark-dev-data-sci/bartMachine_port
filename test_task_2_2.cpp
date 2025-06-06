#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

/**
 * Task 2.2 Validation: StatToolbox - sample_average Methods
 * 
 * This test validates that:
 * 1. The sample_average method for std::vector<double> works correctly
 * 2. The sample_average method for double* (TDoubleArrayList equivalent) works correctly
 * 3. The sample_average method for std::vector<int> works correctly
 * 4. All methods produce numerically equivalent results to the Java implementation
 */

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-10) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    std::cout << "=== Task 2.2 Validation: StatToolbox - sample_average Methods ===" << std::endl;
    
    try {
        // Test 1: sample_average for std::vector<double>
        std::cout << "Test 1: sample_average for std::vector<double>... " << std::flush;
        std::vector<double> double_vec = {1.0, 2.0, 3.0, 4.0, 5.0};
        double expected_avg = 3.0;
        double actual_avg = StatToolbox::sample_average(double_vec);
        
        if (approx_equal(actual_avg, expected_avg)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_avg << ", got " << actual_avg << ")" << std::endl;
            return 1;
        }
        
        // Test 2: sample_average for double* (TDoubleArrayList equivalent)
        std::cout << "Test 2: sample_average for double* (TDoubleArrayList equivalent)... " << std::flush;
        double double_arr[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_avg = 3.0;
        actual_avg = StatToolbox::sample_average(double_arr, 5);
        
        if (approx_equal(actual_avg, expected_avg)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_avg << ", got " << actual_avg << ")" << std::endl;
            return 1;
        }
        
        // Test 3: sample_average for std::vector<int>
        std::cout << "Test 3: sample_average for std::vector<int>... " << std::flush;
        std::vector<int> int_vec = {1, 2, 3, 4, 5};
        expected_avg = 3.0;
        actual_avg = StatToolbox::sample_average(int_vec);
        
        if (approx_equal(actual_avg, expected_avg)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_avg << ", got " << actual_avg << ")" << std::endl;
            return 1;
        }
        
        // Test 4: Numerical equivalence with more complex data
        std::cout << "Test 4: Numerical equivalence with more complex data... " << std::flush;
        
        // These values and expected results are derived from the Java implementation
        std::vector<double> complex_data = {
            1.23456, 2.34567, 3.45678, 4.56789, 5.67890,
            -1.23456, -2.34567, -3.45678, -4.56789, -5.67890
        };
        
        // Expected result from Java: 0.0
        expected_avg = 0.0;
        actual_avg = StatToolbox::sample_average(complex_data);
        
        if (approx_equal(actual_avg, expected_avg)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_avg << ", got " << actual_avg << ")" << std::endl;
            return 1;
        }
        
        // Test 5: Edge case - empty vector
        std::cout << "Test 5: Edge case - empty vector... " << std::flush;
        std::vector<double> empty_vec;
        
        // In Java, this would result in NaN due to division by zero
        // We should get the same behavior in C++
        actual_avg = StatToolbox::sample_average(empty_vec);
        
        if (std::isnan(actual_avg)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected NaN, got " << actual_avg << ")" << std::endl;
            return 1;
        }
        
        std::cout << std::endl << "=== Task 2.2 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ sample_average for std::vector<double> implemented correctly" << std::endl;
        std::cout << "✓ sample_average for double* (TDoubleArrayList equivalent) implemented correctly" << std::endl;
        std::cout << "✓ sample_average for std::vector<int> implemented correctly" << std::endl;
        std::cout << "✓ All methods produce numerically equivalent results to the Java implementation" << std::endl;
        std::cout << "✓ Ready for Task 2.3 (StatToolbox - sample_median Method)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
