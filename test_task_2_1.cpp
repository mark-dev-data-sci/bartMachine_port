#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <cassert>
#include <limits>

/**
 * Task 2.1 Validation: StatToolbox - Class Structure + Constants
 * 
 * This test validates that:
 * 1. The StatToolbox class is properly structured
 * 2. The ILLEGAL_FLAG constant is defined with the correct value
 */

int main() {
    std::cout << "=== Task 2.1 Validation: StatToolbox - Class Structure + Constants ===" << std::endl;
    
    try {
        // Test 1: Verify ILLEGAL_FLAG constant
        std::cout << "Test 1: Verify ILLEGAL_FLAG constant... " << std::flush;
        const double expected_flag_value = -999999999;
        if (StatToolbox::ILLEGAL_FLAG == expected_flag_value) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (expected " << expected_flag_value 
                      << ", got " << StatToolbox::ILLEGAL_FLAG << ")" << std::endl;
            return 1;
        }
        
        // Test 2: Verify RNG methods work (already tested in Task 1.4, but checking again)
        std::cout << "Test 2: Verify RNG methods work... " << std::flush;
        StatToolbox::setSeed(12345);
        double rand_val = StatToolbox::rand();
        if (rand_val > 0.0 && rand_val < 1.0) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (value out of range: " << rand_val << ")" << std::endl;
            return 1;
        }
        
        std::cout << std::endl << "=== Task 2.1 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ StatToolbox class structure implemented correctly" << std::endl;
        std::cout << "✓ ILLEGAL_FLAG constant defined with correct value" << std::endl;
        std::cout << "✓ Ready for Task 2.2 (StatToolbox - sample_average Methods)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
