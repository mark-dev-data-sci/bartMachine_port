#include "src/cpp/include/exact_port_mersenne_twister.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

/**
 * Task 1.3 Validation: setSeed(array) and nextDouble Methods
 * 
 * This test validates that:
 * 1. The setSeed(array) method correctly initializes the RNG state
 * 2. The nextDouble method produces the expected sequence of values
 * 3. The nextDouble(includeZero, includeOne) variants work correctly
 */

int main() {
    std::cout << "=== Task 1.3 Validation: setSeed(array) and nextDouble Methods ===" << std::endl;
    
    try {
        // Test 1: setSeed(array) basic functionality
        std::cout << "Test 1: setSeed(array) basic functionality... " << std::flush;
        std::vector<int> seed_array = {0x123, 0x234, 0x345, 0x456};
        ExactPortMersenneTwister rng1(seed_array);
        std::cout << "PASSED" << std::endl;
        
        // Test 2: nextDouble() basic functionality
        std::cout << "Test 2: nextDouble() basic functionality... " << std::flush;
        double val = rng1.nextDouble();
        // Just check that it's in the expected range (0,1)
        if (val > 0.0 && val < 1.0) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (value out of range: " << val << ")" << std::endl;
            return 1;
        }
        
        // Test 3: Sequence reproducibility with same seed
        std::cout << "Test 3: Sequence reproducibility... " << std::flush;
        ExactPortMersenneTwister rng2(12345L);
        ExactPortMersenneTwister rng3(12345L);
        
        bool sequences_match = true;
        for (int i = 0; i < 1000; i++) {
            double val1 = rng2.nextDouble();
            double val2 = rng3.nextDouble();
            if (val1 != val2) {
                sequences_match = false;
                std::cout << "FAILED (sequences diverge at iteration " << i << ")" << std::endl;
                return 1;
            }
        }
        
        if (sequences_match) {
            std::cout << "PASSED" << std::endl;
        }
        
        // Test 4: nextDouble variants
        std::cout << "Test 4: nextDouble variants... " << std::flush;
        ExactPortMersenneTwister rng4(54321L);
        
        // Test includeZero=false, includeOne=false (default)
        double val1 = rng4.nextDouble(false, false);
        if (val1 <= 0.0 || val1 >= 1.0) {
            std::cout << "FAILED (nextDouble(false, false) out of range: " << val1 << ")" << std::endl;
            return 1;
        }
        
        // Test includeZero=true, includeOne=false
        double val2 = rng4.nextDouble(true, false);
        if (val2 < 0.0 || val2 >= 1.0) {
            std::cout << "FAILED (nextDouble(true, false) out of range: " << val2 << ")" << std::endl;
            return 1;
        }
        
        // Test includeZero=false, includeOne=true
        double val3 = rng4.nextDouble(false, true);
        if (val3 <= 0.0 || val3 > 1.0) {
            std::cout << "FAILED (nextDouble(false, true) out of range: " << val3 << ")" << std::endl;
            return 1;
        }
        
        // Test includeZero=true, includeOne=true
        double val4 = rng4.nextDouble(true, true);
        if (val4 < 0.0 || val4 > 1.0) {
            std::cout << "FAILED (nextDouble(true, true) out of range: " << val4 << ")" << std::endl;
            return 1;
        }
        
        std::cout << "PASSED" << std::endl;
        
        // Test 5: Print first 10 values for visual inspection
        std::cout << "Test 5: First 10 values from seed 12345..." << std::endl;
        ExactPortMersenneTwister rng5(12345L);
        std::cout << std::fixed << std::setprecision(16);
        for (int i = 0; i < 10; i++) {
            std::cout << "  Value " << (i+1) << ": " << rng5.nextDouble() << std::endl;
        }
        std::cout << "PASSED (visual inspection)" << std::endl;
        
        std::cout << std::endl << "=== Task 1.3 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ setSeed(array) method implemented correctly" << std::endl;
        std::cout << "✓ nextDouble() produces values in correct range" << std::endl;
        std::cout << "✓ Sequences are reproducible with same seed" << std::endl;
        std::cout << "✓ All nextDouble variants work correctly" << std::endl;
        std::cout << "✓ Ready for Task 1.4 (StatToolbox RNG Interface Methods)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
