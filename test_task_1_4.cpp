#include "src/cpp/include/stat_toolbox.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

/**
 * Task 1.4 Validation: StatToolbox RNG Interface Methods
 * 
 * This test validates that:
 * 1. The StatToolbox::setSeed method correctly initializes the RNG state
 * 2. The StatToolbox::rand method produces the expected sequence of values
 * 3. The sequences match those from the ExactPortMersenneTwister class
 */

int main() {
    std::cout << "=== Task 1.4 Validation: StatToolbox RNG Interface Methods ===" << std::endl;
    
    try {
        // Test 1: StatToolbox::setSeed basic functionality
        std::cout << "Test 1: StatToolbox::setSeed basic functionality... " << std::flush;
        StatToolbox::setSeed(12345);
        std::cout << "PASSED" << std::endl;
        
        // Test 2: StatToolbox::rand basic functionality
        std::cout << "Test 2: StatToolbox::rand basic functionality... " << std::flush;
        double val = StatToolbox::rand();
        // Just check that it's in the expected range (0,1)
        if (val > 0.0 && val < 1.0) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED (value out of range: " << val << ")" << std::endl;
            return 1;
        }
        
        // Test 3: Sequence reproducibility with same seed
        std::cout << "Test 3: Sequence reproducibility... " << std::flush;
        StatToolbox::setSeed(54321);
        std::vector<double> seq1;
        for (int i = 0; i < 10; i++) {
            seq1.push_back(StatToolbox::rand());
        }
        
        StatToolbox::setSeed(54321);
        std::vector<double> seq2;
        for (int i = 0; i < 10; i++) {
            seq2.push_back(StatToolbox::rand());
        }
        
        bool sequences_match = true;
        for (int i = 0; i < 10; i++) {
            if (seq1[i] != seq2[i]) {
                sequences_match = false;
                std::cout << "FAILED (sequences diverge at iteration " << i << ")" << std::endl;
                return 1;
            }
        }
        
        if (sequences_match) {
            std::cout << "PASSED" << std::endl;
        }
        
        // Test 4: Equivalence with ExactPortMersenneTwister
        std::cout << "Test 4: Equivalence with ExactPortMersenneTwister... " << std::flush;
        StatToolbox::setSeed(98765);
        ExactPortMersenneTwister rng(98765);
        
        bool equivalent = true;
        for (int i = 0; i < 100; i++) {
            double val1 = StatToolbox::rand();
            double val2 = rng.nextDouble(false, false);
            if (val1 != val2) {
                equivalent = false;
                std::cout << "FAILED (values diverge at iteration " << i << ")" << std::endl;
                std::cout << "  StatToolbox::rand() = " << val1 << std::endl;
                std::cout << "  ExactPortMersenneTwister::nextDouble(false, false) = " << val2 << std::endl;
                return 1;
            }
        }
        
        if (equivalent) {
            std::cout << "PASSED" << std::endl;
        }
        
        // Test 5: Print first 10 values for visual inspection
        std::cout << "Test 5: First 10 values from seed 12345..." << std::endl;
        StatToolbox::setSeed(12345);
        std::cout << std::fixed << std::setprecision(16);
        for (int i = 0; i < 10; i++) {
            std::cout << "  Value " << (i+1) << ": " << StatToolbox::rand() << std::endl;
        }
        std::cout << "PASSED (visual inspection)" << std::endl;
        
        std::cout << std::endl << "=== Task 1.4 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ StatToolbox::setSeed method implemented correctly" << std::endl;
        std::cout << "✓ StatToolbox::rand produces values in correct range" << std::endl;
        std::cout << "✓ Sequences are reproducible with same seed" << std::endl;
        std::cout << "✓ StatToolbox::rand matches ExactPortMersenneTwister::nextDouble(false, false)" << std::endl;
        std::cout << "✓ Ready for Task 2.1 (StatToolbox - Class Structure + Basic Stats)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}
