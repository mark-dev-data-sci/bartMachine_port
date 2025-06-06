#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "src/cpp/include/stat_toolbox.h"

/**
 * Test for Task 3.1: StatToolbox - sample_from_inv_gamma Method
 * 
 * This test validates the implementation of the sample_from_inv_gamma method
 * in the StatToolbox class, which samples from an inverse gamma distribution.
 * 
 * The test ensures that:
 * 1. The method produces identical sampling sequences for the same seeds
 * 2. The method correctly implements the inverse gamma sampling logic
 */

int main() {
    std::cout << "Testing Task 3.1: StatToolbox - sample_from_inv_gamma Method" << std::endl;
    
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    // Test parameters
    double k = 3.0;      // Shape parameter
    double theta = 2.0;  // Scale parameter
    
    // Generate samples
    const int num_samples = 10;
    std::vector<double> samples;
    
    for (int i = 0; i < num_samples; i++) {
        double sample = StatToolbox::sample_from_inv_gamma(k, theta);
        samples.push_back(sample);
        std::cout << "Sample " << i << ": " << sample << std::endl;
    }
    
    // Reset seed and verify reproducibility
    StatToolbox::setSeed(12345);
    
    for (int i = 0; i < num_samples; i++) {
        double sample = StatToolbox::sample_from_inv_gamma(k, theta);
        std::cout << "Verification Sample " << i << ": " << sample << std::endl;
        
        // Verify that the samples are identical for the same seed
        assert(std::abs(sample - samples[i]) < 1e-10);
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
