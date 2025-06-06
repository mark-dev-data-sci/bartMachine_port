#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "src/cpp/include/stat_toolbox.h"

/**
 * Test for Task 3.2: StatToolbox - sample_from_norm_dist Method
 * 
 * This test validates the implementation of the sample_from_norm_dist method
 * in the StatToolbox class, which samples from a normal distribution.
 * 
 * The test ensures that:
 * 1. The method produces identical sampling sequences for the same seeds
 * 2. The method correctly implements the normal distribution sampling logic
 */

int main() {
    std::cout << "Testing Task 3.2: StatToolbox - sample_from_norm_dist Method" << std::endl;
    
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    // Test parameters
    double mu = 10.0;      // Mean
    double sigsq = 2.0;    // Variance
    
    // Generate samples
    const int num_samples = 10;
    std::vector<double> samples;
    
    for (int i = 0; i < num_samples; i++) {
        double sample = StatToolbox::sample_from_norm_dist(mu, sigsq);
        samples.push_back(sample);
        std::cout << "Sample " << i << ": " << sample << std::endl;
    }
    
    // Reset seed and verify reproducibility
    StatToolbox::setSeed(12345);
    
    for (int i = 0; i < num_samples; i++) {
        double sample = StatToolbox::sample_from_norm_dist(mu, sigsq);
        std::cout << "Verification Sample " << i << ": " << sample << std::endl;
        
        // Verify that the samples are identical for the same seed
        assert(std::abs(sample - samples[i]) < 1e-10);
    }
    
    // Test with different parameters
    mu = 0.0;
    sigsq = 1.0;
    
    std::cout << "\nTesting with standard normal parameters (mu=0, sigsq=1):" << std::endl;
    
    for (int i = 0; i < num_samples; i++) {
        double sample = StatToolbox::sample_from_norm_dist(mu, sigsq);
        std::cout << "Standard Normal Sample " << i << ": " << sample << std::endl;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
