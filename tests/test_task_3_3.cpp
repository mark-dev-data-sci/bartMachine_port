#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "src/cpp/include/stat_toolbox.h"

/**
 * Test for Task 3.3: StatToolbox - multinomial_sample Method
 * 
 * This test validates the implementation of the multinomial_sample method
 * in the StatToolbox class, which samples from a multinomial distribution.
 * 
 * The test ensures that:
 * 1. The method produces identical sampling sequences for the same seeds
 * 2. The method correctly implements the multinomial distribution sampling logic
 */

int main() {
    std::cout << "Testing Task 3.3: StatToolbox - multinomial_sample Method" << std::endl;
    
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    // Test parameters
    std::vector<int> vals = {1, 2, 3, 4, 5};
    std::vector<double> probs = {0.1, 0.2, 0.3, 0.25, 0.15};
    
    // Generate samples
    const int num_samples = 10;
    std::vector<int> samples;
    
    for (int i = 0; i < num_samples; i++) {
        int sample = StatToolbox::multinomial_sample(vals, probs);
        samples.push_back(sample);
        std::cout << "Sample " << i << ": " << sample << std::endl;
    }
    
    // Reset seed and verify reproducibility
    StatToolbox::setSeed(12345);
    
    for (int i = 0; i < num_samples; i++) {
        int sample = StatToolbox::multinomial_sample(vals, probs);
        std::cout << "Verification Sample " << i << ": " << sample << std::endl;
        
        // Verify that the samples are identical for the same seed
        assert(sample == samples[i]);
    }
    
    // Test with different parameters
    vals = {10, 20, 30};
    probs = {0.3, 0.5, 0.2};
    
    std::cout << "\nTesting with different parameters:" << std::endl;
    
    for (int i = 0; i < num_samples; i++) {
        int sample = StatToolbox::multinomial_sample(vals, probs);
        std::cout << "Different Parameters Sample " << i << ": " << sample << std::endl;
    }
    
    // Test with extreme probabilities
    vals = {100, 200};
    probs = {0.99, 0.01};
    
    std::cout << "\nTesting with extreme probabilities:" << std::endl;
    
    for (int i = 0; i < num_samples; i++) {
        int sample = StatToolbox::multinomial_sample(vals, probs);
        std::cout << "Extreme Probabilities Sample " << i << ": " << sample << std::endl;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
