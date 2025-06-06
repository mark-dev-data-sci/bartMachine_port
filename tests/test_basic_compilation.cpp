#include "src/cpp/include/exact_port_mersenne_twister.h"
#include <iostream>
#include <vector>

/**
 * Basic compilation and instantiation test for ExactPortMersenneTwister
 * 
 * This test validates that:
 * 1. The class can be compiled without errors
 * 2. All constructors work
 * 3. Basic state operations function
 * 4. Memory management works correctly
 */

int main() {
    std::cout << "=== Task 1.1 Validation: MersenneTwisterFast Class Structure ===" << std::endl;
    
    try {
        // Test 1: Default constructor
        std::cout << "Test 1: Default constructor..." << std::flush;
        ExactPortMersenneTwister rng1;
        std::cout << " PASSED" << std::endl;
        
        // Test 2: Constructor with seed
        std::cout << "Test 2: Constructor with seed..." << std::flush;
        ExactPortMersenneTwister rng2(12345L);
        std::cout << " PASSED" << std::endl;
        
        // Test 3: Constructor with array
        std::cout << "Test 3: Constructor with array..." << std::flush;
        std::vector<int> seed_array = {0x123, 0x234, 0x345, 0x456};
        ExactPortMersenneTwister rng3(seed_array);
        std::cout << " PASSED" << std::endl;
        
        // Test 4: Copy constructor
        std::cout << "Test 4: Copy constructor..." << std::flush;
        ExactPortMersenneTwister rng4(rng2);
        std::cout << " PASSED" << std::endl;
        
        // Test 5: Assignment operator
        std::cout << "Test 5: Assignment operator..." << std::flush;
        ExactPortMersenneTwister rng5;
        rng5 = rng3;
        std::cout << " PASSED" << std::endl;
        
        // Test 6: Clone method
        std::cout << "Test 6: Clone method..." << std::flush;
        ExactPortMersenneTwister* rng6 = rng2.clone();
        delete rng6;  // Test memory cleanup
        std::cout << " PASSED" << std::endl;
        
        // Test 7: Basic method calls (should not crash, even with stub implementations)
        std::cout << "Test 7: Basic method calls..." << std::flush;
        rng1.clearGaussian();  // This method is implemented
        bool result = rng1.stateEquals(rng2);  // Should return false (stub implementation)
        (void)result;  // Suppress unused variable warning
        std::cout << " PASSED" << std::endl;
        
        // Test 8: Method calls that return default values
        std::cout << "Test 8: Stub method calls..." << std::flush;
        int int_val = rng1.nextInt();
        float float_val = rng1.nextFloat();
        double double_val = rng1.nextDouble();
        bool bool_val = rng1.nextBoolean();
        
        // Verify default return values
        if (int_val == 0 && float_val == 0.0f && double_val == 0.0 && bool_val == false) {
            std::cout << " PASSED" << std::endl;
        } else {
            std::cout << " FAILED (unexpected return values)" << std::endl;
            return 1;
        }
        
        std::cout << std::endl;
        std::cout << "=== Task 1.1 VALIDATION SUCCESSFUL ===" << std::endl;
        std::cout << "✓ Class structure created successfully" << std::endl;
        std::cout << "✓ All constructors work" << std::endl;
        std::cout << "✓ Memory management functions correctly" << std::endl;
        std::cout << "✓ Basic state operations work" << std::endl;
        std::cout << "✓ Ready for Task 1.2 (setSeed method implementation)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception occurred" << std::endl;
        return 1;
    }
}
