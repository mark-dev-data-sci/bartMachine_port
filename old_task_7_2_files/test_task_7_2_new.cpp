#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>

// Function declaration for initialize_random_samples
extern "C" void initialize_random_samples();

/**
 * Test for Task 7.2: Validation with Original Datasets
 * 
 * This test runs a comparison between the Java and C++ implementations
 * of bartMachine using the original R interface.
 */

int main() {
    std::cout << "Running Task 7.2: Validation with Original Datasets" << std::endl;
    
    // Initialize random samples for chi-squared and standard normal distributions
    std::cout << "Initializing random samples..." << std::endl;
    initialize_random_samples();
    
    // Run the comparison script
    std::cout << "Running comparison script..." << std::endl;
    int result = std::system("cd ../src/r && Rscript compare_java_cpp_simple_fixed.R");
    
    if (result != 0) {
        std::cerr << "Error running comparison script" << std::endl;
        return 1;
    }
    
    // Check if the report file was created
    std::ifstream report_file("../src/r/comparison_report_port.md");
    if (!report_file.is_open()) {
        std::cerr << "Error: Comparison report file not found" << std::endl;
        return 1;
    }
    
    // Read and display the report
    std::cout << "\nComparison Report:" << std::endl;
    std::cout << "==================" << std::endl;
    
    std::string line;
    while (std::getline(report_file, line)) {
        std::cout << line << std::endl;
    }
    
    report_file.close();
    
    std::cout << "\nTask 7.2 completed successfully!" << std::endl;
    return 0;
}
