#include "include/bartmachine_b_hyperparams.h"
#include "include/exact_port_mersenne_twister.h"
#include <random>
#include <cmath>

// Function declaration
extern "C" void initialize_random_samples();

// Initialize the random samples for chi-squared and standard normal distributions
void initialize_random_samples() {
    // Set the seed for reproducibility
    std::mt19937 gen(12345);
    
    // Generate chi-squared samples
    const int chi_sq_samples = 1000;
    double* chi_sq = new double[chi_sq_samples];
    
    // For chi-squared with nu + n degrees of freedom (assuming nu = 3 and n = 100)
    std::chi_squared_distribution<double> chi_sq_dist(103);
    for (int i = 0; i < chi_sq_samples; i++) {
        chi_sq[i] = chi_sq_dist(gen);
    }
    
    // Set the static members
    bartmachine_b_hyperparams::samps_chi_sq_df_eq_nu_plus_n = chi_sq;
    bartmachine_b_hyperparams::samps_chi_sq_df_eq_nu_plus_n_length = chi_sq_samples;
    
    // Generate standard normal samples
    const int std_normal_samples = 1000;
    double* std_normal = new double[std_normal_samples];
    
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    for (int i = 0; i < std_normal_samples; i++) {
        std_normal[i] = normal_dist(gen);
    }
    
    // Set the static members
    bartmachine_b_hyperparams::samps_std_normal = std_normal;
    bartmachine_b_hyperparams::samps_std_normal_length = std_normal_samples;
}
