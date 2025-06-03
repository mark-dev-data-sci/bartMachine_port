#include "include/stat_toolbox.h"
#include <limits>

/**
 * Exact port of StatToolbox from Java to C++
 * 
 * This file contains the implementation of the RNG interface methods
 * and basic statistical functions.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/StatToolbox.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */

// Initialize static member
ExactPortMersenneTwister* StatToolbox::R = new ExactPortMersenneTwister();

/**
 * Sets the seed for the random number generator
 *
 * @param seed The seed
 */
void StatToolbox::setSeed(int64_t seed) {
    R->setSeed(seed);
}

/**
 * A convenience method for a random object
 *
 * @return A random number drawn from a uniform distribution bounded between 0 and 1.
 */
double StatToolbox::rand() {
    return R->nextDouble(false, false);
}

/**
 * Compute the sample average of a vector of data
 *
 * @param y The vector of data values
 * @return The sample average
 */
double StatToolbox::sample_average(const std::vector<double>& y) {
    double y_bar = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        y_bar += y[i];
    }
    return y_bar / static_cast<double>(y.size());
}

/**
 * Compute the sample average of a vector of data
 * This is equivalent to the TDoubleArrayList version in Java
 *
 * @param y The vector of data values
 * @return The sample average
 */
double StatToolbox::sample_average(const double* y, int size) {
    double y_bar = 0.0;
    for (int i = 0; i < size; i++) {
        y_bar += y[i];
    }
    return y_bar / static_cast<double>(size);
}

/**
 * Compute the sample average of a vector of data
 *
 * @param y The vector of data values
 * @return The sample average
 */
double StatToolbox::sample_average(const std::vector<int>& y) {
    double y_bar = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        y_bar += y[i];
    }
    return y_bar / static_cast<double>(y.size());
}

/**
 * Compute the sample median of a vector of data
 *
 * @param arr The vector of data values
 * @return The sample median
 */
double StatToolbox::sample_median(std::vector<double> arr) {
    int n = static_cast<int>(arr.size());
    
    // Handle empty array case
    if (n == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Sort the array (note: we take a copy of the array, not a reference)
    std::sort(arr.begin(), arr.end());
    
    if (n % 2 == 0) {
        // Even number of elements
        double a = arr[n / 2];
        double b = arr[n / 2 - 1];
        return (a + b) / 2.0;
    } else {
        // Odd number of elements
        return arr[(n - 1) / 2];
    }
}

/**
 * Compute the minimum value in a vector of data
 *
 * @param arr The vector of data values
 * @return The minimum value, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_minimum(const std::vector<double>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    double min_val = arr[0];
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

/**
 * Compute the minimum value in an array of data
 * This is equivalent to the TDoubleArrayList version in Java
 *
 * @param arr The array of data values
 * @param size The size of the array
 * @return The minimum value, or ILLEGAL_FLAG if the array is empty
 */
double StatToolbox::sample_minimum(const double* arr, int size) {
    if (size <= 0 || arr == nullptr) {
        return ILLEGAL_FLAG;
    }
    
    double min_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

/**
 * Compute the maximum value in a vector of data
 *
 * @param arr The vector of data values
 * @return The maximum value, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_maximum(const std::vector<double>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    double max_val = arr[0];
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

/**
 * Compute the maximum value in an array of data
 * This is equivalent to the TDoubleArrayList version in Java
 *
 * @param arr The array of data values
 * @param size The size of the array
 * @return The maximum value, or ILLEGAL_FLAG if the array is empty
 */
double StatToolbox::sample_maximum(const double* arr, int size) {
    if (size <= 0 || arr == nullptr) {
        return ILLEGAL_FLAG;
    }
    
    double max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

/**
 * Compute the minimum value in a vector of integer data
 *
 * @param arr The vector of data values
 * @return The minimum value, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_minimum(const std::vector<int>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    int min_val = arr[0];
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return static_cast<double>(min_val);
}

/**
 * Compute the maximum value in a vector of integer data
 *
 * @param arr The vector of data values
 * @return The maximum value, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_maximum(const std::vector<int>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    int max_val = arr[0];
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return static_cast<double>(max_val);
}

/**
 * Compute the sum of squared error (the squared deviation from the sample average) of a vector of data
 *
 * @param arr The vector of data values
 * @return The sum of squared error, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_sum_sq_err(const std::vector<double>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    double y_bar = sample_average(arr);
    double sum_sqd_deviations = 0.0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum_sqd_deviations += std::pow(arr[i] - y_bar, 2);
    }
    return sum_sqd_deviations;
}

/**
 * Compute the sum of squared error (the squared deviation from the sample average) of an array of data
 * This is equivalent to the TDoubleArrayList version in Java
 *
 * @param arr The array of data values
 * @param size The size of the array
 * @return The sum of squared error, or ILLEGAL_FLAG if the array is empty
 */
double StatToolbox::sample_sum_sq_err(const double* arr, int size) {
    if (size <= 0 || arr == nullptr) {
        return ILLEGAL_FLAG;
    }
    
    double y_bar = sample_average(arr, size);
    double sum_sqd_deviations = 0.0;
    for (int i = 0; i < size; i++) {
        sum_sqd_deviations += std::pow(arr[i] - y_bar, 2);
    }
    return sum_sqd_deviations;
}

/**
 * Compute the sample variance of a vector of data
 *
 * @param arr The vector of data values
 * @return The sample variance, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_variance(const std::vector<double>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    if (arr.size() == 1) {
        return 0.0;  // Variance of a single element is 0
    }
    
    return sample_sum_sq_err(arr) / static_cast<double>(arr.size() - 1);
}

/**
 * Compute the sample variance of an array of data
 * This is equivalent to the TDoubleArrayList version in Java
 *
 * @param arr The array of data values
 * @param size The size of the array
 * @return The sample variance, or ILLEGAL_FLAG if the array is empty
 */
double StatToolbox::sample_variance(const double* arr, int size) {
    if (size <= 0 || arr == nullptr) {
        return ILLEGAL_FLAG;
    }
    
    if (size == 1) {
        return 0.0;  // Variance of a single element is 0
    }
    
    return sample_sum_sq_err(arr, size) / static_cast<double>(size - 1);
}

/**
 * Compute the sample standard deviation of a vector of data
 *
 * @param arr The vector of data values
 * @return The sample standard deviation, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_standard_deviation(const std::vector<double>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    if (arr.size() == 1) {
        return 0.0;  // Standard deviation of a single element is 0
    }
    
    return std::sqrt(sample_variance(arr));
}

/**
 * Compute the sample standard deviation of an array of data
 * This is equivalent to the TDoubleArrayList version in Java
 *
 * @param arr The array of data values
 * @param size The size of the array
 * @return The sample standard deviation, or ILLEGAL_FLAG if the array is empty
 */
double StatToolbox::sample_standard_deviation(const double* arr, int size) {
    if (size <= 0 || arr == nullptr) {
        return ILLEGAL_FLAG;
    }
    
    if (size == 1) {
        return 0.0;  // Standard deviation of a single element is 0
    }
    
    return std::sqrt(sample_variance(arr, size));
}

/**
 * Compute the sample standard deviation of a vector of integer data
 *
 * @param arr The vector of data values
 * @return The sample standard deviation, or ILLEGAL_FLAG if the vector is empty
 */
double StatToolbox::sample_standard_deviation(const std::vector<int>& arr) {
    if (arr.empty()) {
        return ILLEGAL_FLAG;
    }
    
    if (arr.size() == 1) {
        return 0.0;  // Standard deviation of a single element is 0
    }
    
    double y_bar = sample_average(arr);
    double sum_sqd_deviations = 0.0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum_sqd_deviations += std::pow(arr[i] - y_bar, 2);
    }
    return std::sqrt(sum_sqd_deviations / static_cast<double>(arr.size() - 1));
}

/**
 * Find the index of the maximum value in a vector of integer data
 *
 * @param arr The vector of data values
 * @return The index of the maximum value
 * @throws std::runtime_error if the vector is empty
 */
int StatToolbox::FindMaxIndex(const std::vector<int>& arr) {
    if (arr.empty()) {
        throw std::runtime_error("Cannot find maximum index in an empty array");
    }
    
    int index = 0;
    int max = std::numeric_limits<int>::min();
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] > max) {
            max = arr[i];
            index = static_cast<int>(i);
        }
    }
    return index;
}

/**
 * Draws a sample from an inverse gamma distribution.
 * 
 * @param k The shape parameter of the inverse gamma distribution of interest
 * @param theta The scale parameter of the inverse gamma distribution of interest
 * @return The sampled value
 */
double StatToolbox::sample_from_inv_gamma(double k, double theta) {
    return (1 / (theta / 2)) / bartmachine_b_hyperparams::samps_chi_sq_df_eq_nu_plus_n[static_cast<int>(std::floor(rand() * bartmachine_b_hyperparams::samps_chi_sq_df_eq_nu_plus_n_length))];
}

/**
 * Draws a sample from a normal distribution.
 * 
 * @param mu The mean of the normal distribution of interest
 * @param sigsq The variance of the normal distribution of interest
 * @return The sample value
 */
double StatToolbox::sample_from_norm_dist(double mu, double sigsq) {
    double std_norm_realization = bartmachine_b_hyperparams::samps_std_normal[static_cast<int>(std::floor(rand() * bartmachine_b_hyperparams::samps_std_normal_length))];
    return mu + std::sqrt(sigsq) * std_norm_realization;
}

/**
 * Sample from a multinomial distribution
 * 
 * @param vals The integer values of the labels in this multinomial distribution
 * @param probs The probabilities of each label
 * @return A sample from the multinomial distribution
 */
int StatToolbox::multinomial_sample(const std::vector<int>& vals, const std::vector<double>& probs) {
    double r = StatToolbox::rand();
    double cum_prob = 0.0;
    int index = 0;
    
    if (r < probs[0]) {
        return vals[0];
    }
    
    while (true) {
        cum_prob += probs[index];
        if (r > cum_prob && r < cum_prob + probs[index + 1]) {
            return vals[index + 1];
        }
        index++;
    }
}
