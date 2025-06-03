#ifndef STAT_TOOLBOX_H
#define STAT_TOOLBOX_H

#include "exact_port_mersenne_twister.h"
#include "bartmachine_b_hyperparams.h"
#include <vector>
#include <algorithm>
#include <cmath>

/**
 * Exact port of StatToolbox from Java to C++
 * 
 * This is a direct port of the Java StatToolbox class from bartMachine.
 * All member variables, constants, and method signatures are preserved exactly
 * to ensure numerical equivalence.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/StatToolbox.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class StatToolbox {
private:
    // RNG instance - exact copy from Java
    static ExactPortMersenneTwister* R;

public:
    /** A flag that indicates an illegal value or failed operation */
    static constexpr double ILLEGAL_FLAG = -999999999;

    // RNG interface methods - exact signatures from Java
    static void setSeed(int64_t seed);
    static double rand();
    
    /**
     * Compute the sample average of a vector of data
     *
     * @param y The vector of data values
     * @return The sample average
     */
    static double sample_average(const std::vector<double>& y);
    
    /**
     * Compute the sample average of a vector of data
     * This is equivalent to the TDoubleArrayList version in Java
     *
     * @param y The vector of data values
     * @return The sample average
     */
    static double sample_average(const double* y, int size);
    
    /**
     * Compute the sample average of a vector of data
     *
     * @param y The vector of data values
     * @return The sample average
     */
    static double sample_average(const std::vector<int>& y);
    
    /**
     * Compute the sample median of a vector of data
     *
     * @param arr The vector of data values
     * @return The sample median
     */
    static double sample_median(std::vector<double> arr);
    
    /**
     * Compute the minimum value in a vector of data
     *
     * @param arr The vector of data values
     * @return The minimum value, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_minimum(const std::vector<double>& arr);
    
    /**
     * Compute the minimum value in an array of data
     * This is equivalent to the TDoubleArrayList version in Java
     *
     * @param arr The array of data values
     * @param size The size of the array
     * @return The minimum value, or ILLEGAL_FLAG if the array is empty
     */
    static double sample_minimum(const double* arr, int size);
    
    /**
     * Compute the maximum value in a vector of data
     *
     * @param arr The vector of data values
     * @return The maximum value, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_maximum(const std::vector<double>& arr);
    
    /**
     * Compute the maximum value in an array of data
     * This is equivalent to the TDoubleArrayList version in Java
     *
     * @param arr The array of data values
     * @param size The size of the array
     * @return The maximum value, or ILLEGAL_FLAG if the array is empty
     */
    static double sample_maximum(const double* arr, int size);
    
    /**
     * Compute the minimum value in a vector of integer data
     *
     * @param arr The vector of data values
     * @return The minimum value, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_minimum(const std::vector<int>& arr);
    
    /**
     * Compute the maximum value in a vector of integer data
     *
     * @param arr The vector of data values
     * @return The maximum value, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_maximum(const std::vector<int>& arr);
    
    /**
     * Compute the sample variance of a vector of data
     *
     * @param arr The vector of data values
     * @return The sample variance, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_variance(const std::vector<double>& arr);
    
    /**
     * Compute the sample variance of an array of data
     * This is equivalent to the TDoubleArrayList version in Java
     *
     * @param arr The array of data values
     * @param size The size of the array
     * @return The sample variance, or ILLEGAL_FLAG if the array is empty
     */
    static double sample_variance(const double* arr, int size);
    
    /**
     * Compute the sample standard deviation of a vector of data
     *
     * @param arr The vector of data values
     * @return The sample standard deviation, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_standard_deviation(const std::vector<double>& arr);
    
    /**
     * Compute the sample standard deviation of an array of data
     * This is equivalent to the TDoubleArrayList version in Java
     *
     * @param arr The array of data values
     * @param size The size of the array
     * @return The sample standard deviation, or ILLEGAL_FLAG if the array is empty
     */
    static double sample_standard_deviation(const double* arr, int size);
    
    /**
     * Compute the sample standard deviation of a vector of integer data
     *
     * @param arr The vector of data values
     * @return The sample standard deviation, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_standard_deviation(const std::vector<int>& arr);
    
    /**
     * Compute the sum of squared error (the squared deviation from the sample average) of a vector of data
     *
     * @param arr The vector of data values
     * @return The sum of squared error, or ILLEGAL_FLAG if the vector is empty
     */
    static double sample_sum_sq_err(const std::vector<double>& arr);
    
    /**
     * Compute the sum of squared error (the squared deviation from the sample average) of an array of data
     * This is equivalent to the TDoubleArrayList version in Java
     *
     * @param arr The array of data values
     * @param size The size of the array
     * @return The sum of squared error, or ILLEGAL_FLAG if the array is empty
     */
    static double sample_sum_sq_err(const double* arr, int size);
    
    /**
     * Find the index of the maximum value in a vector of integer data
     *
     * @param arr The vector of data values
     * @return The index of the maximum value
     * @throws std::runtime_error if the vector is empty
     */
    static int FindMaxIndex(const std::vector<int>& arr);
    
    /**
     * Draws a sample from an inverse gamma distribution.
     * 
     * @param k The shape parameter of the inverse gamma distribution of interest
     * @param theta The scale parameter of the inverse gamma distribution of interest
     * @return The sampled value
     */
    static double sample_from_inv_gamma(double k, double theta);
    
    /**
     * Draws a sample from a normal distribution.
     * 
     * @param mu The mean of the normal distribution of interest
     * @param sigsq The variance of the normal distribution of interest
     * @return The sample value
     */
    static double sample_from_norm_dist(double mu, double sigsq);
    
    /**
     * Sample from a multinomial distribution
     *
     * @param vals The integer values of the labels in this multinomial distribution
     * @param probs The probabilities of each label
     * @return A sample from the multinomial distribution
     */
    static int multinomial_sample(const std::vector<int>& vals, const std::vector<double>& probs);
};

#endif // STAT_TOOLBOX_H
