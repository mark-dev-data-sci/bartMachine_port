#include <Rcpp.h>
#include "../cpp/include/bartmachine_regression.h"
#include "../cpp/include/bartmachine_classification.h"
#include "../cpp/include/stat_toolbox.h"

/**
 * Rcpp interface for bartMachine
 * 
 * This file provides the R-to-C++ bridge for the bartMachine package.
 * It exposes C++ functions to R using Rcpp.
 */

// [[Rcpp::export]]
double rcpp_add(double x, double y) {
    return x + y;
}

// [[Rcpp::export]]
void rcpp_print(std::string message) {
    Rcpp::Rcout << "Message from R: " << message << std::endl;
}

// [[Rcpp::export]]
Rcpp::NumericVector rcpp_vector_double(Rcpp::NumericVector x) {
    Rcpp::NumericVector result(x.size());
    for (int i = 0; i < x.size(); i++) {
        result[i] = x[i] * 2.0;
    }
    return result;
}

// [[Rcpp::export]]
Rcpp::List rcpp_bartmachine_regression(Rcpp::NumericMatrix X, Rcpp::NumericVector y, Rcpp::NumericVector test_point) {
    // Get dimensions
    int n = X.nrow();
    int p = X.ncol();
    
    // Convert R matrix to C++ format
    double** X_cpp = new double*[n];
    for (int i = 0; i < n; i++) {
        X_cpp[i] = new double[p];
        for (int j = 0; j < p; j++) {
            X_cpp[i][j] = X(i, j);
        }
    }
    
    // Convert R vector to C++ format
    double* y_cpp = new double[n];
    for (int i = 0; i < n; i++) {
        y_cpp[i] = y[i];
    }
    
    // Convert test point to C++ format
    double* test_point_cpp = new double[p];
    for (int j = 0; j < p; j++) {
        test_point_cpp[j] = test_point[j];
    }
    
    // Create a bartMachineRegression instance
    bartMachineRegression regression;
    
    // Set hyperparameters
    regression.setNumTrees(50);
    regression.setNumBurnIn(250);
    regression.setNumIterationsAfterBurnIn(1000);
    
    // Build the model
    regression.build(X_cpp, y_cpp, n, p);
    
    // Test prediction
    double prediction = regression.Evaluate(test_point_cpp);
    
    // Test prediction intervals
    double* interval = regression.get95PctPostPredictiveIntervalForPrediction(test_point_cpp);
    
    // Create result list
    Rcpp::List result;
    result["prediction"] = prediction;
    result["lower"] = interval[0];
    result["upper"] = interval[1];
    
    // Clean up
    delete[] interval;
    delete[] test_point_cpp;
    delete[] y_cpp;
    for (int i = 0; i < n; i++) {
        delete[] X_cpp[i];
    }
    delete[] X_cpp;
    
    return result;
}

// [[Rcpp::export]]
Rcpp::List rcpp_bartmachine_classification(Rcpp::NumericMatrix X, Rcpp::IntegerVector y, Rcpp::NumericVector test_point) {
    // Get dimensions
    int n = X.nrow();
    int p = X.ncol();
    
    // Convert R matrix to C++ format
    double** X_cpp = new double*[n];
    for (int i = 0; i < n; i++) {
        X_cpp[i] = new double[p];
        for (int j = 0; j < p; j++) {
            X_cpp[i][j] = X(i, j);
        }
    }
    
    // Convert R vector to C++ format
    int* y_cpp = new int[n];
    for (int i = 0; i < n; i++) {
        y_cpp[i] = y[i];
    }
    
    // Convert test point to C++ format
    double* test_point_cpp = new double[p];
    for (int j = 0; j < p; j++) {
        test_point_cpp[j] = test_point[j];
    }
    
    // Create a bartMachineClassification instance
    bartMachineClassification classification;
    
    // Set hyperparameters
    classification.setNumTrees(50);
    classification.setNumBurnIn(250);
    classification.setNumIterationsAfterBurnIn(1000);
    
    // Build the model
    classification.build(X_cpp, y_cpp, n, p);
    
    // Test prediction
    double probability = classification.getProbability(test_point_cpp);
    int prediction = classification.getPrediction(test_point_cpp);
    
    // Create result list
    Rcpp::List result;
    result["probability"] = probability;
    result["prediction"] = prediction;
    
    // Clean up
    delete[] test_point_cpp;
    delete[] y_cpp;
    for (int i = 0; i < n; i++) {
        delete[] X_cpp[i];
    }
    delete[] X_cpp;
    
    return result;
}

// [[Rcpp::export]]
void rcpp_set_seed(int seed) {
    StatToolbox::setSeed(seed);
}

// [[Rcpp::export]]
double rcpp_rand() {
    return StatToolbox::rand();
}

// [[Rcpp::export]]
Rcpp::NumericVector rcpp_sample_from_norm_dist(double mu, double sigsq, int n) {
    Rcpp::NumericVector result(n);
    for (int i = 0; i < n; i++) {
        result[i] = StatToolbox::sample_from_norm_dist(mu, sigsq);
    }
    return result;
}

// [[Rcpp::export]]
Rcpp::NumericVector rcpp_sample_from_inv_gamma(double k, double theta, int n) {
    Rcpp::NumericVector result(n);
    for (int i = 0; i < n; i++) {
        result[i] = StatToolbox::sample_from_inv_gamma(k, theta);
    }
    return result;
}
