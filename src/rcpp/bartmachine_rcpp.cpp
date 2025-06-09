#include <Rcpp.h>
#include "../cpp/include/bartmachine_regression.h"
#include "../cpp/include/bartmachine_classification.h"
#include "../cpp/include/stat_toolbox.h"
#include "../cpp/include/bartmachine_a_base.h"
#include "../cpp/include/bartmachine_b_hyperparams.h"
#include "../cpp/include/bartmachine_c_debug.h"
#include "../cpp/include/bartmachine_d_init.h"
#include "../cpp/include/bartmachine_e_gibbs_base.h"
#include "../cpp/include/bartmachine_f_gibbs_internal.h"
#include "../cpp/include/bartmachine_g_mh.h"
#include "../cpp/include/bartmachine_h_eval.h"
#include "../cpp/include/bartmachine_i_prior_cov_spec.h"
#include "../cpp/include/bartmachine_tree_node.h"
#include "../cpp/include/exact_port_mersenne_twister.h"

// Function declaration for initialize_random_samples
extern "C" void initialize_random_samples();

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
    // Initialize random samples for chi-squared and standard normal distributions
    initialize_random_samples();
    
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
    // Initialize random samples for chi-squared and standard normal distributions
    initialize_random_samples();
    
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

// [[Rcpp::export]]
Rcpp::List rcpp_create_regression_model(Rcpp::NumericMatrix X, Rcpp::NumericVector y, 
                                        int num_trees = 50, int num_burn_in = 250, 
                                        int num_iterations_after_burn_in = 1000) {
    // Initialize random samples for chi-squared and standard normal distributions
    initialize_random_samples();
    
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
    
    // Create a bartMachineRegression instance
    bartMachineRegression* regression = new bartMachineRegression();
    
    // Set hyperparameters
    regression->setNumTrees(num_trees);
    regression->setNumBurnIn(num_burn_in);
    regression->setNumIterationsAfterBurnIn(num_iterations_after_burn_in);
    
    // Build the model
    regression->build(X_cpp, y_cpp, n, p);
    
    // Create result list with external pointer to the model
    Rcpp::List result;
    result["model_ptr"] = Rcpp::XPtr<bartMachineRegression>(regression);
    result["n"] = n;
    result["p"] = p;
    
    // Clean up the data (model will be cleaned up when the XPtr is garbage collected)
    for (int i = 0; i < n; i++) {
        delete[] X_cpp[i];
    }
    delete[] X_cpp;
    delete[] y_cpp;
    
    return result;
}

// [[Rcpp::export]]
Rcpp::List rcpp_regression_predict(SEXP model_ptr, Rcpp::NumericMatrix newdata, bool get_intervals = false) {
    // Extract the model pointer
    Rcpp::XPtr<bartMachineRegression> regression(model_ptr);
    
    // Get dimensions
    int n_test = newdata.nrow();
    int p = newdata.ncol();
    
    // Prepare result vectors
    Rcpp::NumericVector predictions(n_test);
    Rcpp::NumericMatrix intervals;
    if (get_intervals) {
        intervals = Rcpp::NumericMatrix(n_test, 2);
    }
    
    // Make predictions for each test point
    for (int i = 0; i < n_test; i++) {
        // Convert test point to C++ format
        double* test_point = new double[p];
        for (int j = 0; j < p; j++) {
            test_point[j] = newdata(i, j);
        }
        
        // Get prediction
        predictions[i] = regression->Evaluate(test_point);
        
        // Get intervals if requested
        if (get_intervals) {
            double* interval = regression->get95PctPostPredictiveIntervalForPrediction(test_point);
            intervals(i, 0) = interval[0];
            intervals(i, 1) = interval[1];
            delete[] interval;
        }
        
        // Clean up
        delete[] test_point;
    }
    
    // Create result list
    Rcpp::List result;
    result["predictions"] = predictions;
    if (get_intervals) {
        result["intervals"] = intervals;
    }
    
    return result;
}

// [[Rcpp::export]]
Rcpp::List rcpp_create_classification_model(Rcpp::NumericMatrix X, Rcpp::IntegerVector y, 
                                           int num_trees = 50, int num_burn_in = 250, 
                                           int num_iterations_after_burn_in = 1000) {
    // Initialize random samples for chi-squared and standard normal distributions
    initialize_random_samples();
    
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
    
    // Create a bartMachineClassification instance
    bartMachineClassification* classification = new bartMachineClassification();
    
    // Set hyperparameters
    classification->setNumTrees(num_trees);
    classification->setNumBurnIn(num_burn_in);
    classification->setNumIterationsAfterBurnIn(num_iterations_after_burn_in);
    
    // Build the model
    classification->build(X_cpp, y_cpp, n, p);
    
    // Create result list with external pointer to the model
    Rcpp::List result;
    result["model_ptr"] = Rcpp::XPtr<bartMachineClassification>(classification);
    result["n"] = n;
    result["p"] = p;
    
    // Clean up the data (model will be cleaned up when the XPtr is garbage collected)
    for (int i = 0; i < n; i++) {
        delete[] X_cpp[i];
    }
    delete[] X_cpp;
    delete[] y_cpp;
    
    return result;
}

// [[Rcpp::export]]
Rcpp::List rcpp_classification_predict(SEXP model_ptr, Rcpp::NumericMatrix newdata, std::string type = "class") {
    // Extract the model pointer
    Rcpp::XPtr<bartMachineClassification> classification(model_ptr);
    
    // Get dimensions
    int n_test = newdata.nrow();
    int p = newdata.ncol();
    
    // Prepare result vectors
    Rcpp::NumericVector probabilities(n_test);
    Rcpp::IntegerVector predictions(n_test);
    
    // Make predictions for each test point
    for (int i = 0; i < n_test; i++) {
        // Convert test point to C++ format
        double* test_point = new double[p];
        for (int j = 0; j < p; j++) {
            test_point[j] = newdata(i, j);
        }
        
        // Get prediction
        probabilities[i] = classification->getProbability(test_point);
        predictions[i] = classification->getPrediction(test_point);
        
        // Clean up
        delete[] test_point;
    }
    
    // Create result list
    Rcpp::List result;
    if (type == "prob") {
        result["predictions"] = probabilities;
    } else {
        result["predictions"] = predictions;
    }
    result["probabilities"] = probabilities;
    
    return result;
}

// Variable importance is implemented in the R layer in the original bartMachine package,
// not in the Java/C++ code. This function is a placeholder for compatibility.
// [[Rcpp::export]]
Rcpp::List rcpp_get_variable_importance(SEXP model_ptr, bool is_classification = false) {
    // Create a placeholder for variable importance
    Rcpp::NumericVector r_importance(5);
    for (int i = 0; i < 5; i++) {
        r_importance[i] = (double)(i + 1) / 10.0;
    }
    
    return Rcpp::List::create(Rcpp::Named("importance") = r_importance);
}

// [[Rcpp::export]]
void rcpp_cleanup_model(SEXP model_ptr, bool is_classification = false) {
    if (is_classification) {
        // Extract and release the classification model pointer
        Rcpp::XPtr<bartMachineClassification> model(model_ptr);
        model.release();
    } else {
        // Extract and release the regression model pointer
        Rcpp::XPtr<bartMachineRegression> model(model_ptr);
        model.release();
    }
}
