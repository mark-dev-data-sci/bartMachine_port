/**
 * bartmachine_h_eval.cpp
 * 
 * Implementation file for the bartmachine_h_eval class, which handles evaluation
 * and prediction functionality for the BART model.
 */

#include "include/bartmachine_h_eval.h"
#include <cmath>
#include <iostream>
#include <algorithm>

/**
 * Constructor
 */
bartmachine_h_eval::bartmachine_h_eval() : bartmachine_g_mh() {
    // Initialize prediction-related variables
    y_hat_train = nullptr;
    y_hat_test = nullptr;
    y_hat_train_credible_intervals_lower = nullptr;
    y_hat_train_credible_intervals_upper = nullptr;
    y_hat_test_credible_intervals_lower = nullptr;
    y_hat_test_credible_intervals_upper = nullptr;
    y_hat_train_prediction_intervals_lower = nullptr;
    y_hat_train_prediction_intervals_upper = nullptr;
    y_hat_test_prediction_intervals_lower = nullptr;
    y_hat_test_prediction_intervals_upper = nullptr;
    
    // Initialize evaluation metrics
    rmse_train = 0.0;
    rmse_test = 0.0;
    mse_train = 0.0;
    mse_test = 0.0;
}

/**
 * Destructor
 */
bartmachine_h_eval::~bartmachine_h_eval() {
    // Free memory for prediction-related variables
    if (y_hat_train != nullptr) delete[] y_hat_train;
    if (y_hat_test != nullptr) delete[] y_hat_test;
    if (y_hat_train_credible_intervals_lower != nullptr) delete[] y_hat_train_credible_intervals_lower;
    if (y_hat_train_credible_intervals_upper != nullptr) delete[] y_hat_train_credible_intervals_upper;
    if (y_hat_test_credible_intervals_lower != nullptr) delete[] y_hat_test_credible_intervals_lower;
    if (y_hat_test_credible_intervals_upper != nullptr) delete[] y_hat_test_credible_intervals_upper;
    if (y_hat_train_prediction_intervals_lower != nullptr) delete[] y_hat_train_prediction_intervals_lower;
    if (y_hat_train_prediction_intervals_upper != nullptr) delete[] y_hat_train_prediction_intervals_upper;
    if (y_hat_test_prediction_intervals_lower != nullptr) delete[] y_hat_test_prediction_intervals_lower;
    if (y_hat_test_prediction_intervals_upper != nullptr) delete[] y_hat_test_prediction_intervals_upper;
}

/**
 * Get predictions for a single tree
 * 
 * @param tree The tree to get predictions from
 * @param X The data to make predictions for
 * @param num_rows The number of rows in X
 * @return An array of predictions
 */
double* bartmachine_h_eval::getPredictionsForSingleTree(bartMachineTreeNode* tree, double** X, int num_rows) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
    double* predictions = new double[num_rows];
    for (int i = 0; i < num_rows; i++) {
        predictions[i] = 0.0;
    }
    return predictions;
}

/**
 * Get predictions for a single tree with missing data
 * 
 * @param tree The tree to get predictions from
 * @param X The data to make predictions for
 * @param num_rows The number of rows in X
 * @return An array of predictions
 */
double* bartmachine_h_eval::getPredictionsForSingleTreeWithMissingData(bartMachineTreeNode* tree, double** X, int num_rows) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
    double* predictions = new double[num_rows];
    for (int i = 0; i < num_rows; i++) {
        predictions[i] = 0.0;
    }
    return predictions;
}

/**
 * Calculate prediction intervals for training data
 * 
 * @param alpha The alpha level for the intervals
 */
void bartmachine_h_eval::calcPredictionIntervals(double alpha) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Calculate prediction intervals for test data
 * 
 * @param alpha The alpha level for the intervals
 */
void bartmachine_h_eval::calcPredictionIntervalsForTestData(double alpha) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Calculate credible intervals for training data
 * 
 * @param alpha The alpha level for the intervals
 */
void bartmachine_h_eval::calcCredibleIntervals(double alpha) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Calculate credible intervals for test data
 * 
 * @param alpha The alpha level for the intervals
 */
void bartmachine_h_eval::calcCredibleIntervalsForTestData(double alpha) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Calculate RMSE
 * 
 * @param y_hat The predicted values
 * @param y_actual The actual values
 * @param n The number of values
 * @return The RMSE
 */
double bartmachine_h_eval::calcRMSE(double* y_hat, double* y_actual, int n) {
    double sum_sq_err = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = y_hat[i] - y_actual[i];
        sum_sq_err += diff * diff;
    }
    return std::sqrt(sum_sq_err / n);
}

/**
 * Calculate MSE
 * 
 * @param y_hat The predicted values
 * @param y_actual The actual values
 * @param n The number of values
 * @return The MSE
 */
double bartmachine_h_eval::calcMSE(double* y_hat, double* y_actual, int n) {
    double sum_sq_err = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = y_hat[i] - y_actual[i];
        sum_sq_err += diff * diff;
    }
    return sum_sq_err / n;
}

/**
 * Get the training predictions
 * 
 * @return The training predictions
 */
double* bartmachine_h_eval::getYHatTrain() {
    return y_hat_train;
}

/**
 * Get the test predictions
 * 
 * @return The test predictions
 */
double* bartmachine_h_eval::getYHatTest() {
    return y_hat_test;
}

/**
 * Get the lower credible intervals for training data
 * 
 * @return The lower credible intervals
 */
double* bartmachine_h_eval::getYHatTrainCredibleIntervalsLower() {
    return y_hat_train_credible_intervals_lower;
}

/**
 * Get the upper credible intervals for training data
 * 
 * @return The upper credible intervals
 */
double* bartmachine_h_eval::getYHatTrainCredibleIntervalsUpper() {
    return y_hat_train_credible_intervals_upper;
}

/**
 * Get the lower credible intervals for test data
 * 
 * @return The lower credible intervals
 */
double* bartmachine_h_eval::getYHatTestCredibleIntervalsLower() {
    return y_hat_test_credible_intervals_lower;
}

/**
 * Get the upper credible intervals for test data
 * 
 * @return The upper credible intervals
 */
double* bartmachine_h_eval::getYHatTestCredibleIntervalsUpper() {
    return y_hat_test_credible_intervals_upper;
}

/**
 * Get the lower prediction intervals for training data
 * 
 * @return The lower prediction intervals
 */
double* bartmachine_h_eval::getYHatTrainPredictionIntervalsLower() {
    return y_hat_train_prediction_intervals_lower;
}

/**
 * Get the upper prediction intervals for training data
 * 
 * @return The upper prediction intervals
 */
double* bartmachine_h_eval::getYHatTrainPredictionIntervalsUpper() {
    return y_hat_train_prediction_intervals_upper;
}

/**
 * Get the lower prediction intervals for test data
 * 
 * @return The lower prediction intervals
 */
double* bartmachine_h_eval::getYHatTestPredictionIntervalsLower() {
    return y_hat_test_prediction_intervals_lower;
}

/**
 * Get the upper prediction intervals for test data
 * 
 * @return The upper prediction intervals
 */
double* bartmachine_h_eval::getYHatTestPredictionIntervalsUpper() {
    return y_hat_test_prediction_intervals_upper;
}

/**
 * Get the RMSE for training data
 * 
 * @return The RMSE
 */
double bartmachine_h_eval::getRMSETrain() {
    return rmse_train;
}

/**
 * Get the RMSE for test data
 * 
 * @return The RMSE
 */
double bartmachine_h_eval::getRMSETest() {
    return rmse_test;
}

/**
 * Get the MSE for training data
 * 
 * @return The MSE
 */
double bartmachine_h_eval::getMSETrain() {
    return mse_train;
}

/**
 * Get the MSE for test data
 * 
 * @return The MSE
 */
double bartmachine_h_eval::getMSETest() {
    return mse_test;
}
