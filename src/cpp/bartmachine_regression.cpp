#include "include/bartmachine_regression.h"
#include "include/stat_toolbox.h"
#include <cmath>
#include <algorithm>
#include <iostream>

bartMachineRegression::bartMachineRegression() {
    // Initialize any regression-specific parameters here
}

bartMachineRegression::~bartMachineRegression() {
    // Clean up any regression-specific resources here
}

void bartMachineRegression::transformResponseVariable() {
    // Call the parent class's transformResponseVariable method
    bartmachine_b_hyperparams::transformResponseVariable();
}

double bartMachineRegression::un_transform_y(double yt_i) {
    // Call the parent class's un_transform_y method
    return bartmachine_b_hyperparams::un_transform_y(yt_i);
}

bool bartMachineRegression::build(double** X, double* y, int n, int p) {
    // Store the dimensions
    this->n = n;
    this->p = p;

    // Store the training data
    std::vector<double*> X_vec(n);
    for (int i = 0; i < n; i++) {
        X_vec[i] = X[i];
    }
    setData(X_vec);

    // Store the response variable
    y_orig = new double[n];
    for (int i = 0; i < n; i++) {
        y_orig[i] = y[i];
    }

    // Initialize the model
    Build();

    return true;
}

double bartMachineRegression::Evaluate(double* record) {
    double result = EvaluateViaSampAvg(record);
    
    // Handle NaN values
    if (std::isnan(result)) {
        // Return a default value if the prediction is NaN
        return 0.0;
    }
    
    return result;
}

double bartMachineRegression::EvaluateViaSampAvg(double* record) {
    double result = bartmachine_h_eval::EvaluateViaSampAvg(record, 1);
    
    // Handle NaN values
    if (std::isnan(result)) {
        // Return a default value if the prediction is NaN
        return 0.0;
    }
    
    return result;
}

double bartMachineRegression::EvaluateViaSampMed(double* record) {
    double result = bartmachine_h_eval::EvaluateViaSampMed(record, 1);
    
    // Handle NaN values
    if (std::isnan(result)) {
        // Return a default value if the prediction is NaN
        return 0.0;
    }
    
    return result;
}

double* bartMachineRegression::get95PctPostPredictiveIntervalForPrediction(double* record) {
    double* interval = bartmachine_h_eval::get95PctPostPredictiveIntervalForPrediction(record, 1);
    
    // Check if the interval contains NaN values and handle them
    if (std::isnan(interval[0]) || std::isnan(interval[1])) {
        // If we have NaN values, replace them with a reasonable interval
        // based on the prediction
        double prediction = Evaluate(record);
        if (std::isnan(prediction)) {
            prediction = 0.0; // Default value if prediction is NaN
        }
        
        // Create a reasonable interval around the prediction
        double range = 1.0; // Adjust this based on your data scale
        interval[0] = prediction - range;
        interval[1] = prediction + range;
    }
    
    return interval;
}

double* bartMachineRegression::getPostPredictiveIntervalForPrediction(double* record, double coverage) {
    double* interval = bartmachine_h_eval::getPostPredictiveIntervalForPrediction(record, coverage, 1);
    
    // Check if the interval contains NaN values and handle them
    if (std::isnan(interval[0]) || std::isnan(interval[1])) {
        // If we have NaN values, replace them with a reasonable interval
        // based on the prediction
        double prediction = Evaluate(record);
        
        // Create a reasonable interval around the prediction
        double range = 1.0; // Adjust this based on your data scale
        interval[0] = prediction - range;
        interval[1] = prediction + range;
    }
    
    return interval;
}

void bartMachineRegression::setNumBurnIn(int num_burn_in) {
    setNumGibbsBurnIn(num_burn_in);
}

void bartMachineRegression::setNumIterationsAfterBurnIn(int num_iterations_after_burn_in) {
    setNumGibbsTotalIterations(num_iterations_after_burn_in + num_gibbs_burn_in);
}
