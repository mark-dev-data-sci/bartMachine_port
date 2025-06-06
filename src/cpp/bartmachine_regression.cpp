#include "include/bartmachine_regression.h"
#include "include/stat_toolbox.h"
#include <cmath>
#include <algorithm>

bartMachineRegression::bartMachineRegression() {
    // Initialize any regression-specific parameters here
}

bartMachineRegression::~bartMachineRegression() {
    // Clean up any regression-specific resources here
}

bool bartMachineRegression::build(double** X, double* y, int n, int p) {
    // Store the dimensions
    num_observations = n;
    num_predictors = p;
    
    // Store the training data
    X_train = X;
    y_train = y;
    
    // Initialize the model
    InitializeModel();
    
    // Run the Gibbs sampler
    RunGibbsSampler();
    
    return true;
}

double bartMachineRegression::Evaluate(double* record) {
    // Default to sample average evaluation
    return EvaluateViaSampAvg(record);
}

double bartMachineRegression::EvaluateViaSampAvg(double* record) {
    // Get Gibbs samples for prediction
    double* gibbs_samples = getGibbsSamplesForPrediction(record, 1);
    
    // Calculate the sample average
    double prediction = StatToolbox::sample_average(gibbs_samples, num_gibbs_after_burn_in);
    
    // Clean up
    delete[] gibbs_samples;
    
    return prediction;
}

double bartMachineRegression::EvaluateViaSampMed(double* record) {
    // Get Gibbs samples for prediction
    double* gibbs_samples = getGibbsSamplesForPrediction(record, 1);
    
    // Calculate the sample median
    double prediction = StatToolbox::sample_median(gibbs_samples, num_gibbs_after_burn_in);
    
    // Clean up
    delete[] gibbs_samples;
    
    return prediction;
}

double* bartMachineRegression::get95PctPostPredictiveIntervalForPrediction(double* record) {
    // Use 95% coverage
    return getPostPredictiveIntervalForPrediction(record, 0.95);
}

double* bartMachineRegression::getPostPredictiveIntervalForPrediction(double* record, double coverage) {
    // Get Gibbs samples for prediction
    double* gibbs_samples = getGibbsSamplesForPrediction(record, 1);
    
    // Calculate the interval
    double* interval = new double[2];
    
    // Sort the samples
    std::sort(gibbs_samples, gibbs_samples + num_gibbs_after_burn_in);
    
    // Calculate the lower and upper bounds
    double alpha = (1.0 - coverage) / 2.0;
    int lower_index = static_cast<int>(alpha * num_gibbs_after_burn_in);
    int upper_index = static_cast<int>((1.0 - alpha) * num_gibbs_after_burn_in);
    
    // Ensure indices are within bounds
    lower_index = std::max(0, lower_index);
    upper_index = std::min(num_gibbs_after_burn_in - 1, upper_index);
    
    // Set the interval bounds
    interval[0] = gibbs_samples[lower_index];
    interval[1] = gibbs_samples[upper_index];
    
    // Clean up
    delete[] gibbs_samples;
    
    return interval;
}
