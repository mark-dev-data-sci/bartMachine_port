#include "include/bartmachine_classification.h"
#include "include/stat_toolbox.h"
#include <cmath>
#include <algorithm>

bartMachineClassification::bartMachineClassification() {
    // Initialize any classification-specific parameters here
}

bartMachineClassification::~bartMachineClassification() {
    // Clean up any classification-specific resources here
}

bool bartMachineClassification::build(double** X, int* y, int n, int p) {
    // Store the dimensions
    num_observations = n;
    num_predictors = p;
    
    // Store the training data
    X_train = X;
    
    // Convert binary y to double for BART model
    double* y_double = new double[n];
    for (int i = 0; i < n; i++) {
        y_double[i] = static_cast<double>(y[i]);
    }
    
    y_train = y_double;
    
    // Initialize the model
    InitializeModel();
    
    // Run the Gibbs sampler
    RunGibbsSampler();
    
    return true;
}

double bartMachineClassification::getProbability(double* record) {
    // Get Gibbs samples for prediction
    double* gibbs_samples = getGibbsSamplesForPrediction(record, 1);
    
    // Calculate the average probability
    double sum = 0.0;
    for (int i = 0; i < num_gibbs_after_burn_in; i++) {
        // Apply sigmoid function to convert to probability
        double prob = 1.0 / (1.0 + exp(-gibbs_samples[i]));
        sum += prob;
    }
    
    double avg_prob = sum / num_gibbs_after_burn_in;
    
    // Clean up
    delete[] gibbs_samples;
    
    return avg_prob;
}

int bartMachineClassification::getPrediction(double* record) {
    // Use default threshold of 0.5
    return getPrediction(record, 0.5);
}

int bartMachineClassification::getPrediction(double* record, double threshold) {
    // Get probability
    double prob = getProbability(record);
    
    // Convert to class prediction
    return (prob >= threshold) ? 1 : 0;
}

double* bartMachineClassification::getProbabilities(double** X, int n) {
    // Allocate array for probabilities
    double* probs = new double[n];
    
    // Calculate probability for each observation
    for (int i = 0; i < n; i++) {
        probs[i] = getProbability(X[i]);
    }
    
    return probs;
}

int* bartMachineClassification::getPredictions(double** X, int n) {
    // Use default threshold of 0.5
    return getPredictions(X, n, 0.5);
}

int* bartMachineClassification::getPredictions(double** X, int n, double threshold) {
    // Allocate array for predictions
    int* preds = new int[n];
    
    // Calculate prediction for each observation
    for (int i = 0; i < n; i++) {
        preds[i] = getPrediction(X[i], threshold);
    }
    
    return preds;
}
