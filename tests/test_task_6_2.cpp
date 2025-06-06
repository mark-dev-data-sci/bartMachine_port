#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#include "../src/cpp/include/bartmachine_regression.h"
#include "../src/cpp/include/bartmachine_classification.h"
#include "../src/cpp/include/stat_toolbox.h"

/**
 * Test file for Task 6.2: Regression and Classification
 * 
 * This test validates the implementation of the specialized bartMachine classes
 * for regression and classification.
 */

void test_bartmachine_regression() {
    std::cout << "Testing bartMachineRegression class..." << std::endl;
    
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    // Create a simple regression problem
    int n = 100;  // number of observations
    int p = 5;    // number of predictors
    
    // Create X matrix (n x p)
    std::vector<std::vector<double>> X(n, std::vector<double>(p));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            X[i][j] = StatToolbox::rand();
        }
    }
    
    // Create y vector (n x 1)
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        // Simple linear relationship with noise
        y[i] = 2.0 * X[i][0] + 1.5 * X[i][1] - 0.5 * X[i][2] + 0.1 * StatToolbox::rand();
    }
    
    // Convert to the format expected by bartMachineRegression
    double** X_array = new double*[n];
    for (int i = 0; i < n; i++) {
        X_array[i] = new double[p];
        for (int j = 0; j < p; j++) {
            X_array[i][j] = X[i][j];
        }
    }
    
    double* y_array = new double[n];
    for (int i = 0; i < n; i++) {
        y_array[i] = y[i];
    }
    
    // Create a bartMachineRegression instance
    bartMachineRegression regression;
    
    // Set hyperparameters
    regression.setNumTrees(50);
    regression.setNumBurnIn(250);
    regression.setNumIterationsAfterBurnIn(1000);
    
    // Build the model
    regression.build(X_array, y_array, n, p);
    
    // Test prediction
    double* test_point = new double[p];
    for (int j = 0; j < p; j++) {
        test_point[j] = 0.5;  // Test with all predictors = 0.5
    }
    
    double prediction = regression.Evaluate(test_point);
    std::cout << "Prediction for test point: " << prediction << std::endl;
    
    // Test prediction intervals
    double* interval = regression.get95PctPostPredictiveIntervalForPrediction(test_point);
    std::cout << "95% prediction interval: [" << interval[0] << ", " << interval[1] << "]" << std::endl;
    
    // Verify that the prediction is within the interval
    assert(prediction >= interval[0] && prediction <= interval[1]);
    
    // Clean up
    delete[] interval;
    delete[] test_point;
    for (int i = 0; i < n; i++) {
        delete[] X_array[i];
    }
    delete[] X_array;
    delete[] y_array;
    
    std::cout << "PASSED" << std::endl;
}

void test_bartmachine_classification() {
    std::cout << "Testing bartMachineClassification class..." << std::endl;
    
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    // Create a simple classification problem
    int n = 100;  // number of observations
    int p = 5;    // number of predictors
    
    // Create X matrix (n x p)
    std::vector<std::vector<double>> X(n, std::vector<double>(p));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            X[i][j] = StatToolbox::rand();
        }
    }
    
    // Create y vector (n x 1) with binary labels
    std::vector<int> y(n);
    for (int i = 0; i < n; i++) {
        // Simple decision boundary
        if (X[i][0] + X[i][1] > 1.0) {
            y[i] = 1;  // Class 1
        } else {
            y[i] = 0;  // Class 0
        }
    }
    
    // Convert to the format expected by bartMachineClassification
    double** X_array = new double*[n];
    for (int i = 0; i < n; i++) {
        X_array[i] = new double[p];
        for (int j = 0; j < p; j++) {
            X_array[i][j] = X[i][j];
        }
    }
    
    int* y_array = new int[n];
    for (int i = 0; i < n; i++) {
        y_array[i] = y[i];
    }
    
    // Create a bartMachineClassification instance
    bartMachineClassification classification;
    
    // Set hyperparameters
    classification.setNumTrees(50);
    classification.setNumBurnIn(250);
    classification.setNumIterationsAfterBurnIn(1000);
    
    // Build the model
    classification.build(X_array, y_array, n, p);
    
    // Test prediction
    double* test_point = new double[p];
    for (int j = 0; j < p; j++) {
        test_point[j] = 0.6;  // Test with all predictors = 0.6
    }
    
    // Get class probability
    double prob = classification.getProbability(test_point);
    std::cout << "Probability for test point: " << prob << std::endl;
    
    // Get class prediction
    int pred_class = classification.getPrediction(test_point);
    std::cout << "Predicted class for test point: " << pred_class << std::endl;
    
    // Verify that the prediction is consistent with the probability
    if (prob >= 0.5) {
        assert(pred_class == 1);
    } else {
        assert(pred_class == 0);
    }
    
    // Clean up
    delete[] test_point;
    for (int i = 0; i < n; i++) {
        delete[] X_array[i];
    }
    delete[] X_array;
    delete[] y_array;
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Testing Task 6.2: Regression and Classification" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    test_bartmachine_regression();
    test_bartmachine_classification();
    
    std::cout << std::endl << "All tests PASSED!" << std::endl;
    
    return 0;
}
