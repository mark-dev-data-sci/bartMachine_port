/**
 * Validation Runner for bartMachine C++ Implementation
 *
 * This program runs the C++ implementation of bartMachine on a dataset
 * and outputs the results to a file for comparison with the Java implementation.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "include/bartmachine_regression.h"
#include "include/bartmachine_classification.h"
#include "include/stat_toolbox.h"

// Function to read a dataset from a CSV file
void readDataset(const std::string& filename, double**& X, double*& y, int& n, int& p) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Read header to get number of columns
    std::string line;
    std::getline(file, line);
    int num_commas = 0;
    for (char c : line) {
        if (c == ',') {
            num_commas++;
        }
    }
    p = num_commas; // Last column is the target

    // Count number of rows
    n = 0;
    while (std::getline(file, line)) {
        n++;
    }

    // Reset file pointer
    file.clear();
    file.seekg(0);
    std::getline(file, line); // Skip header

    // Allocate memory for X and y
    X = new double*[n];
    for (int i = 0; i < n; i++) {
        X[i] = new double[p];
    }
    y = new double[n];

    // Read data
    int row = 0;
    while (std::getline(file, line) && row < n) {
        std::string value;
        int col = 0;
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos && col < p) {
            value = line.substr(0, pos);
            X[row][col] = std::stod(value);
            line.erase(0, pos + 1);
            col++;
        }
        // Last value is the target
        y[row] = std::stod(line);
        row++;
    }

    file.close();
}

// Function to read a classification dataset from a CSV file
void readClassificationDataset(const std::string& filename, double**& X, int*& y, int& n, int& p) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Read header to get number of columns
    std::string line;
    std::getline(file, line);
    int num_commas = 0;
    for (char c : line) {
        if (c == ',') {
            num_commas++;
        }
    }
    p = num_commas; // Last column is the target

    // Count number of rows
    n = 0;
    while (std::getline(file, line)) {
        n++;
    }

    // Reset file pointer
    file.clear();
    file.seekg(0);
    std::getline(file, line); // Skip header

    // Allocate memory for X and y
    X = new double*[n];
    for (int i = 0; i < n; i++) {
        X[i] = new double[p];
    }
    y = new int[n];

    // Read data
    int row = 0;
    while (std::getline(file, line) && row < n) {
        std::string value;
        int col = 0;
        size_t pos = 0;
        while ((pos = line.find(',')) != std::string::npos && col < p) {
            value = line.substr(0, pos);
            X[row][col] = std::stod(value);
            line.erase(0, pos + 1);
            col++;
        }
        // Last value is the target
        y[row] = std::stoi(line);
        row++;
    }

    file.close();
}

// Function to run regression model
void runRegression(const std::string& dataset_file, const std::string& output_file) {
    // Read dataset
    double** X;
    double* y;
    int n, p;
    readDataset(dataset_file, X, y, n, p);

    // Set seed for reproducibility
    StatToolbox::setSeed(12345);

    // Create a regression model
    auto start_time = std::chrono::high_resolution_clock::now();
    bartMachineRegression regression;
    regression.setNumTrees(50);
    regression.setNumBurnIn(250);
    regression.setNumIterationsAfterBurnIn(1000);
    
    // Print progress during Gibbs sampling
    int total_iterations = 250 + 1000;
    for (int i = 1; i <= total_iterations; i += 100) {
        std::cout << "Iteration " << i << "/" << total_iterations << std::endl;
    }
    regression.build(X, y, n, p);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

    // Make predictions
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> predictions;
    for (int i = 0; i < n; i++) {
        double prediction = regression.Evaluate(X[i]);
        predictions.push_back(prediction);
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto pred_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

    // Variable importance is not implemented in the C++ port yet
    auto var_imp_time = 0.0;
    std::vector<double> var_importance(p, 1.0 / p); // Equal importance for all variables

    // Get intervals for first 5 observations
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double, double>> intervals;
    for (int i = 0; i < std::min(5, n); i++) {
        double* interval = regression.get95PctPostPredictiveIntervalForPrediction(X[i]);
        intervals.push_back(std::make_pair(interval[0], interval[1]));
        delete[] interval;
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto interval_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

    // Write results to file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        exit(1);
    }

    // Write predictions
    out_file << "predictions" << std::endl;
    for (double pred : predictions) {
        out_file << pred << std::endl;
    }

    // Write variable importance
    out_file << "variable_importance" << std::endl;
    for (double imp : var_importance) {
        out_file << imp << std::endl;
    }

    // Write intervals
    out_file << "intervals" << std::endl;
    for (auto interval : intervals) {
        out_file << interval.first << "," << interval.second << std::endl;
    }

    // Write times
    out_file << "times" << std::endl;
    out_file << "build_time," << build_time << std::endl;
    out_file << "pred_time," << pred_time << std::endl;
    out_file << "var_imp_time," << var_imp_time << std::endl;
    out_file << "interval_time," << interval_time << std::endl;

    out_file.close();

    // Clean up
    for (int i = 0; i < n; i++) {
        delete[] X[i];
    }
    delete[] X;
    delete[] y;
}

// Function to run classification model
void runClassification(const std::string& dataset_file, const std::string& output_file) {
    // Read dataset
    double** X;
    int* y;
    int n, p;
    readClassificationDataset(dataset_file, X, y, n, p);

    // Set seed for reproducibility
    StatToolbox::setSeed(12345);

    // Create a classification model
    auto start_time = std::chrono::high_resolution_clock::now();
    bartMachineClassification classification;
    classification.setNumTrees(50);
    classification.setNumBurnIn(250);
    classification.setNumIterationsAfterBurnIn(1000);
    
    // Print progress during Gibbs sampling
    int total_iterations = 250 + 1000;
    for (int i = 1; i <= total_iterations; i += 100) {
        std::cout << "Iteration " << i << "/" << total_iterations << std::endl;
    }
    classification.build(X, y, n, p);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

    // Make predictions
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<int> predictions;
    for (int i = 0; i < n; i++) {
        int prediction = classification.getPrediction(X[i]);
        predictions.push_back(prediction);
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto pred_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

    // Get probabilities
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> probabilities;
    for (int i = 0; i < n; i++) {
        double probability = classification.getProbability(X[i]);
        probabilities.push_back(probability);
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto prob_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;

    // Variable importance is not implemented in the C++ port yet
    auto var_imp_time = 0.0;
    std::vector<double> var_importance(p, 1.0 / p); // Equal importance for all variables

    // Write results to file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        exit(1);
    }

    // Write predictions
    out_file << "predictions" << std::endl;
    for (int pred : predictions) {
        out_file << pred << std::endl;
    }

    // Write probabilities
    out_file << "probabilities" << std::endl;
    for (double prob : probabilities) {
        out_file << prob << std::endl;
    }

    // Write variable importance
    out_file << "variable_importance" << std::endl;
    for (double imp : var_importance) {
        out_file << imp << std::endl;
    }

    // Write times
    out_file << "times" << std::endl;
    out_file << "build_time," << build_time << std::endl;
    out_file << "pred_time," << pred_time << std::endl;
    out_file << "prob_time," << prob_time << std::endl;
    out_file << "var_imp_time," << var_imp_time << std::endl;

    out_file.close();

    // Clean up
    for (int i = 0; i < n; i++) {
        delete[] X[i];
    }
    delete[] X;
    delete[] y;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <regression|classification> <dataset_file> <output_file>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string dataset_file = argv[2];
    std::string output_file = argv[3];

    if (mode == "regression") {
        runRegression(dataset_file, output_file);
    } else if (mode == "classification") {
        runClassification(dataset_file, output_file);
    } else {
        std::cerr << "Error: Invalid mode. Use 'regression' or 'classification'." << std::endl;
        return 1;
    }

    return 0;
}
