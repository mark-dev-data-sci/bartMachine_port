#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#include "src/cpp/include/exact_port_mersenne_twister.h"
#include "src/cpp/include/stat_toolbox.h"
#include "src/cpp/include/bartmachine_b_hyperparams.h"
#include "src/cpp/include/bartmachine_tree_node.h"
#include "src/cpp/include/classifier.h"

// Test helper functions
bool test_random_split_value();
bool test_random_direction_for_missing_data();
bool test_predictors_that_could_be_used_to_split();
bool test_possible_split_values();

// Main test function
int main() {
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    std::cout << "Testing Task 4.3: bartMachineTreeNode - Random Operations" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    bool all_tests_passed = true;
    
    // Test random split value selection
    std::cout << "Testing pickRandomSplitValue()... ";
    bool split_value_test = test_random_split_value();
    std::cout << (split_value_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= split_value_test;
    
    // Test random direction for missing data
    std::cout << "Testing pickRandomDirectionForMissingData()... ";
    bool direction_test = test_random_direction_for_missing_data();
    std::cout << (direction_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= direction_test;
    
    // Test predictors that could be used to split
    std::cout << "Testing predictorsThatCouldBeUsedToSplitAtNode()... ";
    bool predictors_test = test_predictors_that_could_be_used_to_split();
    std::cout << (predictors_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= predictors_test;
    
    // Test possible split values
    std::cout << "Testing possibleSplitValuesGivenAttribute()... ";
    bool split_values_test = test_possible_split_values();
    std::cout << (split_values_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= split_values_test;
    
    if (all_tests_passed) {
        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests FAILED!" << std::endl;
        return 1;
    }
}

// Test implementation for pickRandomSplitValue
bool test_random_split_value() {
    // Create a simple dataset with 10 observations and 3 features
    int n = 10;
    int p = 3;
    
    // Create X_y data
    std::vector<double*> X_y;
    for (int i = 0; i < n; i++) {
        double* row = new double[p + 1]; // +1 for y
        row[0] = i % 5;          // Feature 1: values 0,1,2,3,4,0,1,2,3,4
        row[1] = i < 5 ? 0 : 1;  // Feature 2: values 0,0,0,0,0,1,1,1,1,1
        row[2] = i;              // Feature 3: values 0,1,2,3,4,5,6,7,8,9
        row[p] = i;              // Response: values 0,1,2,3,4,5,6,7,8,9
        X_y.push_back(row);
    }
    
    // Create X_y_by_col data
    std::vector<double*> X_y_by_col;
    for (int j = 0; j < p + 1; j++) {
        double* col = new double[n];
        for (int i = 0; i < n; i++) {
            col[i] = X_y[i][j];
        }
        X_y_by_col.push_back(col);
    }
    
    // Create hyperparameters
    bartmachine_b_hyperparams* bart = new bartmachine_b_hyperparams();
    bart->setData(X_y);
    bart->setXYByCol(X_y_by_col); // Set X_y_by_col using the setter
    bart->setMemCacheForSpeed(true);
    
    // Create a tree node
    bartMachineTreeNode* node = new bartMachineTreeNode(bart);
    
    // Set up the node with data
    double* y_trans = new double[n];
    for (int i = 0; i < n; i++) {
        y_trans[i] = X_y[i][p];
    }
    node->setStumpData(X_y, y_trans, p);
    
    // Test pickRandomSplitValue for each feature
    bool test_passed = true;
    
    // For feature 0 (values 0,1,2,3,4)
    node->setSplitAttributeM(0);
    double split_val_0 = node->pickRandomSplitValue();
    // Valid split values are 0,1,2,3 (not 4 as it's the max)
    test_passed &= (split_val_0 == 0 || split_val_0 == 1 || split_val_0 == 2 || split_val_0 == 3);
    
    // For feature 1 (values 0,1)
    node->setSplitAttributeM(1);
    double split_val_1 = node->pickRandomSplitValue();
    // Valid split value is only 0 (not 1 as it's the max)
    test_passed &= (split_val_1 == 0);
    
    // For feature 2 (values 0,1,2,3,4,5,6,7,8,9)
    node->setSplitAttributeM(2);
    double split_val_2 = node->pickRandomSplitValue();
    // Valid split values are 0,1,2,3,4,5,6,7,8 (not 9 as it's the max)
    test_passed &= (split_val_2 >= 0 && split_val_2 <= 8);
    
    // Clean up
    delete[] y_trans;
    delete node;
    delete bart;
    for (auto row : X_y) {
        delete[] row;
    }
    for (auto col : X_y_by_col) {
        delete[] col;
    }
    
    return test_passed;
}

// Test implementation for pickRandomDirectionForMissingData
bool test_random_direction_for_missing_data() {
    // Create a tree node
    bartMachineTreeNode* node = new bartMachineTreeNode();
    
    // Call pickRandomDirectionForMissingData multiple times and check distribution
    int num_trials = 1000;
    int count_true = 0;
    
    for (int i = 0; i < num_trials; i++) {
        bool direction = node->pickRandomDirectionForMissingData();
        if (direction) {
            count_true++;
        }
    }
    
    // Check if the distribution is roughly 50/50
    // Allow for some randomness, but it should be close to 50%
    double proportion_true = static_cast<double>(count_true) / num_trials;
    bool test_passed = (proportion_true > 0.45 && proportion_true < 0.55);
    
    // Debug output
    std::cout << "\nDEBUG: pickRandomDirectionForMissingData() test" << std::endl;
    std::cout << "  Proportion of true values: " << proportion_true << std::endl;
    std::cout << "  Count of true values: " << count_true << " out of " << num_trials << std::endl;
    
    // Clean up
    delete node;
    
    return test_passed;
}

// Test implementation for random split attribute selection
// This indirectly tests predictorsThatCouldBeUsedToSplitAtNode
bool test_predictors_that_could_be_used_to_split() {
    // Create a simple dataset with 10 observations and 3 features
    int n = 10;
    int p = 3;
    
    // Create X_y data
    std::vector<double*> X_y;
    for (int i = 0; i < n; i++) {
        double* row = new double[p + 1]; // +1 for y
        row[0] = i % 5;          // Feature 1: values 0,1,2,3,4,0,1,2,3,4
        row[1] = i < 5 ? 0 : 1;  // Feature 2: values 0,0,0,0,0,1,1,1,1,1
        row[2] = 1.0;            // Feature 3: all values are 1 (no split possible)
        row[p] = i;              // Response: values 0,1,2,3,4,5,6,7,8,9
        X_y.push_back(row);
    }
    
    // Create X_y_by_col data
    std::vector<double*> X_y_by_col;
    for (int j = 0; j < p + 1; j++) {
        double* col = new double[n];
        for (int i = 0; i < n; i++) {
            col[i] = X_y[i][j];
        }
        X_y_by_col.push_back(col);
    }
    
    // Create hyperparameters
    bartmachine_b_hyperparams* bart = new bartmachine_b_hyperparams();
    bart->setData(X_y);
    bart->setXYByCol(X_y_by_col); // Set X_y_by_col using the setter
    bart->setMemCacheForSpeed(true);
    
    // Create a tree node
    bartMachineTreeNode* node = new bartMachineTreeNode(bart);
    
    // Set up the node with data
    double* y_trans = new double[n];
    for (int i = 0; i < n; i++) {
        y_trans[i] = X_y[i][p];
    }
    node->setStumpData(X_y, y_trans, p);
    
    // Test multiple random split attribute selections
    // We'll do this 100 times and check that we only get attributes 0 and 1
    // (since attribute 2 has all the same value and can't be used for splitting)
    bool test_passed = true;
    std::unordered_set<int> observed_attributes;
    
    for (int i = 0; i < 100; i++) {
        // Set a random attribute
        node->setSplitAttributeM(i % p); // Cycle through all attributes
        
        // Try to get a random split value
        double split_val = node->pickRandomSplitValue();
        
        // If we got a valid split value, record the attribute
        if (split_val != bartMachineTreeNode::BAD_FLAG_double) {
            observed_attributes.insert(node->getSplitAttributeM());
        }
    }
    
    // We should have observed only attributes 0 and 1
    test_passed = (observed_attributes.size() == 2 && 
                  observed_attributes.count(0) == 1 && 
                  observed_attributes.count(1) == 1);
    
    // Clean up
    delete[] y_trans;
    delete node;
    delete bart;
    for (auto row : X_y) {
        delete[] row;
    }
    for (auto col : X_y_by_col) {
        delete[] col;
    }
    
    return test_passed;
}

// Test implementation for possibleSplitValuesGivenAttribute
// This indirectly tests possibleSplitValuesGivenAttribute by using pickRandomSplitValue
bool test_possible_split_values() {
    // Create a simple dataset with 10 observations and 3 features
    int n = 10;
    int p = 3;
    
    // Create X_y data
    std::vector<double*> X_y;
    for (int i = 0; i < n; i++) {
        double* row = new double[p + 1]; // +1 for y
        row[0] = i % 5;          // Feature 1: values 0,1,2,3,4,0,1,2,3,4
        row[1] = i < 5 ? 0 : 1;  // Feature 2: values 0,0,0,0,0,1,1,1,1,1
        row[2] = 1.0;            // Feature 3: all values are 1 (no split possible)
        row[p] = i;              // Response: values 0,1,2,3,4,5,6,7,8,9
        X_y.push_back(row);
    }
    
    // Create X_y_by_col data
    std::vector<double*> X_y_by_col;
    for (int j = 0; j < p + 1; j++) {
        double* col = new double[n];
        for (int i = 0; i < n; i++) {
            col[i] = X_y[i][j];
        }
        X_y_by_col.push_back(col);
    }
    
    // Create hyperparameters
    bartmachine_b_hyperparams* bart = new bartmachine_b_hyperparams();
    bart->setData(X_y);
    bart->setXYByCol(X_y_by_col); // Set X_y_by_col using the setter
    bart->setMemCacheForSpeed(true);
    
    // Create a tree node
    bartMachineTreeNode* node = new bartMachineTreeNode(bart);
    
    // Set up the node with data
    double* y_trans = new double[n];
    for (int i = 0; i < n; i++) {
        y_trans[i] = X_y[i][p];
    }
    node->setStumpData(X_y, y_trans, p);
    
    bool test_passed = true;
    
    // Test for feature 0 (values 0,1,2,3,4)
    node->setSplitAttributeM(0);
    
    // Call pickRandomSplitValue multiple times and collect the results
    std::unordered_set<double> observed_splits_0;
    for (int i = 0; i < 100; i++) {
        double split_val = node->pickRandomSplitValue();
        if (split_val != bartMachineTreeNode::BAD_FLAG_double) {
            observed_splits_0.insert(split_val);
        }
    }
    
    // We should observe values 0, 1, 2, 3 (not 4 as it's the max)
    test_passed &= (observed_splits_0.size() == 4);
    
    // Debug output
    std::cout << "\nDEBUG: possibleSplitValuesGivenAttribute() test for feature 0" << std::endl;
    std::cout << "  Observed splits: ";
    for (double val : observed_splits_0) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "  Number of unique splits: " << observed_splits_0.size() << std::endl;
    
    if (test_passed && observed_splits_0.size() == 4) {
        test_passed &= (observed_splits_0.count(0) == 1 && 
                        observed_splits_0.count(1) == 1 && 
                        observed_splits_0.count(2) == 1 && 
                        observed_splits_0.count(3) == 1);
    }
    
    // Test for feature 1 (values 0,1)
    node->setSplitAttributeM(1);
    
    // Call pickRandomSplitValue multiple times and collect the results
    std::unordered_set<double> observed_splits_1;
    for (int i = 0; i < 100; i++) {
        double split_val = node->pickRandomSplitValue();
        if (split_val != bartMachineTreeNode::BAD_FLAG_double) {
            observed_splits_1.insert(split_val);
        }
    }
    
    // We should observe only value 0 (not 1 as it's the max)
    test_passed &= (observed_splits_1.size() == 1 && observed_splits_1.count(0) == 1);
    
    // Test for feature 2 (all values are 1)
    node->setSplitAttributeM(2);
    
    // Call pickRandomSplitValue multiple times and collect the results
    bool all_bad_flags = true;
    for (int i = 0; i < 10; i++) {
        double split_val = node->pickRandomSplitValue();
        if (split_val != bartMachineTreeNode::BAD_FLAG_double) {
            all_bad_flags = false;
            break;
        }
    }
    
    // We should only get BAD_FLAG_double since no split is possible
    test_passed &= all_bad_flags;
    
    // Clean up
    delete[] y_trans;
    delete node;
    delete bart;
    for (auto row : X_y) {
        delete[] row;
    }
    for (auto col : X_y_by_col) {
        delete[] col;
    }
    
    return test_passed;
}
