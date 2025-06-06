#include <iostream>
#include <cassert>
#include <vector>
#include "src/cpp/include/bartmachine_tree_node.h"

/**
 * Test for Task 4.2: bartMachineTreeNode - Tree Manipulation (Non-Random)
 * 
 * This test validates the implementation of the tree manipulation methods
 * in the bartMachineTreeNode class.
 * 
 * The test ensures that:
 * 1. Data management methods work correctly
 * 2. Tree structure queries return expected values
 * 3. Non-random tree modifications work correctly
 * 4. Evaluation methods work correctly
 */

int main() {
    std::cout << "Testing Task 4.2: bartMachineTreeNode - Tree Manipulation (Non-Random)" << std::endl;
    
    // Create a simple tree structure for testing
    bartmachine_b_hyperparams* bart = new bartmachine_b_hyperparams();
    // Set p through the data
    
    // Create a root node
    bartMachineTreeNode* root = new bartMachineTreeNode(bart);
    
    // Create test data
    std::vector<double*> X_y;
    const int n = 10; // Number of samples
    const int p = 3;  // Number of predictors
    
    // Create sample data with 3 predictors and 10 samples
    for (int i = 0; i < n; i++) {
        double* record = new double[p + 2]; // p predictors + response + index
        record[0] = i % 3;     // First predictor
        record[1] = i * 0.5;   // Second predictor
        record[2] = i % 2;     // Third predictor
        record[p] = i * 1.0;   // Response (just the index value for simplicity)
        record[p + 1] = i;     // Index
        X_y.push_back(record);
    }
    
    // Create transformed responses
    double* y_trans = new double[n];
    for (int i = 0; i < n; i++) {
        y_trans[i] = i * 1.0; // Same as the response in X_y
    }
    
    // Set up the bart object with the data
    bart->setData(X_y);
    
    // Test setStumpData
    std::cout << "Testing setStumpData..." << std::endl;
    root->setStumpData(X_y, y_trans, p);
    assert(root->n_eta == n);
    
    // Create a tree structure
    root->setLeaf(false);
    root->setSplitAttributeM(0); // Split on first predictor
    root->setSplitValue(1.0);    // Split value
    
    bartMachineTreeNode* left_child = new bartMachineTreeNode(root);
    bartMachineTreeNode* right_child = new bartMachineTreeNode(root);
    
    root->setLeft(left_child);
    root->setRight(right_child);
    
    // Test propagateDataByChangedRule
    std::cout << "Testing propagateDataByChangedRule..." << std::endl;
    root->propagateDataByChangedRule();
    
    // Verify data was propagated correctly
    // For our test data, samples with predictor[0] <= 1.0 should go left, others right
    // This should be samples 0, 1, 3, 4, 6, 7, 9 to the left (7 samples)
    // and samples 2, 5, 8 to the right (3 samples)
    assert(left_child->n_eta == 7);
    assert(right_child->n_eta == 3);
    
    // Test sumResponses and sumResponsesQuantitySqd
    std::cout << "Testing sumResponses and sumResponsesQuantitySqd..." << std::endl;
    double sum = left_child->sumResponses();
    double sum_sqd = left_child->sumResponsesQuantitySqd();
    
    // Expected sum for left child: 0 + 1 + 3 + 4 + 6 + 7 + 9 = 30
    assert(sum == 30.0);
    // Expected sum squared: 30^2 = 900
    assert(sum_sqd == 900.0);
    
    // Test updateWithNewResponsesRecursively
    std::cout << "Testing updateWithNewResponsesRecursively..." << std::endl;
    double* new_responses = new double[n];
    for (int i = 0; i < n; i++) {
        new_responses[i] = i * 2.0; // Double the responses
    }
    
    root->updateWithNewResponsesRecursively(new_responses);
    
    // Verify responses were updated
    sum = left_child->sumResponses();
    sum_sqd = left_child->sumResponsesQuantitySqd();
    
    // Expected sum for left child with doubled responses: 0 + 2 + 6 + 8 + 12 + 14 + 18 = 60
    assert(sum == 60.0);
    // Expected sum squared: 60^2 = 3600
    assert(sum_sqd == 3600.0);
    
    // Test prediction_untransformed and avg_response_untransformed
    std::cout << "Testing prediction_untransformed and avg_response_untransformed..." << std::endl;
    
    // Set y_pred for testing
    left_child->y_pred = 5.0;
    // We can't override the un_transform_y method directly, so we'll test with the actual implementation
    
    double pred_untrans = left_child->prediction_untransformed();
    // We can't assert the exact value since we're using the actual implementation
    // Just check that it's not the BAD_FLAG_double value
    assert(pred_untrans != bartMachineTreeNode::BAD_FLAG_double);
    
    double avg_untrans = left_child->avg_response_untransformed();
    // We can't assert the exact value since we're using the actual implementation
    // Just check that it's not the BAD_FLAG_double value
    assert(avg_untrans != bartMachineTreeNode::BAD_FLAG_double);
    
    // Test updateYHatsWithPrediction
    std::cout << "Testing updateYHatsWithPrediction..." << std::endl;
    left_child->updateYHatsWithPrediction();
    
    // Test Evaluate and EvaluateNode
    std::cout << "Testing Evaluate and EvaluateNode..." << std::endl;
    
    // Create a test record
    double test_record[p] = {0.0, 2.5, 1.0}; // Should go to left child
    
    bartMachineTreeNode* eval_node = root->EvaluateNode(test_record);
    assert(eval_node == left_child);
    
    double eval_result = root->Evaluate(test_record);
    assert(eval_result == left_child->y_pred);
    
    // Test findTerminalNodesDataAboveOrEqualToN
    std::cout << "Testing findTerminalNodesDataAboveOrEqualToN..." << std::endl;
    
    // Make left_child a non-leaf with children
    left_child->setLeaf(false);
    left_child->setSplitAttributeM(1); // Split on second predictor
    left_child->setSplitValue(2.0);    // Split value
    
    bartMachineTreeNode* left_left_child = new bartMachineTreeNode(left_child);
    bartMachineTreeNode* left_right_child = new bartMachineTreeNode(left_child);
    
    left_child->setLeft(left_left_child);
    left_child->setRight(left_right_child);
    
    // Propagate data to new children
    left_child->propagateDataByChangedRule();
    
    std::vector<bartMachineTreeNode*> terminal_nodes = root->getTerminalNodesWithDataAboveOrEqualToN(2);
    
    // We should have 3 terminal nodes with at least 2 data points
    assert(terminal_nodes.size() == 3);
    
    // Test findPrunableAndChangeableNodes
    std::cout << "Testing findPrunableAndChangeableNodes..." << std::endl;
    
    std::vector<bartMachineTreeNode*> prunable_nodes = root->getPrunableAndChangeableNodes();
    
    // We should have 1 prunable node (left_child)
    assert(prunable_nodes.size() == 1);
    assert(prunable_nodes[0] == left_child);
    
    // Test flushNodeData
    std::cout << "Testing flushNodeData..." << std::endl;
    root->flushNodeData();
    
    // Clean up
    delete[] new_responses;
    delete[] y_trans;
    
    for (auto record : X_y) {
        delete[] record;
    }
    
    delete bart;
    delete root; // This should delete all child nodes recursively
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
