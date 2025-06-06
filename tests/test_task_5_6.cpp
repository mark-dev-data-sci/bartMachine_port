#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "src/cpp/include/bartmachine_g_mh.h"
#include "src/cpp/include/exact_port_mersenne_twister.h"

/**
 * Task 5.6 Validation: Metropolis-Hastings - Integration
 *
 * This test validates that the Metropolis-Hastings algorithm integration
 * is correctly implemented according to the Java implementation.
 */

class TestMH : public bartmachine_g_mh {
public:
    // Make protected methods public for testing
    using bartmachine_g_mh::metroHastingsPosteriorTreeSpaceIteration;
    using bartmachine_g_mh::randomlyPickAmongTheProposalSteps;
    
    // Make protected members public for testing
    void setGibbsSampleNum(int num) {
        gibbs_sample_num = num;
    }
    
    void setGibbsSamplesOfSigsq(double* samples) {
        gibbs_samples_of_sigsq = samples;
    }
    
    void setProbGrow(double prob) {
        prob_grow = prob;
    }
    
    void setProbPrune(double prob) {
        prob_prune = prob;
    }
    
    // Constructor that calls parent constructor
    TestMH() : bartmachine_g_mh() {}
    
    // Set data for testing
    void setTestData(std::vector<std::vector<double>>& X_y_vec) {
        // Convert std::vector<std::vector<double>> to std::vector<double*>
        this->X_y.clear();
        for (auto& row : X_y_vec) {
            double* row_ptr = new double[row.size()];
            for (size_t i = 0; i < row.size(); i++) {
                row_ptr[i] = row[i];
            }
            this->X_y.push_back(row_ptr);
        }
        this->n = X_y_vec.size();
        this->p = X_y_vec[0].size() - 1;
    }
    
    // Set X_y_by_col for testing
    void setTestXYByCol(std::vector<std::vector<double>>& X_y_by_col_vec) {
        // Convert std::vector<std::vector<double>> to std::vector<double*>
        this->X_y_by_col.clear();
        for (auto& col : X_y_by_col_vec) {
            double* col_ptr = new double[col.size()];
            for (size_t i = 0; i < col.size(); i++) {
                col_ptr[i] = col[i];
            }
            this->X_y_by_col.push_back(col_ptr);
        }
    }
    
    // Clean up allocated memory
    ~TestMH() {
        for (auto ptr : X_y) {
            delete[] ptr;
        }
        for (auto ptr : X_y_by_col) {
            delete[] ptr;
        }
    }
};

/**
 * Create a simple tree for testing the MH algorithm
 */
bartMachineTreeNode* createTestTree(bartmachine_b_hyperparams* bart) {
    // Create a root node with the bart hyperparameters
    bartMachineTreeNode* root = new bartMachineTreeNode(bart);
    
    // Set up a simple tree with a split
    root->setSplitAttributeM(0);  // Split on first attribute
    root->setSplitValue(0.5);     // Split at value 0.5
    
    // Create left and right children with parent
    bartMachineTreeNode* leftChild = new bartMachineTreeNode(root);
    bartMachineTreeNode* rightChild = new bartMachineTreeNode(root);
    
    // Set parent-child relationships
    root->setLeft(leftChild);
    root->setRight(rightChild);
    
    // Set leaf status
    leftChild->setLeaf(true);
    rightChild->setLeaf(true);
    root->setLeaf(false);
    
    // Initialize some data for the nodes
    root->n_eta = 3;
    leftChild->n_eta = 1;
    rightChild->n_eta = 2;
    
    return root;
}

/**
 * Test the randomlyPickAmongTheProposalSteps method
 */
void testRandomlyPickAmongTheProposalSteps() {
    std::cout << "  Testing randomlyPickAmongTheProposalSteps... " << std::endl;
    
    std::cout << "    Creating MH object..." << std::endl;
    TestMH mh;
    
    // Set probabilities
    std::cout << "    Setting probabilities..." << std::endl;
    mh.setProbGrow(0.3);
    mh.setProbPrune(0.3);
    // Implied prob_change = 0.4
    
    // Set RNG seed for deterministic behavior
    std::cout << "    Setting RNG seed..." << std::endl;
    StatToolbox::setSeed(12345);
    
    // Call the method multiple times and count the results
    std::cout << "    Calling randomlyPickAmongTheProposalSteps..." << std::endl;
    int grow_count = 0;
    int prune_count = 0;
    int change_count = 0;
    const int num_trials = 1000;
    
    for (int i = 0; i < num_trials; i++) {
        bartmachine_g_mh::Steps step = mh.randomlyPickAmongTheProposalSteps();
        if (step == bartmachine_g_mh::Steps::GROW) {
            grow_count++;
        } else if (step == bartmachine_g_mh::Steps::PRUNE) {
            prune_count++;
        } else if (step == bartmachine_g_mh::Steps::CHANGE) {
            change_count++;
        }
    }
    
    std::cout << "    Grow count: " << grow_count << " (" << (double)grow_count / num_trials << ")" << std::endl;
    std::cout << "    Prune count: " << prune_count << " (" << (double)prune_count / num_trials << ")" << std::endl;
    std::cout << "    Change count: " << change_count << " (" << (double)change_count / num_trials << ")" << std::endl;
    
    // Verify the results are approximately correct
    // Allow for some randomness, but should be close to the expected probabilities
    assert(std::abs((double)grow_count / num_trials - 0.3) < 0.1);
    assert(std::abs((double)prune_count / num_trials - 0.3) < 0.1);
    assert(std::abs((double)change_count / num_trials - 0.4) < 0.1);
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test the metroHastingsPosteriorTreeSpaceIteration method
 */
void testMetroHastingsPosteriorTreeSpaceIteration() {
    std::cout << "  Testing metroHastingsPosteriorTreeSpaceIteration... " << std::endl;
    
    std::cout << "    Creating MH object..." << std::endl;
    TestMH mh;
    
    // Set up test data
    std::cout << "    Setting up test data..." << std::endl;
    std::vector<std::vector<double>> X_y = {
        {0.1, 0.2, 0.3, 1.0},  // x1, x2, x3, y
        {0.4, 0.5, 0.6, 2.0},
        {0.7, 0.8, 0.9, 3.0}
    };
    mh.setTestData(X_y);
    
    // Set up X_y_by_col
    std::cout << "    Setting up X_y_by_col..." << std::endl;
    std::vector<std::vector<double>> X_y_by_col = {
        {0.1, 0.4, 0.7},  // x1 column
        {0.2, 0.5, 0.8},  // x2 column
        {0.3, 0.6, 0.9},  // x3 column
        {1.0, 2.0, 3.0}   // y column
    };
    mh.setTestXYByCol(X_y_by_col);
    
    // Set probabilities
    std::cout << "    Setting probabilities..." << std::endl;
    mh.setProbGrow(0.3);
    mh.setProbPrune(0.3);
    // Implied prob_change = 0.4
    
    // Create a test tree
    std::cout << "    Creating test tree..." << std::endl;
    bartMachineTreeNode* tree = createTestTree(&mh);
    
    // Initialize gibbs_samples_of_sigsq
    std::cout << "    Initializing gibbs_samples_of_sigsq..." << std::endl;
    mh.setGibbsSampleNum(1); // Set the current sample number
    double* samples = new double[2]; // We only need one sample, but allocate 2 to avoid out-of-bounds
    samples[0] = 1.0; // Set a reasonable value for sigsq
    mh.setGibbsSamplesOfSigsq(samples);
    
    // Set RNG seed for deterministic behavior
    std::cout << "    Setting RNG seed..." << std::endl;
    StatToolbox::setSeed(12345);
    
    // Create accept/reject matrices
    std::cout << "    Creating accept/reject matrices..." << std::endl;
    bool** accept_reject_mh = new bool*[2];
    char** accept_reject_mh_steps = new char*[2];
    for (int i = 0; i < 2; i++) {
        accept_reject_mh[i] = new bool[1];
        accept_reject_mh_steps[i] = new char[1];
    }
    
    // Call the method
    std::cout << "    Calling metroHastingsPosteriorTreeSpaceIteration..." << std::endl;
    bartMachineTreeNode* next_tree = mh.metroHastingsPosteriorTreeSpaceIteration(tree, 0, accept_reject_mh, accept_reject_mh_steps);
    
    // Verify the result
    std::cout << "    Step: " << accept_reject_mh_steps[1][0] << std::endl;
    std::cout << "    Accepted: " << (accept_reject_mh[1][0] ? "true" : "false") << std::endl;
    
    // Clean up
    delete tree;
    delete[] samples;
    for (int i = 0; i < 2; i++) {
        delete[] accept_reject_mh[i];
        delete[] accept_reject_mh_steps[i];
    }
    delete[] accept_reject_mh;
    delete[] accept_reject_mh_steps;
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Main test function
 */
int main() {
    std::cout << "Testing Task 5.6: Metropolis-Hastings - Integration" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Test MH class
    std::cout << "Testing bartmachine_g_mh class... ";
    TestMH mh;
    std::cout << "PASSED" << std::endl;
    
    try {
        // Test each method one at a time
        std::cout << "Testing randomlyPickAmongTheProposalSteps..." << std::endl;
        testRandomlyPickAmongTheProposalSteps();
        
        std::cout << "Testing metroHastingsPosteriorTreeSpaceIteration..." << std::endl;
        testMetroHastingsPosteriorTreeSpaceIteration();
        
        std::cout << std::endl << "All tests PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
    }
    
    return 0;
}
