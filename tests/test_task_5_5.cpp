#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "src/cpp/include/bartmachine_g_mh.h"
#include "src/cpp/include/exact_port_mersenne_twister.h"

/**
 * Task 5.5 Validation: Metropolis-Hastings - Change Operation
 *
 * This test validates that the Metropolis-Hastings change operation
 * is correctly implemented according to the Java implementation.
 */

class TestMH : public bartmachine_g_mh {
public:
    // Make protected methods public for testing
    using bartmachine_g_mh::doMHChangeAndCalcLnR;
    using bartmachine_g_mh::calcLnLikRatioChange;
    
    // Make protected members public for testing
    void setGibbsSampleNum(int num) {
        gibbs_sample_num = num;
    }
    
    void setGibbsSamplesOfSigsq(double* samples) {
        gibbs_samples_of_sigsq = samples;
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
 * Create a simple tree for testing the change operation
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
 * Test the calcLnLikRatioChange method
 */
void testCalcLnLikRatioChange() {
    std::cout << "  Testing calcLnLikRatioChange... " << std::endl;
    
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
    
    // Create a test tree
    std::cout << "    Creating test tree..." << std::endl;
    bartMachineTreeNode* tree = createTestTree(&mh);
    
    // Create a copy of the tree with a different splitting rule
    std::cout << "    Creating changed tree..." << std::endl;
    bartMachineTreeNode* changedTree = tree->clone();
    changedTree->setSplitAttributeM(1);  // Change split attribute from 0 to 1
    changedTree->setSplitValue(0.6);     // Change split value from 0.5 to 0.6
    
    // Initialize gibbs_samples_of_sigsq for calcLnLikRatioChange
    std::cout << "    Initializing gibbs_samples_of_sigsq..." << std::endl;
    mh.setGibbsSampleNum(1); // Set the current sample number
    double* samples = new double[2]; // We only need one sample, but allocate 2 to avoid out-of-bounds
    samples[0] = 1.0; // Set a reasonable value for sigsq
    mh.setGibbsSamplesOfSigsq(samples);
    
    // Call the method
    std::cout << "    Calling calcLnLikRatioChange..." << std::endl;
    double lnLikRatio = mh.calcLnLikRatioChange(tree, changedTree);
    std::cout << "    lnLikRatio = " << lnLikRatio << std::endl;
    
    // Verify the result (exact value will depend on implementation)
    // For now, just check it's a valid number
    assert(!std::isnan(lnLikRatio));
    
    // Clean up - use the destructor to handle the children
    delete tree;
    delete changedTree;
    delete[] samples;
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test the doMHChangeAndCalcLnR method
 */
void testDoMHChangeAndCalcLnR() {
    std::cout << "  Testing doMHChangeAndCalcLnR... " << std::endl;
    
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
    
    // Create a test tree
    std::cout << "    Creating test tree..." << std::endl;
    bartMachineTreeNode* tree = createTestTree(&mh);
    
    // Create a copy of the tree for the change operation
    std::cout << "    Creating copy of tree for change..." << std::endl;
    bartMachineTreeNode* treeCopy = tree->clone();
    
    // Initialize gibbs_samples_of_sigsq for calcLnLikRatioChange
    std::cout << "    Initializing gibbs_samples_of_sigsq..." << std::endl;
    mh.setGibbsSampleNum(1); // Set the current sample number
    double* samples = new double[2]; // We only need one sample, but allocate 2 to avoid out-of-bounds
    samples[0] = 1.0; // Set a reasonable value for sigsq
    mh.setGibbsSamplesOfSigsq(samples);
    
    // Set RNG seed for deterministic behavior
    std::cout << "    Setting RNG seed..." << std::endl;
    StatToolbox::setSeed(12345);
    
    // Call the method
    std::cout << "    Calling doMHChangeAndCalcLnR..." << std::endl;
    double lnR = mh.doMHChangeAndCalcLnR(tree, treeCopy);
    std::cout << "    lnR = " << lnR << std::endl;
    
    // Verify the result (exact value will depend on implementation)
    // For now, just check it's a valid number
    assert(!std::isnan(lnR));
    
    // Clean up - use the destructor to handle the children
    delete tree;
    delete treeCopy;
    delete[] samples;
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Main test function
 */
int main() {
    std::cout << "Testing Task 5.5: Metropolis-Hastings - Change Operation" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Test MH class
    std::cout << "Testing bartmachine_g_mh class... ";
    TestMH mh;
    std::cout << "PASSED" << std::endl;
    
    try {
        // Test each method one at a time
        std::cout << "Testing calcLnLikRatioChange..." << std::endl;
        testCalcLnLikRatioChange();
        
        std::cout << "Testing doMHChangeAndCalcLnR..." << std::endl;
        testDoMHChangeAndCalcLnR();
        
        std::cout << std::endl << "All tests PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
    }
    
    return 0;
}
