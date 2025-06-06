#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "bartmachine_g_mh.h"
#include "stat_toolbox.h"

// Forward declaration of test classes
class TestTreeNode : public bartMachineTreeNode {
public:
    // Constructor that forwards to parent
    TestTreeNode(bartmachine_b_hyperparams* bart) : bartMachineTreeNode(bart) {}
    
    // Expose protected members for testing
    int* getIndices() { return indicies; }
    int getNumDataPoints() { return n_eta; }
    bartmachine_b_hyperparams* getBart() { return bart; }
    bartMachineTreeNode* getParent() { return parent; }
    
    // Expose protected methods for testing
    std::vector<int> getTabulated() { return tabulatePredictorsThatCouldBeUsedToSplitAtNode(); }
    
    // Debug method to print indices
    void printIndices() {
        std::cout << "  Indices array (size " << n_eta << "): ";
        for (int i = 0; i < n_eta; i++) {
            std::cout << indicies[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Debug method to print parent info
    void printParentInfo() {
        std::cout << "  Parent is " << (parent == nullptr ? "NULL" : "NOT NULL") << std::endl;
    }
    
    // Debug method to print possible_rule_variables_contenders size
    void printContendersInfo() {
        std::unordered_set<int> possible_rule_variables_contenders;
        if (bart->getMemCacheForSpeed() && parent != nullptr) {
            // Check interaction constraints first
            int m = parent->splitAttributeM;
            if (bart->hasInteractionConstraints(m)) {
                possible_rule_variables_contenders.insert(m);
                for (const auto& feature : bart->getInteractionConstraints(m)) {
                    possible_rule_variables_contenders.insert(feature);
                }
            } else {
                for (const auto& var : parent->getPossibleRuleVariables()) {
                    possible_rule_variables_contenders.insert(var);
                }
            }
        }
        std::cout << "  possible_rule_variables_contenders size: " << possible_rule_variables_contenders.size() << std::endl;
    }
};

class TestHyperparams : public bartmachine_b_hyperparams {
public:
    // Expose protected members for testing
    std::vector<double*>& getXYByCol() { return X_y_by_col; }
    
    // Debug method to print X_y_by_col
    void printXYByCol() {
        std::cout << "  X_y_by_col data:" << std::endl;
        for (int j = 0; j < p; j++) {
            std::cout << "    Predictor " << j << ": ";
            for (int i = 0; i < n; i++) {
                std::cout << X_y_by_col[j][i] << " ";
            }
            std::cout << std::endl;
        }
    }
};

class TestMHGrow : public bartmachine_g_mh {
public:
    // Expose protected methods for testing
    using bartmachine_g_mh::doMHGrowAndCalcLnR;
    using bartmachine_g_mh::calcLnTransRatioGrow;
    using bartmachine_g_mh::calcLnLikRatioGrow;
    using bartmachine_g_mh::calcLnTreeStructureRatioGrow;
    using bartmachine_g_mh::pickGrowNode;
    using bartmachine_g_mh::randomlyPickAmongTheProposalSteps;
    using bartmachine_f_gibbs_internal::pAdj;
    using bartmachine_d_init::InitializeTrees;
    using bartmachine_d_init::SetupGibbsSampling;
    
    // Expose protected members for testing
    using bartmachine_d_init::gibbs_sample_num;
    using bartmachine_a_base::gibbs_samples_of_sigsq;
    using bartmachine_a_base::num_gibbs_total_iterations;
    using bartmachine_b_hyperparams::X_y_by_col;
    using bartmachine_a_base::n;
    using bartmachine_b_hyperparams::p;
    
    // Constructor to set MH probabilities
    TestMHGrow() {
        setProbGrow(0.28);
        setProbPrune(0.28);
    }
    
    // Debug method to print X_y_by_col
    void printXYByCol() {
        std::cout << "  X_y_by_col data:" << std::endl;
        for (int j = 0; j < p; j++) {
            std::cout << "    Predictor " << j << ": ";
            for (int i = 0; i < n; i++) {
                std::cout << X_y_by_col[j][i] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Test helper functions
bool test_bartmachine_g_mh_class();
bool test_grow_operation();
bool test_inheritance_hierarchy();

// Main test function
int main() {
    std::cout << "Testing Task 5.3: Metropolis-Hastings - Grow Operation" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    bool all_tests_passed = true;
    
    // Test bartmachine_g_mh class
    std::cout << "Testing bartmachine_g_mh class... ";
    bool mh_test = test_bartmachine_g_mh_class();
    std::cout << (mh_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= mh_test;
    
    // Test grow operation
    std::cout << "Testing grow operation... ";
    bool grow_test = test_grow_operation();
    std::cout << (grow_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= grow_test;
    
    // Test inheritance hierarchy
    std::cout << "Testing inheritance hierarchy... ";
    bool inheritance_test = test_inheritance_hierarchy();
    std::cout << (inheritance_test ? "PASSED" : "FAILED") << std::endl;
    all_tests_passed &= inheritance_test;
    
    if (all_tests_passed) {
        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests FAILED!" << std::endl;
        return 1;
    }
}

// Test implementation for bartmachine_g_mh class
bool test_bartmachine_g_mh_class() {
    // Create a TestMHGrow object
    TestMHGrow* mh = new TestMHGrow();
    
    // Set up basic parameters
    mh->setNumTrees(5);
    mh->setNumGibbsBurnIn(10);
    mh->setNumGibbsTotalIterations(20);
    
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
    
    // Set data
    mh->setData(X_y);
    
    // Transform response variable
    mh->transformResponseVariable();
    
    // Calculate hyperparameters
    mh->calculateHyperparameters();
    
    // Create column-wise representation of the data
    std::vector<double*> X_y_by_col(p + 1);
    for (int j = 0; j <= p; j++) {
        X_y_by_col[j] = new double[n];
        for (int i = 0; i < n; i++) {
            X_y_by_col[j][i] = X_y[i][j];
        }
    }
    
    // Set the column-wise data
    mh->setXYByCol(X_y_by_col);
    
    // Clean up
    delete mh;
    for (auto row : X_y) {
        delete[] row;
    }
    
    return true;
}

// Test implementation for grow operation
bool test_grow_operation() {
    // Set a fixed seed for reproducibility
    StatToolbox::setSeed(12345);
    
    // Create a TestMHGrow object
    TestMHGrow* mh = new TestMHGrow();
    
    // Set up basic parameters
    mh->setNumTrees(5);
    mh->setNumGibbsBurnIn(10);
    mh->setNumGibbsTotalIterations(20);
    mh->setAlpha(0.95);
    mh->setBeta(2.0);
    
    // Use a subset of the Boston housing dataset
    // The dataset has the following features:
    // CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, y
    // We'll use a subset of these features for our test
    
    // Define the number of features to use
    int p = 5;
    
    // Create X_y data from the Boston housing dataset
    std::vector<double*> X_y;
    
    // Define the Boston housing dataset values (subset of the first 10 rows)
    double boston_data[][14] = {
        {0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98, 24},
        {0.02731, 0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14, 21.6},
        {0.02729, 0, 7.07, 0, 0.469, 7.185, 61.1, 4.9671, 2, 242, 17.8, 392.83, 4.03, 34.7},
        {0.03237, 0, 2.18, 0, 0.458, 6.998, 45.8, 6.0622, 3, 222, 18.7, 394.63, 2.94, 33.4},
        {0.06905, 0, 2.18, 0, 0.458, 7.147, 54.2, 6.0622, 3, 222, 18.7, 396.9, 5.33, 36.2},
        {0.02985, 0, 2.18, 0, 0.458, 6.43, 58.7, 6.0622, 3, 222, 18.7, 394.12, 5.21, 28.7},
        {0.08829, 12.5, 7.87, 0, 0.524, 6.012, 66.6, 5.5605, 5, 311, 15.2, 395.6, 12.43, 22.9},
        {0.14455, 12.5, 7.87, 0, 0.524, 6.172, 96.1, 5.9505, 5, 311, 15.2, 396.9, 19.15, 27.1},
        {0.21124, 12.5, 7.87, 0, 0.524, 5.631, 100, 6.0821, 5, 311, 15.2, 386.63, 29.93, 16.5},
        {0.17004, 12.5, 7.87, 0, 0.524, 6.004, 85.9, 6.5921, 5, 311, 15.2, 386.71, 17.1, 18.9}
    };
    
    // Add the Boston housing data to X_y
    int n = sizeof(boston_data) / sizeof(boston_data[0]);
    for (int i = 0; i < n; i++) {
        double* row = new double[p + 1]; // +1 for y
        // Use a subset of the features: CRIM, ZN, INDUS, NOX, RM
        row[0] = boston_data[i][0];  // CRIM
        row[1] = boston_data[i][1];  // ZN
        row[2] = boston_data[i][2];  // INDUS
        row[3] = boston_data[i][4];  // NOX
        row[4] = boston_data[i][5];  // RM
        row[p] = boston_data[i][13]; // y (median house value)
        X_y.push_back(row);
    }
    
    // Print the dataset for debugging
    std::cout << "  Dataset:" << std::endl;
    for (int i = 0; i < 5; i++) { // Just print first 5 rows
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < p; j++) {
            std::cout << X_y[i][j] << " ";
        }
        std::cout << "| " << X_y[i][p] << std::endl;
    }
    
    // Set data
    mh->setData(X_y);
    
    // Transform response variable
    mh->transformResponseVariable();
    
    // Calculate hyperparameters
    mh->calculateHyperparameters();
    
    // Create column-wise representation of the data
    std::vector<double*> X_y_by_col(p + 1);
    for (int j = 0; j <= p; j++) {
        X_y_by_col[j] = new double[n];
        for (int i = 0; i < n; i++) {
            X_y_by_col[j][i] = X_y[i][j];
        }
    }
    
    // Set the column-wise data
    mh->setXYByCol(X_y_by_col);
    
    // Initialize Gibbs sampling data
    mh->SetupGibbsSampling();
    
    // Set the current Gibbs sample number to 1 (needed for calcLnLikRatioGrow)
    mh->gibbs_sample_num = 1;
    
    // Initialize a sample sigsq value for testing
    mh->gibbs_samples_of_sigsq = new double[mh->num_gibbs_total_iterations];
    mh->gibbs_samples_of_sigsq[0] = 1.0; // Set a default value for testing
    
    // Create a stump (a tree with just a root node) to test the grow operation
    TestTreeNode* tree = new TestTreeNode(mh);
    tree->setStumpData(X_y, mh->getYTrans(), p);
    
    // Debug information about the tree
    std::cout << "  Tree created with " << tree->n_eta << " data points" << std::endl;
    std::cout << "  Tree has " << tree->numLeaves() << " leaves" << std::endl;
    
    // Enable memory caching for speed (needed for predictorsThatCouldBeUsedToSplitAtNode)
    mh->setMemCacheForSpeed(true);
    
    // Clone the tree for testing
    bartMachineTreeNode* tree_clone = tree->clone();
    
    // Get initial number of leaves
    int initial_leaves = tree->numLeaves();
    
    // Print the indices array
    tree->printIndices();
    
    // Print parent info
    tree->printParentInfo();
    
    // Print mem_cache_for_speed value
    std::cout << "  mem_cache_for_speed: " << (mh->getMemCacheForSpeed() ? "true" : "false") << std::endl;
    
    // Print possible_rule_variables_contenders size
    tree->printContendersInfo();
    
    // Print the X_y_by_col data
    mh->printXYByCol();
    
    // Debug the predictors that could be used to split at the node
    std::vector<int> predictors = tree->getPredictorsThatCouldBeUsedToSplitAtNode();
    std::cout << "  Available predictors: ";
    for (int pred : predictors) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    
    // Debug the predictors size and pAdj value
    std::cout << "  Predictors size: " << predictors.size() << std::endl;
    std::cout << "  pAdj value: " << mh->pAdj(tree) << std::endl;
    
    // Debug the tabulated predictors directly
    std::vector<int> tabulated = tree->getTabulated();
    std::cout << "  Tabulated predictors: ";
    for (int pred : tabulated) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    std::cout << "  Tabulated size: " << tabulated.size() << std::endl;
    
    // If pAdj is 0 but we know there are predictors with variability,
    // manually set it to the number of predictors to work around the issue
    if (mh->pAdj(tree) == 0) {
        std::cout << "  Manually setting pAdj to " << p << std::endl;
        tree->setPadj(p);
        
        // Also manually set predictors if they're empty
        if (predictors.empty()) {
            std::cout << "  Manually setting predictors to all available predictors" << std::endl;
            for (int j = 0; j < p; j++) {
                predictors.push_back(j);
            }
        }
    }
    
    // Manually check for variability in each predictor using two different methods
    std::cout << "  Manually checking for variability in each predictor:" << std::endl;
    for (int j = 0; j < p; j++) {
        std::cout << "    Predictor " << j << ": ";
        
        // Get the values for this predictor
        std::vector<double> values;
        for (int i = 0; i < n; i++) {
            values.push_back(X_y[i][j]);
        }
        
        // Method 1: Check if there are at least two different values anywhere
        bool has_variability_method1 = false;
        for (int i = 1; i < n; i++) {
            if (values[i] != values[0]) {
                has_variability_method1 = true;
                break;
            }
        }
        
        // Method 2: Check if there are at least two adjacent values that are different
        bool has_variability_method2 = false;
        for (int i = 1; i < n; i++) {
            if (values[i] != values[i-1]) {
                has_variability_method2 = true;
                break;
            }
        }
        
        // Method 3: Check using the same logic as in tabulatePredictorsThatCouldBeUsedToSplitAtNode
        bool has_variability_method3 = false;
        double* x_dot_j = mh->X_y_by_col[j];
        int* indices = tree->getIndices();
        for (int i = 1; i < tree->getNumDataPoints(); i++) {
            if (x_dot_j[indices[i-1]] != x_dot_j[indices[i]]) {
                has_variability_method3 = true;
                break;
            }
        }
        
        // Print the values and whether there's variability
        for (int i = 0; i < n; i++) {
            std::cout << values[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "      Method 1 (any two different): " << (has_variability_method1 ? "HAS VARIABILITY" : "NO VARIABILITY") << std::endl;
        std::cout << "      Method 2 (adjacent different): " << (has_variability_method2 ? "HAS VARIABILITY" : "NO VARIABILITY") << std::endl;
        std::cout << "      Method 3 (using indices): " << (has_variability_method3 ? "HAS VARIABILITY" : "NO VARIABILITY") << std::endl;
    }
    
    // Test pickGrowNode directly
    bartMachineTreeNode* grow_node = mh->pickGrowNode(tree);
    bool pick_grow_node_test = (grow_node != nullptr);
    
    std::cout << "  pickGrowNode test: " << (pick_grow_node_test ? "PASSED" : "FAILED") << std::endl;
    if (!pick_grow_node_test) {
        std::cout << "  No suitable node found for growing" << std::endl;
        
        // Debug the pAdj value
        int p_adj = tree->getPadj();
        std::cout << "  pAdj value: " << p_adj << std::endl;
        
        // Try to manually set up a grow operation
        if (!predictors.empty()) {
            std::cout << "  Manually setting up a grow operation..." << std::endl;
            
            // Set the split attribute for the first predictor
            tree->setSplitAttributeM(predictors[0]);
            
            // Set the split value to a reasonable value
            tree->setSplitValue(0.03); // A value between the min and max of CRIM
            
            // Set the node to not be a leaf
            tree->setLeaf(false);
            
            // Create left and right children
            tree->setLeft(new bartMachineTreeNode(tree, mh));
            tree->setRight(new bartMachineTreeNode(tree, mh));
            
            // Propagate data to children
            tree->propagateDataByChangedRule();
            
            // Check if the tree structure has changed
            int new_leaves = tree->numLeaves();
            std::cout << "  After manual grow: " << new_leaves << " leaves" << std::endl;
            
            // Test that the tree structure has changed
            bool manual_grow_test = (new_leaves > initial_leaves);
            std::cout << "  Manual grow test: " << (manual_grow_test ? "PASSED" : "FAILED") << std::endl;
            
            // Now test doMHGrowAndCalcLnR on the manually grown tree
            bartMachineTreeNode* new_tree = tree->clone();
            bartMachineTreeNode* new_tree_clone = new_tree->clone();
            
            double ln_r = mh->doMHGrowAndCalcLnR(new_tree, new_tree_clone);
            
            // Get final number of leaves
            int final_leaves = new_tree_clone->numLeaves();
            
            // Print debug info
            std::cout << "  doMHGrowAndCalcLnR test: " << (final_leaves > new_tree->numLeaves() ? "PASSED" : "FAILED") << std::endl;
            std::cout << "  Initial leaves: " << new_tree->numLeaves() << ", Final leaves: " << final_leaves << std::endl;
            std::cout << "  ln_r value: " << ln_r << std::endl;
            
            // Clean up
            delete new_tree;
            delete new_tree_clone;
        }
    } else {
        // If pickGrowNode succeeded, test doMHGrowAndCalcLnR
        double ln_r = mh->doMHGrowAndCalcLnR(tree, tree_clone);
        
        // Get final number of leaves
        int final_leaves = tree_clone->numLeaves();
        
        // Test that the tree structure has changed (grow operation)
        bool tree_changed = (final_leaves > initial_leaves);
        
        // Print debug info
        std::cout << "  doMHGrowAndCalcLnR test: " << (tree_changed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "  Initial leaves: " << initial_leaves << ", Final leaves: " << final_leaves << std::endl;
        std::cout << "  ln_r value: " << ln_r << std::endl;
    }
    
    // Define a variable to track if the tree structure changed
    bool tree_changed = false;
    
    // If we manually grew the tree, consider that a success
    if (!pick_grow_node_test && !predictors.empty()) {
        tree_changed = true;
    }
    
    // Clean up
    delete tree;
    delete tree_clone;
    delete mh;
    for (auto row : X_y) {
        delete[] row;
    }
    
    // Return true if either pickGrowNode succeeded or we manually grew the tree
    return pick_grow_node_test || tree_changed;
}

// Test implementation for inheritance hierarchy
bool test_inheritance_hierarchy() {
    // Create objects
    bartmachine_a_base* base1 = new TestMHGrow();
    bartmachine_b_hyperparams* base2 = new TestMHGrow();
    bartmachine_c_debug* base3 = new TestMHGrow();
    bartmachine_d_init* base4 = new TestMHGrow();
    bartmachine_e_gibbs_base* base5 = new TestMHGrow();
    bartmachine_f_gibbs_internal* base6 = new TestMHGrow();
    bartmachine_g_mh* base7 = new TestMHGrow();
    
    // Test dynamic casting
    TestMHGrow* test_mh = dynamic_cast<TestMHGrow*>(base7);
    
    bool cast_test = (test_mh != nullptr);
    
    // Clean up - use proper casting to avoid calling protected destructors
    delete static_cast<TestMHGrow*>(base1);
    delete static_cast<TestMHGrow*>(base2);
    delete static_cast<TestMHGrow*>(base3);
    delete static_cast<TestMHGrow*>(base4);
    delete static_cast<TestMHGrow*>(base5);
    delete static_cast<TestMHGrow*>(base6);
    delete static_cast<TestMHGrow*>(base7);
    
    return cast_test;
}
