#include "include/bartmachine_tree_node.h"
#include "include/classifier.h"
#include <sstream>
#include <iostream>
#include <algorithm>

/**
 * Exact port of bartMachineTreeNode from Java to C++
 * 
 * This file contains the implementation of the class structure and basic methods
 * for the bartMachineTreeNode class.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachineTreeNode.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */

// Initialize static constants
const double bartMachineTreeNode::BAD_FLAG_double = -std::numeric_limits<double>::max();
const int bartMachineTreeNode::BAD_FLAG_int = -std::numeric_limits<int>::max();

// Default constructor - matches Java: public bartMachineTreeNode()
bartMachineTreeNode::bartMachineTreeNode() : 
    parent(nullptr),
    left(nullptr),
    right(nullptr),
    depth(0),
    isLeaf(true),
    splitAttributeM(BAD_FLAG_int),
    splitValue(BAD_FLAG_double),
    sendMissingDataRight(false),
    y_pred(BAD_FLAG_double),
    y_avg(BAD_FLAG_double),
    posterior_var(BAD_FLAG_double),
    posterior_mean(BAD_FLAG_double),
    n_eta(0),
    yhats(nullptr),
    bart(nullptr),
    indicies(nullptr),
    responses(nullptr),
    sum_responses_qty_sqd(0.0),
    sum_responses_qty(0.0),
    padj(0),
    attribute_split_counts(nullptr)
{
}

// Constructor with parent and bart - matches Java: public bartMachineTreeNode(bartMachineTreeNode parent, bartmachine_b_hyperparams bart)
bartMachineTreeNode::bartMachineTreeNode(bartMachineTreeNode* parent, bartmachine_b_hyperparams* bart) :
    parent(parent),
    left(nullptr),
    right(nullptr),
    isLeaf(true),
    splitAttributeM(BAD_FLAG_int),
    splitValue(BAD_FLAG_double),
    sendMissingDataRight(false),
    y_pred(BAD_FLAG_double),
    y_avg(BAD_FLAG_double),
    posterior_var(BAD_FLAG_double),
    posterior_mean(BAD_FLAG_double),
    n_eta(0),
    bart(bart),
    indicies(nullptr),
    responses(nullptr),
    sum_responses_qty_sqd(0.0),
    sum_responses_qty(0.0),
    padj(0),
    attribute_split_counts(nullptr)
{
    if (parent != nullptr) {
        yhats = parent->yhats;
        depth = parent->depth + 1;
    } else {
        yhats = nullptr;
        depth = 0;
    }
}

// Constructor with parent only - matches Java: public bartMachineTreeNode(bartMachineTreeNode parent)
bartMachineTreeNode::bartMachineTreeNode(bartMachineTreeNode* parent) :
    bartMachineTreeNode(parent, parent->bart)
{
}

// Constructor with bart only - matches Java: public bartMachineTreeNode(bartmachine_b_hyperparams bart)
bartMachineTreeNode::bartMachineTreeNode(bartmachine_b_hyperparams* bart) :
    parent(nullptr),
    left(nullptr),
    right(nullptr),
    depth(0),
    isLeaf(true),
    splitAttributeM(BAD_FLAG_int),
    splitValue(BAD_FLAG_double),
    sendMissingDataRight(false),
    y_pred(BAD_FLAG_double),
    y_avg(BAD_FLAG_double),
    posterior_var(BAD_FLAG_double),
    posterior_mean(BAD_FLAG_double),
    n_eta(0),
    yhats(nullptr),
    bart(bart),
    indicies(nullptr),
    responses(nullptr),
    sum_responses_qty_sqd(0.0),
    sum_responses_qty(0.0),
    padj(0),
    attribute_split_counts(nullptr)
{
}

// Destructor
bartMachineTreeNode::~bartMachineTreeNode() {
    // Delete children
    delete left;
    delete right;
    
    // Delete arrays
    delete[] indicies;
    delete[] responses;
    delete[] attribute_split_counts;
    
    // Note: We don't delete yhats here because it's shared among nodes
}

// Copy constructor
bartMachineTreeNode::bartMachineTreeNode(const bartMachineTreeNode& other) :
    parent(other.parent),
    left(nullptr),
    right(nullptr),
    depth(other.depth),
    isLeaf(other.isLeaf),
    splitAttributeM(other.splitAttributeM),
    splitValue(other.splitValue),
    sendMissingDataRight(other.sendMissingDataRight),
    y_pred(other.y_pred),
    y_avg(other.y_avg),
    posterior_var(other.posterior_var),
    posterior_mean(other.posterior_mean),
    n_eta(other.n_eta),
    yhats(other.yhats),
    bart(other.bart),
    sum_responses_qty_sqd(other.sum_responses_qty_sqd),
    sum_responses_qty(other.sum_responses_qty),
    padj(other.padj)
{
    // Copy arrays
    if (other.indicies != nullptr) {
        indicies = new int[n_eta];
        std::copy(other.indicies, other.indicies + n_eta, indicies);
    } else {
        indicies = nullptr;
    }
    
    if (other.responses != nullptr) {
        responses = new double[n_eta];
        std::copy(other.responses, other.responses + n_eta, responses);
    } else {
        responses = nullptr;
    }
    
    // Copy possible_rule_variables
    possible_rule_variables = other.possible_rule_variables;
    
    // Copy possible_split_vals_by_attr
    possible_split_vals_by_attr = other.possible_split_vals_by_attr;
    
    // Copy attribute_split_counts
    if (other.attribute_split_counts != nullptr) {
        // We need to know the size of the array, which should be bart->p
        if (bart != nullptr) {
            attribute_split_counts = new int[bart->p];
            std::copy(other.attribute_split_counts, other.attribute_split_counts + bart->p, attribute_split_counts);
        } else {
            attribute_split_counts = nullptr;
        }
    } else {
        attribute_split_counts = nullptr;
    }
    
    // Copy children
    if (other.left != nullptr) {
        left = other.left->clone();
        left->parent = this;
    }
    
    if (other.right != nullptr) {
        right = other.right->clone();
        right->parent = this;
    }
}

// Assignment operator
bartMachineTreeNode& bartMachineTreeNode::operator=(const bartMachineTreeNode& other) {
    if (this != &other) {
        // Delete existing resources
        delete left;
        delete right;
        delete[] indicies;
        delete[] responses;
        delete[] attribute_split_counts;
        
        // Copy basic members
        parent = other.parent;
        depth = other.depth;
        isLeaf = other.isLeaf;
        splitAttributeM = other.splitAttributeM;
        splitValue = other.splitValue;
        sendMissingDataRight = other.sendMissingDataRight;
        y_pred = other.y_pred;
        y_avg = other.y_avg;
        posterior_var = other.posterior_var;
        posterior_mean = other.posterior_mean;
        n_eta = other.n_eta;
        yhats = other.yhats;
        bart = other.bart;
        sum_responses_qty_sqd = other.sum_responses_qty_sqd;
        sum_responses_qty = other.sum_responses_qty;
        padj = other.padj;
        
        // Copy arrays
        if (other.indicies != nullptr) {
            indicies = new int[n_eta];
            std::copy(other.indicies, other.indicies + n_eta, indicies);
        } else {
            indicies = nullptr;
        }
        
        if (other.responses != nullptr) {
            responses = new double[n_eta];
            std::copy(other.responses, other.responses + n_eta, responses);
        } else {
            responses = nullptr;
        }
        
        // Copy possible_rule_variables
        possible_rule_variables = other.possible_rule_variables;
        
        // Copy possible_split_vals_by_attr
        possible_split_vals_by_attr = other.possible_split_vals_by_attr;
        
        // Copy attribute_split_counts
        if (other.attribute_split_counts != nullptr) {
            // We need to know the size of the array, which should be bart->p
            if (bart != nullptr) {
                attribute_split_counts = new int[bart->p];
                std::copy(other.attribute_split_counts, other.attribute_split_counts + bart->p, attribute_split_counts);
            } else {
                attribute_split_counts = nullptr;
            }
        } else {
            attribute_split_counts = nullptr;
        }
        
        // Copy children
        left = nullptr;
        right = nullptr;
        
        if (other.left != nullptr) {
            left = other.left->clone();
            left->parent = this;
        }
        
        if (other.right != nullptr) {
            right = other.right->clone();
            right->parent = this;
        }
    }
    return *this;
}

// Clone method - matches Java: public bartMachineTreeNode clone()
bartMachineTreeNode* bartMachineTreeNode::clone() const {
    return new bartMachineTreeNode(*this);
}

// Basic tree navigation methods
bartMachineTreeNode* bartMachineTreeNode::getLeft() {
    return left;
}

bartMachineTreeNode* bartMachineTreeNode::getRight() {
    return right;
}

int bartMachineTreeNode::getGeneration() {
    return depth;
}

void bartMachineTreeNode::setGeneration(int generation) {
    depth = generation;
}

bool bartMachineTreeNode::getIsLeaf() {
    return isLeaf;
}

void bartMachineTreeNode::setLeaf(bool isLeaf) {
    this->isLeaf = isLeaf;
}

void bartMachineTreeNode::setLeft(bartMachineTreeNode* left) {
    this->left = left;
}

void bartMachineTreeNode::setRight(bartMachineTreeNode* right) {
    this->right = right;
}

int bartMachineTreeNode::getSplitAttributeM() {
    return splitAttributeM;
}

void bartMachineTreeNode::setSplitAttributeM(int splitAttributeM) {
    this->splitAttributeM = splitAttributeM;
}

double bartMachineTreeNode::getSplitValue() {
    return splitValue;
}

void bartMachineTreeNode::setSplitValue(double splitValue) {
    this->splitValue = splitValue;
}

// Tree property methods
double bartMachineTreeNode::avgResponse() {
    return StatToolbox::sample_average(responses, n_eta);
}

bool bartMachineTreeNode::isStump() {
    return parent == nullptr && left == nullptr && right == nullptr;
}

int bartMachineTreeNode::deepestNode() {
    if (this->isLeaf) {
        return 0;
    } else {
        // In the Java implementation, if a node is not a leaf, it's guaranteed to have both left and right children.
        // In C++, we need to check for null pointers.
        int ldn = 0;
        int rdn = 0;
        
        if (this->left != nullptr) {
            ldn = this->left->deepestNode();
        }
        
        if (this->right != nullptr) {
            rdn = this->right->deepestNode();
        }
        
        // Exact port of Java implementation
        if (ldn > rdn) {
            return 1 + ldn;
        } else {
            return 1 + rdn;
        }
    }
}

int bartMachineTreeNode::numLeaves() {
    if (this->isLeaf) {
        return 1;
    } else {
        // In the Java implementation, if a node is not a leaf, it's guaranteed to have both left and right children.
        // In C++, we need to check for null pointers.
        int leftLeaves = 0;
        int rightLeaves = 0;
        
        if (this->left != nullptr) {
            leftLeaves = this->left->numLeaves();
        }
        
        if (this->right != nullptr) {
            rightLeaves = this->right->numLeaves();
        }
        
        // If both children are null, this node is effectively a leaf
        if (leftLeaves == 0 && rightLeaves == 0) {
            return 1;
        }
        
        return leftLeaves + rightLeaves;
    }
}

int bartMachineTreeNode::numNodesAndLeaves() {
    if (this->isLeaf) {
        return 1;
    } else {
        // In the Java implementation, if a node is not a leaf, it's guaranteed to have both left and right children.
        // In C++, we need to check for null pointers.
        int leftCount = 0;
        int rightCount = 0;
        
        if (this->left != nullptr) {
            leftCount = this->left->numNodesAndLeaves();
        }
        
        if (this->right != nullptr) {
            rightCount = this->right->numNodesAndLeaves();
        }
        
        return 1 + leftCount + rightCount;
    }
}

int bartMachineTreeNode::numPruneNodesAvailable() {
    if (this->isLeaf) {
        return 0;
    }
    
    // Check if both children are leaves (or null)
    bool leftIsLeaf = (this->left == nullptr || this->left->isLeaf);
    bool rightIsLeaf = (this->right == nullptr || this->right->isLeaf);
    
    if (leftIsLeaf && rightIsLeaf) {
        return 1;
    }
    
    int leftCount = 0;
    int rightCount = 0;
    
    if (this->left != nullptr) {
        leftCount = this->left->numPruneNodesAvailable();
    }
    
    if (this->right != nullptr) {
        rightCount = this->right->numPruneNodesAvailable();
    }
    
    return leftCount + rightCount;
}

// Tree manipulation methods
void bartMachineTreeNode::pruneTreeAt(bartMachineTreeNode* node) {
    node->left = nullptr;
    node->right = nullptr;
    node->isLeaf = true;
    node->splitAttributeM = BAD_FLAG_int;
    node->splitValue = BAD_FLAG_double;
}

// Random operations
double bartMachineTreeNode::pickRandomSplitValue() {
    std::vector<double> split_values = possibleSplitValuesGivenAttribute();
    if (split_values.empty()) {
        return bartMachineTreeNode::BAD_FLAG_double;
    }

    double rand_val = StatToolbox::rand();
    int rand_index = static_cast<int>(std::floor(rand_val * split_values.size()));

    if (DEBUG_NODES) {
        std::cout << "DEBUG: pickRandomSplitValue() for feature " << splitAttributeM << std::endl;
        std::cout << "  Split values: ";
        for (double val : split_values) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::cout << "  Random value: " << rand_val << std::endl;
        std::cout << "  Random index: " << rand_index << " (out of " << split_values.size() << ")" << std::endl;
        std::cout << "  Selected split value: " << split_values[rand_index] << std::endl;
    }

    return split_values[rand_index];
}

bool bartMachineTreeNode::pickRandomDirectionForMissingData() {
    double rand_val = StatToolbox::rand();
    if (DEBUG_NODES) {
        std::cout << "DEBUG: pickRandomDirectionForMissingData() rand value: " << rand_val << std::endl;
    }
    return rand_val < 0.5 ? false : true;
}

// Helper methods for random operations
std::vector<int> bartMachineTreeNode::predictorsThatCouldBeUsedToSplitAtNode() {
    if (bart->mem_cache_for_speed) {
        if (possible_rule_variables.empty()) {
            possible_rule_variables = tabulatePredictorsThatCouldBeUsedToSplitAtNode();
        }
        return possible_rule_variables;
    } else {
        return tabulatePredictorsThatCouldBeUsedToSplitAtNode();
    }
}

std::vector<int> bartMachineTreeNode::tabulatePredictorsThatCouldBeUsedToSplitAtNode() {
    // Define debug log function but make it a no-op to avoid excessive output
    auto debug_task_5_3_log = [](const std::string& message) {
        // Comment out debug output to prevent crashes
        // std::cout << "[DEBUG_TASK_5_3] " << message << std::endl;
    };
    
    std::unordered_set<int> possible_rule_variables_contenders;
    if (bart->mem_cache_for_speed && parent != nullptr) {
        // Check interaction constraints first
        int m = parent->splitAttributeM;
        if (bart->hasInteractionConstraints(m)) {
            possible_rule_variables_contenders.insert(m); // You should always be able to split on the same feature as above
            for (const auto& feature : bart->getInteractionConstraints(m)) {
                possible_rule_variables_contenders.insert(feature);
            }
        } else {
            for (const auto& var : parent->possible_rule_variables) {
                possible_rule_variables_contenders.insert(var);
            }
        }
    }

    // Debug output disabled to prevent crashes

    std::vector<int> possible_rule_variables;
    
    // Use bart->X_y_by_col.size() - 1 instead of bart->p
    // This is because in some cases (like in the test code), bart->p might not be set correctly,
    // but bart->X_y_by_col.size() will always reflect the actual number of predictors + 1 (for the response variable)
    int num_predictors = bart->X_y_by_col.size() - 1;
    
    // Debug output disabled to prevent crashes
    
    for (int j = 0; j < num_predictors; j++) {
        // Debug output disabled to prevent crashes
        
        // Only skip if possible_rule_variables_contenders is not empty AND j is not in it
        if (!possible_rule_variables_contenders.empty() && possible_rule_variables_contenders.count(j) == 0) {
            // Debug output disabled to prevent crashes
            continue;
        }
        
        // If size of unique of x_i > 1
        double* x_dot_j = bart->X_y_by_col[j];
        
        // Debug output disabled to prevent crashes
        
        for (int i = 1; i < n_eta; i++) {
            if (x_dot_j[indicies[i - 1]] != x_dot_j[indicies[i]]) {
                // Debug output disabled to prevent crashes
                possible_rule_variables.push_back(j);
                break;
            }
        }
    }
    
    // Debug output disabled to prevent crashes
    
    return possible_rule_variables;
}

int bartMachineTreeNode::nAdj() {
    return possibleSplitValuesGivenAttribute().size();
}

std::vector<double> bartMachineTreeNode::possibleSplitValuesGivenAttribute() {
    if (bart->mem_cache_for_speed) {
        if (possible_split_vals_by_attr.count(splitAttributeM) == 0) {
            possible_split_vals_by_attr[splitAttributeM] = tabulatePossibleSplitValuesGivenAttribute();
        }
        return possible_split_vals_by_attr[splitAttributeM];
    } else {
        return tabulatePossibleSplitValuesGivenAttribute();
    }
}

std::vector<double> bartMachineTreeNode::tabulatePossibleSplitValuesGivenAttribute() {
    double* x_dot_j = bart->X_y_by_col[splitAttributeM];
    std::vector<double> x_dot_j_node(n_eta);

    if (DEBUG_NODES) {
        // Debug output
        std::cout << "\nDEBUG: tabulatePossibleSplitValuesGivenAttribute() for feature " << splitAttributeM << std::endl;
        std::cout << "  Raw values: ";
    }

    for (int i = 0; i < n_eta; i++) {
        double val = x_dot_j[indicies[i]];
        if (Classifier::isMissing(val)) {
            x_dot_j_node[i] = BAD_FLAG_double;
            if (DEBUG_NODES) {
                std::cout << "MISSING ";
            }
        } else {
            x_dot_j_node[i] = val;
            if (DEBUG_NODES) {
                std::cout << val << " ";
            }
        }
    }
    
    if (DEBUG_NODES) {
        std::cout << std::endl;
    }

    // Create a vector of unique values (preserving order of first occurrence)
    std::vector<double> unique_values;
    for (double val : x_dot_j_node) {
        if (val != BAD_FLAG_double) { // Skip missing values
            // Only add if not already in the vector
            if (std::find(unique_values.begin(), unique_values.end(), val) == unique_values.end()) {
                unique_values.push_back(val);
            }
        }
    }

    if (DEBUG_NODES) {
        // Debug output
        std::cout << "  Unique values: ";
        for (double val : unique_values) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Find the maximum value
    double max_val = BAD_FLAG_double;
    for (double val : x_dot_j_node) {
        if (val != BAD_FLAG_double && (max_val == BAD_FLAG_double || val > max_val)) {
            max_val = val;
        }
    }

    if (DEBUG_NODES) {
        std::cout << "  Maximum value: " << max_val << std::endl;
    }

    // Create result vector excluding the maximum value
    std::vector<double> result;
    for (double val : unique_values) {
        if (val != max_val) { // Skip the maximum value
            result.push_back(val);
        }
    }

    if (DEBUG_NODES) {
        // Debug output
        std::cout << "  Result values: ";
        for (double val : result) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return result;
}

void bartMachineTreeNode::clearRulesAndSplitCache() {
    possible_rule_variables.clear();
    possible_split_vals_by_attr.clear();
}

// Debug and utility methods
std::string bartMachineTreeNode::stringID() {
    std::stringstream ss;
    ss << this;
    std::string str = ss.str();
    size_t pos = str.find('@');
    if (pos != std::string::npos) {
        return str.substr(pos + 1);
    }
    return str;
}

std::string bartMachineTreeNode::stringLocation(bool show_parent) {
    if (this->parent == nullptr) {
        return show_parent ? "P" : "";
    } else if (this->parent->left == this) {
        return this->parent->stringLocation(show_parent) + "L";
    } else if (this->parent->right == this) {
        return this->parent->stringLocation(show_parent) + "R";
    } else {
        return this->parent->stringLocation(show_parent) + "?";
    }
}

std::string bartMachineTreeNode::stringLocation() {
    return stringLocation(true);
}

// Data management methods
double bartMachineTreeNode::prediction_untransformed() {
    return y_pred == BAD_FLAG_double ? BAD_FLAG_double : bart->un_transform_y(y_pred);
}

double bartMachineTreeNode::avg_response_untransformed() {
    return bart->un_transform_y(avgResponse());
}

double bartMachineTreeNode::sumResponsesQuantitySqd() {
    if (sum_responses_qty_sqd == 0) {
        sum_responses_qty_sqd = std::pow(sumResponses(), 2);
    }
    return sum_responses_qty_sqd;
}

double bartMachineTreeNode::sumResponses() {
    if (sum_responses_qty == 0) {
        sum_responses_qty = 0.0;
        for (int i = 0; i < n_eta; i++) {
            sum_responses_qty += responses[i];
        }
    }
    return sum_responses_qty;
}

void bartMachineTreeNode::flushNodeData() {
    yhats = nullptr;
    if (bart && bart->flush_indices_to_save_ram) {
        indicies = nullptr;
    }
    responses = nullptr;
    possible_rule_variables.clear();
    possible_split_vals_by_attr.clear();
    
    if (this->left != nullptr) {
        this->left->flushNodeData();
    }
    if (this->right != nullptr) {
        this->right->flushNodeData();
    }
}

void bartMachineTreeNode::propagateDataByChangedRule() {
    if (isLeaf) {
        if (DEBUG_NODES) {
            printNodeDebugInfo("propagateDataByChangedRule LEAF");
        }
        return;
    }
    
    // Split the data correctly
    std::vector<int> left_indices;
    std::vector<int> right_indices;
    std::vector<double> left_responses;
    std::vector<double> right_responses;
    
    left_indices.reserve(n_eta);
    right_indices.reserve(n_eta);
    left_responses.reserve(n_eta);
    right_responses.reserve(n_eta);
    
    for (int i = 0; i < n_eta; i++) {
        double* datum = bart->X_y[indicies[i]];
        // Handle missing data first
        if (Classifier::isMissing(datum[splitAttributeM])) {
            if (sendMissingDataRight) {
                right_indices.push_back(indicies[i]);
                right_responses.push_back(responses[i]);
            } else {
                left_indices.push_back(indicies[i]);
                left_responses.push_back(responses[i]);
            }
        } else if (datum[splitAttributeM] <= splitValue) {
            left_indices.push_back(indicies[i]);
            left_responses.push_back(responses[i]);
        } else {
            right_indices.push_back(indicies[i]);
            right_responses.push_back(responses[i]);
        }
    }
    
    // Populate the left daughter
    left->n_eta = left_responses.size();
    left->responses = new double[left->n_eta];
    left->indicies = new int[left->n_eta];
    
    for (int i = 0; i < left->n_eta; i++) {
        left->responses[i] = left_responses[i];
        left->indicies[i] = left_indices[i];
    }
    
    // Populate the right daughter
    right->n_eta = right_responses.size();
    right->responses = new double[right->n_eta];
    right->indicies = new int[right->n_eta];
    
    for (int i = 0; i < right->n_eta; i++) {
        right->responses[i] = right_responses[i];
        right->indicies[i] = right_indices[i];
    }
    
    // Recursively propagate to children
    left->propagateDataByChangedRule();
    right->propagateDataByChangedRule();
}

void bartMachineTreeNode::updateWithNewResponsesRecursively(double* new_responses) {
    // Nuke previous responses and sums
    delete[] responses;
    responses = new double[n_eta]; // Ensure correct dimension
    sum_responses_qty_sqd = 0; // Need to be primitives
    sum_responses_qty = 0; // Need to be primitives
    
    // Copy all the new data in appropriately
    for (int i = 0; i < n_eta; i++) {
        double y_new = new_responses[indicies[i]];
        responses[i] = y_new;
    }
    
    if (DEBUG_NODES) {
        std::cout << "new_responses: (size " << n_eta << ") [";
        for (int i = 0; i < n_eta; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << bart->un_transform_y(new_responses[i]);
        }
        std::cout << "]" << std::endl;
        printNodeDebugInfo("updateWithNewResponsesRecursively");
    }
    
    if (this->isLeaf) {
        return;
    }
    
    this->left->updateWithNewResponsesRecursively(new_responses);
    this->right->updateWithNewResponsesRecursively(new_responses);
}

void bartMachineTreeNode::setStumpData(std::vector<double*>& X_y, double* y_trans, int p) {
    // Pull out X data, set y's, and indices appropriately
    n_eta = X_y.size();
    
    responses = new double[n_eta];
    indicies = new int[n_eta];
    
    for (int i = 0; i < n_eta; i++) {
        indicies[i] = i;
    }
    
    for (int i = 0; i < n_eta; i++) {
        responses[i] = y_trans[i];
    }
    
    // Initialize the yhats
    yhats = new double[n_eta];
    // Initialize sendMissing
    sendMissingDataRight = pickRandomDirectionForMissingData();
    
    // Initialize attribute_split_counts
    attribute_split_counts = new int[p];
    for (int i = 0; i < p; i++) {
        attribute_split_counts[i] = 0;
    }
    
    
    if (DEBUG_NODES) {
        printNodeDebugInfo("setStumpData");
    }
}

void bartMachineTreeNode::updateYHatsWithPrediction() {
    for (int i = 0; i < n_eta; i++) {
        yhats[indicies[i]] = y_pred;
    }
    
    if (DEBUG_NODES) {
        printNodeDebugInfo("updateYHatsWithPrediction");
    }
}

// Evaluation methods
double bartMachineTreeNode::Evaluate(double* record) {
    return EvaluateNode(record)->y_pred;
}

bartMachineTreeNode* bartMachineTreeNode::EvaluateNode(double* record) {
    bartMachineTreeNode* evalNode = this;
    while (true) {
        if (evalNode->isLeaf) {
            return evalNode;
        }
        
        // All split rules are less than or equals (this is merely a convention)
        // Handle missing data first
        if (Classifier::isMissing(record[evalNode->splitAttributeM])) {
            evalNode = evalNode->sendMissingDataRight ? evalNode->right : evalNode->left;
        } else if (record[evalNode->splitAttributeM] <= evalNode->splitValue) {
            evalNode = evalNode->left;
        } else {
            evalNode = evalNode->right;
        }
    }
}

// Public methods for tree operations
std::vector<bartMachineTreeNode*> bartMachineTreeNode::getTerminalNodesWithDataAboveOrEqualToN(int n_rule) {
    std::vector<bartMachineTreeNode*> terminal_nodes_data_above_n;
    findTerminalNodesDataAboveOrEqualToN(this, terminal_nodes_data_above_n, n_rule);
    return terminal_nodes_data_above_n;
}

std::vector<bartMachineTreeNode*> bartMachineTreeNode::getPrunableAndChangeableNodes() {
    std::vector<bartMachineTreeNode*> prunable_and_changeable_nodes;
    findPrunableAndChangeableNodes(this, prunable_and_changeable_nodes);
    return prunable_and_changeable_nodes;
}

// Helper methods for tree operations (private)
void bartMachineTreeNode::findTerminalNodesDataAboveOrEqualToN(bartMachineTreeNode* node, std::vector<bartMachineTreeNode*>& terminal_nodes, int n_rule) {
    if (node->isLeaf && node->n_eta >= n_rule) {
        terminal_nodes.push_back(node);
    } else if (!node->isLeaf) { // As long as we're not in a leaf we should recurse
        if (node->left == nullptr || node->right == nullptr) {
            std::cerr << "error node no children during findTerminalNodesDataAboveOrEqualToN id: " << node->stringID() << std::endl;
        }
        findTerminalNodesDataAboveOrEqualToN(node->left, terminal_nodes, n_rule);
        findTerminalNodesDataAboveOrEqualToN(node->right, terminal_nodes, n_rule);
    }
}

void bartMachineTreeNode::findPrunableAndChangeableNodes(bartMachineTreeNode* node, std::vector<bartMachineTreeNode*>& prunable_nodes) {
    if (node->isLeaf) {
        return;
    } else if (node->left->isLeaf && node->right->isLeaf) {
        prunable_nodes.push_back(node);
    } else {
        findPrunableAndChangeableNodes(node->left, prunable_nodes);
        findPrunableAndChangeableNodes(node->right, prunable_nodes);
    }
}

void bartMachineTreeNode::printNodeDebugInfo(const std::string& title) {
    std::cout << "\n" << title << " node debug info for " << this->stringLocation(true) << (isLeaf ? " (LEAF) " : " (INTERNAL NODE) ") << " d = " << depth << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "n_eta = " << n_eta << " y_pred = " << (y_pred == BAD_FLAG_double ? "BLANK" : std::to_string(bart->un_transform_y(y_pred))) << std::endl;
    std::cout << "parent = " << parent << " left = " << left << " right = " << right << std::endl;
    
    if (this->parent != nullptr) {
        std::cout << "----- PARENT RULE:   X_" << parent->splitAttributeM << " <= " << parent->splitValue << " & M -> " << (parent->sendMissingDataRight ? "R" : "L") << " ------" << std::endl;
        // Get vals of this x currently here
        if (bart && parent && parent->splitAttributeM >= 0 && 
            static_cast<size_t>(parent->splitAttributeM) < bart->X_y_by_col.size()) {
            double* x_dot_j = bart->X_y_by_col[parent->splitAttributeM];
            std::vector<double> x_dot_j_node(n_eta);
            for (int i = 0; i < n_eta; i++) {
                x_dot_j_node[i] = x_dot_j[indicies[i]];
            }
            std::sort(x_dot_j_node.begin(), x_dot_j_node.end());
            std::cout << "   all X_" << parent->splitAttributeM << " values here: [";
            for (int i = 0; i < n_eta; i++) {
                if (i > 0) std::cout << ", ";
                std::cout << x_dot_j_node[i];
            }
            std::cout << "]" << std::endl;
        }
    }
    
    if (!isLeaf) {
        std::cout << "----- RULE:   X_" << splitAttributeM << " <= " << splitValue << " & M -> " << (sendMissingDataRight ? "R" : "L") << " ------" << std::endl;
        // Get vals of this x currently here
        if (bart && splitAttributeM >= 0 && 
            static_cast<size_t>(splitAttributeM) < bart->X_y_by_col.size()) {
            double* x_dot_j = bart->X_y_by_col[splitAttributeM];
            std::vector<double> x_dot_j_node(n_eta);
            for (int i = 0; i < n_eta; i++) {
                x_dot_j_node[i] = x_dot_j[indicies[i]];
            }
            std::sort(x_dot_j_node.begin(), x_dot_j_node.end());
            std::cout << "   all X_" << splitAttributeM << " values here: [";
            for (int i = 0; i < n_eta; i++) {
                if (i > 0) std::cout << ", ";
                std::cout << x_dot_j_node[i];
            }
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << "sum_responses_qty = " << sum_responses_qty << " sum_responses_qty_sqd = " << sum_responses_qty_sqd << std::endl;
    
    if (bart && bart->mem_cache_for_speed) {
        std::cout << "possible_rule_variables: [";
        for (size_t i = 0; i < possible_rule_variables.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << possible_rule_variables[i];
        }
        std::cout << "]" << std::endl;
        
        std::cout << "possible_split_vals_by_attr: {" << std::endl;
        if (!possible_split_vals_by_attr.empty()) {
            for (const auto& pair : possible_split_vals_by_attr) {
                std::cout << "  " << pair.first << " -> [";
                const auto& values = pair.second;
                for (size_t i = 0; i < values.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << values[i];
                }
                std::cout << "]," << std::endl;
            }
            std::cout << " }" << std::endl;
        } else {
            std::cout << " NULL MAP" << std::endl << "}" << std::endl;
        }
    }
    
    std::cout << "responses: (size " << n_eta << ") [";
    for (int i = 0; i < n_eta; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << bart->un_transform_y(responses[i]);
    }
    std::cout << "]" << std::endl;
    
    std::cout << "indicies: (size " << n_eta << ") [";
    for (int i = 0; i < n_eta; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << indicies[i];
    }
    std::cout << "]" << std::endl;
    
    bool all_zeros = true;
    for (int i = 0; i < n_eta; i++) {
        if (yhats[i] != 0.0) {
            all_zeros = false;
            break;
        }
    }
    
    if (all_zeros) {
        std::cout << "y_hat_vec: (size " << n_eta << ") [ BLANK ]" << std::endl;
    } else {
        std::cout << "y_hat_vec: (size " << n_eta << ") [";
        for (int i = 0; i < n_eta; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << bart->un_transform_y(yhats[i]);
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "-----------------------------------------" << std::endl << std::endl << std::endl;
}
