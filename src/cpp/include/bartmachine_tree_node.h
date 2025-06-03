#ifndef BARTMACHINE_TREE_NODE_H
#define BARTMACHINE_TREE_NODE_H

#include "bartmachine_b_hyperparams.h"
#include "stat_toolbox.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>

/**
 * Exact port of bartMachineTreeNode from Java to C++
 * 
 * The class that stores all the information in one node of the BART trees
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachineTreeNode.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartMachineTreeNode {
public:
    /** Setting this to true will print out debug information at the node level during Gibbs sampling */
    static const bool DEBUG_NODES = false;
    
    /** a flag that represents an invalid double value */
    static const double BAD_FLAG_double;
    /** a flag that represents an invalid integer value */
    static const int BAD_FLAG_int;
    
    /** the parent node of this node */
    bartMachineTreeNode* parent;
    /** the left daughter node */
    bartMachineTreeNode* left;
    /** the right daughter node */
    bartMachineTreeNode* right;
    /** the generation of this node from the top node (root note has generation = 0 by definition) */
    int depth;
    /** is this node a terminal node? */
    bool isLeaf;
    /** the attribute this node makes a decision on */
    int splitAttributeM;
    /** the value this node makes a decision on */
    double splitValue;
    /** send missing data to the right? */ 
    bool sendMissingDataRight;
    /** if this is a leaf node, then the result of the prediction for regression, otherwise null */
    double y_pred;
    /** the average of the node responses if applicable (for research purposes only) */
    double y_avg;
    /** the posterior variance in the conditional distribution if applicable (for research purposes only) */
    double posterior_var;
    /** the posterior mean in the conditional distribution if applicable (for research purposes only) */
    double posterior_mean;
    
    /** the number of data points in this node */
    int n_eta;
    /** these are the yhats in the correct order */
    double* yhats;

    // Constructors - exact signatures from Java
    bartMachineTreeNode();
    bartMachineTreeNode(bartMachineTreeNode* parent, bartmachine_b_hyperparams* bart);
    bartMachineTreeNode(bartMachineTreeNode* parent);
    bartMachineTreeNode(bartmachine_b_hyperparams* bart);
    
    // Destructor
    ~bartMachineTreeNode();
    
    // Copy constructor and assignment operator
    bartMachineTreeNode(const bartMachineTreeNode& other);
    bartMachineTreeNode& operator=(const bartMachineTreeNode& other);
    
    // Clone method - exact signature from Java
    bartMachineTreeNode* clone() const;
    
    // Basic tree navigation methods
    bartMachineTreeNode* getLeft();
    bartMachineTreeNode* getRight();
    int getGeneration();
    void setGeneration(int generation);
    bool getIsLeaf();
    void setLeaf(bool isLeaf);
    void setLeft(bartMachineTreeNode* left);
    void setRight(bartMachineTreeNode* right);
    int getSplitAttributeM();
    void setSplitAttributeM(int splitAttributeM);
    double getSplitValue();
    void setSplitValue(double splitValue);
    
    // Tree property methods
    double avgResponse();
    bool isStump();
    int deepestNode();
    int numLeaves();
    int numNodesAndLeaves();
    int numPruneNodesAvailable();
    
    // Data management methods
    double prediction_untransformed();
    double avg_response_untransformed();
    double sumResponsesQuantitySqd();
    double sumResponses();
    void flushNodeData();
    void propagateDataByChangedRule();
    void updateWithNewResponsesRecursively(double* new_responses);
    void setStumpData(std::vector<double*>& X_y, double* y_trans, int p);
    void updateYHatsWithPrediction();
    
    // Tree manipulation methods
    static void pruneTreeAt(bartMachineTreeNode* node);
    
    // Random operations
    double pickRandomSplitValue();
    static bool pickRandomDirectionForMissingData();
    
    // Debug and utility methods
    std::string stringID();
    std::string stringLocation(bool show_parent);
    std::string stringLocation();
    void printNodeDebugInfo(const std::string& title);
    
    // Evaluation methods
    double Evaluate(double* record);
    bartMachineTreeNode* EvaluateNode(double* record);
    
    // Public methods for tree operations
    std::vector<bartMachineTreeNode*> getTerminalNodesWithDataAboveOrEqualToN(int n_rule);
    std::vector<bartMachineTreeNode*> getPrunableAndChangeableNodes();
    
private:
    /** a link back to the overall bart model */
    bartmachine_b_hyperparams* bart;
    
    /** the indices in {0, 1, ..., n-1} of the data records in this node */
    int* indicies;
    /** the y's in this node */
    double* responses;
    /** the square of the sum of the responses, y */
    double sum_responses_qty_sqd;
    /** the sum of the responses, y */
    double sum_responses_qty;
    /** this caches the possible split variables populated only if the <code>mem_cache_for_speed</code> feature is set to on */
    std::vector<int> possible_rule_variables;
    /** this caches the possible split values BY variable populated only if the <code>mem_cache_for_speed</code> feature is set to on */
    std::unordered_map<int, std::vector<double>> possible_split_vals_by_attr;
    /** this number of possible split variables at this node */
    int padj;
    /** a tabulation of the counts of attributes being used in split points in this tree */
    int* attribute_split_counts;
    
    // Helper methods
    std::vector<int> predictorsThatCouldBeUsedToSplitAtNode();
    std::vector<int> tabulatePredictorsThatCouldBeUsedToSplitAtNode();
    int nAdj();
    std::vector<double> possibleSplitValuesGivenAttribute();
    std::vector<double> tabulatePossibleSplitValuesGivenAttribute();
    void clearRulesAndSplitCache();
    
    // Helper methods for tree operations (private)
    static void findTerminalNodesDataAboveOrEqualToN(bartMachineTreeNode* node, std::vector<bartMachineTreeNode*>& terminal_nodes, int n_rule);
    static void findPrunableAndChangeableNodes(bartMachineTreeNode* node, std::vector<bartMachineTreeNode*>& prunable_nodes);
};

#endif // BARTMACHINE_TREE_NODE_H
