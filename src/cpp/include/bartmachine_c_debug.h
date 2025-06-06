#ifndef BARTMACHINE_C_DEBUG_H
#define BARTMACHINE_C_DEBUG_H

#include "bartmachine_b_hyperparams.h"
#include "bartmachine_tree_node.h"
#include "tree_array_illustration.h"
#include <string>

/**
 * Exact port of bartMachine_c_debug from Java to C++
 * 
 * This portion of the code used to have many debug functions. These have 
 * been removed during the tidy up for release.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_c_debug.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_c_debug : public bartmachine_b_hyperparams {
protected:
    /** should we create illustrations of the trees and save the images to the debug directory? */
    bool tree_illust = false;
    
    /** a unique name for this BART model (used for debugging) */
    std::string unique_name;
    
    /** the hook that gets called to save the tree illustrations when the Gibbs sampler begins */
    void InitTreeIllustrations();
    
    /** the hook that gets called to save the tree illustrations for each Gibbs sample */
    void illustrate(TreeArrayIllustration* tree_array_illustration);

public:
    /**
     * Set the debug status
     * 
     * @param status    The debug status to set
     */
    void setDebugStatus(bool status);
    
    /**
     * Get the debug status
     * 
     * @return  The current debug status
     */
    bool getDebugStatus();
    
    /**
     * Get the untransformed samples of the sigsqs from the Gibbs chain
     * 
     * @return  The vector of untransformed variances over all the Gibbs samples
     */
    double* getGibbsSamplesSigsqs();
    
    /**
     * Queries the depths of the <code>num_trees</code> trees between a range of Gibbs samples
     * 
     * @param n_g_i     The Gibbs sample number to start querying
     * @param n_g_f     The Gibbs sample number (+1) to stop querying
     * @return          The depths of all <code>num_trees</code> trees for each Gibbs sample specified
     */
    int** getDepthsForTrees(int n_g_i, int n_g_f);
    
    /**
     * Queries the number of nodes (terminal and non-terminal) in the <code>num_trees</code> trees between a range of Gibbs samples
     * 
     * @param n_g_i     The Gibbs sample number to start querying
     * @param n_g_f     The Gibbs sample number (+1) to stop querying
     * @return          The number of nodes of all <code>num_trees</code> trees for each Gibbs sample specified
     */
    int** getNumNodesAndLeavesForTrees(int n_g_i, int n_g_f);
    
    // Virtual destructor
    virtual ~bartmachine_c_debug() = default;
};

#endif // BARTMACHINE_C_DEBUG_H
