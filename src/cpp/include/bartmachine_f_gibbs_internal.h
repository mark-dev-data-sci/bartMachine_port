#ifndef BARTMACHINE_F_GIBBS_INTERNAL_H
#define BARTMACHINE_F_GIBBS_INTERNAL_H

#include "bartmachine_e_gibbs_base.h"

/**
 * Exact port of bartMachine_f_gibbs_internal from Java to C++
 * 
 * This portion of the code performs the posterior sampling
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_f_gibbs_internal.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_f_gibbs_internal : public bartmachine_e_gibbs_base {
protected:
    /**
     * Draws a sigsq from the posterior distribution
     * 
     * @param sample_num    The current sample number of the Gibbs sampler
     * @param es            The vector of residuals at this point in the Gibbs chain
     * @return              The drawn sigsq
     */
    double drawSigsqFromPosterior(int sample_num, double* es) override;
    
    /**
     * Calculates the posterior mean of a leaf node
     * 
     * @param node          The leaf node
     * @param current_sigsq The current sigsq
     * @param leaf_var      The leaf variance
     * @return              The posterior mean
     */
    double calcLeafPosteriorMean(bartMachineTreeNode* node, double current_sigsq, double leaf_var);
    
    /**
     * Calculates the posterior variance of a leaf node
     * 
     * @param node          The leaf node
     * @param current_sigsq The current sigsq
     * @return              The posterior variance
     */
    double calcLeafPosteriorVar(bartMachineTreeNode* node, double current_sigsq);
    
    /**
     * Assigns leaf values by sampling from the posterior mean and sigsq and updates the yhats
     * 
     * @param node          The node
     * @param current_sigsq The current sigsq
     */
    void assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats(bartMachineTreeNode* node, double current_sigsq) override;
    
    /**
     * Picks a random predictor that can be assigned to a node
     * 
     * @param node  The node
     * @return      The predictor index
     */
    int pickRandomPredictorThatCanBeAssigned(bartMachineTreeNode* node);
    
    /**
     * Calculates the adjusted probability of a node
     * 
     * @param node  The node
     * @return      The adjusted probability
     */
    double pAdj(bartMachineTreeNode* node);
    
    /**
     * Performs a Metropolis-Hastings step to sample from the posterior distribution of trees
     * 
     * @param copy_of_old_jth_tree  The old tree
     * @param t                     The tree index
     * @param accept_reject_mh      The accept/reject matrix
     * @param accept_reject_mh_steps The accept/reject steps matrix
     * @return                      The new tree
     */
    bartMachineTreeNode* metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode* copy_of_old_jth_tree, int t, bool** accept_reject_mh, char** accept_reject_mh_steps) override;
    
    // Virtual destructor
    virtual ~bartmachine_f_gibbs_internal() = default;
};

#endif // BARTMACHINE_F_GIBBS_INTERNAL_H
