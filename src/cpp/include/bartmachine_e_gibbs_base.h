#ifndef BARTMACHINE_E_GIBBS_BASE_H
#define BARTMACHINE_E_GIBBS_BASE_H

#include "bartmachine_d_init.h"
#include <string>

// Forward declaration
class TreeArrayIllustration;

/**
 * Exact port of bartMachine_e_gibbs_base from Java to C++
 * 
 * This portion of the code performs everything in 
 * the Gibbs sampling except for the posterior sampling itself
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_e_gibbs_base.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_e_gibbs_base : public bartmachine_d_init {
public:
    /** Builds a BART model by unleashing the Gibbs sampler */
    void Build();

protected:
    /** Run the Gibbs sampler for the total number of samples prespecified while flushing unneeded memory from the previous sample */
    void DoGibbsSampling();

    /** Run one Gibbs sample at the current sample number */ 
    void DoOneGibbsSample();

    /** Print a Gibbs sample debug message */
    void GibbsSampleDebugMessage(int t);

    /** 
     * A wrapper for sampling the mus (mean predictions at terminal nodes). This function implements part of the "residual diffing" explained in the paper.
     * 
     * @param sample_num    The current sample number of the Gibbs sampler
     * @param t             The tree index number in 1...<code>num_trees</code>
     */
    void SampleMusWrapper(int sample_num, int t);

    /**
     * A wrapper that is responsible for drawing variance values
     *  
     * @param sample_num    The current sample number of the Gibbs sampler
     * @param es            The vector of residuals at this point in the Gibbs chain
     */
    void SampleSigsq(int sample_num, double* es);
    
    /**
     * This function calculates the residuals from the sum-of-trees model using the diff trick explained in the paper
     * 
     * @param sample_num    The current sample number of the Gibbs sampler
     * @param R_j           The residuals of the model save the last tree's contribution
     * @return              The vector of residuals at this point in the Gibbs chain
     */
    double* getResidualsFromFullSumModel(int sample_num, double* R_j);
    
    /**
     * A wrapper for sampling one tree during the Gibbs sampler
     * 
     * @param sample_num                The current sample number of the Gibbs sampler
     * @param t                         The current tree to be sampled
     * @param trees                     The trees in this Gibbs sampler
     * @param tree_array_illustration   The tree array (for debugging purposes only)
     * @return                          The responses minus the sum of the trees' contribution up to this point
     */
    double* SampleTree(int sample_num, int t, bartMachineTreeNode** trees, TreeArrayIllustration* tree_array_illustration);

    // Abstract methods to be implemented by derived classes
    virtual double drawSigsqFromPosterior(int sample_num, double* es) = 0;
    
    virtual bartMachineTreeNode* metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode* copy_of_old_jth_tree, int t, bool** accept_reject_mh, char** accept_reject_mh_steps) = 0;

    virtual void assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats(bartMachineTreeNode* node, double current_sigsq) = 0;
    
public:
    // Virtual destructor
    virtual ~bartmachine_e_gibbs_base() = default;

private:
    /** deletes from memory tree Gibbs samples in the burn-in portion of the chain */
    void DeleteBurnInsOnPreviousSamples();
};

#endif // BARTMACHINE_E_GIBBS_BASE_H
