#ifndef BARTMACHINE_D_INIT_H
#define BARTMACHINE_D_INIT_H

#include "bartmachine_c_debug.h"
#include "bartmachine_tree_node.h"
#include <string>

// Forward declaration for TreeArrayIllustration
class TreeArrayIllustration;

/**
 * Exact port of bartMachine_d_init from Java to C++
 * 
 * This portion of the code initializes the Gibbs sampler
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_d_init.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_d_init : public bartmachine_c_debug {
protected:
    /** during debugging, we may want to fix sigsq */
    double fixed_sigsq;
    /** the number of the current Gibbs sample */
    int gibbs_sample_num;
    /** cached current sum of residuals vector */
    double* sum_resids_vec;
    /** a unique name for this BART model (used for debugging) */
    std::string unique_name;
    
    /** Initializes the Gibbs sampler setting all zero entries and moves the counter to the first sample */
    void SetupGibbsSampling();
    
    /** Initializes the vectors that hold information across the Gibbs sampler */
    void InitGibbsSamplingData();
    
    /** Initializes the tree structures in the zeroth Gibbs sample to be merely stumps */
    void InitializeTrees();
    
    /** Initializes the leaf structure (the mean predictions) by setting them to zero (in the transformed scale, this is the center of the range) */
    void InitializeMus();
    
    /** Initializes the first variance value by drawing from the prior */
    void InitizializeSigsq();
    
    /** deletes from memory tree Gibbs samples in the burn-in portion of the chain */
    void DeleteBurnInsOnPreviousSamples();
    
    /** the hook that gets called to save the tree illustrations for each Gibbs sample */
    void illustrate(TreeArrayIllustration* tree_array_illustration);

public:
    /** this is the number of posterior Gibbs samples after burn-in (thinning was never implemented) */
    int numSamplesAfterBurningAndThinning();
    
    void setNumGibbsBurnIn(int num_gibbs_burn_in);
    void setNumGibbsTotalIterations(int num_gibbs_total_iterations);
    void setSigsq(double fixed_sigsq);
    
    bool** getAcceptRejectMH();
    
    // Virtual destructor
    virtual ~bartmachine_d_init() = default;
};

#endif // BARTMACHINE_D_INIT_H
