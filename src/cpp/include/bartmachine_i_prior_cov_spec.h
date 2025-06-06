#ifndef BARTMACHINE_I_PRIOR_COV_SPEC_H
#define BARTMACHINE_I_PRIOR_COV_SPEC_H

#include "bartmachine_h_eval.h"
#include <vector>

// Forward declaration
class TIntArrayList;

/**
 * Exact port of bartMachine_i_prior_cov_spec from Java to C++
 * 
 * This portion of the code implements the informed prior information on covariates feature.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_i_prior_cov_spec.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_i_prior_cov_spec : public bartmachine_h_eval {
protected:
    /** Do we use this feature in this BART model? */
    bool use_prior_cov_spec = false;
    
    /** This is a probability vector which is the prior on which covariates to split instead of the uniform discrete distribution by default */
    double* cov_split_prior = nullptr;
    
    /**
     * Pick one predictor from a set of valid predictors that can be part of a split rule at a node
     * while accounting for the covariate prior.
     *
     * @param node  The node of interest
     * @return      The index of the column to split on
     */
    int pickRandomPredictorThatCanBeAssignedF1(bartMachineTreeNode* node);
    
    /**
     * The prior-adjusted number of covariates available to be split at this node
     *
     * @param node      The node of interest
     * @return          The prior-adjusted number of covariates that can be split
     */
    double pAdjF1(bartMachineTreeNode* node);
    
    /**
     * Given a set of valid predictors return the probability vector that corresponds to the
     * elements of <code>cov_split_prior</code> re-normalized because some entries may be deleted
     *
     * @param predictors    The indices of the valid covariates
     * @return              The updated and renormalized prior probability vector on the covariates to split
     */
    double* getWeightedCovSplitPriorSubset(TIntArrayList* predictors);

public:
    /**
     * Set the covariate split prior
     * 
     * @param cov_split_prior   The prior probability vector on the covariates to split
     */
    void setCovSplitPrior(double* cov_split_prior);
    
    /**
     * Pick one predictor from a set of valid predictors that can be part of a split rule at a node
     * 
     * @param node  The node of interest
     * @return      The index of the column to split on
     */
    int pickRandomPredictorThatCanBeAssigned(bartMachineTreeNode* node);
    
    /**
     * The prior-adjusted number of covariates available to be split at this node
     * 
     * @param node      The node of interest
     * @return          The prior-adjusted number of covariates that can be split
     */
    double pAdj(bartMachineTreeNode* node);
    
    // Virtual destructor
    virtual ~bartmachine_i_prior_cov_spec() = default;
};

#endif // BARTMACHINE_I_PRIOR_COV_SPEC_H
