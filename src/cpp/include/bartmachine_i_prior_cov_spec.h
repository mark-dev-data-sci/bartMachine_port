/**
 * bartmachine_i_prior_cov_spec.h
 * 
 * Header file for the bartmachine_i_prior_cov_spec class, which handles
 * prior covariate specifications for the BART model.
 */

#ifndef BARTMACHINE_I_PRIOR_COV_SPEC_H
#define BARTMACHINE_I_PRIOR_COV_SPEC_H

#include "bartmachine_h_eval.h"
#include <vector>
#include <string>

class bartmachine_i_prior_cov_spec : public bartmachine_h_eval {
protected:
    // Protected member variables
    
    // Covariate importance
    double* cov_importance_vec;
    double* cov_importance_sd_vec;
    
    // Interaction constraints
    bool** interaction_constraints;
    int num_interaction_constraints;
    
    // Variable names
    std::vector<std::string> var_names;
    
    // Split counts
    int** split_counts_by_var_and_tree;
    int* total_count_by_var;
    
public:
    // Constructor and destructor
    bartmachine_i_prior_cov_spec();
    virtual ~bartmachine_i_prior_cov_spec();
    
    // Covariate importance methods
    virtual void calcCovariateImportance();
    virtual double* getCovariateImportance();
    virtual double* getCovariateImportanceSD();
    
    // Interaction constraint methods
    virtual void setInteractionConstraints(bool** constraints, int num_constraints);
    virtual bool** getInteractionConstraints();
    virtual int getNumInteractionConstraints();
    virtual bool isInteractionAllowed(int var1, int var2);
    
    // Variable name methods
    virtual void setVarNames(std::vector<std::string> names);
    virtual std::vector<std::string> getVarNames();
    virtual std::string getVarName(int var_index);
    
    // Split count methods
    virtual void calcSplitCounts();
    virtual int** getSplitCountsByVarAndTree();
    virtual int* getTotalCountByVar();
    virtual int getSplitCountForVar(int var_index);
    
    // Covariate selection methods
    virtual void setCovariateSelectionMode(bool use_selection);
    virtual bool getCovariateSelectionMode();
    virtual void setCovariateSelectionThreshold(double threshold);
    virtual double getCovariateSelectionThreshold();
};

#endif // BARTMACHINE_I_PRIOR_COV_SPEC_H
