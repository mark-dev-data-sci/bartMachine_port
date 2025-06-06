#ifndef BARTMACHINE_B_HYPERPARAMS_H
#define BARTMACHINE_B_HYPERPARAMS_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "stat_toolbox.h"
#include "bartmachine_a_base.h"

/**
 * Exact port of bartMachine_b_hyperparams from Java to C++
 * 
 * This portion of the code controls hyperparameters for the BART
 * algorithm as well as properties and transformations of the response variable.
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_b_hyperparams.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_b_hyperparams : public bartmachine_a_base {
protected:
    /** the number of predictors in the training set */
    int p;
    
    /** the training data */
    std::vector<double*> X_y;
    
    /** the training data organized by column */
    std::vector<double*> X_y_by_col;
    
    /** the original response variable */
    double* y_orig;
    
    /** the transformed response variable */
    double* y_trans;
    
    /** the number of observations */
    int n;
public:
    /** The static field that controls the bounds on the transformed y variable which is between negative and positive this value */
    static constexpr double YminAndYmaxHalfDiff = 0.5;
    
    /** A cached library of chi-squared with degrees of freedom nu plus n (used for Gibbs sampling the variance) */
    static double samps_chi_sq_df_eq_nu_plus_n[]; // Default values will be set in cpp file
    /** The number of samples in the cached library of chi-squared values */
    static int samps_chi_sq_df_eq_nu_plus_n_length;
    /** A cached library of standard normal values (used for Gibbs sampling the posterior means of the terminal nodes) */
    static double samps_std_normal[]; // Default values will be set in cpp file
    /** The number of samples in the cached library of standard normal values */
    static int samps_std_normal_length;
    
protected:
    /** the center of the prior of the terminal node prediction distribution */
    double hyper_mu_mu;
    /** the variance of the prior of the terminal node prediction distribution */
    double hyper_sigsq_mu;
    /** half the shape parameter and half the multiplicand of the scale parameter of the inverse gamma prior on the variance */
    double hyper_nu = 3.0;
    /** the multiplier of the scale parameter of the inverse gamma prior on the variance */
    double hyper_lambda;
    /** this controls where to set <code>hyper_sigsq_mu</code> by forcing the variance to be this number of standard deviations on the normal CDF */
    double hyper_k = 2.0;
    /** At a fixed <code>hyper_nu</code>, this controls where to set <code>hyper_lambda</code> by forcing q proportion to be at that value in the inverse gamma CDF */
    double hyper_q = 0.9;
    
    /** A hyperparameter that controls how easy it is to grow new nodes in a tree independent of depth */
    double alpha = 0.95;
    /** A hyperparameter that controls how easy it is to grow new nodes in a tree dependent on depth which makes it more difficult as the tree gets deeper */
    double beta = 2;
    /** the minimum of the response variable on its original scale */
    double y_min;
    /** the maximum of the response variable on its original scale */
    double y_max;
    /** the minimum of the response variable on its original scale */
    double y_range_sq;
    /** the sample variance of the response variable on its original scale */
    double sample_var_y = 0.0;
    /** if a covariate is a key here, the value defines interaction between the variables that are legal */
    std::unordered_map<int, std::unordered_set<int>> interaction_constraints;
    
public:
    // Virtual destructor
    virtual ~bartmachine_b_hyperparams() = default;
    
    // Implementation of virtual methods from bartmachine_a_base
    void setData(std::vector<double*>& X_y) override;
    void transformResponseVariable() override;
    
    // Methods from Java implementation
    void calculateHyperparameters();
    double transform_y(double y_i);
    double* un_transform_y(double* yt, int length);
    virtual double un_transform_y(double yt_i);
    double un_transform_sigsq(double sigsq_t_i);
    double* un_transform_sigsq(double* sigsq_t_is, int length);
    double un_transform_y_and_round(double yt_i);
    double* un_transform_y_and_round(double* yt, int length);
    
    // Setters
    void setInteractionConstraints(std::unordered_map<int, std::unordered_set<int>>& interaction_constraints);
    void setK(double hyper_k);
    void setQ(double hyper_q);
    void setNu(double hyper_nu);
    void setAlpha(double alpha);
    void setBeta(double beta);
    void setXYByCol(const std::vector<double*>& X_y_by_col);
    
    // Getters
    double getHyper_mu_mu() const;
    double getHyper_sigsq_mu() const;
    double getHyper_nu() const;
    double getHyper_lambda() const;
    double getY_min() const;
    double getY_max() const;
    double getY_range_sq() const;
    
    // Accessor methods for interaction constraints
    bool hasInteractionConstraints(int key) const {
        return interaction_constraints.count(key) > 0;
    }
    
    const std::unordered_set<int>& getInteractionConstraints(int key) const {
        return interaction_constraints.at(key);
    }
    
    // Additional accessor methods for protected members (needed for bartMachineTreeNode)
    int getP() const { return p; }
    const std::vector<double*>& getXY() const { return X_y; }
    const std::vector<double*>& getXYByCol() const { return X_y_by_col; }
    double* getYTrans() const { return y_trans; }
    bool getMemCacheForSpeed() const { return mem_cache_for_speed; }
    
    // Friend class declaration to allow bartMachineTreeNode to access protected members
    friend class bartMachineTreeNode;
};

#endif // BARTMACHINE_B_HYPERPARAMS_H
