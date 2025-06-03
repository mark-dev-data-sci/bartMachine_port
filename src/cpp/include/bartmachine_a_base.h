#ifndef BARTMACHINE_A_BASE_H
#define BARTMACHINE_A_BASE_H

#include "classifier.h"
#include <vector>

// Forward declaration to avoid circular dependency
class bartMachineTreeNode;

/**
 * Exact port of bartMachine_a_base from Java to C++
 * 
 * The base class for any BART implementation. Contains
 * mostly instance variables and settings for the algorithm
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_a_base.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_a_base : public Classifier {
protected:
    /** the number of observations */
    int n;
    /** the number of predictors in the training set */
    int p;

    /** all Gibbs samples for burn-in and post burn-in where each entry is a vector of pointers to the <code>num_trees</code> trees in the sum-of-trees model */
    bartMachineTreeNode*** gibbs_samples_of_bart_trees;
    /** Gibbs samples post burn-in where each entry is a vector of pointers to the <code>num_trees</code> trees in the sum-of-trees model */
    bartMachineTreeNode*** gibbs_samples_of_bart_trees_after_burn_in;
    /** Gibbs samples for burn-in and post burn-in of the variances */
    double* gibbs_samples_of_sigsq;
    /** Gibbs samples for post burn-in of the variances */
    double* gibbs_samples_of_sigsq_after_burn_in;
    /** a record of whether each Gibbs sample accepted or rejected the MH step within each of the <code>num_trees</code> trees */
    bool** accept_reject_mh;
    /** a record of the proposal of each Gibbs sample within each of the <code>m</code> trees: G, P or C for "grow", "prune", "change" */
    char** accept_reject_mh_steps;

    /** the number of trees in our sum-of-trees model */
    int num_trees;
    /** how many Gibbs samples we burn-in for */
    int num_gibbs_burn_in;
    /** how many total Gibbs samples in a BART model creation */
    int num_gibbs_total_iterations;

    /** the current thread being used to run this Gibbs sampler */
    int threadNum;
    /** how many CPU cores to use during model creation */
    int num_cores;
    /** 
     * whether or not we use the memory cache feature
     * 
     * @see Section 3.1 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
     */
    bool mem_cache_for_speed;
    /** saves indices in nodes (useful for computing weights) */
    bool flush_indices_to_save_ram;
    /** should we print stuff out to screen? */
    bool verbose = true;

protected:
    /** Remove unnecessary data from the Gibbs chain to conserve RAM */
    void FlushData();
    
    /** Remove unnecessary data from an individual Gibbs sample */
    void FlushDataForSample(bartMachineTreeNode** bart_trees);

public:
    /** Must be implemented, but does nothing */
    void StopBuilding();

    void setThreadNum(int threadNum);
    
    void setVerbose(bool verbose);
    
    void setTotalNumThreads(int num_cores);

    void setMemCacheForSpeed(bool mem_cache_for_speed);
    
    void setFlushIndicesToSaveRAM(bool flush_indices_to_save_ram);

    void setNumTrees(int m);
    
    // Accessor methods for protected members (needed for bartMachineTreeNode)
    bool getMemCacheForSpeed() const { return mem_cache_for_speed; }
    bool getFlushIndicesToSaveRAM() const { return flush_indices_to_save_ram; }
    
    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~bartmachine_a_base() = default;
    
    // Virtual method for setting data
    virtual void setData(std::vector<double*>& X_y);
    
    // Virtual method for transforming response variable - to be implemented by derived classes
    virtual void transformResponseVariable() = 0;
    
    // Friend class declaration to allow bartMachineTreeNode to access protected members
    friend class bartMachineTreeNode;
};

#endif // BARTMACHINE_A_BASE_H
