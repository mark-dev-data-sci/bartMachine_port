/**
 * bartmachine_i_prior_cov_spec.cpp
 * 
 * Implementation file for the bartmachine_i_prior_cov_spec class, which handles
 * prior covariate specifications for the BART model.
 */

#include "include/bartmachine_i_prior_cov_spec.h"
#include <cmath>
#include <iostream>
#include <algorithm>

/**
 * Constructor
 */
bartmachine_i_prior_cov_spec::bartmachine_i_prior_cov_spec() : bartmachine_h_eval() {
    // Initialize covariate importance variables
    cov_importance_vec = nullptr;
    cov_importance_sd_vec = nullptr;
    
    // Initialize interaction constraints
    interaction_constraints = nullptr;
    num_interaction_constraints = 0;
    
    // Initialize split counts
    split_counts_by_var_and_tree = nullptr;
    total_count_by_var = nullptr;
}

/**
 * Destructor
 */
bartmachine_i_prior_cov_spec::~bartmachine_i_prior_cov_spec() {
    // Free memory for covariate importance variables
    if (cov_importance_vec != nullptr) delete[] cov_importance_vec;
    if (cov_importance_sd_vec != nullptr) delete[] cov_importance_sd_vec;
    
    // Free memory for interaction constraints
    if (interaction_constraints != nullptr) {
        for (int i = 0; i < num_interaction_constraints; i++) {
            if (interaction_constraints[i] != nullptr) delete[] interaction_constraints[i];
        }
        delete[] interaction_constraints;
    }
    
    // Free memory for split counts
    if (split_counts_by_var_and_tree != nullptr) {
        for (int i = 0; i < p; i++) {
            if (split_counts_by_var_and_tree[i] != nullptr) delete[] split_counts_by_var_and_tree[i];
        }
        delete[] split_counts_by_var_and_tree;
    }
    if (total_count_by_var != nullptr) delete[] total_count_by_var;
}

/**
 * Calculate covariate importance
 */
void bartmachine_i_prior_cov_spec::calcCovariateImportance() {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Get covariate importance
 * 
 * @return The covariate importance vector
 */
double* bartmachine_i_prior_cov_spec::getCovariateImportance() {
    return cov_importance_vec;
}

/**
 * Get covariate importance standard deviations
 * 
 * @return The covariate importance standard deviation vector
 */
double* bartmachine_i_prior_cov_spec::getCovariateImportanceSD() {
    return cov_importance_sd_vec;
}

/**
 * Set interaction constraints
 * 
 * @param constraints The interaction constraints
 * @param num_constraints The number of constraints
 */
void bartmachine_i_prior_cov_spec::setInteractionConstraints(bool** constraints, int num_constraints) {
    // Free existing constraints
    if (interaction_constraints != nullptr) {
        for (int i = 0; i < num_interaction_constraints; i++) {
            if (interaction_constraints[i] != nullptr) delete[] interaction_constraints[i];
        }
        delete[] interaction_constraints;
    }
    
    // Set new constraints
    num_interaction_constraints = num_constraints;
    interaction_constraints = constraints;
}

/**
 * Get interaction constraints
 * 
 * @return The interaction constraints
 */
bool** bartmachine_i_prior_cov_spec::getInteractionConstraints() {
    return interaction_constraints;
}

/**
 * Get the number of interaction constraints
 * 
 * @return The number of interaction constraints
 */
int bartmachine_i_prior_cov_spec::getNumInteractionConstraints() {
    return num_interaction_constraints;
}

/**
 * Check if an interaction is allowed
 * 
 * @param var1 The first variable
 * @param var2 The second variable
 * @return True if the interaction is allowed, false otherwise
 */
bool bartmachine_i_prior_cov_spec::isInteractionAllowed(int var1, int var2) {
    // If no constraints, all interactions are allowed
    if (interaction_constraints == nullptr) return true;
    
    // Check if the interaction is allowed
    for (int i = 0; i < num_interaction_constraints; i++) {
        if ((interaction_constraints[i][0] == var1 && interaction_constraints[i][1] == var2) ||
            (interaction_constraints[i][0] == var2 && interaction_constraints[i][1] == var1)) {
            return true;
        }
    }
    
    return false;
}

/**
 * Set variable names
 * 
 * @param names The variable names
 */
void bartmachine_i_prior_cov_spec::setVarNames(std::vector<std::string> names) {
    var_names = names;
}

/**
 * Get variable names
 * 
 * @return The variable names
 */
std::vector<std::string> bartmachine_i_prior_cov_spec::getVarNames() {
    return var_names;
}

/**
 * Get a variable name
 * 
 * @param var_index The variable index
 * @return The variable name
 */
std::string bartmachine_i_prior_cov_spec::getVarName(int var_index) {
    if (var_index >= 0 && var_index < (int)var_names.size()) {
        return var_names[var_index];
    }
    return "Unknown";
}

/**
 * Calculate split counts
 */
void bartmachine_i_prior_cov_spec::calcSplitCounts() {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Get split counts by variable and tree
 * 
 * @return The split counts
 */
int** bartmachine_i_prior_cov_spec::getSplitCountsByVarAndTree() {
    return split_counts_by_var_and_tree;
}

/**
 * Get total count by variable
 * 
 * @return The total counts
 */
int* bartmachine_i_prior_cov_spec::getTotalCountByVar() {
    return total_count_by_var;
}

/**
 * Get split count for a variable
 * 
 * @param var_index The variable index
 * @return The split count
 */
int bartmachine_i_prior_cov_spec::getSplitCountForVar(int var_index) {
    if (total_count_by_var != nullptr && var_index >= 0 && var_index < p) {
        return total_count_by_var[var_index];
    }
    return 0;
}

/**
 * Set covariate selection mode
 * 
 * @param use_selection Whether to use covariate selection
 */
void bartmachine_i_prior_cov_spec::setCovariateSelectionMode(bool use_selection) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Get covariate selection mode
 * 
 * @return Whether covariate selection is enabled
 */
bool bartmachine_i_prior_cov_spec::getCovariateSelectionMode() {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
    return false;
}

/**
 * Set covariate selection threshold
 * 
 * @param threshold The threshold
 */
void bartmachine_i_prior_cov_spec::setCovariateSelectionThreshold(double threshold) {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
}

/**
 * Get covariate selection threshold
 * 
 * @return The threshold
 */
double bartmachine_i_prior_cov_spec::getCovariateSelectionThreshold() {
    // Implement this method based on the Java implementation
    // This is a placeholder implementation
    return 0.0;
}
