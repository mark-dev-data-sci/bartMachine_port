#include "include/bartmachine_f_gibbs_internal.h"
#include "include/stat_toolbox.h"
#include <cmath>

double bartmachine_f_gibbs_internal::drawSigsqFromPosterior(int sample_num, double* es) {
    // Calculate the sum of squared residuals
    double sse = 0;
    for (int i = 0; i < n; i++) {
        sse += es[i] * es[i];
    }
    
    // We're sampling from sigsq ~ InvGamma((nu + n) / 2, 1/2 * (sum_i error^2_i + lambda * nu))
    // which is equivalent to sampling (1 / sigsq) ~ Gamma((nu + n) / 2, 2 / (sum_i error^2_i + lambda * nu))
    return StatToolbox::sample_from_inv_gamma((hyper_nu + n) / 2.0, 2.0 / (sse + hyper_nu * hyper_lambda));
}

double bartmachine_f_gibbs_internal::calcLeafPosteriorVar(bartMachineTreeNode* node, double current_sigsq) {
    return 1.0 / (1.0 / hyper_sigsq_mu + node->n_eta / current_sigsq);
}

double bartmachine_f_gibbs_internal::calcLeafPosteriorMean(bartMachineTreeNode* node, double current_sigsq, double leaf_var) {
    // Use avgResponse() method to get the average response
    node->y_avg = node->avgResponse();
    return leaf_var * (hyper_mu_mu / hyper_sigsq_mu + node->n_eta / current_sigsq * node->y_avg);
}

void bartmachine_f_gibbs_internal::assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats(bartMachineTreeNode* node, double current_sigsq) {
    if (node->isLeaf) {
        // Update posterior_var
        node->posterior_var = calcLeafPosteriorVar(node, current_sigsq);
        
        // Draw from posterior distribution
        node->posterior_mean = calcLeafPosteriorMean(node, current_sigsq, node->posterior_var);
        node->y_pred = StatToolbox::sample_from_norm_dist(node->posterior_mean, node->posterior_var);
        
        if (node->y_pred == StatToolbox::ILLEGAL_FLAG) {
            node->y_pred = 0.0; // This could happen on an empty node
            std::cerr << "ERROR assignLeafFINAL " << node->y_pred << " (sigsq = " << current_sigsq << ")" << std::endl;
        }
        
        // Now update yhats
        node->updateYHatsWithPrediction();
    } else {
        assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats(node->left, current_sigsq);
        assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats(node->right, current_sigsq);
    }
}

int bartmachine_f_gibbs_internal::pickRandomPredictorThatCanBeAssigned(bartMachineTreeNode* node) {
    std::vector<int> predictors = node->getPredictorsThatCouldBeUsedToSplitAtNode();
    return predictors[static_cast<int>(std::floor(StatToolbox::rand() * pAdj(node)))];
}

double bartmachine_f_gibbs_internal::pAdj(bartMachineTreeNode* node) {
    if (node->getPadj() == 0) {
        node->setPadj(node->getPredictorsThatCouldBeUsedToSplitAtNode().size());
    }
    return node->getPadj();
}

bartMachineTreeNode* bartmachine_f_gibbs_internal::metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode* copy_of_old_jth_tree, int t, bool** accept_reject_mh, char** accept_reject_mh_steps) {
    // This is a placeholder implementation
    // In the actual implementation, this would perform a Metropolis-Hastings step
    // to sample from the posterior distribution of trees
    
    // For now, just return the old tree
    return copy_of_old_jth_tree;
}
