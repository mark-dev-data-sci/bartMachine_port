#include "include/bartmachine_g_mh.h"
#include "include/stat_toolbox.h"
#include <cmath>
#include <iostream>

bartMachineTreeNode* bartmachine_g_mh::metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode* T_i, int tree_num, bool** accept_reject_mh, char** accept_reject_mh_steps) {
    bartMachineTreeNode* T_star = T_i->clone();
    // Each proposal will calculate its own value, but this has to be initialized atop
    double log_r = 0;

    // If it's a stump force a GROW change, otherwise pick randomly from the steps according to the "hidden parameters"
    switch (T_i->isStump() ? Steps::GROW : randomlyPickAmongTheProposalSteps()) {
        case Steps::GROW:
            accept_reject_mh_steps[gibbs_sample_num][tree_num] = 'G';
            log_r = doMHGrowAndCalcLnR(T_i, T_star);
            break;
        case Steps::PRUNE:
            accept_reject_mh_steps[gibbs_sample_num][tree_num] = 'P';
            log_r = doMHPruneAndCalcLnR(T_i, T_star);
            break;
        case Steps::CHANGE:
            accept_reject_mh_steps[gibbs_sample_num][tree_num] = 'C';
            log_r = doMHChangeAndCalcLnR(T_i, T_star);
            break;
    }
    
    // Draw from a Uniform 0, 1 and log it
    double ln_u_0_1 = std::log(StatToolbox::rand());
    if (DEBUG_MH) {
        std::cout << "ln u = " << ln_u_0_1 <<
                " <? ln(r) = " <<
                (log_r < -99999 ? "very small" : std::to_string(log_r)) << std::endl;
    }

    if (ln_u_0_1 < log_r) { // Accept proposal if the draw is less than the ratio
        if (DEBUG_MH) {
            std::cout << "proposal ACCEPTED\n\n" << std::endl;
        }
        // Mark it was accepted
        accept_reject_mh[gibbs_sample_num][tree_num] = true;
        return T_star;
    }

    // Reject proposal
    if (DEBUG_MH) {
        std::cout << "proposal REJECTED\n\n" << std::endl;
    }
    accept_reject_mh[gibbs_sample_num][tree_num] = false;
    return T_i;
}

double bartmachine_g_mh::doMHGrowAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star) {
    // First select a node that can be grown
    bartMachineTreeNode* grow_node = pickGrowNode(T_star);
    // If we couldn't find a node that be grown, then we can't grow, so reject offhand
    if (grow_node == nullptr) {
        return std::numeric_limits<double>::lowest();
    }
    
    // Now start the growth process
    // First pick the attribute and then the split
    grow_node->splitAttributeM = pickRandomPredictorThatCanBeAssigned(grow_node);
    grow_node->splitValue = grow_node->pickRandomSplitValue();
    // Now pick randomly which way the missing data goes - left (false) or right (true)
    grow_node->sendMissingDataRight = bartMachineTreeNode::pickRandomDirectionForMissingData();
    // Inform the user if things go awry
    if (grow_node->splitValue == bartMachineTreeNode::BAD_FLAG_double) {
        return std::numeric_limits<double>::lowest();
    }
    
    grow_node->isLeaf = false;
    grow_node->left = new bartMachineTreeNode(grow_node);
    grow_node->right = new bartMachineTreeNode(grow_node);
    grow_node->propagateDataByChangedRule();
    
    if (grow_node->left->n_eta <= 0 || grow_node->right->n_eta <= 0) {
        if (DEBUG_MH) {
            std::cerr << "ERROR GROW <<" << grow_node->stringLocation(true) << ">> cannot split a node where daughter only has NO data points   proposal ln(r) = -oo DUE TO CANNOT GROW" << std::endl;
        }
        return std::numeric_limits<double>::lowest();
    }
    
    double ln_transition_ratio_grow = calcLnTransRatioGrow(T_i, T_star, grow_node);
    double ln_likelihood_ratio_grow = calcLnLikRatioGrow(grow_node);
    double ln_tree_structure_ratio_grow = calcLnTreeStructureRatioGrow(grow_node);
    
    if (DEBUG_MH) {
        std::cout << gibbs_sample_num << " GROW  <<" << grow_node->stringLocation(true) << ">> ---- X_" << (grow_node->splitAttributeM) << 
            " < " << grow_node->splitValue << " & " << (grow_node->sendMissingDataRight ? "M -> R" : "M -> L") << 
            "\n  ln trans ratio: " << ln_transition_ratio_grow << " ln lik ratio: " << ln_likelihood_ratio_grow << " ln structure ratio: " << ln_tree_structure_ratio_grow <<
            "\n  trans ratio: " << 
            (std::exp(ln_transition_ratio_grow) < 0.00001 ? "very small" : std::to_string(std::exp(ln_transition_ratio_grow))) <<
            "  lik ratio: " << 
            (std::exp(ln_likelihood_ratio_grow) < 0.00001 ? "very small" : std::to_string(std::exp(ln_likelihood_ratio_grow))) <<
            "  structure ratio: " << 
            (std::exp(ln_tree_structure_ratio_grow) < 0.00001 ? "very small" : std::to_string(std::exp(ln_tree_structure_ratio_grow))) << std::endl;
    }
    
    return ln_transition_ratio_grow + ln_likelihood_ratio_grow + ln_tree_structure_ratio_grow;
}

double bartmachine_g_mh::calcLnTransRatioGrow(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star, bartMachineTreeNode* node_grown_in_Tstar) {
    int b = T_i->numLeaves();
    double p_adj = pAdj(node_grown_in_Tstar);
    int n_adj = node_grown_in_Tstar->getNAdj();
    int w_2_star = T_star->numPruneNodesAvailable();
    return std::log(b) + std::log(p_adj) + std::log(n_adj) - std::log(w_2_star);
}

double bartmachine_g_mh::calcLnLikRatioGrow(bartMachineTreeNode* grow_node) {
    double sigsq = gibbs_samples_of_sigsq[gibbs_sample_num - 1];
    int n_ell = grow_node->n_eta;
    int n_ell_L = grow_node->left->n_eta;
    int n_ell_R = grow_node->right->n_eta;
    
    // Now go ahead and calculate it out in an organized fashion:
    double sigsq_plus_n_ell_hyper_sisgsq_mu = sigsq + n_ell * hyper_sigsq_mu;
    double sigsq_plus_n_ell_L_hyper_sisgsq_mu = sigsq + n_ell_L * hyper_sigsq_mu;
    double sigsq_plus_n_ell_R_hyper_sisgsq_mu = sigsq + n_ell_R * hyper_sigsq_mu;
    double c = 0.5 * (
            std::log(sigsq) 
            + std::log(sigsq_plus_n_ell_hyper_sisgsq_mu) 
            - std::log(sigsq_plus_n_ell_L_hyper_sisgsq_mu) 
            - std::log(sigsq_plus_n_ell_R_hyper_sisgsq_mu));
    double d = hyper_sigsq_mu / (2 * sigsq);
    double e = grow_node->left->sumResponsesQuantitySqd() / sigsq_plus_n_ell_L_hyper_sisgsq_mu
            + grow_node->right->sumResponsesQuantitySqd() / sigsq_plus_n_ell_R_hyper_sisgsq_mu
            - grow_node->sumResponsesQuantitySqd() / sigsq_plus_n_ell_hyper_sisgsq_mu;
    return c + d * e;
}

double bartmachine_g_mh::calcLnTreeStructureRatioGrow(bartMachineTreeNode* grow_node) {
    int d_eta = grow_node->depth;
    double p_adj = pAdj(grow_node);
    int n_adj = grow_node->getNAdj();
    return std::log(alpha) 
            + 2 * std::log(1 - alpha / std::pow(2 + d_eta, beta))
            - std::log(std::pow(1 + d_eta, beta) - alpha)
            - std::log(p_adj) 
            - std::log(n_adj);
}

bartMachineTreeNode* bartmachine_g_mh::pickGrowNode(bartMachineTreeNode* T) {
    std::vector<bartMachineTreeNode*> growth_nodes = T->getTerminalNodesWithDataAboveOrEqualToN(2);
    
    // 2 checks
    // a) If there is no nodes to grow, return null
    // b) If the node we picked CANNOT grow due to no available predictors, return null as well
    
    // Do check a
    if (growth_nodes.size() == 0) {
        return nullptr;
    }
    
    // Now we pick one of these nodes with enough data points randomly
    bartMachineTreeNode* growth_node = growth_nodes[static_cast<int>(std::floor(StatToolbox::rand() * growth_nodes.size()))];
    
    // Do check b
    if (pAdj(growth_node) == 0) {
        return nullptr;
    }
    // If we passed, we can use this node
    return growth_node;
}

bartmachine_g_mh::Steps bartmachine_g_mh::randomlyPickAmongTheProposalSteps() {
    double roll = StatToolbox::rand();
    if (roll < prob_grow) {
        return Steps::GROW;
    }
    if (roll < prob_grow + prob_prune) {
        return Steps::PRUNE;
    }
    return Steps::CHANGE;
}

double bartmachine_g_mh::doMHPruneAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star) {
    // First select a node that can be pruned
    bartMachineTreeNode* prune_node = pickPruneNodeOrChangeNode(T_star);
    // If we didn't find one to prune, then we can't prune, so reject offhand
    if (prune_node == nullptr) {
        return std::numeric_limits<double>::lowest();
    }
    
    double ln_transition_ratio_prune = calcLnTransRatioPrune(T_i, T_star, prune_node);
    double ln_likelihood_ratio_prune = -calcLnLikRatioGrow(prune_node); // Inverse of before (will speed up later)
    double ln_tree_structure_ratio_prune = -calcLnTreeStructureRatioGrow(prune_node);
    
    if (DEBUG_MH) {
        std::cout << gibbs_sample_num << " PRUNE <<" << prune_node->stringLocation(true) << 
                ">> ---- X_" << (prune_node->splitAttributeM == bartMachineTreeNode::BAD_FLAG_int ? "null" : std::to_string(prune_node->splitAttributeM)) << 
                " < " << (prune_node->splitValue == bartMachineTreeNode::BAD_FLAG_double ? "NaN" : std::to_string(prune_node->splitValue)) << 
                "\n  ln trans ratio: " << ln_transition_ratio_prune << " ln lik ratio: " << ln_likelihood_ratio_prune << " ln structure ratio: " << ln_tree_structure_ratio_prune <<
                "\n  trans ratio: " << 
                (std::exp(ln_transition_ratio_prune) < 0.00001 ? "very small" : std::to_string(std::exp(ln_transition_ratio_prune))) <<
                "  lik ratio: " << 
                (std::exp(ln_likelihood_ratio_prune) < 0.00001 ? "very small" : std::to_string(std::exp(ln_likelihood_ratio_prune))) <<
                "  structure ratio: " << 
                (std::exp(ln_tree_structure_ratio_prune) < 0.00001 ? "very small" : std::to_string(std::exp(ln_tree_structure_ratio_prune))) << std::endl;
    }
    
    bartMachineTreeNode::pruneTreeAt(prune_node);
    return ln_transition_ratio_prune + ln_likelihood_ratio_prune + ln_tree_structure_ratio_prune;
}

double bartmachine_g_mh::calcLnTransRatioPrune(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star, bartMachineTreeNode* prune_node) {
    int w_2 = T_i->numPruneNodesAvailable();
    int b = T_i->numLeaves();
    double p_adj = pAdj(prune_node);
    int n_adj = prune_node->getNAdj();
    return std::log(w_2) - std::log(b - 1) - std::log(p_adj) - std::log(n_adj);
}

bartMachineTreeNode* bartmachine_g_mh::pickPruneNodeOrChangeNode(bartMachineTreeNode* T) {
    // Two checks need to be performed first before we run a search on the tree structure
    // a) If this is the root, we can't prune so return null
    // b) If there are no prunable nodes (not sure how that could happen), return null as well
    
    if (T->isStump()) {
        return nullptr;
    }
    
    std::vector<bartMachineTreeNode*> prunable_and_changeable_nodes = T->getPrunableAndChangeableNodes();
    if (prunable_and_changeable_nodes.size() == 0) {
        return nullptr;
    }
    
    // Now we pick one of these nodes randomly
    return prunable_and_changeable_nodes[static_cast<int>(std::floor(StatToolbox::rand() * prunable_and_changeable_nodes.size()))];
}

/**
 * Perform the change step on a tree and return the log Metropolis-Hastings ratio
 * 
 * @param T_i       The root node of the original tree 
 * @param T_star    The root node of a copy of the original tree where one node will be changed
 * @return          The log Metropolis-Hastings ratio
 * @see             Section A.3 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
 */
double bartmachine_g_mh::doMHChangeAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star) {
    // Select a node suitable for changing
    bartMachineTreeNode* eta_star = pickPruneNodeOrChangeNode(T_star);
    // If we didn't find one to change, then we can't change, so reject offhand
    if (eta_star == nullptr) {
        return std::numeric_limits<double>::lowest();
    }
    
    // Create a clone of the node for calculation purposes
    bartMachineTreeNode* eta_just_for_calculation = eta_star->clone();
    
    // Now start the change process
    // First pick the attribute and then the split and then which way to send the missing data
    eta_star->splitAttributeM = pickRandomPredictorThatCanBeAssigned(eta_star);
    eta_star->splitValue = eta_star->pickRandomSplitValue();
    eta_star->sendMissingDataRight = bartMachineTreeNode::pickRandomDirectionForMissingData();
    // Inform the user if things go awry
    if (eta_star->splitValue == bartMachineTreeNode::BAD_FLAG_double) {
        return std::numeric_limits<double>::lowest();
    }
    
    eta_star->propagateDataByChangedRule();
    // The children no longer have the right data!
    // In the Java implementation, clearRulesAndSplitCache is called directly
    // In C++, it's protected, so we need to use a different approach
    // We'll set the possible rule variables to an empty vector to achieve the same effect
    eta_star->left->setPadj(0);
    eta_star->right->setPadj(0);
    
    double ln_tree_structure_ratio_change = calcLnLikRatioChange(eta_just_for_calculation, eta_star);
    if (DEBUG_MH) {
        std::cout << gibbs_sample_num << " CHANGE  <<" << eta_star->stringLocation(true) << ">> ---- X_" << (eta_star->splitAttributeM) << 
            " < " << eta_star->splitValue << " & " << (eta_star->sendMissingDataRight ? "M -> R" : "M -> L") << " from " << 
            "X_" << (eta_just_for_calculation->splitAttributeM) << 
            " < " << eta_just_for_calculation->splitValue << " & " << (eta_just_for_calculation->sendMissingDataRight ? "M -> R" : "M -> L") << 
            "\n  ln lik ratio: " << ln_tree_structure_ratio_change << 
            "  lik ratio: " << 
            (std::exp(ln_tree_structure_ratio_change) < 0.00001 ? "very small" : std::to_string(std::exp(ln_tree_structure_ratio_change))) << std::endl;
    }
    
    // Clean up the clone
    delete eta_just_for_calculation;
    
    return ln_tree_structure_ratio_change; // The transition ratio cancels out the tree structure ratio
}

/**
 * Calculates the log likelihood ratio for a change step
 * 
 * @param eta       The node in the original tree that was targeted for a change in the splitting rule
 * @param eta_star  The same node but now with a different splitting rule
 * @return          The log likelihood ratio
 * @see             Section A.3.2 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
 */
double bartmachine_g_mh::calcLnLikRatioChange(bartMachineTreeNode* eta, bartMachineTreeNode* eta_star) {
    int n_1_star = eta_star->left->n_eta;
    int n_2_star = eta_star->right->n_eta;
    int n_1 = eta->left->n_eta;
    int n_2 = eta->right->n_eta;
    
    double sigsq = gibbs_samples_of_sigsq[gibbs_sample_num - 1];
    double ratio_sigsqs = sigsq / hyper_sigsq_mu;
    double n_1_plus_ratio_sigsqs = n_1 + ratio_sigsqs;
    double n_2_plus_ratio_sigsqs = n_2 + ratio_sigsqs;
    
    // NOTE: this can be sped up by just taking the diffs
    double sum_sq_1_star = eta_star->left->sumResponsesQuantitySqd();
    double sum_sq_2_star = eta_star->right->sumResponsesQuantitySqd();
    double sum_sq_1 = eta->left->sumResponsesQuantitySqd();
    double sum_sq_2 = eta->right->sumResponsesQuantitySqd();
    
    // Couple checks
    if (n_1_star == 0 || n_2_star == 0) {
        if (DEBUG_MH) {
            eta->printNodeDebugInfo("PARENT BEFORE");
            eta_star->printNodeDebugInfo("PARENT AFTER");
            eta->left->printNodeDebugInfo("LEFT BEFORE");
            eta->right->printNodeDebugInfo("RIGHT BEFORE");
            eta_star->left->printNodeDebugInfo("LEFT AFTER");
            eta_star->right->printNodeDebugInfo("RIGHT AFTER");
        }
        return std::numeric_limits<double>::lowest();
    }
    
    // Do simplified calculation if the n's remain the same
    if (n_1_star == n_1) {
        return 1 / (2 * sigsq) * (
                (sum_sq_1_star - sum_sq_1) / n_1_plus_ratio_sigsqs + 
                (sum_sq_2_star - sum_sq_2) / n_2_plus_ratio_sigsqs
            );
    }
    // Otherwise do the lengthy calculation
    else {
        double n_1_star_plus_ratio_sigsqs = n_1_star + ratio_sigsqs;
        double n_2_star_plus_ratio_sigsqs = n_2_star + ratio_sigsqs;
        
        double a = std::log(n_1_plus_ratio_sigsqs) + 
                   std::log(n_2_plus_ratio_sigsqs) - 
                   std::log(n_1_star_plus_ratio_sigsqs) - 
                   std::log(n_2_star_plus_ratio_sigsqs);
        double b = (
                sum_sq_1_star / n_1_star_plus_ratio_sigsqs + 
                sum_sq_2_star / n_2_star_plus_ratio_sigsqs -
                sum_sq_1 / n_1_plus_ratio_sigsqs - 
                sum_sq_2 / n_2_plus_ratio_sigsqs                 
            );
        
        return 0.5 * a + 1 / (2 * sigsq) * b;
    }
}
