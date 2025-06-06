#include "include/bartmachine_g_mh.h"
#include "include/stat_toolbox.h"
#include <cmath>
#include <iostream>

bartMachineTreeNode* bartmachine_g_mh::metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode* T_i, int tree_num, bool** accept_reject_mh, char** accept_reject_mh_steps) {
    bartMachineTreeNode* T_star = T_i->clone();
    // Each proposal will calculate its own value, but this has to be initialized atop
    double log_r = 0;
    
    // If it's a stump force a GROW change, otherwise pick randomly from the steps according to the "hidden parameters"
    if (T_i->isStump()) {
        accept_reject_mh_steps[gibbs_sample_num][tree_num] = 'G';
        log_r = doMHGrowAndCalcLnR(T_i, T_star);
    } else {
        switch (randomlyPickAmongTheProposalSteps()) {
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

// Placeholder implementations for other methods
// These will be implemented in future tasks

double bartmachine_g_mh::doMHPruneAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star) {
    // This is a placeholder implementation
    // Will be implemented in Task 5.4
    return std::numeric_limits<double>::lowest();
}

double bartmachine_g_mh::calcLnTransRatioPrune(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star, bartMachineTreeNode* prune_node) {
    // This is a placeholder implementation
    // Will be implemented in Task 5.4
    return 0.0;
}

bartMachineTreeNode* bartmachine_g_mh::pickPruneNodeOrChangeNode(bartMachineTreeNode* T) {
    // This is a placeholder implementation
    // Will be implemented in Task 5.4
    return nullptr;
}

double bartmachine_g_mh::doMHChangeAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star) {
    // This is a placeholder implementation
    // Will be implemented in Task 5.5
    return std::numeric_limits<double>::lowest();
}

double bartmachine_g_mh::calcLnLikRatioChange(bartMachineTreeNode* eta, bartMachineTreeNode* eta_star) {
    // This is a placeholder implementation
    // Will be implemented in Task 5.5
    return 0.0;
}
