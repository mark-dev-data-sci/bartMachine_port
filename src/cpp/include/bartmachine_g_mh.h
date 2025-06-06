#ifndef BARTMACHINE_G_MH_H
#define BARTMACHINE_G_MH_H

#include "bartmachine_f_gibbs_internal.h"
#include <limits>

/**
 * Exact port of bartMachine_g_mh from Java to C++
 * 
 * This portion of the code performs the Metropolis-Hastings tree search step
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_g_mh.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_g_mh : public bartmachine_f_gibbs_internal {
protected:
    /** turning this flag on prints out debugging information about the Metropolis-Hastings tree search step */
    static const bool DEBUG_MH = false;
    
    /** the hyperparameter of the probability of picking a grow step during the Metropolis-Hastings tree proposal */
    double prob_grow;
    /** the hyperparameter of the probability of picking a prune step during the Metropolis-Hastings tree proposal */
    double prob_prune;

    /** the types of steps in the Metropolis-Hastings tree search */
    enum class Steps {GROW, PRUNE, CHANGE};
    
    /**
     * Performs a Metropolis-Hastings step to sample from the posterior distribution of trees
     * 
     * @param T_i               The original tree to be changed by a proposal step
     * @param tree_num          The tree index
     * @param accept_reject_mh  The accept/reject matrix
     * @param accept_reject_mh_steps The accept/reject steps matrix
     * @return                  The next tree (T_{i+1}) via one iteration of M-H which can be the proposal tree (if the step was accepted) or the original tree (if the step was rejected)
     */
    bartMachineTreeNode* metroHastingsPosteriorTreeSpaceIteration(bartMachineTreeNode* T_i, int tree_num, bool** accept_reject_mh, char** accept_reject_mh_steps) override;
    
    /**
     * Perform the grow step on a tree and return the log Metropolis-Hastings ratio
     * 
     * @param T_i       The root node of the original tree 
     * @param T_star    The root node of a copy of the original tree where one node will be grown
     * @return          The log Metropolis-Hastings ratio
     */
    double doMHGrowAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star);
    
    /**
     * Calculates the log transition ratio for a grow step
     * 
     * @param T_i                   The root node of the original tree
     * @param T_star                The root node of the proposal tree
     * @param node_grown_in_Tstar   The node that was grown in the proposal tree
     * @return                      The log of the transition ratio
     */
    double calcLnTransRatioGrow(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star, bartMachineTreeNode* node_grown_in_Tstar);
    
    /**
     * Calculates the log of the likelihood ratio for a grow step
     * 
     * @param grow_node     The node that was grown in the proposal tree
     * @return              The log of the likelihood ratio
     */
    double calcLnLikRatioGrow(bartMachineTreeNode* grow_node);
    
    /**
     * Calculates the log transition ratio for a grow step
     * 
     * @param grow_node     The node that was grown in the proposal tree
     * @return              The log of the transition ratio
     */
    double calcLnTreeStructureRatioGrow(bartMachineTreeNode* grow_node);
    
    /**
     * Selects a node in a tree that is eligible for being grown with two children
     * 
     * @param T     The root node of the tree to be searched
     * @return      The node that is viable for growing
     */
    bartMachineTreeNode* pickGrowNode(bartMachineTreeNode* T);
    
    /**
     * Perform the prune step on a tree and return the log Metropolis-Hastings ratio
     * 
     * @param T_i       The root node of the original tree 
     * @param T_star    The root node of a copy of the original tree where one set of terminal nodes will be pruned
     * @return          The log Metropolis-Hastings ratio
     */
    double doMHPruneAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star);
    
    /**
     * This calculates the log transition ratio for the prune step.
     * 
     * @param T_i           The root node of the original tree
     * @param T_star        The root node of the proposal tree
     * @param prune_node    The node that was pruned
     * @return              The log transition ratio
     */
    double calcLnTransRatioPrune(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star, bartMachineTreeNode* prune_node);
    
    /**
     * This function picks a node suitable for pruning or changing. In our implementation this is a 
     * node that is "singly internal" (ie it has two children and its children are both terminal nodes)
     * 
     * @param T                 The root of the tree we wish to find singly internal nodes    
     * @return                  A singly internal node selected at random from all candididates or null if none exist
     */
    bartMachineTreeNode* pickPruneNodeOrChangeNode(bartMachineTreeNode* T);
    
    /**
     * Perform the change step on a tree and return the log Metropolis-Hastings ratio
     * 
     * @param T_i       The root node of the original tree 
     * @param T_star    The root node of a copy of the original tree where one node will be changed
     * @return          The log Metropolis-Hastings ratio
     */
    double doMHChangeAndCalcLnR(bartMachineTreeNode* T_i, bartMachineTreeNode* T_star);
    
    /**
     * Calculates the log likelihood ratio for a change step
     * 
     * @param eta       The node in the original tree that was targeted for a change in the splitting rule
     * @param eta_star  The same node but now with a different splitting rule
     * @return          The log likelihood ratio
     */
    double calcLnLikRatioChange(bartMachineTreeNode* eta, bartMachineTreeNode* eta_star);
    
    /**
     * Randomly chooses among the valid tree proposal steps from a multinomial distribution
     * 
     * @return  The step that was chosen
     */
    Steps randomlyPickAmongTheProposalSteps();
    
public:
    // Setter methods for MH probabilities
    void setProbGrow(double prob_grow) { this->prob_grow = prob_grow; }
    void setProbPrune(double prob_prune) { this->prob_prune = prob_prune; }
    
    // Virtual destructor
    virtual ~bartmachine_g_mh() = default;
};

#endif // BARTMACHINE_G_MH_H
