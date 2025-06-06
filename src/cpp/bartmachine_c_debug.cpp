#include "include/bartmachine_c_debug.h"
#include <iostream>

// Forward declaration of TreeArrayIllustration class
class TreeArrayIllustration;

void bartmachine_c_debug::InitTreeIllustrations() {
    bartMachineTreeNode** initial_trees = gibbs_samples_of_bart_trees[0];
    TreeArrayIllustration* tree_array_illustration = new TreeArrayIllustration(0, unique_name);
    
    for (int t = 0; t < num_trees; t++) {
        tree_array_illustration->AddTree(initial_trees[t]);
        tree_array_illustration->addLikelihood(0);
    }
    tree_array_illustration->CreateIllustrationAndSaveImage();
    
    delete tree_array_illustration;
}

void bartmachine_c_debug::illustrate(TreeArrayIllustration* tree_array_illustration) {
    if (tree_illust) {
        tree_array_illustration->CreateIllustrationAndSaveImage();
    }
}

double* bartmachine_c_debug::getGibbsSamplesSigsqs() {
    double* sigsqs_to_export = new double[num_gibbs_total_iterations + 1];
    for (int n_g = 0; n_g <= num_gibbs_total_iterations; n_g++) {
        sigsqs_to_export[n_g] = un_transform_sigsq(gibbs_samples_of_sigsq[n_g]);
    }
    return sigsqs_to_export;
}

int** bartmachine_c_debug::getDepthsForTrees(int n_g_i, int n_g_f) {
    int** all_depths = new int*[n_g_f - n_g_i];
    for (int g = n_g_i; g < n_g_f; g++) {
        all_depths[g - n_g_i] = new int[num_trees];
        for (int t = 0; t < num_trees; t++) {
            all_depths[g - n_g_i][t] = gibbs_samples_of_bart_trees[g][t]->deepestNode();
        }
    }
    return all_depths;
}

int** bartmachine_c_debug::getNumNodesAndLeavesForTrees(int n_g_i, int n_g_f) {
    int** all_new_nodes = new int*[n_g_f - n_g_i];
    for (int g = n_g_i; g < n_g_f; g++) {
        all_new_nodes[g - n_g_i] = new int[num_trees];
        for (int t = 0; t < num_trees; t++) {
            all_new_nodes[g - n_g_i][t] = gibbs_samples_of_bart_trees[g][t]->numNodesAndLeaves();
        }
    }
    return all_new_nodes;
}
