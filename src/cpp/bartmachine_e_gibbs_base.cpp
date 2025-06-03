#include "include/bartmachine_e_gibbs_base.h"
#include "include/tree_array_illustration.h"
#include <iostream>
#include <cstdlib>
#include <vector>

// Helper class for array operations (ported from Tools.java)
class Tools {
public:
    static double* subtract_arrays(double* arr1, double* arr2, int length) {
        double* diff = new double[length];
        for (int i = 0; i < length; i++) {
            diff[i] = arr1[i] - arr2[i];
        }
        return diff;
    }
    
    static double* add_arrays(double* arr1, double* arr2, int length) {
        double* sum = new double[length];
        for (int i = 0; i < length; i++) {
            sum[i] = arr1[i] + arr2[i];
        }
        return sum;
    }
};

void bartmachine_e_gibbs_base::Build() {
    SetupGibbsSampling();
    DoGibbsSampling();
}

void bartmachine_e_gibbs_base::DoGibbsSampling() {
    while (gibbs_sample_num <= num_gibbs_total_iterations) {
        DoOneGibbsSample();
        // Now flush the previous previous gibbs sample to not leak memory
        FlushDataForSample(gibbs_samples_of_bart_trees[gibbs_sample_num - 1]);
        // Keep one previous for interaction constraints
        // if (gibbs_sample_num > 1) {
        //     FlushDataForSample(gibbs_samples_of_bart_trees[gibbs_sample_num - 2]);
        // }
        DeleteBurnInsOnPreviousSamples();
        gibbs_sample_num++;
    }
}

void bartmachine_e_gibbs_base::DoOneGibbsSample() {
    // This array is the array of trees for this given sample
    bartMachineTreeNode** bart_trees = new bartMachineTreeNode*[num_trees];
    TreeArrayIllustration* tree_array_illustration = new TreeArrayIllustration(gibbs_sample_num, unique_name);

    // We cycle over each tree and update it according to formulas 15, 16 on p274
    double* R_j = new double[n];
    for (int t = 0; t < num_trees; t++) {
        if (verbose) {
            GibbsSampleDebugMessage(t);
        }
        R_j = SampleTree(gibbs_sample_num, t, bart_trees, tree_array_illustration);
        SampleMusWrapper(gibbs_sample_num, t);
    }
    // Now we have the last residual vector which we pass on to sample sigsq
    double* residuals = getResidualsFromFullSumModel(gibbs_sample_num, R_j);
    SampleSigsq(gibbs_sample_num, residuals);
    if (tree_illust) {
        illustrate(tree_array_illustration);
    }
    
    // Clean up
    delete[] residuals;
    delete tree_array_illustration;
}

void bartmachine_e_gibbs_base::GibbsSampleDebugMessage(int t) {
    if (t == 0 && gibbs_sample_num % 100 == 0) {
        std::string message = "Iteration " + std::to_string(gibbs_sample_num) + "/" + std::to_string(num_gibbs_total_iterations);
        if (num_cores > 1) {
            message += "  thread: " + std::to_string(threadNum + 1);
        }
        // Note: We're not implementing the Windows-specific memory reporting
        std::cout << message << std::endl;
    }
}

void bartmachine_e_gibbs_base::SampleMusWrapper(int sample_num, int t) {
    bartMachineTreeNode* previous_tree = gibbs_samples_of_bart_trees[sample_num - 1][t];
    // Subtract out previous tree's yhats
    sum_resids_vec = Tools::subtract_arrays(sum_resids_vec, previous_tree->yhats, n);
    bartMachineTreeNode* tree = gibbs_samples_of_bart_trees[sample_num][t];

    double current_sigsq = gibbs_samples_of_sigsq[sample_num - 1];
    assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats(tree, current_sigsq);
    
    // After mus are sampled, we need to update the sum_resids_vec
    // Add in current tree's yhats
    sum_resids_vec = Tools::add_arrays(sum_resids_vec, tree->yhats, n);
}

void bartmachine_e_gibbs_base::SampleSigsq(int sample_num, double* es) {
    double sigsq = drawSigsqFromPosterior(sample_num, es);
    gibbs_samples_of_sigsq[sample_num] = sigsq;
}

double* bartmachine_e_gibbs_base::getResidualsFromFullSumModel(int sample_num, double* R_j) {
    // All we need to do is subtract the last tree's yhats now
    bartMachineTreeNode* last_tree = gibbs_samples_of_bart_trees[sample_num][num_trees - 1];
    double* result = new double[n];
    for (int i = 0; i < n; i++) {
        result[i] = R_j[i] - last_tree->yhats[i];
    }
    return result;
}

double* bartmachine_e_gibbs_base::SampleTree(int sample_num, int t, bartMachineTreeNode** trees, TreeArrayIllustration* tree_array_illustration) {
    // First copy the tree from the previous gibbs position
    bartMachineTreeNode* copy_of_old_jth_tree = gibbs_samples_of_bart_trees[sample_num - 1][t]->clone();
    
    // Okay so first we need to get "y" that this tree sees. This is defined as R_j in formula 12 on p274
    // Just go to sum_residual_vec and subtract it from y_trans
    double* temp1 = Tools::subtract_arrays(y_trans, sum_resids_vec, n);
    double* R_j = Tools::add_arrays(temp1, copy_of_old_jth_tree->yhats, n);
    delete[] temp1;
    
    // Now, (important!) set the R_j's as this tree's data.
    copy_of_old_jth_tree->updateWithNewResponsesRecursively(R_j);
    
    // Sample from T_j | R_j, \sigma
    // Now we will run one M-H step on this tree with the y as the R_j
    bartMachineTreeNode* new_jth_tree = metroHastingsPosteriorTreeSpaceIteration(copy_of_old_jth_tree, t, accept_reject_mh, accept_reject_mh_steps);
    
    // Add it to the vector of current sample's trees
    trees[t] = new_jth_tree;
    
    // Now set the new trees in the gibbs sample pantheon
    gibbs_samples_of_bart_trees[sample_num] = trees;
    tree_array_illustration->AddTree(new_jth_tree);
    
    // Return the updated residuals
    return R_j;
}
