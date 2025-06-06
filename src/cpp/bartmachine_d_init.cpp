#include "include/bartmachine_d_init.h"
#include "include/stat_toolbox.h"
#include <iostream>

// Forward declaration of TreeArrayIllustration class
class TreeArrayIllustration;

void bartmachine_d_init::SetupGibbsSampling() {
    InitGibbsSamplingData();
    InitizializeSigsq();
    InitializeTrees();
    InitializeMus();
    if (tree_illust) {
        InitTreeIllustrations();
    }
    // The zeroth gibbs sample is the initialization we just did; now we're onto the first in the chain
    gibbs_sample_num = 1;
    
    sum_resids_vec = new double[n];
    for (int i = 0; i < n; i++) {
        sum_resids_vec[i] = 0.0;
    }
}

void bartmachine_d_init::InitGibbsSamplingData() {
    // Now initialize the gibbs sampler array for trees and error variances
    gibbs_samples_of_bart_trees = new bartMachineTreeNode**[num_gibbs_total_iterations + 1];
    for (int i = 0; i <= num_gibbs_total_iterations; i++) {
        gibbs_samples_of_bart_trees[i] = new bartMachineTreeNode*[num_trees];
    }
    
    gibbs_samples_of_bart_trees_after_burn_in = new bartMachineTreeNode**[num_gibbs_total_iterations - num_gibbs_burn_in + 1];
    for (int i = 0; i <= num_gibbs_total_iterations - num_gibbs_burn_in; i++) {
        gibbs_samples_of_bart_trees_after_burn_in[i] = new bartMachineTreeNode*[num_trees];
    }
    
    gibbs_samples_of_sigsq = new double[num_gibbs_total_iterations + 1];
    gibbs_samples_of_sigsq_after_burn_in = new double[num_gibbs_total_iterations - num_gibbs_burn_in];
    
    accept_reject_mh = new bool*[num_gibbs_total_iterations + 1];
    for (int i = 0; i <= num_gibbs_total_iterations; i++) {
        accept_reject_mh[i] = new bool[num_trees];
    }
    
    accept_reject_mh_steps = new char*[num_gibbs_total_iterations + 1];
    for (int i = 0; i <= num_gibbs_total_iterations; i++) {
        accept_reject_mh_steps[i] = new char[num_trees];
    }
}

void bartmachine_d_init::InitializeTrees() {
    // Create the array of trees for the zeroth gibbs sample
    bartMachineTreeNode** bart_trees = new bartMachineTreeNode*[num_trees];
    for (int i = 0; i < num_trees; i++) {
        bartMachineTreeNode* stump = new bartMachineTreeNode(this);
        stump->setStumpData(X_y, y_trans, p);
        bart_trees[i] = stump;
    }
    gibbs_samples_of_bart_trees[0] = bart_trees;
}

void bartmachine_d_init::InitializeMus() {
    for (int i = 0; i < num_trees; i++) {
        gibbs_samples_of_bart_trees[0][i]->y_pred = 0;
    }
}

void bartmachine_d_init::InitizializeSigsq() {
    gibbs_samples_of_sigsq[0] = StatToolbox::sample_from_inv_gamma(hyper_nu / 2, 2 / (hyper_nu * hyper_lambda));
}

void bartmachine_d_init::DeleteBurnInsOnPreviousSamples() {
    if (gibbs_sample_num <= num_gibbs_burn_in + 1 && gibbs_sample_num >= 2) {
        // Clean up memory for trees
        for (int t = 0; t < num_trees; t++) {
            delete gibbs_samples_of_bart_trees[gibbs_sample_num - 2][t];
        }
        gibbs_samples_of_bart_trees[gibbs_sample_num - 2] = nullptr;
    }
}

void bartmachine_d_init::illustrate(TreeArrayIllustration* tree_array_illustration) {
    if (tree_illust) {
        tree_array_illustration->CreateIllustrationAndSaveImage();
    }
}

int bartmachine_d_init::numSamplesAfterBurningAndThinning() {
    return num_gibbs_total_iterations - num_gibbs_burn_in;
}

void bartmachine_d_init::setNumGibbsBurnIn(int num_gibbs_burn_in) {
    this->num_gibbs_burn_in = num_gibbs_burn_in;
}

void bartmachine_d_init::setNumGibbsTotalIterations(int num_gibbs_total_iterations) {
    this->num_gibbs_total_iterations = num_gibbs_total_iterations;
}

void bartmachine_d_init::setSigsq(double fixed_sigsq) {
    this->fixed_sigsq = fixed_sigsq;
}

bool** bartmachine_d_init::getAcceptRejectMH() {
    return accept_reject_mh;
}
