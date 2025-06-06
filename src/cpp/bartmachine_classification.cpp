#include "include/bartmachine_classification.h"
#include "include/stat_toolbox.h"
#include "include/stat_util.h"
#include "include/tree_array_illustration.h"
#include <cmath>
#include <algorithm>
#include <iostream>

bartMachineClassification::bartMachineClassification() {
    // Initialize any classification-specific parameters here
}

bartMachineClassification::~bartMachineClassification() {
    // Clean up any classification-specific resources here
}

bool bartMachineClassification::build(double** X, int* y, int n, int p) {
    // Store the dimensions
    this->n = n;
    this->p = p;

    // Store the training data
    std::vector<double*> X_vec(n);
    for (int i = 0; i < n; i++) {
        X_vec[i] = X[i];
    }
    setData(X_vec);

    // Convert binary y to double for BART model
    y_orig = new double[n];
    for (int i = 0; i < n; i++) {
        y_orig[i] = static_cast<double>(y[i]);
    }

    // Initialize the model
    Build();

    return true;
}

void bartMachineClassification::DoOneGibbsSample() {
    // This array is the array of trees for this given sample
    bartMachineTreeNode** bart_trees = new bartMachineTreeNode*[num_trees];
    for (int t = 0; t < num_trees; t++) {
        bart_trees[t] = new bartMachineTreeNode();
    }
    
    TreeArrayIllustration* tree_array_illustration = new TreeArrayIllustration(gibbs_sample_num, unique_name);

    // Get Z's
    if (SampleZs()) {
        for (int t = 0; t < num_trees; t++) {
            delete bart_trees[t];
        }
        delete[] bart_trees;
        delete tree_array_illustration;
        return;
    }

    for (int t = 0; t < num_trees; t++) {
        if (verbose) {
            GibbsSampleDebugMessage(t);
        }
        
        SampleTree(gibbs_sample_num, t, bart_trees, tree_array_illustration);
        SampleMusWrapper(gibbs_sample_num, t);
    }

    if (tree_illust) {
        illustrate(tree_array_illustration);
    }

    for (int t = 0; t < num_trees; t++) {
        delete bart_trees[t];
    }
    delete[] bart_trees;
    delete tree_array_illustration;
}

bool bartMachineClassification::SampleZs() {
    for (int i = 0; i < n; i++) {
        double g_x_i = 0;
        bartMachineTreeNode** trees = gibbs_samples_of_bart_trees[gibbs_sample_num - 1];
        for (int t = 0; t < num_trees; t++) {
            // Get the training data for this observation
            double* x_i = getXY()[i];
            double g_x_i_t = trees[t]->Evaluate(x_i);
            if (std::isinf(g_x_i_t) || std::isnan(g_x_i_t)) {
                return true;
            }
            g_x_i += g_x_i_t;
        }
        // y_trans is the Z's from the paper
        y_trans[i] = SampleZi(g_x_i, y_orig[i]);
    }
    return false;
}

double bartMachineClassification::SampleZi(double g_x_i, double y_i) {
    double u = StatToolbox::rand();
    
    if (y_i == 1) {
        double p_i = StatUtil::normal_cdf(-g_x_i);
        return g_x_i + StatUtil::getInvCDF((1 - u) * p_i + u, false);
    }
    else if (y_i == 0) {
        double p_i = StatUtil::normal_cdf(g_x_i);
        return g_x_i - StatUtil::getInvCDF((1 - u) * p_i + u, false);
    }
    
    std::cerr << "SampleZi RESPONSE NOT ZERO / ONE" << std::endl;
    exit(0);
    return -1;
}

void bartMachineClassification::SetupGibbsSampling() {
    bartmachine_i_prior_cov_spec::SetupGibbsSampling();
    
    // All sigsqs are now 1 all the time
    for (int g = 0; g < num_gibbs_total_iterations; g++) {
        gibbs_samples_of_sigsq[g] = SIGSQ_FOR_PROBIT;
    }
}

void bartMachineClassification::calculateHyperparameters() {
    hyper_mu_mu = 0;
    hyper_sigsq_mu = std::pow(3 / (hyper_k * std::sqrt(num_trees)), 2);
}

void bartMachineClassification::transformResponseVariable() {
    y_trans = new double[n]; // Do nothing
}

double bartMachineClassification::un_transform_y(double yt_i) {
    return yt_i; // Do nothing
}

double bartMachineClassification::getProbability(double* record) {
    // Get Gibbs samples for prediction
    double* gibbs_samples = getGibbsSamplesForPrediction(record, 1);

    // Calculate the average probability
    double sum = 0.0;
    int num_samples = numSamplesAfterBurningAndThinning();
    for (int i = 0; i < num_samples; i++) {
        // Apply sigmoid function to convert to probability
        double prob = StatUtil::normal_cdf(gibbs_samples[i]);
        sum += prob;
    }

    double avg_prob = sum / num_samples;

    // Clean up
    delete[] gibbs_samples;

    return avg_prob;
}

int bartMachineClassification::getPrediction(double* record) {
    // Use default threshold of 0.5
    return getPrediction(record, 0.5);
}

int bartMachineClassification::getPrediction(double* record, double threshold) {
    // Get probability
    double prob = getProbability(record);

    // Convert to class prediction
    return (prob >= threshold) ? 1 : 0;
}

double* bartMachineClassification::getProbabilities(double** X, int n) {
    // Allocate array for probabilities
    double* probs = new double[n];

    // Calculate probability for each observation
    for (int i = 0; i < n; i++) {
        probs[i] = getProbability(X[i]);
    }

    return probs;
}

int* bartMachineClassification::getPredictions(double** X, int n) {
    // Use default threshold of 0.5
    return getPredictions(X, n, 0.5);
}

int* bartMachineClassification::getPredictions(double** X, int n, double threshold) {
    // Allocate array for predictions
    int* preds = new int[n];

    // Calculate prediction for each observation
    for (int i = 0; i < n; i++) {
        preds[i] = getPrediction(X[i], threshold);
    }

    return preds;
}

void bartMachineClassification::setNumBurnIn(int num_burn_in) {
    setNumGibbsBurnIn(num_burn_in);
}

void bartMachineClassification::setNumIterationsAfterBurnIn(int num_iterations_after_burn_in) {
    setNumGibbsTotalIterations(num_iterations_after_burn_in + num_gibbs_burn_in);
}
