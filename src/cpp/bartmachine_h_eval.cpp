#include "include/bartmachine_h_eval.h"
#include "include/stat_toolbox.h"
#include <algorithm>

// Implementation of bartmachine_h_eval.cpp

double bartmachine_h_eval::Evaluate(double* record, int num_cores_evaluate) {
    // The default BART evaluation of a new observations is done via sample average of the
    // posterior predictions. Other functions can be used here such as median, mode, etc.
    return EvaluateViaSampAvg(record, num_cores_evaluate);
}

double bartmachine_h_eval::EvaluateViaSampAvg(double* record, int num_cores_evaluate) {
    // Evaluates a new observations via sample average of the posterior predictions.
    double* samples = getGibbsSamplesForPrediction(record, num_cores_evaluate);
    double result = StatToolbox::sample_average(samples, numSamplesAfterBurningAndThinning());
    delete[] samples;
    return result;
}

double bartmachine_h_eval::EvaluateViaSampMed(double* record, int num_cores_evaluate) {
    // Evaluates a new observations via sample median of the posterior predictions.
    double* samples = getGibbsSamplesForPrediction(record, num_cores_evaluate);
    std::vector<double> samples_vec(samples, samples + numSamplesAfterBurningAndThinning());
    double result = StatToolbox::sample_median(samples_vec);
    delete[] samples;
    return result;
}

double* bartmachine_h_eval::getGibbsSamplesForPrediction(double* record, int num_cores_evaluate) {
    // For each sum-of-trees in each posterior of the Gibbs samples, evaluate / predict these new records by summing over
    // the prediction for each tree
    
    // The results for each of the gibbs samples
    double* y_gibbs_samples = new double[numSamplesAfterBurningAndThinning()];
    for (int g = 0; g < numSamplesAfterBurningAndThinning(); g++) {
        bartMachineTreeNode** bart_trees = gibbs_samples_of_bart_trees_after_burn_in[g];
        double yt_g = 0;
        for (int t = 0; t < num_trees; t++) { // Sum of trees right?
            yt_g += bart_trees[t]->Evaluate(record);
        }
        y_gibbs_samples[g] = un_transform_y(yt_g);
    }
    return y_gibbs_samples;
}

double* bartmachine_h_eval::getPostPredictiveIntervalForPrediction(double* record, double coverage, int num_cores_evaluate) {
    // For each sum-of-trees in each posterior of the Gibbs samples, evaluate / predict these new records by summing over
    // the prediction for each tree then order these by value and create an uncertainty interval
    
    // Get all gibbs samples sorted
    double* y_gibbs_samples = getGibbsSamplesForPrediction(record, num_cores_evaluate);
    std::sort(y_gibbs_samples, y_gibbs_samples + numSamplesAfterBurningAndThinning());
    
    // Calculate index of the CI_a and CI_b
    int n_bottom = static_cast<int>(std::round((1 - coverage) / 2 * numSamplesAfterBurningAndThinning())) - 1; // -1 because arrays start at zero
    int n_top = static_cast<int>(std::round(((1 - coverage) / 2 + coverage) * numSamplesAfterBurningAndThinning())) - 1; // -1 because arrays start at zero
    
    double* conf_interval = new double[2];
    conf_interval[0] = y_gibbs_samples[n_bottom];
    conf_interval[1] = y_gibbs_samples[n_top];
    
    delete[] y_gibbs_samples;
    return conf_interval;
}

double* bartmachine_h_eval::get95PctPostPredictiveIntervalForPrediction(double* record, int num_cores_evaluate) {
    // For each sum-of-trees in each posterior of the Gibbs samples, evaluate / predict these new records by summing over
    // the prediction for each tree then order these by value and create a 95% uncertainty interval
    return getPostPredictiveIntervalForPrediction(record, 0.95, num_cores_evaluate);
}
