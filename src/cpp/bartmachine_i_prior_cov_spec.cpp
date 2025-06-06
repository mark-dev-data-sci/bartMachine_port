#include "include/bartmachine_i_prior_cov_spec.h"
#include "include/stat_toolbox.h"
#include <numeric>

// Forward declaration for TIntArrayList
class TIntArrayList {
public:
    std::vector<int> data;
    
    TIntArrayList() {}
    
    void add(int value) {
        data.push_back(value);
    }
    
    int get(int index) const {
        return data[index];
    }
    
    int size() const {
        return data.size();
    }
};

// Helper function to normalize an array (equivalent to Tools.normalize_array in Java)
void normalize_array(double* arr, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            arr[i] /= sum;
        }
    }
}

int bartmachine_i_prior_cov_spec::pickRandomPredictorThatCanBeAssignedF1(bartMachineTreeNode* node) {
    // Get predictors that could be used to split at node
    std::vector<int> predictors = node->getPredictorsThatCouldBeUsedToSplitAtNode();
    
    // Convert std::vector<int> to TIntArrayList for compatibility
    TIntArrayList predictors_list;
    for (int predictor : predictors) {
        predictors_list.add(predictor);
    }
    
    // Get probs of split prior based on predictors that can be used and weight it accordingly
    double* weighted_cov_split_prior_subset = getWeightedCovSplitPriorSubset(&predictors_list);
    
    // Convert TIntArrayList to std::vector<int> for StatToolbox::multinomial_sample
    std::vector<int> vals(predictors_list.size());
    std::vector<double> probs(predictors_list.size());
    
    for (int i = 0; i < predictors_list.size(); i++) {
        vals[i] = predictors_list.get(i);
        probs[i] = weighted_cov_split_prior_subset[i];
    }
    
    // Choose predictor based on random prior value
    int result = StatToolbox::multinomial_sample(vals, probs);
    
    // Clean up
    delete[] weighted_cov_split_prior_subset;
    
    return result;
}

double bartmachine_i_prior_cov_spec::pAdjF1(bartMachineTreeNode* node) {
    if (node->getPadj() == 0) {
        node->setPadj(node->getPredictorsThatCouldBeUsedToSplitAtNode().size());
    }
    
    if (node->getPadj() == 0) {
        return 0;
    }
    
    if (node->getIsLeaf()) {
        return node->getPadj();
    }
    
    // Pull out weighted cov split prior subset vector
    std::vector<int> predictors = node->getPredictorsThatCouldBeUsedToSplitAtNode();
    
    // Convert std::vector<int> to TIntArrayList for compatibility
    TIntArrayList predictors_list;
    for (int predictor : predictors) {
        predictors_list.add(predictor);
    }
    
    // Get probs of split prior based on predictors that can be used and weight it accordingly
    double* weighted_cov_split_prior_subset = getWeightedCovSplitPriorSubset(&predictors_list);
    
    // Find index inside predictor vector
    int index = bartMachineTreeNode::BAD_FLAG_int;
    for (int i = 0; i < predictors_list.size(); i++) {
        if (predictors_list.get(i) == node->getSplitAttributeM()) {
            index = i;
            break;
        }
    }
    
    // Return inverse probability
    double result = 1.0 / weighted_cov_split_prior_subset[index];
    
    // Clean up
    delete[] weighted_cov_split_prior_subset;
    
    return result;
}

double* bartmachine_i_prior_cov_spec::getWeightedCovSplitPriorSubset(TIntArrayList* predictors) {
    double* weighted_cov_split_prior_subset = new double[predictors->size()];
    
    for (int i = 0; i < predictors->size(); i++) {
        weighted_cov_split_prior_subset[i] = cov_split_prior[predictors->get(i)];
    }
    
    normalize_array(weighted_cov_split_prior_subset, predictors->size());
    
    return weighted_cov_split_prior_subset;
}

void bartmachine_i_prior_cov_spec::setCovSplitPrior(double* cov_split_prior) {
    this->cov_split_prior = cov_split_prior;
    // If we're setting the vector, we're using this feature
    use_prior_cov_spec = true;
}

int bartmachine_i_prior_cov_spec::pickRandomPredictorThatCanBeAssigned(bartMachineTreeNode* node) {
    if (use_prior_cov_spec) {
        return pickRandomPredictorThatCanBeAssignedF1(node);
    }
    return bartmachine_h_eval::pickRandomPredictorThatCanBeAssigned(node);
}

double bartmachine_i_prior_cov_spec::pAdj(bartMachineTreeNode* node) {
    if (use_prior_cov_spec) {
        return pAdjF1(node);
    }
    return bartmachine_h_eval::pAdj(node);
}
