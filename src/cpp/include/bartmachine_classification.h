#ifndef BARTMACHINE_CLASSIFICATION_H
#define BARTMACHINE_CLASSIFICATION_H

#include "bartmachine_i_prior_cov_spec.h"

/**
 * This class extends the bartMachine implementation to provide classification-specific functionality.
 * It includes methods for predicting class probabilities and converting probabilities to class labels.
 */
class bartMachineClassification : public bartmachine_i_prior_cov_spec {
public:
    /**
     * Default constructor
     */
    bartMachineClassification();
    
    /**
     * Destructor
     */
    ~bartMachineClassification();
    
    /**
     * Build the BART model for classification
     * 
     * @param X The training data matrix (n x p)
     * @param y The training response vector (n x 1) with binary labels (0/1)
     * @param n The number of observations
     * @param p The number of predictors
     * @return true if the model was built successfully, false otherwise
     */
    bool build(double** X, int* y, int n, int p);
    
    /**
     * Get the probability of class 1 for a new observation
     * 
     * @param record The new observation (p x 1)
     * @return The probability of class 1
     */
    double getProbability(double* record);
    
    /**
     * Get the predicted class (0 or 1) for a new observation
     * 
     * @param record The new observation (p x 1)
     * @return The predicted class (0 or 1)
     */
    int getPrediction(double* record);
    
    /**
     * Get the predicted class (0 or 1) for a new observation using a custom threshold
     * 
     * @param record The new observation (p x 1)
     * @param threshold The probability threshold for class 1 (default is 0.5)
     * @return The predicted class (0 or 1)
     */
    int getPrediction(double* record, double threshold);
    
    /**
     * Get the probabilities of class 1 for multiple observations
     * 
     * @param X The data matrix (n x p)
     * @param n The number of observations
     * @return An array of length n containing the probabilities of class 1
     */
    double* getProbabilities(double** X, int n);
    
    /**
     * Get the predicted classes (0 or 1) for multiple observations
     * 
     * @param X The data matrix (n x p)
     * @param n The number of observations
     * @return An array of length n containing the predicted classes (0 or 1)
     */
    int* getPredictions(double** X, int n);
    
    /**
     * Get the predicted classes (0 or 1) for multiple observations using a custom threshold
     * 
     * @param X The data matrix (n x p)
     * @param n The number of observations
     * @param threshold The probability threshold for class 1 (default is 0.5)
     * @return An array of length n containing the predicted classes (0 or 1)
     */
    int* getPredictions(double** X, int n, double threshold);
};

#endif // BARTMACHINE_CLASSIFICATION_H
