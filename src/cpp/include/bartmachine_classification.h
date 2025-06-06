#ifndef BARTMACHINE_CLASSIFICATION_H
#define BARTMACHINE_CLASSIFICATION_H

#include "bartmachine_i_prior_cov_spec.h"

/**
 * This class extends the bartMachine implementation to provide classification-specific functionality.
 * It includes methods for predicting class probabilities and converting probabilities to class labels.
 */
class bartMachineClassification : public bartmachine_i_prior_cov_spec {
private:
    /** A dummy value for the unused sigsq's in binary classification BART */
    static constexpr double SIGSQ_FOR_PROBIT = 1.0;

    /**
     * We sample the latent variables, Z, for each of the n observations
     *
     * @see Section 2.3 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
     * @return true if there was an error, false otherwise
     */
    bool SampleZs();

    /**
     * We sample one latent variable, Z_i
     *
     * @see Section 2.3 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
     * @param g_x_i The sum of tree predictions for observation i
     * @param y_i The original response value for observation i
     * @return The sampled Z value
     */
    double SampleZi(double g_x_i, double y_i);

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

    /**
     * Set the number of burn-in iterations
     *
     * @param num_burn_in The number of burn-in iterations
     */
    void setNumBurnIn(int num_burn_in);

    /**
     * Set the number of iterations after burn-in
     *
     * @param num_iterations_after_burn_in The number of iterations after burn-in
     */
    void setNumIterationsAfterBurnIn(int num_iterations_after_burn_in);

    /**
     * A Gibbs sample for binary classification BART is a little different
     * than for regression BART. We no longer sample sigsq's. We instead sample Z's,
     * the latent variables that allow us to estimate the prob(Y = 1).
     */
    void DoOneGibbsSample();

    /**
     * Sets up Gibbs sampling. We should also blank out the vector gibbs_samples_of_sigsq with dummy values.
     */
    void SetupGibbsSampling();

    /**
     * Calculates the hyperparameters needed for binary classification BART.
     * This only need hyper_sigsq_mu
     */
    void calculateHyperparameters();

    /**
     * Transform the response variable for classification
     */
    void transformResponseVariable();

    /**
     * Un-transform the response variable for classification
     *
     * @param yt_i The transformed response value
     * @return The original response value
     */
    double un_transform_y(double yt_i);
};

#endif // BARTMACHINE_CLASSIFICATION_H
