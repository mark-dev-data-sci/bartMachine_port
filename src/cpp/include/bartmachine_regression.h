#ifndef BARTMACHINE_REGRESSION_H
#define BARTMACHINE_REGRESSION_H

#include "bartmachine_i_prior_cov_spec.h"

/**
 * This class extends the bartMachine implementation to provide regression-specific functionality.
 * It includes methods for predicting continuous outcomes and calculating prediction intervals.
 */
class bartMachineRegression : public bartmachine_i_prior_cov_spec {
protected:
    /**
     * Transform the response variable for regression
     */
    void transformResponseVariable() override;

    /**
     * Un-transform the response variable for regression
     *
     * @param yt_i The transformed response value
     * @return The original response value
     */
    double un_transform_y(double yt_i) override;

public:
    /**
     * Default constructor
     */
    bartMachineRegression();

    /**
     * Destructor
     */
    ~bartMachineRegression();

    /**
     * Build the BART model for regression
     *
     * @param X The training data matrix (n x p)
     * @param y The training response vector (n x 1)
     * @param n The number of observations
     * @param p The number of predictors
     * @return true if the model was built successfully, false otherwise
     */
    bool build(double** X, double* y, int n, int p);

    /**
     * Predict the response for a new observation
     *
     * @param record The new observation (p x 1)
     * @return The predicted response
     */
    double Evaluate(double* record);

    /**
     * Predict the response for a new observation using sample average
     *
     * @param record The new observation (p x 1)
     * @return The predicted response
     */
    double EvaluateViaSampAvg(double* record);

    /**
     * Predict the response for a new observation using sample median
     *
     * @param record The new observation (p x 1)
     * @return The predicted response
     */
    double EvaluateViaSampMed(double* record);

    /**
     * Get the 95% posterior predictive interval for a new observation
     *
     * @param record The new observation (p x 1)
     * @return An array of length 2 containing the lower and upper bounds of the interval
     */
    double* get95PctPostPredictiveIntervalForPrediction(double* record);

    /**
     * Get the posterior predictive interval with specified coverage for a new observation
     *
     * @param record The new observation (p x 1)
     * @param coverage The desired coverage probability (e.g., 0.95 for 95% interval)
     * @return An array of length 2 containing the lower and upper bounds of the interval
     */
    double* getPostPredictiveIntervalForPrediction(double* record, double coverage);

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
};

#endif // BARTMACHINE_REGRESSION_H
