#ifndef BARTMACHINE_H_EVAL_H
#define BARTMACHINE_H_EVAL_H

#include "bartmachine_g_mh.h"
#include <vector>

/**
 * Exact port of bartMachine_h_eval from Java to C++
 * 
 * This portion of the code performs the evaluation / prediction on the BART model
 * 
 * Original Java source: /Users/mark/Documents/Cline/bartMachine/src/bartMachine/bartMachine_h_eval.java
 * Port repository: /Users/mark/Documents/Cline/bartMachine_port
 */
class bartmachine_h_eval : public bartmachine_g_mh {
public:
    /**
     * The default BART evaluation of a new observations is done via sample average of the
     * posterior predictions. Other functions can be used here such as median, mode, etc.
     *
     * @param record                The observation to be evaluated / predicted
     * @param num_cores_evaluate    The number of CPU cores to use during evaluation
     */
    double Evaluate(double* record, int num_cores_evaluate);

    /**
     * Evaluates a new observations via sample average of the posterior predictions.
     *
     * @param record                The observation to be evaluated / predicted
     * @param num_cores_evaluate    The number of CPU cores to use during evaluation
     */
    double EvaluateViaSampAvg(double* record, int num_cores_evaluate);

    /**
     * Evaluates a new observations via sample median of the posterior predictions.
     *
     * @param record                The observation to be evaluated / predicted
     * @param num_cores_evaluate    The number of CPU cores to use during evaluation
     */
    double EvaluateViaSampMed(double* record, int num_cores_evaluate);

    // Virtual destructor
    virtual ~bartmachine_h_eval() = default;

protected:
    /**
     * For each sum-of-trees in each posterior of the Gibbs samples, evaluate / predict these new records by summing over
     * the prediction for each tree
     *
     * @param record                The observation to be evaluated / predicted
     * @param num_cores_evaluate    The number of CPU cores to use during evaluation
     * @return                      The predicted values as the result of the sum over many trees for all posterior gibbs samples
     */
    double* getGibbsSamplesForPrediction(double* record, int num_cores_evaluate);

    /**
     * For each sum-of-trees in each posterior of the Gibbs samples, evaluate / predict these new records by summing over
     * the prediction for each tree then order these by value and create an uncertainty interval
     *
     * @param record                The observation for which to create an uncertainty interval
     * @param coverage              The percent coverage (between 0 and 1)
     * @param num_cores_evaluate    The number of CPU cores to use during evaluation
     * @return                      A tuple which is the lower value in the interval followed by the higher value
     */
    double* getPostPredictiveIntervalForPrediction(double* record, double coverage, int num_cores_evaluate);

    /**
     * For each sum-of-trees in each posterior of the Gibbs samples, evaluate / predict these new records by summing over
     * the prediction for each tree then order these by value and create a 95% uncertainty interval
     *
     * @param record                The observation for which to create an uncertainty interval
     * @param num_cores_evaluate    The number of CPU cores to use during evaluation
     * @return                      A tuple which is the lower value in the 95% interval followed by the higher value
     */
    double* get95PctPostPredictiveIntervalForPrediction(double* record, int num_cores_evaluate);
};

#endif // BARTMACHINE_H_EVAL_H
