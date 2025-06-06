/**
 * bartmachine_h_eval.h
 * 
 * Header file for the bartmachine_h_eval class, which handles evaluation
 * and prediction functionality for the BART model.
 */

#ifndef BARTMACHINE_H_EVAL_H
#define BARTMACHINE_H_EVAL_H

#include "bartmachine_g_mh.h"

class bartmachine_h_eval : public bartmachine_g_mh {
protected:
    // Protected member variables
    
    // Prediction-related variables
    double* y_hat_train;
    double* y_hat_test;
    double* y_hat_train_credible_intervals_lower;
    double* y_hat_train_credible_intervals_upper;
    double* y_hat_test_credible_intervals_lower;
    double* y_hat_test_credible_intervals_upper;
    double* y_hat_train_prediction_intervals_lower;
    double* y_hat_train_prediction_intervals_upper;
    double* y_hat_test_prediction_intervals_lower;
    double* y_hat_test_prediction_intervals_upper;
    
    // Evaluation metrics
    double rmse_train;
    double rmse_test;
    double mse_train;
    double mse_test;
    
public:
    // Constructor and destructor
    bartmachine_h_eval();
    virtual ~bartmachine_h_eval();
    
    // Prediction methods
    virtual double* getPredictionsForSingleTree(bartMachineTreeNode* tree, double** X, int num_rows);
    virtual double* getPredictionsForSingleTreeWithMissingData(bartMachineTreeNode* tree, double** X, int num_rows);
    
    // Prediction interval methods
    virtual void calcPredictionIntervals(double alpha);
    virtual void calcPredictionIntervalsForTestData(double alpha);
    
    // Credible interval methods
    virtual void calcCredibleIntervals(double alpha);
    virtual void calcCredibleIntervalsForTestData(double alpha);
    
    // Evaluation methods
    virtual double calcRMSE(double* y_hat, double* y_actual, int n);
    virtual double calcMSE(double* y_hat, double* y_actual, int n);
    
    // Getters for predictions and intervals
    virtual double* getYHatTrain();
    virtual double* getYHatTest();
    virtual double* getYHatTrainCredibleIntervalsLower();
    virtual double* getYHatTrainCredibleIntervalsUpper();
    virtual double* getYHatTestCredibleIntervalsLower();
    virtual double* getYHatTestCredibleIntervalsUpper();
    virtual double* getYHatTrainPredictionIntervalsLower();
    virtual double* getYHatTrainPredictionIntervalsUpper();
    virtual double* getYHatTestPredictionIntervalsLower();
    virtual double* getYHatTestPredictionIntervalsUpper();
    
    // Getters for evaluation metrics
    virtual double getRMSETrain();
    virtual double getRMSETest();
    virtual double getMSETrain();
    virtual double getMSETest();
};

#endif // BARTMACHINE_H_EVAL_H
