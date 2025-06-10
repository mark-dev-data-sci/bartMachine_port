"""
Model building utilities for bartMachine.

This module provides functions for building BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bart_package_builders.R' 
    in the original R package.

Role in Port:
    This module handles the creation and building of BART models, including
    setting hyperparameters and training the model.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, Union, List, Dict, Any, Tuple
import logging
import os
import sys
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

from .bart_package_inits import initialize_jvm, is_jvm_running
from .data_preprocessing import preprocess_training_data, preprocess_new_data
from .zzz import (
    create_bart_machine_object,
    build_bart_machine,
    predict_bart_machine,
    get_variable_importance,
    get_variable_inclusion_proportions,
    get_gibbs_samples_sigsqs,
    extract_raw_node_information
)

# Set up logging
logger = logging.getLogger(__name__)

class BartMachine:
    """
    A class for building and using BART models.
    
    This class provides a Pythonic interface to the Java implementation of BART.
    It handles data preprocessing, model building, prediction, and other tasks.
    
    Attributes:
        X: The predictor variables.
        y: The response variable.
        java_bart_machine: The Java BART machine object.
        num_trees: Number of trees in the BART model.
        num_burn_in: Number of burn-in MCMC iterations.
        num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
        alpha: Prior parameter for tree structure.
        beta: Prior parameter for tree structure.
        k: Prior parameter for leaf node parameters.
        q: Prior parameter for tree structure.
        nu: Prior parameter for error variance.
        prob_rule_class: Probability of using a classification rule.
        prob_rule_avg: Probability of using an average rule.
        prob_split_not_decision: Probability of splitting on a non-decision rule.
        prob_rule_quad: Probability of using a quadratic rule.
        use_missing_data: Whether to use missing data in the model.
        use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
        impute_missingness_with_rf_impute: Whether to impute missing values with random forest.
        replace_missing_data_with_x_j_bar: Whether to replace missing data with column means.
        impute_missingness_with_x_j_bar_for_lm: Whether to impute missing values with column means for linear model.
        verbose: Whether to print verbose output.
        seed: Random seed for reproducibility.
        cov_prior_vec: Prior vector for covariate selection.
        pred_type: Type of prediction ("regression" or "classification").
        n: Number of observations.
        p: Number of predictors.
        training_data_features: Names of the training data features.
        time_to_build: Time to build the model.
        y_hat_train: Predicted values for the training data.
        rmse_train: Root mean squared error for the training data.
        PseudoRsq: Pseudo R-squared for the training data.
        confusion_matrix: Confusion matrix for the training data.
        misclassification_error: Misclassification error for the training data.
    """
    
    def __init__(self, X=None, y=None, Xy=None, 
                num_trees: int = 50, 
                num_burn_in: int = 250, 
                num_iterations_after_burn_in: int = 1000,
                alpha: float = 0.95, 
                beta: float = 2.0, 
                k: float = 2.0, 
                q: float = 0.9, 
                nu: float = 3.0,
                prob_rule_class: float = 0.5, 
                prob_rule_avg: float = 0.5, 
                prob_split_not_decision: float = 0.0, 
                prob_rule_quad: float = 0.1,
                use_missing_data: bool = False, 
                use_missing_data_dummies_as_covars: bool = False, 
                impute_missingness_with_rf_impute: bool = False,
                replace_missing_data_with_x_j_bar: bool = False, 
                impute_missingness_with_x_j_bar_for_lm: bool = False,
                verbose: bool = False, 
                seed: Optional[int] = None, 
                cov_prior_vec: Optional[np.ndarray] = None,
                pred_type: str = "regression"):
        """
        Initialize a BartMachine object.
        
        Args:
            X: The predictor variables.
            y: The response variable.
            Xy: Combined predictor and response variables.
            num_trees: Number of trees in the BART model.
            num_burn_in: Number of burn-in MCMC iterations.
            num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
            alpha: Prior parameter for tree structure.
            beta: Prior parameter for tree structure.
            k: Prior parameter for leaf node parameters.
            q: Prior parameter for tree structure.
            nu: Prior parameter for error variance.
            prob_rule_class: Probability of using a classification rule.
            prob_rule_avg: Probability of using an average rule.
            prob_split_not_decision: Probability of splitting on a non-decision rule.
            prob_rule_quad: Probability of using a quadratic rule.
            use_missing_data: Whether to use missing data in the model.
            use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
            impute_missingness_with_rf_impute: Whether to impute missing values with random forest.
            replace_missing_data_with_x_j_bar: Whether to replace missing data with column means.
            impute_missingness_with_x_j_bar_for_lm: Whether to impute missing values with column means for linear model.
            verbose: Whether to print verbose output.
            seed: Random seed for reproducibility.
            cov_prior_vec: Prior vector for covariate selection.
            pred_type: Type of prediction ("regression" or "classification").
        """
        # Check input parameters
        if (X is None and Xy is None) or (y is None and Xy is None):
            # No data provided, just initialize the object
            self.X = None
            self.y = None
        elif X is not None and y is not None and Xy is not None:
            raise ValueError("You cannot specify both X,y and Xy simultaneously.")
        elif X is None and y is None:  # they specified Xy, so now just pull out X,y
            # First ensure it's a dataframe
            if not isinstance(Xy, pd.DataFrame):
                raise ValueError("The training data Xy must be a data frame.")
            
            y = Xy.iloc[:, -1]
            X = Xy.iloc[:, :-1]
            
            self.X = X
            self.y = y
        else:  # they specified X and y
            # Make sure X is a dataframe
            if not isinstance(X, pd.DataFrame):
                raise ValueError("The training data X must be a data frame.")
            
            self.X = X
            self.y = y
        
        # Initialize Java BART machine object
        self.java_bart_machine = None
        
        # Store hyperparameters
        self.num_trees = num_trees
        self.num_burn_in = num_burn_in
        self.num_iterations_after_burn_in = num_iterations_after_burn_in
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.q = q
        self.nu = nu
        self.prob_rule_class = prob_rule_class
        self.prob_rule_avg = prob_rule_avg
        self.prob_split_not_decision = prob_split_not_decision
        self.prob_rule_quad = prob_rule_quad
        self.use_missing_data = use_missing_data
        self.use_missing_data_dummies_as_covars = use_missing_data_dummies_as_covars
        self.impute_missingness_with_rf_impute = impute_missingness_with_rf_impute
        self.replace_missing_data_with_x_j_bar = replace_missing_data_with_x_j_bar
        self.impute_missingness_with_x_j_bar_for_lm = impute_missingness_with_x_j_bar_for_lm
        self.verbose = verbose
        self.seed = seed
        self.cov_prior_vec = cov_prior_vec
        self.pred_type = pred_type
        
        # Initialize other attributes
        self.n = None
        self.p = None
        self.training_data_features = None
        self.time_to_build = None
        self.y_hat_train = None
        self.rmse_train = None
        self.PseudoRsq = None
        self.confusion_matrix = None
        self.misclassification_error = None
        
        # If data is provided, build the model
        if self.X is not None and self.y is not None:
            self.build()
    
    def build(self) -> None:
        """
        Build the BART model.
        
        This function builds the BART model using the provided data and hyperparameters.
        """
        # Check if data is provided
        if self.X is None or self.y is None:
            raise ValueError("No data provided. Please provide X and y before building the model.")
        
        # Initialize JVM if not already running
        if not is_jvm_running():
            initialize_jvm(verbose=self.verbose)
        
        # Start timer
        start_time = time.time()
        
        # Get the number of observations and predictors
        self.n = len(self.X)
        self.p = len(self.X.columns)
        
        # Store the training data features
        self.training_data_features = self.X.columns.tolist()
        
        # Preprocess training data
        preprocessed_data = preprocess_training_data(
            X=self.X,
            use_missing_data_dummies_as_covars=self.use_missing_data_dummies_as_covars
        )
        
        # Extract preprocessed data
        X_preprocessed = preprocessed_data["data"]
        
        # Convert y to numpy array
        y_array = self.y.to_numpy()
        
        # Create Java BART machine object
        self.java_bart_machine = create_bart_machine_object(
            X=X_preprocessed,
            y=y_array,
            num_trees=self.num_trees,
            num_burn_in=self.num_burn_in,
            num_iterations_after_burn_in=self.num_iterations_after_burn_in,
            alpha=self.alpha,
            beta=self.beta,
            k=self.k,
            q=self.q,
            nu=self.nu,
            prob_rule_class=self.prob_rule_class,
            prob_rule_avg=self.prob_rule_avg,
            prob_split_not_decision=self.prob_split_not_decision,
            prob_rule_quad=self.prob_rule_quad,
            use_missing_data=self.use_missing_data,
            use_missing_data_dummies_as_covars=self.use_missing_data_dummies_as_covars,
            impute_missingness_with_rf_impute=self.impute_missingness_with_rf_impute,
            replace_missing_data_with_x_j_bar=self.replace_missing_data_with_x_j_bar,
            impute_missingness_with_x_j_bar_for_lm=self.impute_missingness_with_x_j_bar_for_lm,
            verbose=self.verbose,
            seed=self.seed,
            cov_prior_vec=self.cov_prior_vec,
            pred_type=self.pred_type
        )
        
        # Build the model
        build_bart_machine(self.java_bart_machine)
        
        # End timer
        end_time = time.time()
        self.time_to_build = end_time - start_time
        
        # Get training predictions
        self.y_hat_train = self.predict(self.X)
        
        # Calculate training metrics
        if self.pred_type == "regression":
            # Calculate RMSE
            self.rmse_train = np.sqrt(mean_squared_error(self.y, self.y_hat_train))
            
            # Calculate pseudo R-squared
            y_mean = np.mean(self.y)
            ss_total = np.sum((self.y - y_mean) ** 2)
            ss_residual = np.sum((self.y - self.y_hat_train) ** 2)
            self.PseudoRsq = 1 - ss_residual / ss_total
        else:
            # Calculate confusion matrix
            self.confusion_matrix = confusion_matrix(self.y, self.y_hat_train)
            
            # Calculate misclassification error
            self.misclassification_error = 1 - accuracy_score(self.y, self.y_hat_train)
        
        if self.verbose:
            print(f"BART model built in {self.time_to_build:.2f} seconds.")
            if self.pred_type == "regression":
                print(f"Training RMSE: {self.rmse_train:.4f}")
                print(f"Training pseudo R-squared: {self.PseudoRsq:.4f}")
            else:
                print(f"Training misclassification error: {self.misclassification_error:.4f}")
                print("Training confusion matrix:")
                print(self.confusion_matrix)
    
    def predict(self, new_data: pd.DataFrame, 
               type: str = "response") -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the BART model.
        
        This function makes predictions with the BART model for new data.
        
        Args:
            new_data: The new predictor variables.
            type: Type of prediction to return.
                "response": Return the predicted response.
                "samples": Return posterior samples.
        
        Returns:
            Predicted values or posterior samples.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Preprocess new data
        X_preprocessed = preprocess_new_data(new_data, self)
        
        # Make predictions
        if type == "response":
            # Get predicted response
            predictions = predict_bart_machine(self.java_bart_machine, X_preprocessed)
            
            # Convert to appropriate type
            if self.pred_type == "classification":
                # Convert to binary predictions
                predictions = (predictions > 0.5).astype(int)
            
            return predictions
        elif type == "samples":
            # Get posterior samples
            samples = predict_bart_machine(self.java_bart_machine, X_preprocessed, return_samples=True)
            
            return samples
        else:
            raise ValueError(f"Unknown prediction type: {type}")
    
    def get_var_importance(self, type: str = "splits") -> pd.Series:
        """
        Get variable importance measures from the BART model.
        
        This function returns the variable importance measures for all variables
        in the BART model. The importance measure is based on the frequency with which
        variables are used in splitting rules.
        
        Args:
            type: Type of importance measure to use ("splits" or "trees").
                "splits" counts the total number of splitting rules using each variable.
                "trees" counts the number of trees that use each variable at least once.
        
        Returns:
            A pandas Series containing variable importance measures.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Get variable importance from the Java backend
        var_importance = get_variable_importance(self.java_bart_machine, type)
        
        # Convert to pandas Series
        var_importance_series = pd.Series(var_importance, index=self.training_data_features)
        
        return var_importance_series
    
    def get_var_props_over_chain(self, type: str = "splits") -> pd.Series:
        """
        Get variable inclusion proportions over the MCMC chain.
        
        This function returns the proportion of times each variable is used
        in splitting rules across all trees and MCMC iterations.
        
        Args:
            type: Type of proportion to use ("splits" or "trees").
                "splits" uses the proportion of splitting rules using each variable.
                "trees" uses the proportion of trees that use each variable at least once.
        
        Returns:
            A pandas Series containing variable inclusion proportions.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Get variable inclusion proportions from the Java backend
        var_props = get_variable_inclusion_proportions(self.java_bart_machine)
        
        # Convert to pandas Series
        var_props_series = pd.Series(var_props, index=self.training_data_features)
        
        # Normalize to sum to 1
        var_props_series = var_props_series / var_props_series.sum()
        
        return var_props_series
    
    def get_sigsqs(self) -> np.ndarray:
        """
        Get the sigma squared values from the MCMC chain.
        
        This function returns the sigma squared values from the MCMC chain,
        which can be used to assess convergence.
        
        Returns:
            An array of sigma squared values.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Check if this is a regression model
        if self.pred_type != "regression":
            raise ValueError("This function is only available for regression models.")
        
        # Get sigma squared values from the Java backend
        sigsqs = get_gibbs_samples_sigsqs(self.java_bart_machine)
        
        return np.array(sigsqs)
    
    def calc_credible_intervals(self, new_data: pd.DataFrame, 
                               ci_conf: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate credible intervals for predictions.
        
        This function calculates credible intervals for predictions based on the
        posterior distribution of the BART model.
        
        Args:
            new_data: New data to predict on.
            ci_conf: Confidence level for credible intervals.
        
        Returns:
            A dictionary containing credible intervals.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Check if this is a regression model
        if self.pred_type != "regression":
            raise ValueError("This function is only available for regression models.")
        
        # Get posterior samples
        samples = self.predict(new_data, type="samples")
        
        # Calculate credible intervals
        alpha = 1 - ci_conf
        ci_lower = np.percentile(samples, alpha / 2 * 100, axis=1)
        ci_upper = np.percentile(samples, (1 - alpha / 2) * 100, axis=1)
        
        # Calculate mean prediction
        y_hat = np.mean(samples, axis=1)
        
        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "y_hat": y_hat
        }
    
    def calc_prediction_intervals(self, new_data: pd.DataFrame, 
                                 pi_conf: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals for predictions.
        
        This function calculates prediction intervals for predictions based on the
        posterior predictive distribution of the BART model.
        
        Args:
            new_data: New data to predict on.
            pi_conf: Confidence level for prediction intervals.
        
        Returns:
            A dictionary containing prediction intervals.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Check if this is a regression model
        if self.pred_type != "regression":
            raise ValueError("This function is only available for regression models.")
        
        # Get posterior samples
        samples = self.predict(new_data, type="samples")
        
        # Get sigma squared values
        sigsqs = self.get_sigsqs()
        
        # Calculate prediction intervals
        alpha = 1 - pi_conf
        
        # For each observation, generate samples from the posterior predictive distribution
        n_obs = len(new_data)
        n_samples = samples.shape[1]
        
        # Initialize arrays for prediction intervals
        pi_lower = np.zeros(n_obs)
        pi_upper = np.zeros(n_obs)
        
        # For each observation
        for i in range(n_obs):
            # Generate samples from the posterior predictive distribution
            # For each posterior sample, add noise from N(0, sigma^2)
            posterior_predictive_samples = samples[i, :] + np.random.normal(0, np.sqrt(sigsqs), n_samples)
            
            # Calculate prediction intervals
            pi_lower[i] = np.percentile(posterior_predictive_samples, alpha / 2 * 100)
            pi_upper[i] = np.percentile(posterior_predictive_samples, (1 - alpha / 2) * 100)
        
        # Calculate mean prediction
        y_hat = np.mean(samples, axis=1)
        
        return {
            "pi_lower": pi_lower,
            "pi_upper": pi_upper,
            "y_hat": y_hat
        }
    
    def summary(self) -> None:
        """
        Print a summary of the BART model.
        
        This function prints a summary of the BART model, including the number of trees,
        the number of MCMC iterations, and other model parameters.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Print model information
        print("BART Machine Summary")
        print("====================")
        print(f"Prediction type: {self.pred_type}")
        print(f"Number of trees: {self.num_trees}")
        print(f"Number of burn-in iterations: {self.num_burn_in}")
        print(f"Number of MCMC iterations after burn-in: {self.num_iterations_after_burn_in}")
        print(f"Number of predictors: {self.p}")
        print(f"Number of observations: {self.n}")
        print(f"Time to build: {self.time_to_build:.2f} seconds")
        
        # Print model parameters
        print("\nModel Parameters")
        print("---------------")
        print(f"alpha: {self.alpha}")
        print(f"beta: {self.beta}")
        print(f"k: {self.k}")
        print(f"q: {self.q}")
        print(f"nu: {self.nu}")
        
        # Print variable importance
        print("\nVariable Importance (top 10)")
        print("---------------------------")
        var_props = self.get_var_props_over_chain()
        var_props = var_props.sort_values(ascending=False)
        for i, (var, prop) in enumerate(var_props.items()):
            if i >= 10:
                break
            print(f"{var}: {prop:.4f}")
        
        # Print model fit
        print("\nModel Fit")
        print("---------")
        if self.pred_type == "regression":
            print(f"RMSE (training): {self.rmse_train:.4f}")
            print(f"Pseudo R^2 (training): {self.PseudoRsq:.4f}")
        else:
            print(f"Misclassification error (training): {self.misclassification_error:.4f}")
            print("\nConfusion Matrix (training)")
            print(self.confusion_matrix)
    
    def __str__(self) -> str:
        """
        Return a string representation of the BART model.
        
        Returns:
            A string representation of the BART model.
        """
        if self.java_bart_machine is None:
            return "BART Machine (not built)"
        else:
            return f"BART Machine ({self.pred_type}, {self.num_trees} trees, {self.n} observations, {self.p} predictors)"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the BART model.
        
        Returns:
            A string representation of the BART model.
        """
        return self.__str__()


def bart_machine(X=None, y=None, Xy=None, 
                num_trees: int = 50, 
                num_burn_in: int = 250, 
                num_iterations_after_burn_in: int = 1000,
                alpha: float = 0.95, 
                beta: float = 2.0, 
                k: float = 2.0, 
                q: float = 0.9, 
                nu: float = 3.0,
                prob_rule_class: float = 0.5, 
                prob_rule_avg: float = 0.5, 
                prob_split_not_decision: float = 0.0, 
                prob_rule_quad: float = 0.1,
                use_missing_data: bool = False, 
                use_missing_data_dummies_as_covars: bool = False, 
                impute_missingness_with_rf_impute: bool = False,
                replace_missing_data_with_x_j_bar: bool = False, 
                impute_missingness_with_x_j_bar_for_lm: bool = False,
                verbose: bool = False, 
                seed: Optional[int] = None, 
                cov_prior_vec: Optional[np.ndarray] = None,
                pred_type: str = "regression") -> BartMachine:
    """
    Create and build a BART machine model.
    
    This function creates a BartMachine object and builds the model.
    
    Args:
        X: The predictor variables.
        y: The response variable.
        Xy: Combined predictor and response variables.
        num_trees: Number of trees in the BART model.
        num_burn_in: Number of burn-in MCMC iterations.
        num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
        alpha: Prior parameter for tree structure.
        beta: Prior parameter for tree structure.
        k: Prior parameter for leaf node parameters.
        q: Prior parameter for tree structure.
        nu: Prior parameter for error variance.
        prob_rule_class: Probability of using a classification rule.
        prob_rule_avg: Probability of using an average rule.
        prob_split_not_decision: Probability of splitting on a non-decision rule.
        prob_rule_quad: Probability of using a quadratic rule.
        use_missing_data: Whether to use missing data in the model.
        use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
        impute_missingness_with_rf_impute: Whether to impute missing values with random forest.
        replace_missing_data_with_x_j_bar: Whether to replace missing data with column means.
        impute_missingness_with_x_j_bar_for_lm: Whether to impute missing values with column means for linear model.
        verbose: Whether to print verbose output.
        seed: Random seed for reproducibility.
        cov_prior_vec: Prior vector for covariate selection.
        pred_type: Type of prediction ("regression" or "classification").
    
    Returns:
        A built BartMachine object.
    """
    # Create a BartMachine object
    model = BartMachine(
        X=X,
        y=y,
        Xy=Xy,
        num_trees=num_trees,
        num_burn_in=num_burn_in,
        num_iterations_after_burn_in=num_iterations_after_burn_in,
        alpha=alpha,
        beta=beta,
        k=k,
        q=q,
        nu=nu,
        prob_rule_class=prob_rule_class,
        prob_rule_avg=prob_rule_avg,
        prob_split_not_decision=prob_split_not_decision,
        prob_rule_quad=prob_rule_quad,
        use_missing_data=use_missing_data,
        use_missing_data_dummies_as_covars=use_missing_data_dummies_as_covars,
        impute_missingness_with_rf_impute=impute_missingness_with_rf_impute,
        replace_missing_data_with_x_j_bar=replace_missing_data_with_x_j_bar,
        impute_missingness_with_x_j_bar_for_lm=impute_missingness_with_x_j_bar_for_lm,
        verbose=verbose,
        seed=seed,
        cov_prior_vec=cov_prior_vec,
        pred_type=pred_type
    )
    
    # Build the model if data is provided
    if (X is not None and y is not None) or Xy is not None:
        model.build()
    
    return model
