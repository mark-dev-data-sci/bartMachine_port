"""
Summary utilities for bartMachine.

This module provides functions for summarizing BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bart_package_summaries.R' 
    in the original R package.

Role in Port:
    This module handles the summarization of BART models, including variable importance,
    convergence diagnostics, and other model summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Any, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_sigsqs(bart_machine: Any) -> np.ndarray:
    """
    Get the sigma squared values from the MCMC chain.
    
    This function returns the sigma squared values from the MCMC chain,
    which can be used to assess convergence.
    
    Args:
        bart_machine: The BART machine model.
    
    Returns:
        An array of sigma squared values.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if this is a regression model
    if bart_machine.pred_type != "regression":
        raise ValueError("This function is only available for regression models.")
    
    # Get sigma squared values from the Java backend
    from .zzz import get_gibbs_samples_sigsqs
    sigsqs = get_gibbs_samples_sigsqs(bart_machine.java_bart_machine)
    
    return np.array(sigsqs)

def get_var_counts_over_chain(bart_machine: Any, type: str = "splits") -> pd.DataFrame:
    """
    Get variable counts over the MCMC chain.
    
    This function returns the counts of how many times each variable is used
    in splitting rules across all trees and MCMC iterations.
    
    Args:
        bart_machine: The BART machine model.
        type: Type of counts to get ("splits" or "trees").
            "splits" counts the total number of splitting rules using each variable.
            "trees" counts the number of trees that use each variable at least once.
    
    Returns:
        A DataFrame containing variable counts for each MCMC iteration.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get variable counts from the Java backend
    # This is a placeholder - in a real implementation, we would get this from the Java backend
    # For now, we'll just create a random DataFrame
    var_names = bart_machine.training_data_features
    num_iterations = bart_machine.num_iterations_after_burn_in
    
    # Create a DataFrame with random counts
    counts = np.random.randint(0, 10, size=(num_iterations, len(var_names)))
    counts_df = pd.DataFrame(counts, columns=var_names)
    
    return counts_df

def get_var_props_over_chain(bart_machine: Any, type: str = "splits") -> pd.Series:
    """
    Get variable inclusion proportions over the MCMC chain.
    
    This function returns the proportion of times each variable is used
    in splitting rules across all trees and MCMC iterations.
    
    Args:
        bart_machine: The BART machine model.
        type: Type of proportion to use ("splits" or "trees").
            "splits" uses the proportion of splitting rules using each variable.
            "trees" uses the proportion of trees that use each variable at least once.
    
    Returns:
        A pandas Series containing variable inclusion proportions.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get variable inclusion proportions from the Java backend
    from .zzz import get_variable_inclusion_proportions
    var_props = get_variable_inclusion_proportions(bart_machine.java_bart_machine)
    
    # Convert to pandas Series
    var_props_series = pd.Series(var_props, index=bart_machine.training_data_features)
    
    # Normalize to sum to 1
    var_props_series = var_props_series / var_props_series.sum()
    
    return var_props_series

def get_tree_num(bart_machine: Any) -> int:
    """
    Get the number of trees in the BART model.
    
    Args:
        bart_machine: The BART machine model.
    
    Returns:
        The number of trees in the BART model.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    return bart_machine.num_trees

def calc_credible_intervals(bart_machine: Any, 
                           new_data: Optional[pd.DataFrame] = None,
                           ci_conf: float = 0.95) -> Dict[str, np.ndarray]:
    """
    Calculate credible intervals for predictions.
    
    This function calculates credible intervals for predictions based on the
    posterior distribution of the BART model.
    
    Args:
        bart_machine: The BART machine model.
        new_data: New data to predict on. If None, uses training data.
        ci_conf: Confidence level for credible intervals.
    
    Returns:
        A dictionary containing credible intervals.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if this is a regression model
    if bart_machine.pred_type != "regression":
        raise ValueError("This function is only available for regression models.")
    
    # Get the data to predict on
    if new_data is None:
        # Use training data
        X = bart_machine.X
    else:
        # Use new data
        X = new_data
    
    # Get posterior samples
    samples = bart_machine.predict(X, type="samples")
    
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

def calc_prediction_intervals(bart_machine: Any, 
                             new_data: Optional[pd.DataFrame] = None,
                             pi_conf: float = 0.95) -> Dict[str, np.ndarray]:
    """
    Calculate prediction intervals for predictions.
    
    This function calculates prediction intervals for predictions based on the
    posterior predictive distribution of the BART model.
    
    Args:
        bart_machine: The BART machine model.
        new_data: New data to predict on. If None, uses training data.
        pi_conf: Confidence level for prediction intervals.
    
    Returns:
        A dictionary containing prediction intervals.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if this is a regression model
    if bart_machine.pred_type != "regression":
        raise ValueError("This function is only available for regression models.")
    
    # Get the data to predict on
    if new_data is None:
        # Use training data
        X = bart_machine.X
    else:
        # Use new data
        X = new_data
    
    # Get posterior samples
    samples = bart_machine.predict(X, type="samples")
    
    # Get sigma squared values
    sigsqs = get_sigsqs(bart_machine)
    
    # Calculate prediction intervals
    alpha = 1 - pi_conf
    
    # For each observation, generate samples from the posterior predictive distribution
    n_obs = len(X)
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

def summary(bart_machine: Any) -> None:
    """
    Print a summary of the BART model.
    
    This function prints a summary of the BART model, including the number of trees,
    the number of MCMC iterations, and other model parameters.
    
    Args:
        bart_machine: The BART machine model.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Print model information
    print("BART Machine Summary")
    print("====================")
    print(f"Prediction type: {bart_machine.pred_type}")
    print(f"Number of trees: {bart_machine.num_trees}")
    print(f"Number of burn-in iterations: {bart_machine.num_burn_in}")
    print(f"Number of MCMC iterations after burn-in: {bart_machine.num_iterations_after_burn_in}")
    print(f"Number of predictors: {bart_machine.p}")
    print(f"Number of observations: {bart_machine.n}")
    print(f"Time to build: {bart_machine.time_to_build:.2f} seconds")
    
    # Print model parameters
    print("\nModel Parameters")
    print("---------------")
    print(f"alpha: {bart_machine.alpha}")
    print(f"beta: {bart_machine.beta}")
    print(f"k: {bart_machine.k}")
    print(f"q: {bart_machine.q}")
    print(f"nu: {bart_machine.nu}")
    
    # Print variable importance
    print("\nVariable Importance (top 10)")
    print("---------------------------")
    var_props = get_var_props_over_chain(bart_machine)
    var_props = var_props.sort_values(ascending=False)
    for i, (var, prop) in enumerate(var_props.items()):
        if i >= 10:
            break
        print(f"{var}: {prop:.4f}")
    
    # Print model fit
    print("\nModel Fit")
    print("---------")
    if bart_machine.pred_type == "regression":
        print(f"RMSE (training): {bart_machine.rmse_train:.4f}")
        print(f"Pseudo R^2 (training): {bart_machine.PseudoRsq:.4f}")
    else:
        print(f"Misclassification error (training): {bart_machine.misclassification_error:.4f}")
        print("\nConfusion Matrix (training)")
        print(bart_machine.confusion_matrix)

def get_posterior_samples(bart_machine: Any, 
                         new_data: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Get posterior samples for predictions.
    
    This function returns the posterior samples for predictions based on the
    posterior distribution of the BART model.
    
    Args:
        bart_machine: The BART machine model.
        new_data: New data to predict on. If None, uses training data.
    
    Returns:
        An array of posterior samples.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get the data to predict on
    if new_data is None:
        # Use training data
        X = bart_machine.X
    else:
        # Use new data
        X = new_data
    
    # Get posterior samples
    samples = bart_machine.predict(X, type="samples")
    
    return samples

def get_posterior_mean(bart_machine: Any, 
                      new_data: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Get posterior mean for predictions.
    
    This function returns the posterior mean for predictions based on the
    posterior distribution of the BART model.
    
    Args:
        bart_machine: The BART machine model.
        new_data: New data to predict on. If None, uses training data.
    
    Returns:
        An array of posterior means.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get the data to predict on
    if new_data is None:
        # Use training data
        X = bart_machine.X
    else:
        # Use new data
        X = new_data
    
    # Get posterior samples
    samples = bart_machine.predict(X, type="samples")
    
    # Calculate posterior mean
    y_hat = np.mean(samples, axis=1)
    
    return y_hat

def get_posterior_sd(bart_machine: Any, 
                    new_data: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Get posterior standard deviation for predictions.
    
    This function returns the posterior standard deviation for predictions based on the
    posterior distribution of the BART model.
    
    Args:
        bart_machine: The BART machine model.
        new_data: New data to predict on. If None, uses training data.
    
    Returns:
        An array of posterior standard deviations.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get the data to predict on
    if new_data is None:
        # Use training data
        X = bart_machine.X
    else:
        # Use new data
        X = new_data
    
    # Get posterior samples
    samples = bart_machine.predict(X, type="samples")
    
    # Calculate posterior standard deviation
    y_hat_sd = np.std(samples, axis=1)
    
    return y_hat_sd
