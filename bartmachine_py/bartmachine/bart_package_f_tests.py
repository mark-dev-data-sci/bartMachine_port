"""
F-test utilities for bartMachine.

This module provides functions for performing F-tests with BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bart_package_f_tests.R' 
    in the original R package.

Role in Port:
    This module implements F-tests for variable importance and significance testing
    in BART models. It provides statistical tests to assess the importance of variables
    and their interactions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple
import logging
from scipy import stats
from .bart_package_builders import bart_machine as bart_machine_func

# Set up logging
logger = logging.getLogger(__name__)

def cov_importance_test(bart_machine: Any,
                       covariates: Optional[Union[str, List[str]]] = None,
                       num_permutation_samples: int = 100,
                       plot: bool = True) -> Dict[str, Any]:
    """
    Test the importance of covariates in a BART model.
    
    This function tests the importance of one or more covariates by permuting them
    and measuring the impact on model performance. If no covariates are specified,
    an omnibus test is performed by permuting all covariates.
    
    Args:
        bart_machine: The BART machine model.
        covariates: Covariate or list of covariates to test. If None, all covariates are tested.
        num_permutation_samples: Number of permutation samples.
        plot: Whether to plot the results.
    
    Returns:
        A dictionary containing covariate importance test results.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get all covariates
    all_covariates = bart_machine.training_data_features
    
    # Set up title for output
    if covariates is None:
        title = "bartMachine omnibus test for covariate importance\n"
    elif isinstance(covariates, (list, tuple)) and len(covariates) <= 3:
        cov_names = ", ".join(covariates)
        title = f"bartMachine test for importance of covariate(s): {cov_names}\n"
    else:
        if isinstance(covariates, (list, tuple)):
            title = f"bartMachine test for importance of {len(covariates)} covariates\n"
        else:
            title = f"bartMachine test for importance of covariate: {covariates}\n"
    
    print(title)
    
    # Convert single covariate to list
    if isinstance(covariates, str):
        covariates = [covariates]
    
    # Get the observed error estimate
    if bart_machine.pred_type == "regression":
        observed_error_estimate = bart_machine.PseudoRsq
    else:
        observed_error_estimate = bart_machine.misclassification_error
    
    # Initialize array for permutation samples of error
    permutation_samples_of_error = np.zeros(num_permutation_samples)
    
    # For each permutation sample
    for nsim in range(num_permutation_samples):
        print(".", end="")
        if (nsim + 1) % 50 == 0:
            print("\n", end="")
        
        # Omnibus F-like test - just permute y (same as permuting ALL the columns of X and it's faster)
        if covariates is None:
            # Create a new BART model with permuted y
            bart_machine_samp = bart_machine_func(
                X=bart_machine.X,
                y=np.random.permutation(bart_machine.y),
                num_trees=bart_machine.num_trees,
                num_burn_in=bart_machine.num_burn_in,
                num_iterations_after_burn_in=bart_machine.num_iterations_after_burn_in,
                alpha=bart_machine.alpha,
                beta=bart_machine.beta,
                k=bart_machine.k,
                q=bart_machine.q,
                nu=bart_machine.nu,
                verbose=False
            )
        # Partial F-like test - permute the columns that we're interested in seeing if they matter
        else:
            # Create a copy of the design matrix
            X_samp = bart_machine.X.copy()
            
            # Permute the specified covariates
            for cov in covariates:
                if cov in X_samp.columns:
                    X_samp[cov] = np.random.permutation(X_samp[cov].values)
            
            # Create a new BART model with permuted X
            bart_machine_samp = bart_machine_func(
                X=X_samp,
                y=bart_machine.y,
                num_trees=bart_machine.num_trees,
                num_burn_in=bart_machine.num_burn_in,
                num_iterations_after_burn_in=bart_machine.num_iterations_after_burn_in,
                alpha=bart_machine.alpha,
                beta=bart_machine.beta,
                k=bart_machine.k,
                q=bart_machine.q,
                nu=bart_machine.nu,
                verbose=False
            )
        
        # Record permutation result
        if bart_machine.pred_type == "regression":
            permutation_samples_of_error[nsim] = bart_machine_samp.PseudoRsq
        else:
            permutation_samples_of_error[nsim] = bart_machine_samp.misclassification_error
    
    print("\n")
    
    # Compute p-value
    if bart_machine.pred_type == "regression":
        # For regression, lower Pseudo-R^2 is worse
        p_value = np.sum(observed_error_estimate < permutation_samples_of_error) / (num_permutation_samples + 1)
    else:
        # For classification, higher misclassification error is worse
        p_value = np.sum(observed_error_estimate > permutation_samples_of_error) / (num_permutation_samples + 1)
    
    # Plot the results if requested
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the permutation errors
        ax.hist(permutation_samples_of_error, bins=int(num_permutation_samples/10), alpha=0.5)
        
        # Add a vertical line at the observed error estimate
        ax.axvline(observed_error_estimate, color='blue', linewidth=3)
        
        # Add labels and title
        if bart_machine.pred_type == "regression":
            xlabel = f"permutation samples\np-value = {p_value:.3f}"
            title_suffix = "Null Samples of Pseudo-R^2's"
        else:
            xlabel = f"permutation samples\np-value = {p_value:.3f}"
            title_suffix = "Null Samples of Misclassification Errors"
        
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title} {title_suffix}")
        
        # Set x-limits
        ax.set_xlim(min(permutation_samples_of_error, 0.99 * observed_error_estimate),
                   max(permutation_samples_of_error, 1.01 * observed_error_estimate))
        
        plt.tight_layout()
        plt.show()
    
    print(f"p_val = {p_value}")
    
    # Return the results
    return {
        "permutation_samples_of_error": permutation_samples_of_error,
        "observed_error_estimate": observed_error_estimate,
        "p_value": p_value
    }

def linearity_test(lin_mod: Optional[Any] = None,
                  X: Optional[pd.DataFrame] = None,
                  y: Optional[Union[pd.Series, np.ndarray]] = None,
                  num_permutation_samples: int = 100,
                  plot: bool = True,
                  **kwargs) -> Dict[str, Any]:
    """
    Test for linearity in a BART model.
    
    This function tests for linearity by fitting a linear model and then using BART
    to model the residuals. If the residuals can be modeled by BART, then the
    relationship is nonlinear.
    
    Args:
        lin_mod: A fitted linear model. If None, a linear model is fit using X and y.
        X: The predictor variables. Required if lin_mod is None.
        y: The response variable. Required if lin_mod is None.
        num_permutation_samples: Number of permutation samples.
        plot: Whether to plot the results.
        **kwargs: Additional arguments to pass to bartMachine.
    
    Returns:
        A dictionary containing linearity test results.
    """
    # Check if we need to fit a linear model
    if lin_mod is None:
        if X is None or y is None:
            raise ValueError("If lin_mod is None, X and y must be provided.")
        
        # Fit a linear model
        from sklearn.linear_model import LinearRegression
        lin_mod = LinearRegression().fit(X, y)
    
    # Get the predictions from the linear model
    if X is None:
        raise ValueError("X must be provided to make predictions.")
    
    y_hat = lin_mod.predict(X)
    
    # Create a BART model for the residuals
    bart_mod = bart_machine_func(X=X, y=y - y_hat, **kwargs)
    
    # Test if BART can model the residuals
    return cov_importance_test(bart_mod, num_permutation_samples=num_permutation_samples, plot=plot)

def bart_machine_f_test(bart_machine: Any, covariates: List[str], 
                       test_data: Optional[pd.DataFrame] = None,
                       num_permute_samples: int = 100,
                       alpha: float = 0.05,
                       seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform an F-test for variable importance in a BART model.
    
    Args:
        bart_machine: The BART machine model.
        covariates: List of covariates to test.
        test_data: Test data to use for the test. If None, uses training data.
        num_permute_samples: Number of permutation samples to use.
        alpha: Significance level.
        seed: Random seed.
    
    Returns:
        A dictionary containing the test results.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Use cov_importance_test to perform the test
    result = cov_importance_test(
        bart_machine=bart_machine,
        covariates=covariates,
        num_permutation_samples=num_permute_samples,
        plot=False
    )
    
    # Return the results
    return result
