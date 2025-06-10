"""
Cross-validation utilities for bartMachine.

This module provides functions for cross-validating BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bart_package_cross_validation.R' 
    in the original R package.

Role in Port:
    This module handles cross-validation for BART models, including hyperparameter tuning
    and model selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Any, Tuple
import logging
from sklearn.model_selection import KFold
import time

from .bart_package_builders import BartMachine, bart_machine

# Set up logging
logger = logging.getLogger(__name__)

def bart_machine_cv(X=None, y=None, Xy=None, 
                   num_tree_cvs: List[int] = [50, 200],
                   k_cvs: List[float] = [2, 3, 5],
                   nu_q_cvs: Optional[List[Tuple[float, float]]] = None,
                   k_folds: int = 5, 
                   folds_vec: Optional[np.ndarray] = None,    
                   verbose: bool = False, **kwargs) -> BartMachine:
    """
    Create and build a BART machine model with cross-validated hyperparameters.
    
    This function creates a BartMachine object with cross-validated hyperparameters.
    It performs k-fold cross-validation to find the best hyperparameters for the model.
    
    Args:
        X: The predictor variables.
        y: The response variable.
        Xy: Combined predictor and response variables.
        num_tree_cvs: List of number of trees to try.
        k_cvs: List of k values to try.
        nu_q_cvs: List of (nu, q) tuples to try.
        k_folds: Number of folds for cross-validation.
        folds_vec: Vector of fold indices.
        verbose: Whether to print verbose output.
        **kwargs: Additional arguments to pass to BartMachine.
    
    Returns:
        A built BartMachine object with cross-validated hyperparameters.
    """
    # Check input parameters
    if (X is None and Xy is None) or (y is None and Xy is None):
        raise ValueError("You need to give bartMachine a training set either by specifying X and y or by specifying a matrix Xy which contains the response named 'y'.")
    elif X is not None and y is not None and Xy is not None:
        raise ValueError("You cannot specify both X,y and Xy simultaneously.")
    elif X is None and y is None:  # they specified Xy, so now just pull out X,y
        # First ensure it's a dataframe
        if not isinstance(Xy, pd.DataFrame):
            raise ValueError("The training data Xy must be a data frame.")
        
        y = Xy.iloc[:, -1]
        X = Xy.iloc[:, :-1]
    
    # Make sure X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("The training data X must be a data frame.")
    
    # Check if nu_q_cvs is None
    if nu_q_cvs is None:
        nu_q_cvs = [(3, 0.9), (3, 0.99), (10, 0.75)]
    
    # Create a grid of hyperparameters to try
    param_grid = []
    for num_trees in num_tree_cvs:
        for k in k_cvs:
            for nu, q in nu_q_cvs:
                param_grid.append({
                    "num_trees": num_trees,
                    "k": k,
                    "nu": nu,
                    "q": q
                })
    
    # Create folds for cross-validation
    if folds_vec is None:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        folds_vec = np.zeros(len(X))
        for i, (_, test_idx) in enumerate(kf.split(X)):
            folds_vec[test_idx] = i
    
    # Initialize arrays for cross-validation results
    cv_results = []
    
    # For each set of hyperparameters
    for params in param_grid:
        if verbose:
            print(f"Cross-validating with parameters: {params}")
        
        # Initialize array for fold errors
        fold_errors = np.zeros(k_folds)
        
        # For each fold
        for fold in range(k_folds):
            # Get training and validation indices
            train_idx = folds_vec != fold
            val_idx = folds_vec == fold
            
            # Get training and validation data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Build the model
            model = bart_machine(
                X=X_train,
                y=y_train,
                num_trees=params["num_trees"],
                k=params["k"],
                nu=params["nu"],
                q=params["q"],
                verbose=False,
                **kwargs
            )
            
            # Make predictions
            y_hat = model.predict(X_val)
            
            # Calculate error
            if model.pred_type == "regression":
                fold_errors[fold] = np.mean((y_val - y_hat) ** 2)
            else:
                fold_errors[fold] = np.mean(y_val != y_hat)
        
        # Calculate mean error
        mean_error = np.mean(fold_errors)
        
        # Store results
        cv_results.append({
            "params": params,
            "mean_error": mean_error,
            "fold_errors": fold_errors
        })
        
        if verbose:
            print(f"Mean error: {mean_error:.4f}")
    
    # Find the best hyperparameters
    best_idx = np.argmin([result["mean_error"] for result in cv_results])
    best_params = cv_results[best_idx]["params"]
    
    if verbose:
        print(f"Best hyperparameters: {best_params}")
        print(f"Best mean error: {cv_results[best_idx]['mean_error']:.4f}")
    
    # Build the final model with the best hyperparameters
    final_model = bart_machine(
        X=X,
        y=y,
        num_trees=best_params["num_trees"],
        k=best_params["k"],
        nu=best_params["nu"],
        q=best_params["q"],
        verbose=verbose,
        **kwargs
    )
    
    # Store cross-validation results
    final_model.cv_results = cv_results
    final_model.best_params = best_params
    
    return final_model

def k_fold_cv(bart_machine: BartMachine, 
             k_folds: int = 5, 
             folds_vec: Optional[np.ndarray] = None,
             verbose: bool = False) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation on a BART model.
    
    This function performs k-fold cross-validation on a BART model to estimate
    its generalization performance.
    
    Args:
        bart_machine: The BART machine model.
        k_folds: Number of folds for cross-validation.
        folds_vec: Vector of fold indices.
        verbose: Whether to print verbose output.
    
    Returns:
        A dictionary containing cross-validation results.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get the data
    X = bart_machine.X
    y = bart_machine.y
    
    # Create folds for cross-validation
    if folds_vec is None:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        folds_vec = np.zeros(len(X))
        for i, (_, test_idx) in enumerate(kf.split(X)):
            folds_vec[test_idx] = i
    
    # Initialize arrays for cross-validation results
    fold_errors = np.zeros(k_folds)
    fold_predictions = []
    
    # For each fold
    for fold in range(k_folds):
        if verbose:
            print(f"Cross-validating fold {fold+1}/{k_folds}")
        
        # Get training and validation indices
        train_idx = folds_vec != fold
        val_idx = folds_vec == fold
        
        # Get training and validation data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Build the model
        model = bart_machine(
            X=X_train,
            y=y_train,
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
        
        # Make predictions
        y_hat = model.predict(X_val)
        
        # Store predictions
        fold_predictions.append({
            "fold": fold,
            "y_val": y_val,
            "y_hat": y_hat
        })
        
        # Calculate error
        if model.pred_type == "regression":
            fold_errors[fold] = np.mean((y_val - y_hat) ** 2)
        else:
            fold_errors[fold] = np.mean(y_val != y_hat)
    
    # Calculate mean error
    mean_error = np.mean(fold_errors)
    
    if verbose:
        print(f"Mean error: {mean_error:.4f}")
    
    # Return results
    return {
        "fold_errors": fold_errors,
        "mean_error": mean_error,
        "fold_predictions": fold_predictions
    }

def plot_cv_results(bart_machine: BartMachine, 
                   param_name: str = "num_trees",
                   log_scale: bool = False) -> plt.Figure:
    """
    Plot cross-validation results.
    
    This function plots the cross-validation results for a BART model,
    showing the mean error for different values of a hyperparameter.
    
    Args:
        bart_machine: The BART machine model.
        param_name: The name of the hyperparameter to plot.
        log_scale: Whether to use a log scale for the x-axis.
    
    Returns:
        The matplotlib figure containing the plot.
    """
    # Check if the model has cross-validation results
    if not hasattr(bart_machine, "cv_results"):
        raise ValueError("The model does not have cross-validation results.")
    
    # Get unique values of the parameter
    param_values = []
    mean_errors = []
    
    # For each set of hyperparameters
    for result in bart_machine.cv_results:
        if param_name in result["params"]:
            param_values.append(result["params"][param_name])
            mean_errors.append(result["mean_error"])
    
    # Create a DataFrame with the results
    cv_df = pd.DataFrame({
        param_name: param_values,
        "mean_error": mean_errors
    })
    
    # Group by parameter value and calculate mean error
    cv_df = cv_df.groupby(param_name).mean().reset_index()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the results
    ax.plot(cv_df[param_name], cv_df["mean_error"], "o-")
    
    # Add labels and title
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean Error")
    ax.set_title(f"Cross-Validation Results for {param_name}")
    
    # Use log scale if requested
    if log_scale:
        ax.set_xscale("log")
    
    # Add a vertical line at the best parameter value
    best_param_value = bart_machine.best_params[param_name]
    ax.axvline(best_param_value, color="red", linestyle="--")
    
    # Add a text annotation for the best parameter value
    ax.text(best_param_value, ax.get_ylim()[1] * 0.9, f"Best: {best_param_value}",
           ha="center", va="top", color="red")
    
    return fig

def get_cv_error(bart_machine: BartMachine) -> float:
    """
    Get the cross-validation error for a BART model.
    
    This function returns the mean cross-validation error for a BART model.
    
    Args:
        bart_machine: The BART machine model.
    
    Returns:
        The mean cross-validation error.
    """
    # Check if the model has cross-validation results
    if not hasattr(bart_machine, "cv_results"):
        raise ValueError("The model does not have cross-validation results.")
    
    # Find the best hyperparameters
    best_idx = np.argmin([result["mean_error"] for result in bart_machine.cv_results])
    
    # Return the mean error
    return bart_machine.cv_results[best_idx]["mean_error"]

def get_cv_predictions(bart_machine: BartMachine) -> pd.DataFrame:
    """
    Get the cross-validation predictions for a BART model.
    
    This function returns the cross-validation predictions for a BART model.
    
    Args:
        bart_machine: The BART machine model.
    
    Returns:
        A DataFrame containing the cross-validation predictions.
    """
    # Check if the model has cross-validation results
    if not hasattr(bart_machine, "cv_results"):
        raise ValueError("The model does not have cross-validation results.")
    
    # Initialize lists for predictions
    fold_list = []
    y_val_list = []
    y_hat_list = []
    
    # For each fold
    for fold_result in bart_machine.cv_results[0]["fold_predictions"]:
        fold = fold_result["fold"]
        y_val = fold_result["y_val"]
        y_hat = fold_result["y_hat"]
        
        # Append to lists
        fold_list.extend([fold] * len(y_val))
        y_val_list.extend(y_val)
        y_hat_list.extend(y_hat)
    
    # Create a DataFrame
    cv_df = pd.DataFrame({
        "fold": fold_list,
        "y_val": y_val_list,
        "y_hat": y_hat_list
    })
    
    return cv_df
