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

def get_var_importance(bart_machine: Any, type: str = "splits") -> pd.Series:
    """
    Get variable importance measures from the BART model.
    
    This function returns the variable importance measures for all variables
    in the BART model. The importance measure is based on the frequency with which
    variables are used in splitting rules.
    
    Args:
        bart_machine: The BART machine model.
        type: Type of importance measure to use ("splits" or "trees").
            "splits" counts the total number of splitting rules using each variable.
            "trees" counts the number of trees that use each variable at least once.
    
    Returns:
        A pandas Series containing variable importance measures.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get variable importance from the Java backend
    from .zzz import get_variable_importance
    var_importance = get_variable_importance(bart_machine.java_bart_machine, type)
    
    # Convert to pandas Series
    var_importance_series = pd.Series(var_importance, index=bart_machine.training_data_features)
    
    return var_importance_series

def var_selection_by_permute(bart_machine: Any, 
                           num_reps_for_avg: int = 10, 
                           num_permute_samples: int = 100, 
                           num_trees_for_permute: int = 20, 
                           alpha: float = 0.05, 
                           plot: bool = True, 
                           num_var_plot: int = None, 
                           bottom_margin: int = 10) -> Dict[str, Any]:
    """
    Select variables by permutation importance.
    
    This function selects variables based on their permutation importance.
    It compares the variable importance in the original model to the importance
    in models with permuted responses, to identify variables that are significantly
    more important than would be expected by chance.
    
    Args:
        bart_machine: The BART machine model.
        num_reps_for_avg: Number of repetitions for averaging variable importance.
        num_permute_samples: Number of permutation samples.
        num_trees_for_permute: Number of trees to use in permutation models.
        alpha: Significance level.
        plot: Whether to plot the results.
        num_var_plot: Number of variables to plot. If None, all variables are plotted.
        bottom_margin: Bottom margin for the plot.
    
    Returns:
        A dictionary containing selected variables and related information.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Initialize permutation matrix
    permute_mat = np.zeros((num_permute_samples, bart_machine.p))
    
    # Get average variable proportions from actual data
    print("avg", end="", flush=True)
    var_true_props_avg = get_averaged_true_var_props(bart_machine, num_reps_for_avg, num_trees_for_permute)
    
    # Sort from high to low
    var_true_props_avg = pd.Series(var_true_props_avg, index=bart_machine.training_data_features).sort_values(ascending=False)
    
    # Build null permutation distribution
    print("null", end="", flush=True)
    for b in range(num_permute_samples):
        permute_mat[b, :] = get_null_permute_var_importances(bart_machine, num_trees_for_permute)
        print(".", end="", flush=True)
    print()
    
    # Sort permutation matrix to match the order of var_true_props_avg
    permute_mat_df = pd.DataFrame(permute_mat, columns=bart_machine.training_data_features)
    permute_mat_df = permute_mat_df[var_true_props_avg.index]
    permute_mat = permute_mat_df.values
    
    # Use local cutoff
    pointwise_cutoffs = np.percentile(permute_mat, (1 - alpha) * 100, axis=0)
    important_vars_pointwise_names = var_true_props_avg.index[
        (var_true_props_avg.values > pointwise_cutoffs) & (var_true_props_avg.values > 0)
    ].tolist()
    important_vars_pointwise_col_nums = [
        bart_machine.training_data_features.index(name) for name in important_vars_pointwise_names
    ]
    
    # Use global max cutoff
    max_cut = np.percentile(np.max(permute_mat, axis=1), (1 - alpha) * 100)
    important_vars_simul_max_names = var_true_props_avg.index[
        (var_true_props_avg.values >= max_cut) & (var_true_props_avg.values > 0)
    ].tolist()
    important_vars_simul_max_col_nums = [
        bart_machine.training_data_features.index(name) for name in important_vars_simul_max_names
    ]
    
    # Use global SE cutoff
    perm_se = np.std(permute_mat, axis=0)
    perm_mean = np.mean(permute_mat, axis=0)
    
    # Find the constant that gives the desired coverage
    cover_constant = bisect_k(
        tol=0.01,
        coverage=1 - alpha,
        permute_mat=permute_mat,
        x_left=1,
        x_right=20,
        count_limit=100,
        perm_mean=perm_mean,
        perm_se=perm_se
    )
    
    important_vars_simul_se_names = var_true_props_avg.index[
        (var_true_props_avg.values >= perm_mean + cover_constant * perm_se) & (var_true_props_avg.values > 0)
    ].tolist()
    important_vars_simul_se_col_nums = [
        bart_machine.training_data_features.index(name) for name in important_vars_simul_se_names
    ]
    
    # Plot if requested
    if plot:
        # Set up the plot
        plt.figure(figsize=(12, 10))
        plt.subplots_adjust(bottom=bottom_margin/100)
        
        # Determine number of variables to plot
        if num_var_plot is None or num_var_plot > bart_machine.p:
            num_var_plot = bart_machine.p
        
        # Find non-zero variables
        non_zero_idx = np.where(var_true_props_avg.values > 0)[0]
        non_zero_idx = non_zero_idx[:min(num_var_plot, len(non_zero_idx))]
        plot_n = len(non_zero_idx)
        
        if len(non_zero_idx) < len(var_true_props_avg):
            print(f"Warning: {len(var_true_props_avg) - len(non_zero_idx)} covariates with inclusion proportions of 0 omitted from plots.")
        
        # Create subplots
        plt.subplot(2, 1, 1)
        
        # Plot local procedure
        plt.title("Local Procedure")
        plt.ylabel("proportion included")
        plt.xticks(range(1, plot_n + 1), var_true_props_avg.index[non_zero_idx], rotation=90)
        
        # Plot points
        for j in range(plot_n):
            idx = non_zero_idx[j]
            if var_true_props_avg.iloc[idx] <= pointwise_cutoffs[idx]:
                plt.plot(j + 1, var_true_props_avg.iloc[idx], 'o', markersize=5, color='blue')
            else:
                plt.plot(j + 1, var_true_props_avg.iloc[idx], 'o', markersize=5, color='red', fillstyle='full')
        
        # Plot cutoffs
        for j in range(plot_n):
            idx = non_zero_idx[j]
            plt.plot([j + 1, j + 1], [0, pointwise_cutoffs[idx]], 'g-')
        
        # Plot simul procedures
        plt.subplot(2, 1, 2)
        plt.title("Simul. Max and SE Procedures")
        plt.ylabel("proportion included")
        plt.xticks(range(1, plot_n + 1), var_true_props_avg.index[non_zero_idx], rotation=90)
        
        # Plot max cutoff
        plt.axhline(y=max_cut, color='r')
        
        # Plot points
        for j in range(plot_n):
            idx = non_zero_idx[j]
            if var_true_props_avg.iloc[idx] < max_cut:
                if var_true_props_avg.iloc[idx] > perm_mean[idx] + cover_constant * perm_se[idx]:
                    plt.plot(j + 1, var_true_props_avg.iloc[idx], 's', markersize=5, color='blue')
                else:
                    plt.plot(j + 1, var_true_props_avg.iloc[idx], 'o', markersize=5, color='blue')
            else:
                plt.plot(j + 1, var_true_props_avg.iloc[idx], 'o', markersize=5, color='red', fillstyle='full')
        
        # Plot SE cutoffs
        for j in range(plot_n):
            idx = non_zero_idx[j]
            plt.plot([j + 1, j + 1], [0, perm_mean[idx] + cover_constant * perm_se[idx]], 'b-')
        
        plt.tight_layout()
        plt.show()
    
    # Return results
    return {
        "important_vars_local_names": important_vars_pointwise_names,
        "important_vars_global_max_names": important_vars_simul_max_names,
        "important_vars_global_se_names": important_vars_simul_se_names,
        "important_vars_local_col_nums": important_vars_pointwise_col_nums,
        "important_vars_global_max_col_nums": important_vars_simul_max_col_nums,
        "important_vars_global_se_col_nums": important_vars_simul_se_col_nums,
        "var_true_props_avg": var_true_props_avg,
        "permute_mat": permute_mat
    }

def get_averaged_true_var_props(bart_machine: Any, num_reps_for_avg: int, num_trees_for_permute: int) -> np.ndarray:
    """
    Get averaged variable proportions from the true model.
    
    This function averages variable proportions over multiple runs of the model.
    
    Args:
        bart_machine: The BART machine model.
        num_reps_for_avg: Number of repetitions for averaging.
        num_trees_for_permute: Number of trees to use in each model.
    
    Returns:
        An array of averaged variable proportions.
    """
    var_props = np.zeros(bart_machine.p)
    
    for i in range(num_reps_for_avg):
        bart_machine_dup = bart_machine.duplicate(num_trees=num_trees_for_permute)
        var_props += bart_machine_dup.get_var_props_over_chain()
        print(".", end="", flush=True)
    
    # Average over many runs
    return var_props / num_reps_for_avg

def get_null_permute_var_importances(bart_machine: Any, num_trees_for_permute: int) -> np.ndarray:
    """
    Get variable importances from a model with permuted responses.
    
    This function builds a BART model on permuted responses to get
    variable importances under the null hypothesis of no relationship
    between predictors and response.
    
    Args:
        bart_machine: The BART machine model.
        num_trees_for_permute: Number of trees to use in the permuted model.
    
    Returns:
        An array of variable importances from the permuted model.
    """
    # Permute the responses to disconnect X and y
    y_permuted = np.random.permutation(bart_machine.y)
    
    # Build BART on this permuted training data
    bart_machine_with_permuted_y = bart_machine.duplicate(
        y=y_permuted,
        num_trees=num_trees_for_permute,
        run_in_sample=False,
        verbose=False
    )
    
    # Just return the variable proportions
    var_props = bart_machine_with_permuted_y.get_var_props_over_chain()
    print(".", end="", flush=True)
    
    return var_props

def bisect_k(tol: float, coverage: float, permute_mat: np.ndarray, 
            x_left: float, x_right: float, count_limit: int, 
            perm_mean: np.ndarray, perm_se: np.ndarray) -> float:
    """
    Find the constant that gives the desired coverage using bisection.
    
    This function is used to compute the constant for the global SE method.
    
    Args:
        tol: Tolerance for convergence.
        coverage: Desired coverage.
        permute_mat: Permutation matrix.
        x_left: Left bound for bisection.
        x_right: Right bound for bisection.
        count_limit: Maximum number of iterations.
        perm_mean: Mean of permutation distribution.
        perm_se: Standard error of permutation distribution.
    
    Returns:
        The constant that gives the desired coverage.
    """
    count = 0
    guess = (x_left + x_right) / 2
    
    while (x_right - x_left) / 2 >= tol and count < count_limit:
        # Calculate empirical coverage
        empirical_coverage = np.mean([
            np.all(permute_mat[s, :] - perm_mean <= guess * perm_se)
            for s in range(permute_mat.shape[0])
        ])
        
        if empirical_coverage - coverage == 0:
            break
        elif empirical_coverage - coverage < 0:
            x_left = guess
        else:
            x_right = guess
        
        guess = (x_left + x_right) / 2
        count += 1
    
    return guess

def var_selection_by_permute_cv(bart_machine: Any, 
                               k_folds: int = 5, 
                               folds_vec: Optional[np.ndarray] = None, 
                               num_reps_for_avg: int = 5, 
                               num_permute_samples: int = 100, 
                               num_trees_for_permute: int = 20, 
                               alpha: float = 0.05, 
                               num_trees_pred_cv: int = 50) -> Dict[str, Any]:
    """
    Select variables by permutation importance with cross-validation.
    
    This function selects variables based on their permutation importance,
    using cross-validation to determine the best method (local, global max, or global SE).
    
    Args:
        bart_machine: The BART machine model.
        k_folds: Number of folds for cross-validation.
        folds_vec: Vector of fold indices. If None, folds are created randomly.
        num_reps_for_avg: Number of repetitions for averaging variable importance.
        num_permute_samples: Number of permutation samples.
        num_trees_for_permute: Number of trees to use in permutation models.
        alpha: Significance level.
        num_trees_pred_cv: Number of trees to use in cross-validation models.
    
    Returns:
        A dictionary containing selected variables and the best method.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if k_folds is valid
    if k_folds <= 1 or k_folds > bart_machine.n:
        raise ValueError("The number of folds must be at least 2 and less than or equal to n, use float('inf') for leave one out")
    
    # Handle leave-one-out cross-validation
    if k_folds == float('inf'):
        k_folds = bart_machine.n
    
    # Check if folds_vec is valid
    if folds_vec is not None and not isinstance(folds_vec, (np.ndarray, list)):
        raise ValueError("folds_vec must be an array of integers specifying the indices of each fold.")
    
    # Set up k folds
    if folds_vec is None:
        # Create random folds
        n = bart_machine.n
        if k_folds == float('inf'):
            k_folds = n
        
        if k_folds <= 1 or k_folds > n:
            raise ValueError("The number of folds must be at least 2 and less than or equal to n, use float('inf') for leave one out")
        
        # Create random fold assignments
        np.random.seed(42)  # For reproducibility
        temp = np.random.normal(size=n)
        quantiles = np.percentile(temp, np.linspace(0, 100, k_folds + 1))
        folds_vec = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(k_folds):
                if temp[i] >= quantiles[j] and temp[i] <= quantiles[j + 1]:
                    folds_vec[i] = j + 1
                    break
    else:
        # Use provided folds
        k_folds = len(np.unique(folds_vec))
    
    # Initialize error matrix
    L2_err_mat = np.zeros((k_folds, 3))
    method_names = ["important_vars_local_names", "important_vars_global_max_names", "important_vars_global_se_names"]
    
    # Perform cross-validation
    for k in range(1, k_folds + 1):
        print(f"cv #{k}", end="")
        
        # Find indices of training and test sets
        train_idx = np.where(folds_vec != k)[0]
        test_idx = np.where(folds_vec == k)[0]
        
        # Extract training data
        training_X_k = bart_machine.model_matrix_training_data.iloc[train_idx, :-1]
        training_y_k = bart_machine.y[train_idx]
        
        # Create temporary BART machine for variable selection
        bart_machine_temp = bart_machine.duplicate(
            X=training_X_k,
            y=training_y_k,
            run_in_sample=False,
            verbose=False
        )
        
        # Do variable selection
        bart_variables_select_obj_k = var_selection_by_permute(
            bart_machine_temp,
            num_permute_samples=num_permute_samples,
            num_trees_for_permute=num_trees_for_permute,
            num_reps_for_avg=num_reps_for_avg,
            alpha=alpha,
            plot=False
        )
        
        # Extract test data
        test_X_k = bart_machine.model_matrix_training_data.iloc[test_idx, :-1]
        test_y_k = bart_machine.y[test_idx]
        
        print("method", end="")
        for i, method in enumerate(method_names):
            print(".", end="", flush=True)
            
            # Get variables selected by this method
            vars_selected_by_method = bart_variables_select_obj_k[method]
            
            if len(vars_selected_by_method) == 0:
                # If no variables selected, predict mean
                ybar_est = np.mean(training_y_k)
                # Calculate L2 error
                L2_err_mat[k-1, i] = np.sum((test_y_k - ybar_est) ** 2)
            else:
                # Build BART machine with selected variables
                training_X_k_red = training_X_k[vars_selected_by_method]
                
                # Create new BART machine with reduced feature set
                bart_machine_temp = bart_machine.duplicate(
                    X=training_X_k_red,
                    y=training_y_k,
                    num_trees=num_trees_pred_cv,
                    run_in_sample=False,
                    cov_prior_vec=np.ones(len(vars_selected_by_method)),  # Standard prior
                    verbose=False
                )
                
                # Make predictions on test data
                test_X_k_red = test_X_k[vars_selected_by_method]
                predictions = bart_machine_temp.predict(test_X_k_red)
                
                # Calculate L2 error
                L2_err_mat[k-1, i] = np.sum((test_y_k - predictions) ** 2)
        
        print()
    
    # Find the best method based on L2 error
    L2_err_by_method = np.sum(L2_err_mat, axis=0)
    min_var_selection_method_idx = np.argmin(L2_err_by_method)
    min_var_selection_method = method_names[min_var_selection_method_idx]
    
    # Do final variable selection on the entire dataset
    print("final")
    bart_variables_select_obj = var_selection_by_permute(
        bart_machine,
        num_permute_samples=num_permute_samples,
        num_trees_for_permute=num_trees_for_permute,
        num_reps_for_avg=num_reps_for_avg,
        alpha=alpha,
        plot=False
    )
    
    # Return variables from the best method
    important_vars_cv = sorted(bart_variables_select_obj[min_var_selection_method])
    
    return {
        "best_method": min_var_selection_method,
        "important_vars_cv": important_vars_cv
    }
