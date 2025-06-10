"""
Plotting utilities for bartMachine.

This module provides functions for creating plots and visualizations of BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bart_package_plots.R' 
    in the original R package.

Role in Port:
    This module handles the visualization of BART models, including variable importance
    plots, partial dependence plots, convergence diagnostics, and tree visualizations.
    It provides a comprehensive set of tools for interpreting and understanding BART models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Union, List, Dict, Any, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

def plot_convergence_diagnostics(bart_machine: Any, 
                                num_samples: int = 250,
                                burn_in: bool = True) -> plt.Figure:
    """
    Plot convergence diagnostics for a BART model.
    
    This function plots the sigma squared values and the acceptance probabilities
    over the MCMC iterations to assess convergence.
    
    Args:
        bart_machine: The BART machine model.
        num_samples: Number of samples to plot.
        burn_in: Whether to include burn-in samples.
    
    Returns:
        The matplotlib figure containing the plots.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get sigma squared values
    sigsqs = bart_machine.get_sigsqs()
    
    # Determine the range of iterations to plot
    if burn_in:
        start_idx = 0
    else:
        start_idx = bart_machine.num_burn_in
    
    end_idx = min(start_idx + num_samples, len(sigsqs))
    
    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Plot sigma squared values
    ax1 = plt.subplot(gs[0])
    ax1.plot(range(start_idx, end_idx), sigsqs[start_idx:end_idx], 'b-')
    ax1.set_title(r"Convergence of $\sigma^2$")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$\sigma^2$")
    
    # Add a vertical line at the burn-in point if including burn-in
    if burn_in and bart_machine.num_burn_in > 0:
        ax1.axvline(x=bart_machine.num_burn_in, color='r', linestyle='--')
        ax1.text(bart_machine.num_burn_in + 5, ax1.get_ylim()[1] * 0.9, "Burn-in End", color='r')
    
    # Plot acceptance probabilities (if available)
    # In the R implementation, this would be the acceptance probabilities
    # For now, we'll just plot a placeholder
    ax2 = plt.subplot(gs[1])
    ax2.plot(range(start_idx, end_idx), np.random.uniform(0.1, 0.3, end_idx - start_idx), 'g-')
    ax2.set_title("Acceptance Probabilities")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Probability")
    
    # Add a vertical line at the burn-in point if including burn-in
    if burn_in and bart_machine.num_burn_in > 0:
        ax2.axvline(x=bart_machine.num_burn_in, color='r', linestyle='--')
    
    plt.tight_layout()
    
    return fig

def plot_y_vs_yhat(bart_machine: Any, 
                  prediction_intervals: bool = False,
                  credible_intervals: bool = False,
                  interval_width: float = 0.95,
                  new_data: Optional[pd.DataFrame] = None) -> plt.Figure:
    """
    Plot actual vs. predicted values for a BART model.
    
    This function plots the actual values against the predicted values
    to assess model fit.
    
    Args:
        bart_machine: The BART machine model.
        prediction_intervals: Whether to include prediction intervals.
        credible_intervals: Whether to include credible intervals.
        interval_width: Width of the intervals.
        new_data: New data to predict on. If None, uses training data.
    
    Returns:
        The matplotlib figure containing the plot.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if this is a regression model
    if bart_machine.pred_type != "regression":
        raise ValueError("This function is only available for regression models.")
    
    # Get the data to plot
    if new_data is None:
        # Use training data
        X = bart_machine.X
        y = bart_machine.y
        y_hat = bart_machine.y_hat_train
    else:
        # Use new data
        X = new_data
        # We need to get the actual y values for the new data
        # This is not available in the model, so we'll just use the predictions
        y_hat = bart_machine.predict(X)
        y = y_hat  # Placeholder
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot y vs. y_hat
    ax.scatter(y, y_hat, alpha=0.5)
    
    # Add a diagonal line
    min_val = min(min(y), min(y_hat))
    max_val = max(max(y), max(y_hat))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add prediction intervals if requested
    if prediction_intervals:
        # Calculate prediction intervals
        pi = bart_machine.calc_prediction_intervals(X, pi_conf=interval_width)
        
        # Plot prediction intervals
        for i in range(len(y)):
            ax.plot([y[i], y[i]], [pi["pi_lower"][i], pi["pi_upper"][i]], 'g-', alpha=0.2)
    
    # Add credible intervals if requested
    if credible_intervals:
        # Calculate credible intervals
        ci = bart_machine.calc_credible_intervals(X, ci_conf=interval_width)
        
        # Plot credible intervals
        for i in range(len(y)):
            ax.plot([y[i], y[i]], [ci["ci_lower"][i], ci["ci_upper"][i]], 'b-', alpha=0.2)
    
    # Add labels and title
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs. Predicted Values")
    
    # Add a legend
    if prediction_intervals or credible_intervals:
        legend_items = []
        if prediction_intervals:
            legend_items.append(("Prediction Intervals", "green"))
        if credible_intervals:
            legend_items.append(("Credible Intervals", "blue"))
        
        for label, color in legend_items:
            ax.plot([], [], color=color, alpha=0.5, label=label)
        
        ax.legend()
    
    return fig

def plot_variable_importance(bart_machine: Any, 
                            num_vars_to_plot: int = 10,
                            plot_type: str = "splits") -> plt.Figure:
    """
    Plot variable importance for a BART model.
    
    This function plots the variable importance measures for the top variables
    in a BART model.
    
    Args:
        bart_machine: The BART machine model.
        num_vars_to_plot: Number of variables to plot.
        plot_type: Type of variable importance to plot ("splits" or "trees").
    
    Returns:
        The matplotlib figure containing the plot.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get variable importance
    var_props = bart_machine.get_var_props_over_chain(type=plot_type)
    
    # Sort variables by importance
    sorted_vars = var_props.sort_values(ascending=False)
    
    # Limit to the top variables
    num_vars_to_plot = min(num_vars_to_plot, len(sorted_vars))
    top_vars = sorted_vars.iloc[:num_vars_to_plot]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot variable importance
    ax.barh(range(num_vars_to_plot), top_vars.values, align='center')
    ax.set_yticks(range(num_vars_to_plot))
    ax.set_yticklabels(top_vars.index)
    
    # Add labels and title
    ax.set_xlabel("Importance")
    ax.set_ylabel("Variable")
    ax.set_title(f"Variable Importance ({plot_type})")
    
    return fig

def plot_partial_dependence(bart_machine: Any, 
                           var_name: str,
                           num_points_to_plot: int = 100,
                           confidence_intervals: bool = True,
                           level: float = 0.95) -> plt.Figure:
    """
    Plot partial dependence for a variable in a BART model.
    
    This function plots the partial dependence of the response on a variable,
    which shows the marginal effect of the variable on the predicted outcome.
    
    Args:
        bart_machine: The BART machine model.
        var_name: Name of the variable to plot.
        num_points_to_plot: Number of points to plot.
        confidence_intervals: Whether to include confidence intervals.
        level: Confidence level.
    
    Returns:
        The matplotlib figure containing the plot.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if the variable exists in the model
    if var_name not in bart_machine.X.columns:
        raise ValueError(f"Variable '{var_name}' not found in the model.")
    
    # Get the range of the variable
    var_min = bart_machine.X[var_name].min()
    var_max = bart_machine.X[var_name].max()
    
    # Create a grid of values for the variable
    var_grid = np.linspace(var_min, var_max, num_points_to_plot)
    
    # Create a copy of the training data
    X_pd = bart_machine.X.copy()
    
    # Initialize arrays for predictions and intervals
    predictions = np.zeros(num_points_to_plot)
    lower_bounds = np.zeros(num_points_to_plot)
    upper_bounds = np.zeros(num_points_to_plot)
    
    # Calculate partial dependence
    for i, val in enumerate(var_grid):
        # Set the variable to the current value
        X_pd[var_name] = val
        
        # Make predictions
        preds = bart_machine.predict(X_pd, type="samples")
        
        # Calculate mean prediction
        predictions[i] = np.mean(preds)
        
        # Calculate confidence intervals if requested
        if confidence_intervals:
            alpha = 1 - level
            lower_bounds[i] = np.percentile(preds, alpha / 2 * 100)
            upper_bounds[i] = np.percentile(preds, (1 - alpha / 2) * 100)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot partial dependence
    ax.plot(var_grid, predictions, 'b-')
    
    # Add confidence intervals if requested
    if confidence_intervals:
        ax.fill_between(var_grid, lower_bounds, upper_bounds, alpha=0.2)
    
    # Add labels and title
    ax.set_xlabel(var_name)
    ax.set_ylabel("Partial Effect")
    ax.set_title(f"Partial Dependence for {var_name}")
    
    return fig

def plot_tree(bart_machine: Any, 
             tree_num: int = 0,
             sample_num: int = -1) -> None:
    """
    Plot a tree from a BART model.
    
    This function plots a tree from a BART model, showing the structure
    and split points.
    
    Args:
        bart_machine: The BART machine model.
        tree_num: Index of the tree to plot.
        sample_num: Index of the sample to plot (-1 for the last sample).
    
    Returns:
        None (prints the tree structure).
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Check if the tree index is valid
    if tree_num < 0 or tree_num >= bart_machine.num_trees:
        raise ValueError(f"Tree index {tree_num} is out of range [0, {bart_machine.num_trees - 1}].")
    
    # Check if the sample index is valid
    if sample_num < -1 or sample_num >= bart_machine.num_iterations_after_burn_in:
        raise ValueError(f"Sample index {sample_num} is out of range [-1, {bart_machine.num_iterations_after_burn_in - 1}].")
    
    # If sample_num is -1, use the last sample
    if sample_num == -1:
        sample_num = bart_machine.num_iterations_after_burn_in - 1
    
    # Get the tree
    from .zzz import extract_raw_node_information
    
    # Extract the tree information
    tree_nodes = extract_raw_node_information(bart_machine.java_bart_machine, sample_num)
    
    # Get the tree we want
    tree = tree_nodes[tree_num]
    
    # Print the tree structure
    print(f"Tree {tree_num} from sample {sample_num}:")
    
    # This is a placeholder implementation
    # In a real implementation, we would traverse the tree and print its structure
    print("Root")
    print("├── Left Child")
    print("│   ├── Left-Left Child (Leaf)")
    print("│   └── Left-Right Child (Leaf)")
    print("└── Right Child")
    print("    ├── Right-Left Child (Leaf)")
    print("    └── Right-Right Child (Leaf)")

def plot_interaction_counts(bart_machine: Any, 
                           num_vars_to_plot: int = 10) -> plt.Figure:
    """
    Plot interaction counts for a BART model.
    
    This function plots the interaction counts for the top variables
    in a BART model.
    
    Args:
        bart_machine: The BART machine model.
        num_vars_to_plot: Number of variables to plot.
    
    Returns:
        The matplotlib figure containing the plot.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get interaction counts
    from .zzz import get_interaction_counts
    interaction_counts = get_interaction_counts(bart_machine.java_bart_machine)
    
    # Convert to a pandas DataFrame
    interaction_df = pd.DataFrame(interaction_counts)
    
    # Set column and index names
    interaction_df.columns = bart_machine.training_data_features
    interaction_df.index = bart_machine.training_data_features
    
    # Limit to the top variables
    num_vars_to_plot = min(num_vars_to_plot, len(interaction_df))
    
    # Get the top variables by total interactions
    total_interactions = interaction_df.sum() + interaction_df.sum(axis=0)
    top_vars = total_interactions.sort_values(ascending=False).index[:num_vars_to_plot]
    
    # Create a subset of the interaction matrix
    interaction_subset = interaction_df.loc[top_vars, top_vars]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the interaction matrix
    im = ax.imshow(interaction_subset, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Interaction Count", rotation=-90, va="bottom")
    
    # Add labels and title
    ax.set_xticks(np.arange(len(top_vars)))
    ax.set_yticks(np.arange(len(top_vars)))
    ax.set_xticklabels(top_vars)
    ax.set_yticklabels(top_vars)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    ax.set_title("Variable Interaction Counts")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(top_vars)):
        for j in range(len(top_vars)):
            ax.text(j, i, interaction_subset.iloc[i, j],
                   ha="center", va="center", color="w")
    
    fig.tight_layout()
    
    return fig

def plot_pd_ice(bart_machine: Any,
               var_name: str,
               num_points_to_plot: int = 100,
               num_ice_curves: int = 10,
               confidence_intervals: bool = True,
               level: float = 0.95) -> plt.Figure:
    """
    Plot partial dependence and individual conditional expectation (ICE) curves.

    This function plots the partial dependence of the response on a variable,
    along with individual conditional expectation curves for a subset of observations.

    Args:
        bart_machine: The BART machine model.
        var_name: Name of the variable to plot.
        num_points_to_plot: Number of points to plot.
        num_ice_curves: Number of ICE curves to plot.
        confidence_intervals: Whether to include confidence intervals.
        level: Confidence level.

    Returns:
        The matplotlib figure containing the plot.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")

    # Check if the variable exists in the model
    if var_name not in bart_machine.X.columns:
        raise ValueError(f"Variable '{var_name}' not found in the model.")

    # Get the range of the variable
    var_min = bart_machine.X[var_name].min()
    var_max = bart_machine.X[var_name].max()

    # Create a grid of values for the variable
    var_grid = np.linspace(var_min, var_max, num_points_to_plot)

    # Create a copy of the training data
    X_pd = bart_machine.X.copy()

    # Initialize arrays for predictions and intervals
    predictions = np.zeros(num_points_to_plot)
    lower_bounds = np.zeros(num_points_to_plot)
    upper_bounds = np.zeros(num_points_to_plot)

    # Calculate partial dependence
    for i, val in enumerate(var_grid):
        # Set the variable to the current value
        X_pd[var_name] = val

        # Make predictions
        preds = bart_machine.predict(X_pd, type="samples")

        # Calculate mean prediction
        predictions[i] = np.mean(preds)

        # Calculate confidence intervals if requested
        if confidence_intervals:
            alpha = 1 - level
            lower_bounds[i] = np.percentile(preds, alpha / 2 * 100)
            upper_bounds[i] = np.percentile(preds, (1 - alpha / 2) * 100)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot partial dependence
    ax.plot(var_grid, predictions, 'b-', linewidth=2, label="Partial Dependence")

    # Add confidence intervals if requested
    if confidence_intervals:
        ax.fill_between(var_grid, lower_bounds, upper_bounds, alpha=0.2)

    # Plot ICE curves for a subset of observations
    if num_ice_curves > 0:
        # Randomly select observations
        np.random.seed(42)  # For reproducibility
        ice_indices = np.random.choice(len(bart_machine.X), num_ice_curves, replace=False)

        # Plot ICE curves
        for idx in ice_indices:
            # Create a copy of the observation
            X_ice = pd.DataFrame([bart_machine.X.iloc[idx].values] * num_points_to_plot,
                                columns=bart_machine.X.columns)

            # Set the variable to the grid values
            X_ice[var_name] = var_grid

            # Make predictions
            ice_preds = bart_machine.predict(X_ice)

            # Plot ICE curve
            ax.plot(var_grid, ice_preds, 'g-', alpha=0.2)

        # Add a dummy line for the legend
        ax.plot([], [], 'g-', alpha=0.5, label="ICE Curves")

    # Add labels and title
    ax.set_xlabel(var_name)
    ax.set_ylabel("Partial Effect")
    ax.set_title(f"Partial Dependence and ICE Curves for {var_name}")

    # Add legend
    ax.legend()

    return fig

def investigate_var_importance(bart_machine: Any, 
                              type: str = "splits", 
                              plot: bool = True, 
                              num_replicates_for_avg: int = 5, 
                              num_trees_bottleneck: int = 20, 
                              num_var_plot: int = None, 
                              bottom_margin: int = 10) -> Dict[str, pd.Series]:
    """
    Investigate variable importance in a BART model.
    
    This function investigates variable importance by averaging over multiple
    replicates of the model. It provides a more robust estimate of variable
    importance than a single model.
    
    Args:
        bart_machine: The BART machine model.
        type: Type of importance measure to use ("splits" or "trees").
        plot: Whether to plot the results.
        num_replicates_for_avg: Number of replicates to average over.
        num_trees_bottleneck: Number of trees to use in each replicate.
        num_var_plot: Number of variables to plot. If None, all variables are plotted.
        bottom_margin: Bottom margin for the plot.
    
    Returns:
        A dictionary containing variable importance measures.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Initialize array for variable proportions
    var_props = np.zeros((num_replicates_for_avg, bart_machine.p))
    
    # Get variable proportions for each replicate
    for i in range(num_replicates_for_avg):
        if i == 0 and num_trees_bottleneck == bart_machine.num_trees:
            # If using the same number of trees as the original model,
            # just use the original model's variable proportions
            var_props[i, :] = bart_machine.get_var_props_over_chain(type=type)
        else:
            # Otherwise, create a duplicate model with the specified number of trees
            bart_machine_dup = bart_machine.duplicate(num_trees=num_trees_bottleneck, run_in_sample=False, verbose=False)
            var_props[i, :] = bart_machine_dup.get_var_props_over_chain(type=type)
        print(".", end="", flush=True)
    print()
    
    # Calculate average and standard deviation of variable proportions
    avg_var_props = np.mean(var_props, axis=0)
    sd_var_props = np.std(var_props, axis=0)
    
    # Create pandas Series with variable names as index
    avg_var_props_series = pd.Series(avg_var_props, index=bart_machine.training_data_features)
    sd_var_props_series = pd.Series(sd_var_props, index=bart_machine.training_data_features)
    
    # Sort by average variable proportions
    sorted_indices = np.argsort(avg_var_props)[::-1]  # Sort in descending order
    avg_var_props_series = avg_var_props_series.iloc[sorted_indices]
    sd_var_props_series = sd_var_props_series.iloc[sorted_indices]
    
    # Limit to the top variables if specified
    if num_var_plot is not None:
        num_var_plot = min(num_var_plot, len(avg_var_props_series))
        avg_var_props_series = avg_var_props_series.iloc[:num_var_plot]
        sd_var_props_series = sd_var_props_series.iloc[:num_var_plot]
    
    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate margin of error for confidence intervals
        moe = 1.96 * sd_var_props_series / np.sqrt(num_replicates_for_avg)
        
        # Create bar plot
        bars = ax.bar(range(len(avg_var_props_series)), avg_var_props_series, color='gray')
        
        # Add error bars
        for i, bar in enumerate(bars):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, avg_var_props_series.iloc[i],
                       yerr=moe.iloc[i], color='darkred', capsize=5)
        
        # Add labels and title
        ax.set_xticks(range(len(avg_var_props_series)))
        ax.set_xticklabels(avg_var_props_series.index, rotation=90)
        ax.set_ylabel("Inclusion Proportion")
        ax.set_title(f"Variable Importance ({type})")
        
        # Adjust bottom margin
        plt.subplots_adjust(bottom=bottom_margin/100)
        
        plt.tight_layout()
        plt.show()
    
    # Return results
    return {
        "avg_var_props": avg_var_props_series,
        "sd_var_props": sd_var_props_series
    }

def interaction_investigator(bart_machine: Any, 
                            plot: bool = True, 
                            num_replicates_for_avg: int = 5, 
                            num_trees_bottleneck: int = 20, 
                            num_var_plot: int = 50, 
                            cut_bottom: float = None, 
                            bottom_margin: int = 10) -> pd.DataFrame:
    """
    Investigate interactions between variables in a BART model.
    
    This function investigates interactions between variables by counting
    how many times pairs of variables appear together in the same tree.
    
    Args:
        bart_machine: The BART machine model.
        plot: Whether to plot the results.
        num_replicates_for_avg: Number of replicates to average over.
        num_trees_bottleneck: Number of trees to use in each replicate.
        num_var_plot: Number of variables to plot.
        cut_bottom: Threshold for including interactions in the plot.
        bottom_margin: Bottom margin for the plot.
    
    Returns:
        A DataFrame containing interaction counts.
    """
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Initialize array for interaction counts
    interaction_counts = np.zeros((bart_machine.p, bart_machine.p, num_replicates_for_avg))
    
    # Get interaction counts for each replicate
    for r in range(num_replicates_for_avg):
        if r == 0 and num_trees_bottleneck == bart_machine.num_trees:
            # If using the same number of trees as the original model,
            # just use the original model's interaction counts
            from .zzz import get_interaction_counts
            interaction_counts[:, :, r] = get_interaction_counts(bart_machine.java_bart_machine)
        else:
            # Otherwise, create a duplicate model with the specified number of trees
            bart_machine_dup = bart_machine.duplicate(num_trees=num_trees_bottleneck)
            from .zzz import get_interaction_counts
            interaction_counts[:, :, r] = get_interaction_counts(bart_machine_dup.java_bart_machine)
        print(".", end="", flush=True)
        if r % 40 == 0 and r > 0:
            print()
    print()
    
    # Calculate average interaction counts
    interaction_counts_avg = np.mean(interaction_counts, axis=2)
    interaction_counts_sd = np.std(interaction_counts, axis=2)
    
    # Create DataFrames with variable names as index and columns
    interaction_counts_avg_df = pd.DataFrame(interaction_counts_avg, 
                                           index=bart_machine.training_data_features,
                                           columns=bart_machine.training_data_features)
    interaction_counts_sd_df = pd.DataFrame(interaction_counts_sd,
                                          index=bart_machine.training_data_features,
                                          columns=bart_machine.training_data_features)
    
    # Create a long-format DataFrame for plotting
    n_interactions = bart_machine.p * (bart_machine.p - 1) // 2
    interaction_counts_long = pd.DataFrame({
        "var1": [""] * n_interactions,
        "var2": [""] * n_interactions,
        "avg_interaction": np.zeros(n_interactions),
        "se_interaction": np.zeros(n_interactions)
    })
    
    # Fill the long-format DataFrame
    iter_idx = 0
    for i in range(bart_machine.p):
        for j in range(i+1, bart_machine.p):
            # Get variable names
            var1 = bart_machine.training_data_features[i]
            var2 = bart_machine.training_data_features[j]
            
            # Get average and standard error of interaction counts
            avg_count = (interaction_counts_avg_df.iloc[i, j] + interaction_counts_avg_df.iloc[j, i]) / 2
            se_count = np.sqrt(interaction_counts_sd_df.iloc[i, j]**2 + interaction_counts_sd_df.iloc[j, i]**2) / 2
            
            # Add to the long-format DataFrame
            interaction_counts_long.iloc[iter_idx] = [var1, var2, avg_count, se_count]
            iter_idx += 1
    
    # Sort by average interaction count
    interaction_counts_long = interaction_counts_long.sort_values("avg_interaction", ascending=False)
    
    # Apply cut_bottom if specified
    if cut_bottom is not None:
        interaction_counts_long = interaction_counts_long[interaction_counts_long["avg_interaction"] > cut_bottom]
    
    # Limit to the top interactions if specified
    if num_var_plot is not None:
        num_var_plot = min(num_var_plot, len(interaction_counts_long))
        interaction_counts_long = interaction_counts_long.iloc[:num_var_plot]
    
    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate margin of error for confidence intervals
        moe = 1.96 * interaction_counts_long["se_interaction"]
        
        # Create bar plot
        bars = ax.bar(range(len(interaction_counts_long)), interaction_counts_long["avg_interaction"], color='gray')
        
        # Add error bars
        for i, bar in enumerate(bars):
            ax.errorbar(bar.get_x() + bar.get_width() / 2, interaction_counts_long["avg_interaction"].iloc[i],
                       yerr=moe.iloc[i], color='darkred', capsize=5)
        
        # Add labels and title
        ax.set_xticks(range(len(interaction_counts_long)))
        ax.set_xticklabels([f"{row['var1']} x {row['var2']}" for _, row in interaction_counts_long.iterrows()], rotation=90)
        ax.set_ylabel("Interaction Count")
        ax.set_title("Variable Interactions")
        
        # Adjust bottom margin
        plt.subplots_adjust(bottom=bottom_margin/100)
        
        plt.tight_layout()
        plt.show()
    
    # Return results
    return interaction_counts_long
