"""
Direct comparison between R and Python implementations of bartMachine with very long MCMC chains.

This script provides a direct comparison between the R and Python implementations
of bartMachine, ensuring that the same data, parameters, and random seed are used,
with very long MCMC chains (25x more iterations than the original).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

# Import the Python implementation
from bartmachine.bartMachine import bart_machine
from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm

def main():
    """
    Main function to run the direct comparison with very long MCMC chains.
    """
    # Initialize the JVM for Python implementation
    initialize_jvm(max_memory="4g")  # Increase memory for longer chains
    
    # Load the Boston housing dataset
    boston_path = "/Users/mark/Documents/Cline/bartMachine/datasets/r_boston.csv"
    boston_data = pd.read_csv(boston_path)
    
    # Split into features and target
    X = boston_data.drop('y', axis=1)
    y = boston_data['y']
    
    # Generate train/test split
    np.random.seed(123)  # Set seed for reproducibility
    train_idx = np.random.choice(len(X), int(0.8 * len(X)), replace=False)
    test_idx = np.array([i for i in range(len(X)) if i not in train_idx])
    
    # Save train/test indices for R script
    pd.DataFrame({'train_idx': train_idx}).to_csv('py_train_idx_very_long.csv', index=False)
    pd.DataFrame({'test_idx': test_idx}).to_csv('py_test_idx_very_long.csv', index=False)
    
    # Use the same train/test split as R
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Set model parameters - run MCMC for much longer (25x more iterations)
    num_trees = 50
    num_burn_in = 2500  # 25x more burn-in iterations
    num_iterations_after_burn_in = 5000  # 25x more iterations after burn-in
    seed = 123
    alpha = 0.95
    beta = 2.0
    k = 2.0
    q = 0.9
    nu = 3.0
    
    # Set the random seed in numpy to match R's set.seed
    np.random.seed(seed)
    
    # Run Python implementation
    print("\nRunning Python implementation with very long MCMC chains...")
    start_time = pd.Timestamp.now()
    
    py_bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=num_trees,
        num_burn_in=num_burn_in,
        num_iterations_after_burn_in=num_iterations_after_burn_in,
        seed=seed,
        alpha=alpha,
        beta=beta,
        k=k,
        q=q,
        nu=nu,
        verbose=True  # Set to True to see progress
    )
    
    end_time = pd.Timestamp.now()
    build_time = (end_time - start_time).total_seconds()
    print(f"Python build time: {build_time:.2f} seconds")
    
    # Make predictions with Python model
    py_pred_start_time = pd.Timestamp.now()
    py_pred = py_bart.predict(X_test)
    py_pred_end_time = pd.Timestamp.now()
    py_pred_time = (py_pred_end_time - py_pred_start_time).total_seconds()
    print(f"Python prediction time: {py_pred_time:.2f} seconds")
    
    # Get variable importance from Python model
    py_var_imp_start_time = pd.Timestamp.now()
    py_var_imp = py_bart.get_var_props_over_chain()
    py_var_imp_end_time = pd.Timestamp.now()
    py_var_imp_time = (py_var_imp_end_time - py_var_imp_start_time).total_seconds()
    print(f"Python variable importance time: {py_var_imp_time:.2f} seconds")
    
    # Save Python results
    pd.DataFrame({
        'index': test_idx,
        'prediction': py_pred
    }).to_csv('py_direct_predictions_very_long.csv', index=False)
    
    pd.DataFrame({
        'variable': py_var_imp.index,
        'importance': py_var_imp.values
    }).to_csv('py_direct_var_importance_very_long.csv', index=False)
    
    # Shutdown the JVM
    shutdown_jvm()
    
    print("\nPython implementation with very long MCMC chains completed. Results saved to:")
    print("- py_direct_predictions_very_long.csv")
    print("- py_direct_var_importance_very_long.csv")
    print("- py_train_idx_very_long.csv")
    print("- py_test_idx_very_long.csv")
    
    print("\nNow run the R script with very long MCMC chains:")
    print("Rscript r_bart_boston_very_long.R")

if __name__ == "__main__":
    main()
