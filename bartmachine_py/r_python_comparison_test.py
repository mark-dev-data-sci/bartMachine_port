"""
Python Implementation Test for bartMachine

This script runs the Python implementation of bartMachine on the Boston housing dataset
and saves the results to files for comparison with the R implementation.

Instructions:
1. Run this script to generate Python results
2. Run the accompanying R script (r_bart_boston.R) to generate R results
3. Compare the results manually or using the comparison script
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
    Main function to run the Python implementation and save results.
    """
    # Initialize the JVM for Python implementation
    initialize_jvm(max_memory="2g")
    
    # Load the Boston housing dataset
    boston_path = "/Users/mark/Documents/Cline/bartMachine/datasets/r_boston.csv"
    boston_data = pd.read_csv(boston_path)
    
    # Split into features and target
    X = boston_data.drop('y', axis=1)
    y = boston_data['y']
    
    # Split into train and test sets (80% train, 20% test)
    np.random.seed(123)
    train_idx = np.random.choice(len(X), int(0.8 * len(X)), replace=False)
    test_idx = np.array([i for i in range(len(X)) if i not in train_idx])
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Save train/test indices for R script
    pd.DataFrame({'train_idx': train_idx}).to_csv('py_train_idx.csv', index=False)
    pd.DataFrame({'test_idx': test_idx}).to_csv('py_test_idx.csv', index=False)
    
    # Set model parameters - exactly match R parameters
    num_trees = 50
    num_burn_in = 100
    num_iterations_after_burn_in = 200
    seed = 123
    
    # Run Python implementation
    print("\nRunning Python implementation...")
    py_start_time = pd.Timestamp.now()
    
    # Set the random seed in numpy to match R's set.seed
    np.random.seed(seed)
    
    py_bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=num_trees,
        num_burn_in=num_burn_in,
        num_iterations_after_burn_in=num_iterations_after_burn_in,
        seed=seed,
        # Add additional parameters to match R exactly
        alpha=0.95,
        # beta must be a float, not an integer
        beta=2.0,
        k=2.0,
        q=0.9,
        nu=3.0,
        verbose=False
    )
    
    py_end_time = pd.Timestamp.now()
    py_build_time = (py_end_time - py_start_time).total_seconds()
    print(f"Python build time: {py_build_time:.2f} seconds")
    
    # Make predictions with Python model
    py_pred_start_time = pd.Timestamp.now()
    py_pred = py_bart.predict(X_test)
    py_pred_end_time = pd.Timestamp.now()
    py_pred_time = (py_pred_end_time - py_pred_start_time).total_seconds()
    print(f"Python prediction time: {py_pred_time:.2f} seconds")
    
    # Get variable importance from Python model
    py_var_imp_start_time = pd.Timestamp.now()
    py_var_imp = py_bart.get_var_importance()
    py_var_imp_end_time = pd.Timestamp.now()
    py_var_imp_time = (py_var_imp_end_time - py_var_imp_start_time).total_seconds()
    print(f"Python variable importance time: {py_var_imp_time:.2f} seconds")
    
    # Save Python results
    pd.DataFrame({
        'index': test_idx,
        'prediction': py_pred
    }).to_csv('py_bart_predictions.csv', index=False)
    
    pd.DataFrame({
        'variable': py_var_imp.index,
        'importance': py_var_imp.values
    }).to_csv('py_bart_var_importance.csv', index=False)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, py_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Python BART: Actual vs Predicted')
    plt.savefig('py_actual_vs_predicted.png')
    
    # Plot variable importance
    plt.figure(figsize=(12, 8))
    py_var_imp.sort_values(ascending=False).plot(kind='bar')
    plt.xlabel('Variables')
    plt.ylabel('Importance')
    plt.title('Python BART: Variable Importance')
    plt.tight_layout()
    plt.savefig('py_var_importance.png')
    
    # Shutdown the JVM
    shutdown_jvm()
    
    print("\nPython implementation completed. Results saved to:")
    print("- py_train_idx.csv")
    print("- py_test_idx.csv")
    print("- py_bart_predictions.csv")
    print("- py_bart_var_importance.csv")
    print("- py_actual_vs_predicted.png")
    print("- py_var_importance.png")
    
    print("\nTo compare with R implementation:")
    print("1. Create an R script (r_bart_boston.R) that:")
    print("   - Loads the same dataset")
    print("   - Uses the same train/test split (from py_train_idx.csv and py_test_idx.csv)")
    print("   - Runs bartMachine with the same parameters")
    print("   - Saves predictions and variable importance to CSV files")
    print("2. Run the R script")
    print("3. Compare the results using a comparison script or manually")

if __name__ == "__main__":
    main()
