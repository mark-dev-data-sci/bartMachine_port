"""
Direct comparison between R and Python implementations of bartMachine.

This script provides a direct comparison between the R and Python implementations
of bartMachine, ensuring that the same data, parameters, and random seed are used.
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
    Main function to run the direct comparison.
    """
    # Initialize the JVM for Python implementation
    initialize_jvm(max_memory="2g")
    
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
    pd.DataFrame({'train_idx': train_idx}).to_csv('py_train_idx.csv', index=False)
    pd.DataFrame({'test_idx': test_idx}).to_csv('py_test_idx.csv', index=False)
    
    # Use the same train/test split as R
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Set model parameters - run MCMC for longer
    num_trees = 50
    num_burn_in = 500  # 5x more burn-in iterations
    num_iterations_after_burn_in = 1000  # 5x more iterations after burn-in
    seed = 123
    alpha = 0.95
    beta = 2.0
    k = 2.0
    q = 0.9
    nu = 3.0
    
    # Set the random seed in numpy to match R's set.seed
    np.random.seed(seed)
    
    # Run Python implementation
    print("\nRunning Python implementation...")
    
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
        verbose=False
    )
    
    # Make predictions with Python model
    py_pred = py_bart.predict(X_test)
    
    # Get variable importance from Python model
    py_var_imp = py_bart.get_var_props_over_chain()
    
    # Save Python results
    pd.DataFrame({
        'index': test_idx,
        'prediction': py_pred
    }).to_csv('py_direct_predictions.csv', index=False)
    
    pd.DataFrame({
        'variable': py_var_imp.index,
        'importance': py_var_imp.values
    }).to_csv('py_direct_var_importance.csv', index=False)
    
    # Load R results
    r_pred = pd.read_csv("r_bart_predictions.csv")["prediction"].values
    r_var_imp = pd.read_csv("r_bart_var_importance.csv")
    
    # Compare predictions
    pred_corr = np.corrcoef(py_pred, r_pred)[0, 1]
    pred_rmse = np.sqrt(np.mean((py_pred - r_pred) ** 2))
    pred_mae = np.mean(np.abs(py_pred - r_pred))
    pred_max_diff = np.max(np.abs(py_pred - r_pred))
    
    print("\nPrediction Comparison:")
    print(f"Correlation: {pred_corr:.4f}")
    print(f"RMSE: {pred_rmse:.4f}")
    print(f"MAE: {pred_mae:.4f}")
    print(f"Max Absolute Difference: {pred_max_diff:.4f}")
    
    # Compare variable importance
    # Create a DataFrame with both R and Python variable importance
    var_imp_df = pd.DataFrame({
        'variable': py_var_imp.index,
        'Python': py_var_imp.values
    })
    var_imp_df = var_imp_df.merge(r_var_imp, on='variable', how='left', suffixes=('', '_r'))
    var_imp_df.rename(columns={'importance': 'R'}, inplace=True)
    
    # Calculate differences
    var_imp_df['Difference'] = var_imp_df['Python'] - var_imp_df['R']
    var_imp_df['Abs_Difference'] = np.abs(var_imp_df['Difference'])
    
    # Calculate correlation
    var_imp_corr = np.corrcoef(var_imp_df['Python'], var_imp_df['R'])[0, 1]
    
    print("\nVariable Importance Comparison:")
    print(f"Correlation: {var_imp_corr:.4f}")
    
    # Print the top 5 variables with the largest differences
    print("\nTop 5 variables with largest differences:")
    print(var_imp_df.sort_values('Abs_Difference', ascending=False).head(5)[['variable', 'Python', 'R', 'Difference', 'Abs_Difference']])
    
    # Save comparison results
    var_imp_df.to_csv('direct_var_importance_comparison.csv', index=False)
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(r_pred, py_pred, alpha=0.5)
    plt.plot([min(r_pred), max(r_pred)], [min(r_pred), max(r_pred)], 'r--')
    plt.xlabel('R Predictions')
    plt.ylabel('Python Predictions')
    plt.title('R vs Python Predictions')
    plt.savefig('direct_r_vs_python_predictions.png')
    
    # Plot variable importance
    plt.figure(figsize=(12, 8))
    plt.scatter(var_imp_df['R'], var_imp_df['Python'], alpha=0.7)
    for i, txt in enumerate(var_imp_df['variable']):
        plt.annotate(txt, (var_imp_df['R'].iloc[i], var_imp_df['Python'].iloc[i]))
    plt.plot([0, max(var_imp_df['R'].max(), var_imp_df['Python'].max())], 
             [0, max(var_imp_df['R'].max(), var_imp_df['Python'].max())], 'r--')
    plt.xlabel('R Variable Importance')
    plt.ylabel('Python Variable Importance')
    plt.title('R vs Python Variable Importance')
    plt.savefig('direct_r_vs_python_var_importance.png')
    
    # Shutdown the JVM
    shutdown_jvm()
    
    print("\nComparison completed. Results saved to:")
    print("- py_direct_predictions.csv")
    print("- py_direct_var_importance.csv")
    print("- direct_var_importance_comparison.csv")
    print("- direct_r_vs_python_predictions.png")
    print("- direct_r_vs_python_var_importance.png")

if __name__ == "__main__":
    main()
