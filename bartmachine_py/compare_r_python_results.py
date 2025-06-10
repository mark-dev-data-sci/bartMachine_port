"""
Compare R and Python Results

This script compares the results from the R and Python implementations of bartMachine.
It loads the prediction and variable importance results from both implementations,
calculates comparison metrics, and generates comparison plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    """
    Main function to compare R and Python results.
    """
    # Check if result files exist
    required_files = [
        'py_bart_predictions.csv',
        'py_bart_var_importance.csv',
        'r_bart_predictions.csv',
        'r_bart_var_importance.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found. Please run both R and Python implementations first.")
            sys.exit(1)
    
    # Load prediction results
    py_pred_df = pd.read_csv('py_bart_predictions.csv')
    r_pred_df = pd.read_csv('r_bart_predictions.csv')
    
    # Ensure predictions are for the same test indices
    if not np.array_equal(py_pred_df['index'], r_pred_df['index']):
        print("Warning: Test indices do not match between Python and R results.")
        # Align predictions by index
        merged_df = pd.merge(py_pred_df, r_pred_df, on='index', suffixes=('_py', '_r'))
        py_pred = merged_df['prediction_py'].values
        r_pred = merged_df['prediction_r'].values
    else:
        py_pred = py_pred_df['prediction'].values
        r_pred = r_pred_df['prediction'].values
    
    # Load variable importance results
    py_var_imp_df = pd.read_csv('py_bart_var_importance.csv')
    r_var_imp_df = pd.read_csv('r_bart_var_importance.csv')
    
    # Create dictionaries for easier comparison
    py_var_imp = dict(zip(py_var_imp_df['variable'], py_var_imp_df['importance']))
    r_var_imp = dict(zip(r_var_imp_df['variable'], r_var_imp_df['importance']))
    
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
    # Get all variables from both implementations
    all_vars = sorted(set(list(py_var_imp.keys()) + list(r_var_imp.keys())))
    
    # Create a DataFrame with variable importance from both implementations
    var_imp_comparison = pd.DataFrame({
        'Variable': all_vars,
        'Python': [py_var_imp.get(var, 0) for var in all_vars],
        'R': [r_var_imp.get(var, 0) for var in all_vars],
    })
    
    # Calculate differences
    var_imp_comparison['Difference'] = var_imp_comparison['Python'] - var_imp_comparison['R']
    var_imp_comparison['Abs_Difference'] = np.abs(var_imp_comparison['Difference'])
    
    # Sort by absolute difference
    var_imp_comparison = var_imp_comparison.sort_values('Abs_Difference', ascending=False)
    
    # Calculate correlation between variable importance measures
    var_imp_corr = np.corrcoef(
        [py_var_imp.get(var, 0) for var in all_vars],
        [r_var_imp.get(var, 0) for var in all_vars]
    )[0, 1]
    
    print("\nVariable Importance Comparison:")
    print(f"Correlation: {var_imp_corr:.4f}")
    print("\nTop 5 variables with largest differences:")
    print(var_imp_comparison.head(5).to_string(index=False))
    
    # Plot predictions comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(r_pred, py_pred, alpha=0.5)
    plt.plot([min(r_pred), max(r_pred)], [min(r_pred), max(r_pred)], 'r--')
    plt.xlabel('R Predictions')
    plt.ylabel('Python Predictions')
    plt.title(f'R vs Python Predictions (Correlation: {pred_corr:.4f})')
    plt.savefig('r_vs_python_predictions.png')
    
    # Plot variable importance comparison
    plt.figure(figsize=(12, 8))
    
    # Sort variables by Python importance
    sorted_vars = sorted(py_var_imp.keys(), key=lambda x: py_var_imp[x], reverse=True)
    
    x = np.arange(len(sorted_vars))
    width = 0.35
    
    py_vals = [py_var_imp[var] for var in sorted_vars]
    r_vals = [r_var_imp.get(var, 0) for var in sorted_vars]
    
    plt.bar(x - width/2, py_vals, width, label='Python')
    plt.bar(x + width/2, r_vals, width, label='R')
    
    plt.xlabel('Variables')
    plt.ylabel('Importance')
    plt.title('Variable Importance: Python vs R')
    plt.xticks(x, sorted_vars, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('r_vs_python_var_importance.png')
    
    # Save comparison results to CSV
    var_imp_comparison.to_csv('var_importance_comparison.csv', index=False)
    
    pred_comparison = pd.DataFrame({
        'Index': py_pred_df['index'],
        'Python_Prediction': py_pred,
        'R_Prediction': r_pred,
        'Difference': py_pred - r_pred,
        'Abs_Difference': np.abs(py_pred - r_pred)
    })
    pred_comparison.to_csv('prediction_comparison.csv', index=False)
    
    print("\nComparison completed. Results saved to:")
    print("- r_vs_python_predictions.png")
    print("- r_vs_python_var_importance.png")
    print("- var_importance_comparison.csv")
    print("- prediction_comparison.csv")

if __name__ == "__main__":
    main()
