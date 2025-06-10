"""
Compare results from longer MCMC runs of R and Python implementations.

This script compares the results from the longer MCMC runs of the R and Python
implementations of bartMachine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Main function to compare results from longer MCMC runs.
    """
    # Load Python results
    py_pred = pd.read_csv("py_direct_predictions.csv")["prediction"].values
    py_var_imp = pd.read_csv("py_direct_var_importance.csv")
    
    # Load R results
    r_pred = pd.read_csv("r_bart_predictions_long.csv")["prediction"].values
    r_var_imp = pd.read_csv("r_bart_var_importance_long.csv")
    
    # Compare predictions
    pred_corr = np.corrcoef(py_pred, r_pred)[0, 1]
    pred_rmse = np.sqrt(np.mean((py_pred - r_pred) ** 2))
    pred_mae = np.mean(np.abs(py_pred - r_pred))
    pred_max_diff = np.max(np.abs(py_pred - r_pred))
    
    print("\nPrediction Comparison (Long MCMC Runs):")
    print(f"Correlation: {pred_corr:.4f}")
    print(f"RMSE: {pred_rmse:.4f}")
    print(f"MAE: {pred_mae:.4f}")
    print(f"Max Absolute Difference: {pred_max_diff:.4f}")
    
    # Compare variable importance
    # Create a DataFrame with both R and Python variable importance
    var_imp_df = pd.DataFrame({
        'variable': py_var_imp['variable'],
        'Python': py_var_imp['importance']
    })
    var_imp_df = var_imp_df.merge(r_var_imp, on='variable', how='left', suffixes=('', '_r'))
    var_imp_df.rename(columns={'importance': 'R'}, inplace=True)
    
    # Calculate differences
    var_imp_df['Difference'] = var_imp_df['Python'] - var_imp_df['R']
    var_imp_df['Abs_Difference'] = np.abs(var_imp_df['Difference'])
    
    # Calculate correlation
    var_imp_corr = np.corrcoef(var_imp_df['Python'], var_imp_df['R'])[0, 1]
    
    print("\nVariable Importance Comparison (Long MCMC Runs):")
    print(f"Correlation: {var_imp_corr:.4f}")
    
    # Print the top 5 variables with the largest differences
    print("\nTop 5 variables with largest differences:")
    print(var_imp_df.sort_values('Abs_Difference', ascending=False).head(5)[['variable', 'Python', 'R', 'Difference', 'Abs_Difference']])
    
    # Save comparison results
    var_imp_df.to_csv('long_var_importance_comparison.csv', index=False)
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(r_pred, py_pred, alpha=0.5)
    plt.plot([min(r_pred), max(r_pred)], [min(r_pred), max(r_pred)], 'r--')
    plt.xlabel('R Predictions (Long MCMC)')
    plt.ylabel('Python Predictions (Long MCMC)')
    plt.title('R vs Python Predictions (Long MCMC Runs)')
    plt.savefig('long_r_vs_python_predictions.png')
    
    # Plot variable importance
    plt.figure(figsize=(12, 8))
    plt.scatter(var_imp_df['R'], var_imp_df['Python'], alpha=0.7)
    for i, txt in enumerate(var_imp_df['variable']):
        plt.annotate(txt, (var_imp_df['R'].iloc[i], var_imp_df['Python'].iloc[i]))
    plt.plot([0, max(var_imp_df['R'].max(), var_imp_df['Python'].max())], 
             [0, max(var_imp_df['R'].max(), var_imp_df['Python'].max())], 'r--')
    plt.xlabel('R Variable Importance (Long MCMC)')
    plt.ylabel('Python Variable Importance (Long MCMC)')
    plt.title('R vs Python Variable Importance (Long MCMC Runs)')
    plt.savefig('long_r_vs_python_var_importance.png')
    
    print("\nComparison completed. Results saved to:")
    print("- long_var_importance_comparison.csv")
    print("- long_r_vs_python_predictions.png")
    print("- long_r_vs_python_var_importance.png")

if __name__ == "__main__":
    main()
