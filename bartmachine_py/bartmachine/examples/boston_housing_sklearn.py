"""
Example using the California Housing dataset from scikit-learn with bartMachine Python API.

This script demonstrates the use of the bartMachine Python API with the California Housing dataset.
It loads the dataset from scikit-learn, builds a BART model, makes predictions, and visualizes the results.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import bartMachine
from bartmachine import initialize_jvm, shutdown_jvm, bart_machine

# Initialize JVM
initialize_jvm(max_memory="2g")

try:
    # Load the California Housing dataset from scikit-learn
    print("Loading California Housing dataset from scikit-learn...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Columns: {X.columns.tolist()}")
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(X.describe())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build a BART model
    print("\nBuilding BART model...")
    bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=20,  # Using fewer trees for faster execution
        num_burn_in=100,  # Using fewer burn-in iterations for faster execution
        num_iterations_after_burn_in=200,  # Using fewer iterations for faster execution
        alpha=0.95,
        beta=2.0,
        k=2.0,
        q=0.9,
        nu=3.0,
        verbose=True
    )
    
    # Print model summary
    print("\nModel Summary:")
    bart.summary()
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = bart.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    # Get variable importance
    var_importance = bart.get_var_importance()
    print("\nVariable Importance:")
    print(var_importance.sort_values(ascending=False))
    
    # Calculate credible intervals
    print("\nCalculating credible intervals...")
    ci = bart.calc_credible_intervals(X_test, ci_conf=0.95)
    
    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('Actual vs. Predicted Median House Values')
    plt.savefig('california_actual_vs_predicted.png')
    print("Saved plot to 'california_actual_vs_predicted.png'")
    
    # Plot credible intervals for the first 20 test samples
    plt.figure(figsize=(12, 6))
    indices = np.arange(20)
    plt.errorbar(indices, ci['y_hat'][:20],
                 yerr=[(ci['y_hat'][:20] - ci['ci_lower'][:20]),
                       (ci['ci_upper'][:20] - ci['y_hat'][:20])],
                 fmt='o', capsize=5, elinewidth=1, markeredgewidth=1)
    plt.plot(indices, y_test.iloc[:20], 'rx', markersize=8, label='Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Median House Value')
    plt.title('Predictions with 95% Credible Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('california_credible_intervals.png')
    print("Saved plot to 'california_credible_intervals.png'")
    
    # Plot variable importance
    plt.figure(figsize=(12, 6))
    var_importance.sort_values().plot(kind='barh')
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.savefig('california_variable_importance.png')
    print("Saved plot to 'california_variable_importance.png'")
    
    print("\nExample completed successfully!")

finally:
    # Shutdown JVM
    shutdown_jvm()
