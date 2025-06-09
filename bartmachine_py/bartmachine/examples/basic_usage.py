"""
Basic usage example for bartMachine.

This script demonstrates how to use the bartMachine package for regression and classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from bartmachine import bart_machine, initialize_jvm, shutdown_jvm

def regression_example():
    """
    Example of using bartMachine for regression.
    """
    print("\n=== Regression Example ===\n")
    
    # Generate synthetic data
    np.random.seed(123)
    n = 200
    p = 5
    X = np.random.normal(0, 1, (n, p))
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 1, n)
    
    # Convert to pandas DataFrame and Series
    X_df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    y_series = pd.Series(y, name="y")
    
    # Split data into training and test sets
    train_size = int(0.8 * n)
    X_train = X_df.iloc[:train_size]
    y_train = y_series.iloc[:train_size]
    X_test = X_df.iloc[train_size:]
    y_test = y_series.iloc[train_size:]
    
    # Create and build a BART machine model
    print("Building BART regression model...")
    bart = bart_machine(
        X=X_train, 
        y=y_train,
        num_trees=50,
        num_burn_in=100,
        num_iterations_after_burn_in=200,
        seed=123
    )
    
    # Make predictions
    print("Making predictions...")
    y_pred = bart.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"RMSE: {rmse:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('BART Regression: Actual vs Predicted')
    plt.savefig('regression_results.png')
    print("Plot saved as 'regression_results.png'")

def classification_example():
    """
    Example of using bartMachine for classification.
    """
    print("\n=== Classification Example ===\n")
    
    # Generate synthetic data
    np.random.seed(123)
    n = 200
    p = 5
    X = np.random.normal(0, 1, (n, p))
    # Create a binary outcome based on a nonlinear function
    y_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1]**2 - X[:, 2] * X[:, 3] + X[:, 4])))
    y = (np.random.uniform(0, 1, n) < y_prob).astype(int)
    
    # Convert to pandas DataFrame and Series
    X_df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    y_series = pd.Series(y, name="y").astype('category')
    
    # Split data into training and test sets
    train_size = int(0.8 * n)
    X_train = X_df.iloc[:train_size]
    y_train = y_series.iloc[:train_size]
    X_test = X_df.iloc[train_size:]
    y_test = y_series.iloc[train_size:]
    
    # Create and build a BART machine model
    print("Building BART classification model...")
    bart = bart_machine(
        X=X_train, 
        y=y_train,
        num_trees=50,
        num_burn_in=100,
        num_iterations_after_burn_in=200,
        seed=123
    )
    
    # Make predictions
    print("Making predictions...")
    y_pred = bart.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_test == y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Get probability predictions
    y_prob = bart.predict(X_test, type="prob")
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BART Classification: ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('classification_results.png')
    print("Plot saved as 'classification_results.png'")

if __name__ == "__main__":
    # Initialize the JVM
    print("Initializing JVM...")
    initialize_jvm()
    
    try:
        # Run examples
        regression_example()
        classification_example()
    finally:
        # Shutdown the JVM
        print("\nShutting down JVM...")
        shutdown_jvm()
