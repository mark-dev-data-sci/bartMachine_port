"""
Java Bridge Example

This example demonstrates how to use the Java bridge to interact with the Java backend
of the bartMachine package.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import the bartmachine package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bartmachine.zzz import (
    initialize_jvm, shutdown_jvm, create_bart_machine, predict_bart_machine,
    get_variable_importance, get_variable_inclusion_proportions
)

def main():
    """
    Main function to demonstrate the Java bridge.
    """
    print("Initializing JVM...")
    initialize_jvm(debug=True)
    
    print("\nCreating synthetic data...")
    # Create synthetic data
    np.random.seed(123)
    n = 100
    p = 5
    X = np.random.randn(n, p)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n)
    
    # Convert to list for Java bridge
    X_list = X.tolist()
    y_list = y.tolist()
    
    print("\nCreating BART machine...")
    # Create a BART machine
    bart = create_bart_machine(
        X=X_list,
        y=y_list,
        num_trees=10,
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        debug_log=True
    )
    
    print("\nMaking predictions...")
    # Create test data
    X_test = np.random.randn(10, p).tolist()
    
    # Make predictions
    predictions = predict_bart_machine(bart, X_test)
    print(f"Predictions: {predictions}")
    
    print("\nGetting variable importance...")
    # Get variable importance
    var_importance = get_variable_importance(bart)
    print(f"Variable importance: {var_importance}")
    
    print("\nGetting variable inclusion proportions...")
    # Get variable inclusion proportions
    var_inclusion = get_variable_inclusion_proportions(bart)
    print(f"Variable inclusion proportions: {var_inclusion}")
    
    print("\nShutting down JVM...")
    shutdown_jvm()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
