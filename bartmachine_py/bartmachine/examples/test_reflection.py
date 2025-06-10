"""
Test script for Java reflection approach to access protected methods.

This script tests the Java reflection approach to access the protected
getGibbsSamplesForPrediction method in the bartMachineRegressionMultThread class.
"""

import numpy as np
import pandas as pd
from bartmachine.zzz import (
    initialize_jvm, 
    create_bart_machine, 
    convert_to_java_2d_array,
    convert_from_java_2d_array,
    shutdown_jvm
)
import bartmachine.zzz as jb

def main():
    # Initialize JVM
    print("Initializing JVM...")
    initialize_jvm(debug=True)
    
    # Get the gateway
    gateway = jb._gateway
    
    # Create sample data
    print("Creating sample data...")
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(20)
    
    # Create BART machine
    print("Creating BART machine...")
    bart = create_bart_machine(
        X.tolist(), 
        y.tolist(), 
        num_trees=10, 
        num_burn_in=10, 
        num_iterations_after_burn_in=10
    )
    
    # Create test data
    print("Creating test data...")
    X_test = np.random.randn(5, 5)
    X_test_java = convert_to_java_2d_array(X_test.tolist(), "double")
    
    # Get posterior samples using reflection
    print("Getting posterior samples using reflection...")
    try:
        # Get the method name as a string
        method_name = "getGibbsSamplesForPrediction"
        
        # Get the method using reflection
        method = bart.getClass().getDeclaredMethod(
            method_name,
            X_test_java.getClass(),
            gateway.jvm.java.lang.Integer.TYPE
        )
        
        # Make the method accessible
        method.setAccessible(True)
        
        # Invoke the method
        y_hat_posterior_samples_java = method.invoke(
            bart, 
            X_test_java, 
            1
        )
        
        # Convert the result to Python
        y_hat_posterior_samples = convert_from_java_2d_array(y_hat_posterior_samples_java)
        
        # Calculate y_hat as the mean of posterior samples
        y_hat = [np.mean(samples) for samples in y_hat_posterior_samples]
        
        print("Success! Got posterior samples.")
        print(f"Shape of posterior samples: {np.array(y_hat_posterior_samples).shape}")
        print(f"Mean prediction: {np.mean(y_hat)}")
    except Exception as e:
        print(f"Error getting posterior samples: {str(e)}")
    
    # Shutdown JVM
    print("Shutting down JVM...")
    shutdown_jvm()

if __name__ == "__main__":
    main()
