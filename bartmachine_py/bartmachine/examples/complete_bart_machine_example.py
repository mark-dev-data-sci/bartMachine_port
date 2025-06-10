"""
Complete BART Machine Example

This script demonstrates a complete example of creating a bartMachine object,
building the model, and making predictions.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
    from bartmachine.zzz import create_bart_machine, bart_machine_get_posterior, get_variable_importance
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def generate_synthetic_data(n_samples=100, n_features=5, seed=42):
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        seed: Random seed.
    
    Returns:
        X: Feature matrix.
        y: Target vector.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    
    # Create a non-linear relationship
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + X[:, 2] * X[:, 3] + 0.1 * np.random.randn(n_samples)
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return X_train, y_train, X_test, y_test

def main():
    """
    Main function to demonstrate a complete bartMachine example.
    """
    try:
        # Initialize the JVM
        logger.info("Initializing JVM...")
        initialize_jvm(max_heap_size="1g")
        
        # Check if the JVM is running
        if is_jvm_running():
            logger.info("JVM is running.")
        else:
            logger.error("JVM is not running.")
            return
        
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        X_train, y_train, X_test, y_test = generate_synthetic_data()
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Create a bartMachine object
        logger.info("Creating a bartMachine object...")
        bart_machine = create_bart_machine(
            X=X_train.tolist(),
            y=y_train.tolist(),
            num_trees=50,
            num_burn_in=250,
            num_iterations_after_burn_in=1000,
            debug_log=True
        )
        logger.info("Successfully created bartMachine object.")
        
        # Make predictions
        logger.info("Making predictions...")
        # Print the class of the bart_machine object
        logger.info(f"bart_machine class: {bart_machine.getClass().getName()}")
        posterior_result = bart_machine_get_posterior(bart_machine, X_test.tolist())
        predictions = posterior_result["y_hat"]
        logger.info(f"Predictions shape: {len(predictions)}")
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((np.array(predictions) - y_test)**2))
        logger.info(f"RMSE: {rmse}")
        
        # Get variable importance
        logger.info("Getting variable importance...")
        var_importance = get_variable_importance(bart_machine)
        logger.info(f"Variable importance: {var_importance}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('BART Machine Predictions vs Actual')
        plt.savefig('bart_predictions.png')
        logger.info("Saved predictions plot to bart_predictions.png")
        
        # Plot variable importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(var_importance)), var_importance)
        plt.xlabel('Variable Index')
        plt.ylabel('Importance')
        plt.title('BART Machine Variable Importance')
        plt.savefig('bart_var_importance.png')
        logger.info("Saved variable importance plot to bart_var_importance.png")
        
        # Shutdown the JVM
        logger.info("Shutting down JVM...")
        shutdown_jvm()
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to shutdown the JVM if it's running
        if is_jvm_running():
            try:
                logger.info("Attempting to shutdown JVM after error...")
                shutdown_jvm()
            except Exception as e2:
                logger.error(f"Error shutting down JVM: {str(e2)}")

if __name__ == "__main__":
    main()
