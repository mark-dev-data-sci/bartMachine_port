"""
Simple Predict Example

This script demonstrates a simple example of creating a bartMachine object,
building the model, and making predictions using the predict method.
"""

import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
    from bartmachine.zzz import create_bart_machine_object, convert_to_java_2d_array, convert_from_java_array, convert_to_java_array
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
    Main function to demonstrate a simple predict example.
    """
    try:
        # Initialize the JVM
        logger.info("Initializing JVM...")
        initialize_jvm(max_memory="1g")
        
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
        bart_machine = create_bart_machine_object(
            X=X_train,
            y=y_train,
            num_trees=50,
            num_burn_in=250,
            num_iterations_after_burn_in=1000,
            verbose=True
        )
        logger.info("Successfully created bartMachine object.")
        
        # Explicitly call Build() to make sure the model is built
        logger.info("Building the BART machine model...")
        bart_machine.Build()
        logger.info("Successfully built the BART machine model.")
        
        # Print the class of the bart_machine object
        logger.info(f"bart_machine class: {bart_machine.getClass().getName()}")
        
        # List available methods
        logger.info("Available methods:")
        for method in bart_machine.getClass().getMethods():
            logger.info(f"  {method.getName()}")
        
        # Make predictions using the getGibbsSamplesForPrediction method
        logger.info("Making predictions...")
        
        # Convert X_test to a Java array
        logger.info("Converting X_test to a Java array...")
        X_test_java = convert_to_java_2d_array(X_test.tolist(), "double")
        
        # Call getGibbsSamplesForPrediction
        logger.info("Calling getGibbsSamplesForPrediction...")
        y_hat_posterior_samples = bart_machine.getGibbsSamplesForPrediction(X_test_java, 1)
        
        # Convert to numpy array
        logger.info("Converting posterior samples to numpy array...")
        y_hat_posterior_samples_np = convert_from_java_array(y_hat_posterior_samples)
        
        # Take the mean of the posterior samples to get predictions
        logger.info("Calculating predictions...")
        predictions = np.mean(y_hat_posterior_samples_np, axis=1)
        
        logger.info(f"Predictions shape: {len(predictions)}")
        logger.info(f"First 5 predictions: {predictions[:5]}")
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((np.array(predictions) - y_test)**2))
        logger.info(f"RMSE: {rmse}")
        
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
