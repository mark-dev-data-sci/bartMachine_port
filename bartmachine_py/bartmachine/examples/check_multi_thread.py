"""
Multi-threaded BART Machine Example

This script demonstrates the use of multi-threading in the BART machine implementation.
It compares the performance of single-threaded and multi-threaded BART machines.
"""

import os
import sys
import time
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
    from bartmachine.zzz import (
        create_bart_machine, convert_to_java_2d_array, convert_from_java_array, 
        set_num_cores, bart_machine_get_posterior
    )
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def generate_synthetic_data(n_samples=500, n_features=10, seed=42):
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

def time_prediction(bart_machine, X_test, num_cores):
    """
    Time the prediction process.
    
    Args:
        bart_machine: The BART machine object.
        X_test: The test data.
        num_cores: Number of cores to use for prediction.
    
    Returns:
        elapsed_time: The time taken for prediction.
        predictions: The predictions.
    """
    # Time the prediction
    start_time = time.time()
    posterior_result = bart_machine_get_posterior(bart_machine, X_test.tolist(), num_cores)
    elapsed_time = time.time() - start_time
    
    # Get predictions from posterior samples
    predictions = posterior_result["y_hat"]
    
    return elapsed_time, predictions

def main():
    """
    Main function to demonstrate multi-threading.
    """
    try:
        # Initialize the JVM with more memory
        logger.info("Initializing JVM...")
        initialize_jvm(max_heap_size="2g")
        
        # Check if the JVM is running
        if is_jvm_running():
            logger.info("JVM is running.")
        else:
            logger.error("JVM is not running.")
            return
        
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        X_train, y_train, X_test, y_test = generate_synthetic_data(n_samples=1000, n_features=20)
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Create a single-threaded BART machine
        logger.info("Creating a single-threaded BART machine...")
        bart_machine_single = create_bart_machine(
            X=X_train.tolist(),
            y=y_train.tolist(),
            num_trees=50,
            num_burn_in=100,
            num_iterations_after_burn_in=100,
            debug_log=False,
            use_multithreaded=False
        )
        logger.info("Successfully created single-threaded BART machine.")
        
        # Create a multi-threaded BART machine
        logger.info("Creating a multi-threaded BART machine...")
        bart_machine_multi = create_bart_machine(
            X=X_train.tolist(),
            y=y_train.tolist(),
            num_trees=50,
            num_burn_in=100,
            num_iterations_after_burn_in=100,
            debug_log=False,
            use_multithreaded=True
        )
        logger.info("Successfully created multi-threaded BART machine.")
        
        # Time single-threaded prediction
        logger.info("Timing single-threaded prediction...")
        single_time, single_predictions = time_prediction(bart_machine_single, X_test, 1)
        logger.info(f"Single-threaded prediction time: {single_time:.4f} seconds")
        
        # Time multi-threaded prediction with 2 cores
        logger.info("Timing multi-threaded prediction with 2 cores...")
        multi_time_2, multi_predictions_2 = time_prediction(bart_machine_multi, X_test, 2)
        logger.info(f"Multi-threaded prediction time (2 cores): {multi_time_2:.4f} seconds")
        
        # Time multi-threaded prediction with 4 cores
        logger.info("Timing multi-threaded prediction with 4 cores...")
        multi_time_4, multi_predictions_4 = time_prediction(bart_machine_multi, X_test, 4)
        logger.info(f"Multi-threaded prediction time (4 cores): {multi_time_4:.4f} seconds")
        
        # Calculate speedup
        speedup_2 = single_time / multi_time_2
        speedup_4 = single_time / multi_time_4
        logger.info(f"Speedup with 2 cores: {speedup_2:.2f}x")
        logger.info(f"Speedup with 4 cores: {speedup_4:.2f}x")
        
        # Calculate RMSE for each method
        rmse_single = np.sqrt(np.mean((single_predictions - y_test)**2))
        rmse_multi_2 = np.sqrt(np.mean((multi_predictions_2 - y_test)**2))
        rmse_multi_4 = np.sqrt(np.mean((multi_predictions_4 - y_test)**2))
        
        logger.info(f"RMSE (single-threaded): {rmse_single:.4f}")
        logger.info(f"RMSE (multi-threaded, 2 cores): {rmse_multi_2:.4f}")
        logger.info(f"RMSE (multi-threaded, 4 cores): {rmse_multi_4:.4f}")
        
        # Check if the predictions are the same
        logger.info("Checking if predictions are the same...")
        max_diff_2 = np.max(np.abs(single_predictions - multi_predictions_2))
        max_diff_4 = np.max(np.abs(single_predictions - multi_predictions_4))
        
        logger.info(f"Maximum difference between single and multi (2 cores): {max_diff_2:.6f}")
        logger.info(f"Maximum difference between single and multi (4 cores): {max_diff_4:.6f}")
        
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
