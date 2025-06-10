"""
Example of using the BART Machine with the updated Java bridge.

This script demonstrates how to create a BART Machine model using the updated Java bridge,
which uses setter methods instead of constructor parameters.
"""

import os
import sys
import logging
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
    from bartmachine.zzz import create_bart_machine, predict_bart_machine, get_variable_importance
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def main():
    """
    Main function to demonstrate BART Machine usage.
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
        X, y = make_friedman1(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to lists for Java bridge
        X_train_list = X_train.tolist()
        y_train_list = y_train.tolist()
        X_test_list = X_test.tolist()
        
        # Create a BART Machine model
        logger.info("Creating BART Machine model...")
        bart_machine = create_bart_machine(
            X=X_train_list,
            y=y_train_list,
            num_trees=50,
            num_burn_in=200,
            num_iterations_after_burn_in=1000,
            alpha=0.95,
            beta=2,
            k=2,
            q=0.9,
            nu=3,
            debug_log=True
        )
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = predict_bart_machine(bart_machine, X_test_list)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((np.array(predictions) - y_test) ** 2))
        logger.info(f"RMSE: {rmse:.4f}")
        
        # Get variable importance
        logger.info("Getting variable importance...")
        var_importance = get_variable_importance(bart_machine)
        for i, importance in enumerate(var_importance):
            logger.info(f"Variable {i}: {importance:.4f}")
        
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
