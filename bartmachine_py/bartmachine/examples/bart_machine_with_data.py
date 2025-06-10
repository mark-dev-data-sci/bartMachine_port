"""
BART Machine With Data

This script creates a bartMachine object with synthetic data and builds the model.
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
    from bartmachine.zzz import convert_to_java_2d_array, convert_to_java_array
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
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)
    return X, y

def main():
    """
    Main function to test bartMachine with data.
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
        
        # Get the gateway
        from bartmachine.zzz import _gateway
        jvm = _gateway.jvm
        
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        X, y = generate_synthetic_data()
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Convert data to Java arrays
        logger.info("Converting data to Java arrays...")
        X_java = convert_to_java_2d_array(X.tolist(), "double")
        y_java = convert_to_java_array(y.tolist(), "double")
        
        # Get the bartMachine class
        logger.info("Getting bartMachine class...")
        bart_machine_class = jvm.bartMachine.bartMachineRegression
        
        # Try different approaches to create a bartMachine object with data
        
        # Approach 1: Try using the constructor with X and y
        logger.info("Approach 1: Using constructor with X and y...")
        try:
            bart_machine1 = bart_machine_class(X_java, y_java)
            logger.info("Successfully created bartMachine object with constructor.")
            
            # Set some parameters
            bart_machine1.setNumTrees(50)
            bart_machine1.setNumGibbsBurnIn(250)
            bart_machine1.setNumGibbsTotalIterations(1000)
            
            # Build the model
            logger.info("Building the model...")
            bart_machine1.Build()
            logger.info("Successfully built the model.")
        except Exception as e:
            logger.error(f"Failed with Approach 1: {str(e)}")
        
        # Approach 2: Try creating the object first, then setting the data
        logger.info("Approach 2: Creating object first, then setting the data...")
        try:
            bart_machine2 = bart_machine_class()
            
            # Try to set the data using setData
            logger.info("Setting data using setData...")
            
            # Create a Java ArrayList to hold the data
            data_list = jvm.java.util.ArrayList()
            data_list.add(X_java)
            data_list.add(y_java)
            
            # Set the data
            bart_machine2.setData(data_list)
            logger.info("Successfully set data using setData.")
            
            # Set some parameters
            bart_machine2.setNumTrees(50)
            bart_machine2.setNumGibbsBurnIn(250)
            bart_machine2.setNumGibbsTotalIterations(1000)
            
            # Build the model
            logger.info("Building the model...")
            bart_machine2.Build()
            logger.info("Successfully built the model.")
        except Exception as e:
            logger.error(f"Failed with Approach 2: {str(e)}")
        
        # Approach 3: Try adding data row by row
        logger.info("Approach 3: Adding data row by row...")
        try:
            bart_machine3 = bart_machine_class()
            
            # Add data row by row
            logger.info("Adding data row by row...")
            for i in range(X.shape[0]):
                row = X[i].tolist()
                row_java = convert_to_java_array(row, "double")
                bart_machine3.addTrainingDataRow(row_java, float(y[i]))
            
            # Finalize the data
            logger.info("Finalizing the data...")
            bart_machine3.finalizeTrainingData()
            logger.info("Successfully finalized the data.")
            
            # Set some parameters
            bart_machine3.setNumTrees(50)
            bart_machine3.setNumGibbsBurnIn(250)
            bart_machine3.setNumGibbsTotalIterations(1000)
            
            # Build the model
            logger.info("Building the model...")
            bart_machine3.Build()
            logger.info("Successfully built the model.")
        except Exception as e:
            logger.error(f"Failed with Approach 3: {str(e)}")
        
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
