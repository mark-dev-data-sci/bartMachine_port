"""
BART Machine Data Format

This script tries to understand the exact format expected by the setData method.
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

def generate_synthetic_data(n_samples=10, n_features=2, seed=42):
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
    Main function to test bartMachine data format.
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
        
        # Print the data for debugging
        logger.info("X data:")
        for i in range(X.shape[0]):
            logger.info(f"  Row {i}: {X[i]}")
        logger.info("y data:")
        logger.info(f"  {y}")
        
        # Try different approaches to set the data
        
        # Approach 1: Try to understand the format expected by setData
        logger.info("Approach 1: Understanding the format expected by setData...")
        try:
            # Create a bartMachine object
            bart_machine_class = jvm.bartMachine.bartMachineRegression
            bart_machine = bart_machine_class()
            
            # Create a Java ArrayList to hold the data
            data_list = jvm.java.util.ArrayList()
            
            # Try to add X as a 1D array of 1D arrays
            logger.info("Adding X as a 1D array of 1D arrays...")
            X_rows = jvm.java.util.ArrayList()
            for i in range(X.shape[0]):
                row = X[i].tolist()
                row_java = convert_to_java_array(row, "double")
                X_rows.add(row_java)
            data_list.add(X_rows)
            
            # Add y as a 1D array
            logger.info("Adding y as a 1D array...")
            y_java = convert_to_java_array(y.tolist(), "double")
            data_list.add(y_java)
            
            # Set the data
            logger.info("Setting the data...")
            bart_machine.setData(data_list)
            logger.info("Successfully set the data.")
            
            # Set some parameters
            bart_machine.setNumTrees(50)
            bart_machine.setNumGibbsBurnIn(250)
            bart_machine.setNumGibbsTotalIterations(1000)
            
            # Build the model
            logger.info("Building the model...")
            bart_machine.Build()
            logger.info("Successfully built the model.")
        except Exception as e:
            logger.error(f"Failed with Approach 1: {str(e)}")
        
        # Approach 2: Try using string arrays for addTrainingDataRow
        logger.info("Approach 2: Using string arrays for addTrainingDataRow...")
        try:
            # Create a bartMachine object
            bart_machine_class = jvm.bartMachine.bartMachineRegression
            bart_machine = bart_machine_class()
            
            # Add data row by row using string arrays
            logger.info("Adding data row by row using string arrays...")
            for i in range(X.shape[0]):
                # Convert row to string array
                row = X[i].tolist()
                row_str = [str(x) for x in row]
                row_str.append(str(y[i]))  # Add y value as the last element
                
                # Convert to Java string array
                row_java = _gateway.new_array(jvm.java.lang.String, len(row_str))
                for j, val in enumerate(row_str):
                    row_java[j] = val
                
                # Add the row
                bart_machine.addTrainingDataRow(row_java)
            
            # Finalize the data
            logger.info("Finalizing the data...")
            bart_machine.finalizeTrainingData()
            logger.info("Successfully finalized the data.")
            
            # Set some parameters
            bart_machine.setNumTrees(50)
            bart_machine.setNumGibbsBurnIn(250)
            bart_machine.setNumGibbsTotalIterations(1000)
            
            # Build the model
            logger.info("Building the model...")
            bart_machine.Build()
            logger.info("Successfully built the model.")
        except Exception as e:
            logger.error(f"Failed with Approach 2: {str(e)}")
        
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
