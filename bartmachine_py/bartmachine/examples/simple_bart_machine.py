"""
Simple example to create a BartMachine object without setting any data or parameters.

This script initializes the JVM, creates a BartMachine object, and explores its methods.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
    from bartmachine.zzz import get_java_class
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def main():
    """
    Main function to create a simple BartMachine object.
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
        
        # Get the BartMachine class
        logger.info("Getting BartMachine class...")
        bart_machine_class = get_java_class("bartMachine.bartMachineRegression")
        
        # Create a BartMachine object
        logger.info("Creating BartMachine object...")
        bart_machine = bart_machine_class()
        
        # Print the methods of the BartMachine object
        logger.info("Methods of BartMachine object:")
        for method_name in dir(bart_machine):
            if not method_name.startswith("_"):
                logger.info(f"  {method_name}")
        
        # Try to get the signature of the addTrainingDataRow method
        logger.info("Trying to get the signature of the addTrainingDataRow method...")
        try:
            method = getattr(bart_machine, "addTrainingDataRow")
            logger.info(f"  addTrainingDataRow: {method}")
            
            # Try to get the parameter types
            logger.info("Trying to get the parameter types of the addTrainingDataRow method...")
            try:
                # Use Java reflection to get the method
                method_obj = bart_machine.getClass().getMethod("addTrainingDataRow", None)
                logger.info(f"  Parameter types: {method_obj.getParameterTypes()}")
            except Exception as e:
                logger.error(f"Error getting parameter types: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting method: {str(e)}")
        
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
