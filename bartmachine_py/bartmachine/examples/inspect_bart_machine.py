"""
Inspect BART Machine

This script inspects the bartMachine class to understand its methods and how to properly set data on it.
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
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def main():
    """
    Main function to inspect the bartMachine class.
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
        
        # Get the bartMachine class
        logger.info("Getting bartMachine class...")
        bart_machine_class = jvm.bartMachine.bartMachineRegression
        
        # Create an instance of the class
        logger.info("Creating an instance of the class...")
        bart_machine = bart_machine_class()
        
        # Inspect the class
        logger.info("Inspecting bartMachine class...")
        
        # Get the class object
        class_obj = bart_machine.getClass()
        
        # Get all constructors
        logger.info("Constructors:")
        constructors = class_obj.getConstructors()
        for constructor in constructors:
            logger.info(f"  {constructor}")
        
        # Get all methods
        logger.info("Methods:")
        methods = class_obj.getMethods()
        for method in methods:
            logger.info(f"  {method}")
        
        # Get all fields
        logger.info("Fields:")
        fields = class_obj.getFields()
        for field in fields:
            logger.info(f"  {field}")
        
        # Try to find methods related to setting data
        logger.info("Methods related to setting data:")
        data_methods = [method for method in methods if "data" in method.getName().lower() or "train" in method.getName().lower()]
        for method in data_methods:
            logger.info(f"  {method}")
        
        # Try to find methods related to building the model
        logger.info("Methods related to building the model:")
        build_methods = [method for method in methods if "build" in method.getName().lower()]
        for method in build_methods:
            logger.info(f"  {method}")
        
        # Try to find methods related to prediction
        logger.info("Methods related to prediction:")
        predict_methods = [method for method in methods if "predict" in method.getName().lower()]
        for method in predict_methods:
            logger.info(f"  {method}")
        
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
