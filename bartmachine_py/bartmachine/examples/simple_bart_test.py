"""
Simple BART Test

This script tests basic functionality of the bartMachine Java classes.
It creates a simple bartMachine object and checks if it can be instantiated.
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
    Main function to test basic bartMachine functionality.
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
        
        # Test 1: Check if we can access the bartMachine package
        logger.info("Test 1: Checking if we can access the bartMachine package...")
        try:
            # Try to get the bartMachine class
            bart_machine_class = jvm.bartMachine.bartMachineRegression
            logger.info("Successfully accessed bartMachine.bartMachineRegression class.")
        except Exception as e:
            logger.error(f"Failed to access bartMachine.bartMachineRegression class: {str(e)}")
        
        # Test 2: Create a simple bartMachine object
        logger.info("Test 2: Creating a simple bartMachine object...")
        try:
            # Create a simple bartMachine object
            bart_machine = bart_machine_class()
            logger.info("Successfully created bartMachine object.")
            
            # Print some information about the object
            logger.info(f"bartMachine object: {bart_machine}")
            logger.info(f"bartMachine class: {bart_machine.getClass().getName()}")
        except Exception as e:
            logger.error(f"Failed to create bartMachine object: {str(e)}")
        
        # Test 3: Try to set some parameters
        logger.info("Test 3: Setting some parameters...")
        try:
            # Set some parameters
            bart_machine.setNumTrees(50)
            logger.info("Successfully set numTrees parameter.")
            
            bart_machine.setNumGibbsBurnIn(250)
            logger.info("Successfully set numGibbsBurnIn parameter.")
            
            bart_machine.setNumGibbsTotalIterations(1000)
            logger.info("Successfully set numGibbsTotalIterations parameter.")
            
            bart_machine.setVerbose(True)
            logger.info("Successfully set verbose parameter.")
        except Exception as e:
            logger.error(f"Failed to set parameters: {str(e)}")
        
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
