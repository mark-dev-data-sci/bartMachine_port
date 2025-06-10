"""
Explore the BartMachine class to understand its methods and constructors.

This script initializes the JVM, loads the BartMachine class, and explores its methods
and constructors to understand how to properly create and configure a BartMachine object.
"""

import os
import sys
import logging
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
    from bartmachine.zzz import get_java_class  # Import the function to get Java classes
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def explore_instance(class_name):
    """
    Create an instance of a Java class and explore its methods.
    
    Args:
        class_name: The name of the class to explore.
    """
    try:
        # Get the class using our utility function
        java_class = get_java_class(class_name)
        
        # Create an instance of the class
        instance = java_class()
        
        # Get the instance's methods
        logger.info(f"Methods for {class_name}:")
        for method_name in dir(instance):
            if not method_name.startswith("_"):
                logger.info(f"  {method_name}")
        
        return instance
    except Exception as e:
        logger.error(f"Error exploring class {class_name}: {str(e)}")
        return None

def main():
    """
    Main function to explore the BartMachine class.
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
        
        # Explore the BartMachine class
        logger.info("Exploring BartMachine class...")
        bart_machine = explore_instance("bartMachine.bartMachineRegression")
        
        # Try to get the method signatures
        if bart_machine is not None:
            logger.info("Method signatures:")
            for method_name in dir(bart_machine):
                if not method_name.startswith("_"):
                    try:
                        method = getattr(bart_machine, method_name)
                        if callable(method):
                            logger.info(f"  {method_name}: {method}")
                    except Exception as e:
                        logger.error(f"Error getting method {method_name}: {str(e)}")
        
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
