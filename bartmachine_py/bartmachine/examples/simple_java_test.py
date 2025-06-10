"""
Simple test for Java interoperability.

This script tests basic Java interoperability with Py4J.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway
    from bartmachine.bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def main():
    """
    Main function to test Java interoperability.
    """
    try:
        # Initialize the JVM
        logger.info("Initializing JVM...")
        initialize_jvm(max_heap_size="1g", debug=True, verbose=True)
        
        # Check if the JVM is running
        if is_jvm_running():
            logger.info("JVM is running.")
        else:
            logger.error("JVM is not running.")
            return
        
        # Get the gateway
        from bartmachine.zzz import _gateway
        
        # Try to create a simple Java object
        logger.info("Creating a simple Java object...")
        
        # Create a Java ArrayList
        array_list = _gateway.jvm.java.util.ArrayList()
        array_list.add("Hello")
        array_list.add("World")
        
        logger.info(f"ArrayList contents: {list(array_list)}")
        
        # Try to get the bartMachine class
        logger.info("Getting bartMachine class...")
        bart_machine_class = _gateway.jvm.bartMachine.bartMachineRegression
        
        # Print the class methods
        logger.info("Printing class methods...")
        methods = dir(bart_machine_class)
        logger.info(f"Methods: {methods}")
        
        # Try to get the constructor using Java reflection
        logger.info("Getting constructor using Java reflection...")
        class_object = _gateway.jvm.java.lang.Class.forName("bartMachine.bartMachineRegression")
        constructors = class_object.getConstructors()
        
        for constructor in constructors:
            logger.info(f"Constructor: {constructor}")
            param_types = constructor.getParameterTypes()
            logger.info(f"Parameter types: {[param_type.getName() for param_type in param_types]}")
        
        # Get all methods of the class
        logger.info("Getting all methods of the class...")
        methods = class_object.getMethods()
        
        for method in methods:
            method_name = method.getName()
            if method_name.startswith("set") or method_name.startswith("init") or method_name.startswith("build"):
                logger.info(f"Method: {method}")
                param_types = method.getParameterTypes()
                logger.info(f"Parameter types: {[param_type.getName() for param_type in param_types]}")
        
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
