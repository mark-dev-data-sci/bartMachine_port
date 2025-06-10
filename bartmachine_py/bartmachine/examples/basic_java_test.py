"""
Basic Java Test

This script tests basic Java functionality through the Py4J bridge.
It creates simple Java objects and performs basic operations on them.
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
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def main():
    """
    Main function to test basic Java functionality.
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
        
        # Test 1: Create a Java StringBuilder (won't be auto-converted to Python)
        logger.info("Test 1: Creating a Java StringBuilder...")
        string_builder = jvm.java.lang.StringBuilder("Hello from Java!")
        logger.info(f"Java StringBuilder: {string_builder}")
        logger.info(f"Java StringBuilder length: {string_builder.length()}")
        string_builder.append(" More text.")
        logger.info(f"Java StringBuilder after append: {string_builder}")
        
        # Test 2: Create a Java ArrayList
        logger.info("Test 2: Creating a Java ArrayList...")
        array_list = jvm.java.util.ArrayList()
        array_list.add("Item 1")
        array_list.add("Item 2")
        array_list.add("Item 3")
        logger.info(f"Java ArrayList size: {array_list.size()}")
        logger.info(f"Java ArrayList contents: {[item for item in array_list]}")
        
        # Test 3: Create a Java primitive array
        logger.info("Test 3: Creating a Java primitive array...")
        int_array = _gateway.new_array(jvm.int, 5)
        for i in range(5):
            int_array[i] = i * 10
        logger.info(f"Java int array: {[int_array[i] for i in range(5)]}")
        
        # Test 4: Create a Java 2D array
        logger.info("Test 4: Creating a Java 2D array...")
        double_2d_array = _gateway.new_array(jvm.double, 3, 2)
        for i in range(3):
            for j in range(2):
                double_2d_array[i][j] = i + j * 0.5
        
        # Print the 2D array
        logger.info("Java 2D array contents:")
        for i in range(3):
            row = [double_2d_array[i][j] for j in range(2)]
            logger.info(f"  Row {i}: {row}")
        
        # Test 5: Call a static Java method
        logger.info("Test 5: Calling a static Java method...")
        math = jvm.java.lang.Math
        # Pass a double value explicitly
        logger.info(f"Math.sqrt(16.0): {math.sqrt(16.0)}")
        # For max, we need to specify the types
        logger.info(f"Math.max(10, 20): {math.max(10, 20)}")
        
        # Test 6: Create a Java Date
        logger.info("Test 6: Creating a Java Date...")
        date = jvm.java.util.Date()
        logger.info(f"Current date: {date}")
        logger.info(f"Current time in milliseconds: {date.getTime()}")
        
        # Test 7: Test Java exception handling
        logger.info("Test 7: Testing Java exception handling...")
        try:
            # This will cause an ArrayIndexOutOfBoundsException
            int_array[10] = 100
        except Exception as e:
            logger.info(f"Caught expected exception: {str(e)}")
        
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
