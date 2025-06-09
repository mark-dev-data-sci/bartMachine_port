"""
Java Bridge for bartMachine

This module provides the bridge between Python and Java for the bartMachine package.
It handles the JVM lifecycle, loading Java classes from the bartMachine JAR,
and provides wrapper functions for key Java methods.
"""

import os
import sys
import logging
from typing import Optional, Union, List, Dict, Any, Tuple

# We'll use Py4J as our Java-Python bridge
try:
    from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway, JavaObject
    from py4j.protocol import Py4JJavaError
except ImportError:
    raise ImportError("Py4J is required for Java interoperability. Please install it with 'pip install py4j'.")

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
_gateway = None
_java_port = None
_java_process = None
_is_jvm_running = False

# Path to the bartMachine JAR file
# This will be set during initialization
_bart_jar_path = None

def find_jar_path() -> str:
    """
    Find the path to the bartMachine JAR file.
    
    Returns:
        str: The path to the bartMachine JAR file.
    
    Raises:
        FileNotFoundError: If the JAR file cannot be found.
    """
    # Check common locations for the JAR file
    possible_locations = [
        # Current directory
        os.path.join(os.getcwd(), "bart_java.jar"),
        # Package directory
        os.path.join(os.path.dirname(__file__), "java", "bart_java.jar"),
        # User-specified location through environment variable
        os.environ.get("BART_JAR_PATH"),
        # Original R package location
        "/Users/mark/Documents/Cline/bartMachine/bartMachine/inst/java/bart_java.jar"
    ]
    
    for location in possible_locations:
        if location and os.path.exists(location):
            return location
    
    raise FileNotFoundError(
        "Could not find the bartMachine JAR file. Please set the BART_JAR_PATH environment variable "
        "to the location of the JAR file, or place the JAR file in the current directory."
    )

def initialize_jvm(jar_path: Optional[str] = None, max_memory: str = "1024m", 
                  debug: bool = False) -> None:
    """
    Initialize the JVM and set up the Java gateway.
    
    Args:
        jar_path: Path to the bartMachine JAR file. If None, will try to find it automatically.
        max_memory: Maximum memory allocation for the JVM.
        debug: Whether to print debug information.
    
    Raises:
        RuntimeError: If the JVM cannot be initialized.
    """
    global _gateway, _java_port, _java_process, _is_jvm_running, _bart_jar_path
    
    if _is_jvm_running:
        logger.warning("JVM is already running. Call shutdown_jvm() first if you want to restart it.")
        return
    
    try:
        # Find the JAR path if not provided
        if jar_path is None:
            jar_path = find_jar_path()
        
        _bart_jar_path = jar_path
        
        if debug:
            logger.info(f"Using bartMachine JAR at: {_bart_jar_path}")
        
        # Set up the classpath with all JAR files
        jar_dir = os.path.dirname(_bart_jar_path)
        jar_files = [os.path.join(jar_dir, f) for f in os.listdir(jar_dir) if f.endswith('.jar')]
        classpath = os.pathsep.join(jar_files)
        
        # Launch the JVM
        _java_port = launch_gateway(
            classpath=classpath,
            die_on_exit=True,
            java_options=[f"-Xmx{max_memory}"]
        )
        
        # Connect to the gateway
        _gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=_java_port)
        )
        
        _is_jvm_running = True
        
        if debug:
            logger.info("JVM initialized successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize JVM: {str(e)}")
        raise RuntimeError(f"Failed to initialize JVM: {str(e)}")

def shutdown_jvm() -> None:
    """
    Shutdown the JVM and clean up resources.
    """
    global _gateway, _java_port, _java_process, _is_jvm_running
    
    if not _is_jvm_running:
        logger.warning("JVM is not running.")
        return
    
    try:
        if _gateway is not None:
            _gateway.shutdown()
            _gateway = None
        
        _is_jvm_running = False
        _java_port = None
        _java_process = None
        
        logger.info("JVM shutdown successfully.")
        
    except Exception as e:
        logger.error(f"Failed to shutdown JVM: {str(e)}")
        raise RuntimeError(f"Failed to shutdown JVM: {str(e)}")

def is_jvm_running() -> bool:
    """
    Check if the JVM is running.
    
    Returns:
        bool: True if the JVM is running, False otherwise.
    """
    return _is_jvm_running

def get_java_class(class_name: str) -> Any:
    """
    Get a Java class by name.
    
    Args:
        class_name: The fully qualified name of the Java class.
    
    Returns:
        The Java class.
    
    Raises:
        RuntimeError: If the JVM is not running or the class cannot be found.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        return _gateway.jvm.__getattr__(class_name)
    except Py4JJavaError as e:
        logger.error(f"Failed to get Java class {class_name}: {str(e)}")
        raise RuntimeError(f"Failed to get Java class {class_name}: {str(e)}")

def convert_to_java_array(python_array: List, element_type: str) -> JavaObject:
    """
    Convert a Python list to a Java array.
    
    Args:
        python_array: The Python list to convert.
        element_type: The type of the elements in the array (e.g., "double", "int", "String").
    
    Returns:
        The Java array.
    
    Raises:
        RuntimeError: If the JVM is not running or the conversion fails.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        array_class = get_java_class(f"[L{element_type};")
        java_array = _gateway.new_array(array_class, len(python_array))
        
        for i, item in enumerate(python_array):
            java_array[i] = item
        
        return java_array
    
    except Exception as e:
        logger.error(f"Failed to convert Python array to Java array: {str(e)}")
        raise RuntimeError(f"Failed to convert Python array to Java array: {str(e)}")

def convert_to_java_2d_array(python_2d_array: List[List], element_type: str) -> JavaObject:
    """
    Convert a Python 2D list to a Java 2D array.
    
    Args:
        python_2d_array: The Python 2D list to convert.
        element_type: The type of the elements in the array (e.g., "double", "int", "String").
    
    Returns:
        The Java 2D array.
    
    Raises:
        RuntimeError: If the JVM is not running or the conversion fails.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        rows = len(python_2d_array)
        cols = len(python_2d_array[0]) if rows > 0 else 0
        
        array_class = get_java_class(f"[[L{element_type};")
        java_array = _gateway.new_array(array_class, rows, cols)
        
        for i, row in enumerate(python_2d_array):
            for j, item in enumerate(row):
                java_array[i][j] = item
        
        return java_array
    
    except Exception as e:
        logger.error(f"Failed to convert Python 2D array to Java 2D array: {str(e)}")
        raise RuntimeError(f"Failed to convert Python 2D array to Java 2D array: {str(e)}")

def convert_from_java_array(java_array: JavaObject) -> List:
    """
    Convert a Java array to a Python list.
    
    Args:
        java_array: The Java array to convert.
    
    Returns:
        The Python list.
    
    Raises:
        RuntimeError: If the conversion fails.
    """
    try:
        return list(java_array)
    except Exception as e:
        logger.error(f"Failed to convert Java array to Python list: {str(e)}")
        raise RuntimeError(f"Failed to convert Java array to Python list: {str(e)}")

def convert_from_java_2d_array(java_array: JavaObject) -> List[List]:
    """
    Convert a Java 2D array to a Python 2D list.
    
    Args:
        java_array: The Java 2D array to convert.
    
    Returns:
        The Python 2D list.
    
    Raises:
        RuntimeError: If the conversion fails.
    """
    try:
        return [list(row) for row in java_array]
    except Exception as e:
        logger.error(f"Failed to convert Java 2D array to Python 2D list: {str(e)}")
        raise RuntimeError(f"Failed to convert Java 2D array to Python 2D list: {str(e)}")

# Wrapper functions for key Java methods

def set_seed(seed: int) -> None:
    """
    Set the random seed for the Java implementation.
    
    Args:
        seed: The random seed.
    
    Raises:
        RuntimeError: If the JVM is not running or the method call fails.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        stat_toolbox = get_java_class("bartMachine.StatToolbox")
        stat_toolbox.setSeed(seed)
    except Exception as e:
        logger.error(f"Failed to set random seed: {str(e)}")
        raise RuntimeError(f"Failed to set random seed: {str(e)}")

def create_bart_machine(X: List[List[float]], y: List[float], num_trees: int = 50,
                       num_burn_in: int = 250, num_iterations_after_burn_in: int = 1000,
                       alpha: float = 0.95, beta: float = 2, k: float = 2, q: float = 0.9,
                       nu: float = 3, prob_rule_class: float = 0.5,
                       mh_prob_steps: List[float] = None, debug_log: bool = False,
                       run_in_sample: bool = True, s_sq_y: float = None,
                       sig_sq: float = None, seed: int = None) -> JavaObject:
    """
    Create a BartMachine object in Java.
    
    Args:
        X: The predictor variables.
        y: The response variable.
        num_trees: Number of trees in the ensemble.
        num_burn_in: Number of burn-in MCMC iterations.
        num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
        alpha: Prior parameter for tree structure.
        beta: Prior parameter for tree structure.
        k: Prior parameter for leaf values.
        q: Prior parameter for leaf values.
        nu: Prior parameter for error variance.
        prob_rule_class: Probability of using a classification rule.
        mh_prob_steps: Metropolis-Hastings proposal step probabilities.
        debug_log: Whether to print debug information.
        run_in_sample: Whether to run in-sample prediction.
        s_sq_y: Sample variance of y.
        sig_sq: Error variance.
        seed: Random seed.
    
    Returns:
        The Java BartMachine object.
    
    Raises:
        RuntimeError: If the JVM is not running or the method call fails.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        # Set default for mh_prob_steps if None
        if mh_prob_steps is None:
            mh_prob_steps = [2.5/9, 2.5/9, 4/9]
        
        # Convert Python arrays to Java arrays
        X_java = convert_to_java_2d_array(X, "double")
        y_java = convert_to_java_array(y, "double")
        mh_prob_steps_java = convert_to_java_array(mh_prob_steps, "double")
        
        # Get the BartMachine class
        bart_machine_class = get_java_class("bartMachine.bartMachineRegression")
        
        # Create the BartMachine object
        bart_machine = bart_machine_class(
            X_java, y_java, num_trees, num_burn_in, num_iterations_after_burn_in,
            alpha, beta, k, q, nu, prob_rule_class,
            mh_prob_steps_java, debug_log, run_in_sample,
            s_sq_y, sig_sq, seed
        )
        
        return bart_machine
    
    except Exception as e:
        logger.error(f"Failed to create BartMachine: {str(e)}")
        raise RuntimeError(f"Failed to create BartMachine: {str(e)}")

def build_bart_machine(bart_machine: JavaObject) -> None:
    """
    Build a BartMachine model.
    
    Args:
        bart_machine: The Java BartMachine object.
    
    Raises:
        RuntimeError: If the JVM is not running or the method call fails.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        bart_machine.build()
    except Exception as e:
        logger.error(f"Failed to build BartMachine: {str(e)}")
        raise RuntimeError(f"Failed to build BartMachine: {str(e)}")

def predict_bart_machine(bart_machine: JavaObject, X_test: List[List[float]]) -> List[float]:
    """
    Make predictions with a BartMachine model.
    
    Args:
        bart_machine: The Java BartMachine object.
        X_test: The test data.
    
    Returns:
        The predictions.
    
    Raises:
        RuntimeError: If the JVM is not running or the method call fails.
    """
    if not _is_jvm_running:
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    try:
        # Convert Python array to Java array
        X_test_java = convert_to_java_2d_array(X_test, "double")
        
        # Make predictions
        predictions_java = bart_machine.predict(X_test_java)
        
        # Convert Java array to Python list
        predictions = convert_from_java_array(predictions_java)
        
        return predictions
    
    except Exception as e:
        logger.error(f"Failed to make predictions: {str(e)}")
        raise RuntimeError(f"Failed to make predictions: {str(e)}")
