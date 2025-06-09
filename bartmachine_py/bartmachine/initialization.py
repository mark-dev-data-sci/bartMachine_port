"""
Initialization for bartMachine

This module provides initialization functions for the bartMachine package,
including setting up global variables and utility functions.
"""

import numpy as np
import random
from typing import Optional, Union, List, Dict, Any, Tuple

# Global variables
bartMachine_globals = {}

# Color array
COLORS = np.empty(500, dtype=object)
for i in range(500):
    COLORS[i] = (random.uniform(0, 0.7), random.uniform(0, 0.7), random.uniform(0, 0.7))

# Default number of cores
DEFAULT_BART_NUM_CORES = 1

def set_bart_machine_num_cores(num_cores: int) -> None:
    """
    Set the number of cores to use for bartMachine.
    
    Args:
        num_cores: The number of cores to use.
    """
    bartMachine_globals["BART_NUM_CORES"] = num_cores
    print(f"bartMachine now using {num_cores} cores.")

def bart_machine_num_cores() -> int:
    """
    Get the number of cores in use for bartMachine.
    
    Returns:
        The number of cores in use.
    """
    if "BART_NUM_CORES" in bartMachine_globals:
        return bartMachine_globals["BART_NUM_CORES"]
    else:
        return DEFAULT_BART_NUM_CORES

def set_bart_machine_memory(bart_max_mem: int) -> None:
    """
    Set the maximum memory for bartMachine (deprecated).
    
    Args:
        bart_max_mem: The maximum memory in MB.
    """
    print(f"This method has been deprecated. Please set the JVM memory directly when initializing the JVM.")

def get_var_counts_over_chain(bart_machine: Any, type: str = "splits") -> np.ndarray:
    """
    Get variable counts over the MCMC chain.
    
    Args:
        bart_machine: The BART machine model.
        type: The type of counts to get ("trees" or "splits").
    
    Returns:
        The variable counts.
    """
    check_serialization(bart_machine)  # ensure the Java object exists and fire an error if not

    if type not in ["trees", "splits"]:
        raise ValueError('type must be "trees" or "splits"')
    
    # Call the Java method through the bridge
    from . import is_jvm_running
    if not is_jvm_running():
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    C = bart_machine.java_bart_machine.getCountsForAllAttribute(type)
    
    # Convert Java array to numpy array
    C_np = np.array(C)
    
    # Set column names
    column_names = bart_machine.model_matrix_training_data.columns[:bart_machine.p]
    
    return C_np

def get_var_props_over_chain(bart_machine: Any, type: str = "splits") -> np.ndarray:
    """
    Get variable inclusion proportions over the MCMC chain.
    
    Args:
        bart_machine: The BART machine model.
        type: The type of proportions to get ("trees" or "splits").
    
    Returns:
        The variable inclusion proportions.
    """
    check_serialization(bart_machine)  # ensure the Java object exists and fire an error if not
    
    if type not in ["trees", "splits"]:
        raise ValueError('type must be "trees" or "splits"')
    
    # Call the Java method through the bridge
    from . import is_jvm_running
    if not is_jvm_running():
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    attribute_props = bart_machine.java_bart_machine.getAttributeProps(type)
    
    # Convert Java array to numpy array
    attribute_props_np = np.array(attribute_props)
    
    # Set names
    column_names = bart_machine.model_matrix_training_data.columns[:bart_machine.p]
    
    return attribute_props_np

def sigsq_est(bart_machine: Any) -> float:
    """
    Private function called in summary() to estimate sigma squared.
    
    Args:
        bart_machine: The BART machine model.
    
    Returns:
        The estimated sigma squared.
    """
    # Call the Java method through the bridge
    from . import is_jvm_running
    if not is_jvm_running():
        raise RuntimeError("JVM is not running. Call initialize_jvm() first.")
    
    sigsqs = bart_machine.java_bart_machine.getGibbsSamplesSigsqs()
    
    # Convert Java array to numpy array
    sigsqs_np = np.array(sigsqs)
    
    # Get samples after burn-in
    sigsqs_after_burnin = sigsqs_np[-(bart_machine.num_iterations_after_burn_in):]
    
    return np.mean(sigsqs_after_burnin)

def sample_mode(data: np.ndarray) -> float:
    """
    Calculate the mode of a sample.
    
    Args:
        data: The data to calculate the mode for.
    
    Returns:
        The mode of the data.
    """
    values, counts = np.unique(data, return_counts=True)
    return values[np.argmax(counts)]

def check_serialization(bart_machine: Any) -> None:
    """
    Check if a BART machine is properly serialized.
    
    Args:
        bart_machine: The BART machine model.
    
    Raises:
        RuntimeError: If the BART machine is not serialized.
    """
    if bart_machine.java_bart_machine is None:
        raise RuntimeError(
            "This bartMachine object was loaded but the Java object is not available.\n"
            "Please build bartMachine with proper Java initialization next time.\n"
        )
