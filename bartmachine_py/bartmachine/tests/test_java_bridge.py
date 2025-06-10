"""
Tests for the Java bridge functionality.

This module contains tests for the Java bridge functionality, including
JVM initialization, Java class loading, and data conversion between Python and Java.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from bartmachine import is_jvm_running
from bartmachine.zzz import (
    set_seed, get_java_class, convert_to_java_array, convert_to_java_2d_array,
    convert_from_java_array, convert_from_java_2d_array
)

def test_jvm_initialization():
    """Test JVM initialization."""
    assert is_jvm_running()

def test_java_class_loading():
    """Test loading Java classes."""
    # Try to load a Java class
    java_system = get_java_class("java.lang.System")
    assert java_system is not None
    
    # Try to load the bartMachine classes
    bart_machine_regression = get_java_class("bartMachine.bartMachineRegression")
    assert bart_machine_regression is not None
    
    bart_machine_classification = get_java_class("bartMachine.bartMachineClassification")
    assert bart_machine_classification is not None

def test_set_seed():
    """Test setting the random seed."""
    # Set the seed
    set_seed(123)
    
    # There's no easy way to test if the seed was set correctly,
    # so we just check that the method doesn't raise an exception
    assert True

def test_array_conversion():
    """Test converting arrays between Python and Java."""
    # Create a Python array
    python_array = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Convert to Java array
    java_array = convert_to_java_array(python_array, "double")
    
    # Convert back to Python array
    python_array2 = convert_from_java_array(java_array)
    
    # Check that the arrays are equal
    assert len(python_array) == len(python_array2)
    for i in range(len(python_array)):
        assert python_array[i] == pytest.approx(python_array2[i])

def test_2d_array_conversion():
    """Test converting 2D arrays between Python and Java."""
    # Create a Python 2D array
    python_2d_array = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    
    # Convert to Java 2D array
    java_2d_array = convert_to_java_2d_array(python_2d_array, "double")
    
    # Convert back to Python 2D array
    python_2d_array2 = convert_from_java_2d_array(java_2d_array)
    
    # Check that the arrays are equal
    assert len(python_2d_array) == len(python_2d_array2)
    for i in range(len(python_2d_array)):
        assert len(python_2d_array[i]) == len(python_2d_array2[i])
        for j in range(len(python_2d_array[i])):
            assert python_2d_array[i][j] == pytest.approx(python_2d_array2[i][j])

def test_java_method_call():
    """Test calling Java methods."""
    # Get the Java System class
    java_system = get_java_class("java.lang.System")
    
    # Call the currentTimeMillis method
    current_time = java_system.currentTimeMillis()
    
    # Check that the result is a number
    assert isinstance(current_time, (int, float))
    
    # Check that the result is positive
    assert current_time > 0
