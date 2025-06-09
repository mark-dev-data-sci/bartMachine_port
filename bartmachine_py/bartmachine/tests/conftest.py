"""
Pytest configuration file for bartmachine tests.

This file contains fixtures that are shared across multiple test files.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from bartmachine import initialize_jvm, shutdown_jvm, is_jvm_running

# Setup and teardown fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_jvm():
    """Initialize the JVM before all tests and shut it down after all tests."""
    if not is_jvm_running():
        initialize_jvm()
    yield
    if is_jvm_running():
        shutdown_jvm()

@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    # Set random seed
    np.random.seed(123)
    
    # Generate predictors
    n = 100
    p = 5
    X = np.random.normal(0, 1, (n, p))
    
    # Generate response
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 1, n)
    
    # Convert to pandas DataFrame and Series
    X_df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    y_series = pd.Series(y, name="y")
    
    # Split data into training and test sets
    train_size = int(0.8 * n)
    X_train = X_df.iloc[:train_size]
    y_train = y_series.iloc[:train_size]
    X_test = X_df.iloc[train_size:]
    y_test = y_series.iloc[train_size:]
    
    return X_train, y_train, X_test, y_test

@pytest.fixture
def classification_data(synthetic_data):
    """Generate synthetic classification data for testing."""
    X_train, _, X_test, _ = synthetic_data
    
    # Create binary response
    np.random.seed(123)
    y_prob = 1 / (1 + np.exp(-(X_train.iloc[:, 0] + X_train.iloc[:, 1]**2)))
    y_train = pd.Series((np.random.uniform(0, 1, len(X_train)) < y_prob).astype(int), name="y").astype('category')
    
    y_prob_test = 1 / (1 + np.exp(-(X_test.iloc[:, 0] + X_test.iloc[:, 1]**2)))
    y_test = pd.Series((np.random.uniform(0, 1, len(X_test)) < y_prob_test).astype(int), name="y").astype('category')
    
    return X_train, y_train, X_test, y_test
