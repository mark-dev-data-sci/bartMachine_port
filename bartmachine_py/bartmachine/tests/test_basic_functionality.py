"""
Tests for basic functionality of the bartMachine package.

This module contains tests for the basic functionality of the bartMachine package,
including model building, prediction, and other core features.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add the parent directory to the path so we can import the bartmachine package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from bartmachine import bart_machine

def test_regression_model_building(synthetic_data):
    """Test building a regression model."""
    X_train, y_train, _, _ = synthetic_data
    
    # Create and build a BART machine model
    bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=10,  # Use a small number of trees for faster testing
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        seed=123,
        pred_type="regression"  # Explicitly set prediction type
    )
    
    # Check that the model was built successfully
    assert bart.java_bart_machine is not None
    assert bart.num_trees == 10
    assert bart.num_burn_in == 10
    assert bart.num_iterations_after_burn_in == 20
    assert bart.pred_type == "regression"

def test_classification_model_building(classification_data):
    """Test building a classification model."""
    X_train, y_train, _, _ = classification_data
    
    # Create and build a BART machine model
    bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=10,  # Use a small number of trees for faster testing
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        seed=123,
        pred_type="classification"  # Explicitly set prediction type
    )
    
    # Check that the model was built successfully
    assert bart.java_bart_machine is not None
    assert bart.num_trees == 10
    assert bart.num_burn_in == 10
    assert bart.num_iterations_after_burn_in == 20
    assert bart.pred_type == "classification"

def test_regression_prediction(synthetic_data):
    """Test making predictions with a regression model."""
    X_train, y_train, X_test, y_test = synthetic_data
    
    # Create and build a BART machine model
    bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=10,  # Use a small number of trees for faster testing
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        seed=123
    )
    
    # Make predictions
    y_pred = bart.predict(X_test)
    
    # Check that the predictions have the right shape
    assert len(y_pred) == len(X_test)
    
    # Check that the predictions are reasonable
    # (i.e., they should be in the same range as the training data)
    assert np.min(y_pred) >= np.min(y_train) - 5
    assert np.max(y_pred) <= np.max(y_train) + 5

def test_classification_prediction(classification_data):
    """Test making predictions with a classification model."""
    X_train, y_train, X_test, y_test = classification_data
    
    # Create and build a BART machine model
    bart = bart_machine(
        X=X_train,
        y=y_train,
        num_trees=10,  # Use a small number of trees for faster testing
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        seed=123,
        pred_type="classification"  # Explicitly set prediction type
    )
    
    # Make predictions
    y_pred = bart.predict(X_test)
    
    # Check that the predictions have the right shape
    assert len(y_pred) == len(X_test)
    
    # Check that the predictions are probabilities (between 0 and 1)
    assert np.all(y_pred >= 0) and np.all(y_pred <= 1)
    
    # Make probability predictions
    y_prob_pred = bart.predict(X_test, type="prob")
    
    # Check that the probability predictions have the right shape
    assert len(y_prob_pred) == len(X_test)
    
    # Check that the probability predictions are between 0 and 1
    assert np.all(y_prob_pred >= 0)
    assert np.all(y_prob_pred <= 1)

def test_missing_data_handling(synthetic_data):
    """Test handling of missing data."""
    X_train, y_train, X_test, _ = synthetic_data
    
    # Introduce missing values
    X_train_missing = X_train.copy()
    X_test_missing = X_test.copy()
    X_train_missing.iloc[0, 0] = np.nan
    X_train_missing.iloc[1, 1] = np.nan
    X_test_missing.iloc[0, 0] = np.nan
    
    # Create and build a BART machine model with missing data handling
    bart = bart_machine(
        X=X_train_missing,
        y=y_train,
        num_trees=10,  # Use a small number of trees for faster testing
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        replace_missing_data_with_x_j_bar=True,  # Replace missing values with column means
        seed=123
    )
    
    # Make predictions
    y_pred = bart.predict(X_test_missing)
    
    # Check that the predictions have the right shape
    assert len(y_pred) == len(X_test)

def test_categorical_data_handling(synthetic_data):
    """Test handling of categorical data."""
    X_train, y_train, X_test, _ = synthetic_data
    
    # Convert one column to categorical
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    X_train_cat["X1"] = pd.Categorical(["A" if x < 0 else "B" for x in X_train["X1"]])
    X_test_cat["X1"] = pd.Categorical(["A" if x < 0 else "B" for x in X_test["X1"]])
    
    # Create and build a BART machine model
    bart = bart_machine(
        X=X_train_cat,
        y=y_train,
        num_trees=10,  # Use a small number of trees for faster testing
        num_burn_in=10,
        num_iterations_after_burn_in=20,
        seed=123
    )
    
    # Make predictions
    y_pred = bart.predict(X_test_cat)
    
    # Check that the predictions have the right shape
    assert len(y_pred) == len(X_test)
