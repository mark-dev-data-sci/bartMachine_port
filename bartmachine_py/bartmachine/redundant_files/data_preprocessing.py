"""
Data preprocessing utilities for bartMachine.

This module provides functions for preprocessing data for BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bart_package_data_preprocessing.R' 
    in the original R package.

Role in Port:
    This module handles data preprocessing for BART models, including handling missing values,
    factors, and other data types.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

def preprocess_training_data(X: pd.DataFrame, 
                            use_missing_data_dummies_as_covars: bool = False,
                            rf_imputations_for_missing: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Preprocess training data for BART models.
    
    This function preprocesses training data for BART models, handling missing values,
    factors, and other data types.
    
    Args:
        X: The predictor variables.
        use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
        rf_imputations_for_missing: Random forest imputations for missing values.
    
    Returns:
        A dictionary containing preprocessed data and metadata.
    """
    # Check if X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    
    # Get the number of observations and predictors
    n = X.shape[0]
    p = X.shape[1]
    
    # Initialize lists for factor variables and their levels
    factor_vars = []
    factor_levels = []
    factor_lengths = []
    
    # Initialize list for preprocessed data
    X_preprocessed = []
    
    # For each column in X
    for col_name in X.columns:
        col = X[col_name]
        
        # Check if the column is a factor
        if pd.api.types.is_categorical_dtype(col):
            # Add to factor variables
            factor_vars.append(col_name)
            
            # Get the levels
            levels = list(col.cat.categories)
            factor_levels.append(levels)
            factor_lengths.append(len(levels))
            
            # Create dummy variables
            dummies = pd.get_dummies(col, prefix=col_name, drop_first=True)
            
            # Add to preprocessed data
            X_preprocessed.append(dummies)
        else:
            # Add to preprocessed data
            X_preprocessed.append(col.to_frame())
    
    # Concatenate preprocessed data
    if X_preprocessed:
        X_preprocessed = pd.concat(X_preprocessed, axis=1)
    else:
        X_preprocessed = pd.DataFrame()
    
    # Handle missing values
    if use_missing_data_dummies_as_covars:
        # Create missing data dummies
        missing_dummies = pd.DataFrame()
        
        # For each column in X
        for col_name in X.columns:
            col = X[col_name]
            
            # Check if the column has missing values
            if col.isna().any():
                # Create a dummy variable for missing values
                missing_dummies[f"{col_name}_missing"] = col.isna().astype(int)
        
        # Add missing data dummies to preprocessed data
        if not missing_dummies.empty:
            X_preprocessed = pd.concat([X_preprocessed, missing_dummies], axis=1)
    
    # Add random forest imputations for missing values
    if rf_imputations_for_missing is not None:
        # Add random forest imputations to preprocessed data
        X_preprocessed = pd.concat([X_preprocessed, rf_imputations_for_missing], axis=1)
    
    # Convert to numpy array
    X_preprocessed_array = X_preprocessed.to_numpy()
    
    # Return preprocessed data and metadata
    return {
        "data": X_preprocessed_array,
        "factor_vars": factor_vars,
        "factor_levels": factor_levels,
        "factor_lengths": factor_lengths,
        "column_names": X_preprocessed.columns.tolist()
    }

def preprocess_new_data(X_new: pd.DataFrame, 
                       bart_machine: Any) -> np.ndarray:
    """
    Preprocess new data for prediction with a BART model.
    
    This function preprocesses new data for prediction with a BART model,
    ensuring that it has the same structure as the training data.
    
    Args:
        X_new: The new predictor variables.
        bart_machine: The BART machine model.
    
    Returns:
        A numpy array containing preprocessed data.
    """
    # Check if X_new is a dataframe
    if not isinstance(X_new, pd.DataFrame):
        raise ValueError("X_new must be a pandas DataFrame.")
    
    # Check if the model has been built
    if bart_machine.java_bart_machine is None:
        raise ValueError("The model has not been built. Call build() first.")
    
    # Get the training data
    X_train = bart_machine.X
    
    # Check if X_new has the same columns as X_train
    if not set(X_train.columns).issubset(set(X_new.columns)):
        missing_cols = set(X_train.columns) - set(X_new.columns)
        raise ValueError(f"X_new is missing columns: {missing_cols}")
    
    # Initialize lists for preprocessed data
    X_preprocessed = []
    
    # For each column in X_train
    for col_name in X_train.columns:
        col_train = X_train[col_name]
        col_new = X_new[col_name]
        
        # Check if the column is a factor
        if pd.api.types.is_categorical_dtype(col_train):
            # Get the levels
            levels_train = list(col_train.cat.categories)
            
            # Check if X_new has the same levels
            if not pd.api.types.is_categorical_dtype(col_new):
                # Convert to categorical
                col_new = pd.Categorical(col_new, categories=levels_train)
            else:
                # Check if X_new has the same levels
                levels_new = list(col_new.cat.categories)
                if set(levels_new) != set(levels_train):
                    # Convert to categorical with the same levels
                    col_new = pd.Categorical(col_new, categories=levels_train)
            
            # Create dummy variables
            dummies = pd.get_dummies(col_new, prefix=col_name, drop_first=True)
            
            # Add to preprocessed data
            X_preprocessed.append(dummies)
        else:
            # Add to preprocessed data
            X_preprocessed.append(col_new.to_frame())
    
    # Concatenate preprocessed data
    if X_preprocessed:
        X_preprocessed = pd.concat(X_preprocessed, axis=1)
    else:
        X_preprocessed = pd.DataFrame()
    
    # Handle missing values
    if bart_machine.use_missing_data_dummies_as_covars:
        # Create missing data dummies
        missing_dummies = pd.DataFrame()
        
        # For each column in X_train
        for col_name in X_train.columns:
            col_new = X_new[col_name]
            
            # Create a dummy variable for missing values
            missing_dummies[f"{col_name}_missing"] = col_new.isna().astype(int)
        
        # Add missing data dummies to preprocessed data
        if not missing_dummies.empty:
            X_preprocessed = pd.concat([X_preprocessed, missing_dummies], axis=1)
    
    # Handle random forest imputations for missing values
    if bart_machine.impute_missingness_with_rf_impute:
        # This is a placeholder - in a real implementation, we would use the same
        # random forest imputation model as in the training data
        # For now, we'll just create empty columns
        rf_imputations = pd.DataFrame()
        
        # For each column in X_train
        for col_name in X_train.columns:
            # Check if the column has missing values
            if X_new[col_name].isna().any():
                # Create a column for random forest imputations
                rf_imputations[f"{col_name}_imp"] = X_new[col_name].fillna(X_new[col_name].mean())
        
        # Add random forest imputations to preprocessed data
        if not rf_imputations.empty:
            X_preprocessed = pd.concat([X_preprocessed, rf_imputations], axis=1)
    
    # Convert to numpy array
    X_preprocessed_array = X_preprocessed.to_numpy()
    
    return X_preprocessed_array

def handle_missing_values(X: pd.DataFrame, 
                         strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    This function handles missing values in a DataFrame using various strategies.
    
    Args:
        X: The DataFrame with missing values.
        strategy: The strategy to use for handling missing values.
            "mean": Replace missing values with column means.
            "median": Replace missing values with column medians.
            "mode": Replace missing values with column modes.
            "drop": Drop rows with missing values.
    
    Returns:
        A DataFrame with missing values handled.
    """
    # Check if X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    
    # Check if X has missing values
    if not X.isna().any().any():
        return X
    
    # Handle missing values
    if strategy == "mean":
        # Replace missing values with column means
        return X.fillna(X.mean())
    elif strategy == "median":
        # Replace missing values with column medians
        return X.fillna(X.median())
    elif strategy == "mode":
        # Replace missing values with column modes
        return X.fillna(X.mode().iloc[0])
    elif strategy == "drop":
        # Drop rows with missing values
        return X.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def create_dummy_variables(X: pd.DataFrame, 
                          drop_first: bool = True) -> pd.DataFrame:
    """
    Create dummy variables for categorical columns in a DataFrame.
    
    This function creates dummy variables for categorical columns in a DataFrame.
    
    Args:
        X: The DataFrame with categorical columns.
        drop_first: Whether to drop the first level of each categorical column.
    
    Returns:
        A DataFrame with dummy variables.
    """
    # Check if X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    
    # Check if X has categorical columns
    if not any(pd.api.types.is_categorical_dtype(X[col]) for col in X.columns):
        return X
    
    # Create dummy variables
    return pd.get_dummies(X, drop_first=drop_first)

def standardize_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data in a DataFrame.
    
    This function standardizes data in a DataFrame by subtracting the mean
    and dividing by the standard deviation.
    
    Args:
        X: The DataFrame to standardize.
    
    Returns:
        A standardized DataFrame.
    """
    # Check if X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    
    # Standardize data
    return (X - X.mean()) / X.std()

def normalize_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data in a DataFrame.
    
    This function normalizes data in a DataFrame by scaling each column
    to the range [0, 1].
    
    Args:
        X: The DataFrame to normalize.
    
    Returns:
        A normalized DataFrame.
    """
    # Check if X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    
    # Normalize data
    return (X - X.min()) / (X.max() - X.min())

def check_data_quality(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Check the quality of data for BART models.
    
    This function checks the quality of data for BART models, including
    missing values, categorical variables, and other data quality issues.
    
    Args:
        X: The predictor variables.
        y: The response variable.
    
    Returns:
        A dictionary containing data quality information.
    """
    # Check if X is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    
    # Check if y is a series
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series.")
    
    # Check if X and y have the same number of observations
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of observations.")
    
    # Check for missing values
    missing_X = X.isna().sum()
    missing_y = y.isna().sum()
    
    # Check for categorical variables
    categorical_X = [col for col in X.columns if pd.api.types.is_categorical_dtype(X[col])]
    categorical_y = pd.api.types.is_categorical_dtype(y)
    
    # Check for constant columns
    constant_X = [col for col in X.columns if X[col].nunique() == 1]
    
    # Check for highly correlated columns
    corr_X = X.corr()
    high_corr_X = []
    for i in range(len(corr_X.columns)):
        for j in range(i+1, len(corr_X.columns)):
            if abs(corr_X.iloc[i, j]) > 0.9:
                high_corr_X.append((corr_X.columns[i], corr_X.columns[j], corr_X.iloc[i, j]))
    
    # Return data quality information
    return {
        "missing_X": missing_X,
        "missing_y": missing_y,
        "categorical_X": categorical_X,
        "categorical_y": categorical_y,
        "constant_X": constant_X,
        "high_corr_X": high_corr_X
    }
