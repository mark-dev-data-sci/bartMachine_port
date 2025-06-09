"""
Data preprocessing module for bartMachine.

This module provides functions for preprocessing data before building BART models.
It handles dummification of categorical variables and missing data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple

def dummify_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame to a dummified DataFrame.
    
    Args:
        data: The DataFrame to dummify.
    
    Returns:
        The dummified DataFrame.
    """
    return pd.DataFrame(preprocess_training_data(data)['data'])

def preprocess_training_data(data: pd.DataFrame, 
                            use_missing_data_dummies_as_covars: bool = False,
                            imputations: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Preprocess training data for BART.
    
    This function handles all pre-processing (dummification, missing data, etc.)
    
    Args:
        data: The DataFrame to preprocess.
        use_missing_data_dummies_as_covars: Whether to use missing data as covariates.
        imputations: Imputed values for missing data.
    
    Returns:
        A dictionary containing the preprocessed data and factor lengths.
    """
    # First convert objects to categories (equivalent to R's character to factor)
    object_cols = data.select_dtypes(include=['object']).columns
    for col in object_cols:
        data[col] = data[col].astype('category')
    
    # Get categorical columns
    categorical_cols = data.select_dtypes(include=['category']).columns
    
    factor_lengths = []
    for col in categorical_cols:
        # Create dummies for this categorical variable
        dummies = pd.get_dummies(data[col], prefix=col)
        # Append them to the data
        data = pd.concat([data, dummies], axis=1)
        # Delete the original categorical column
        data = data.drop(columns=[col])
        # Record the length of this factor
        factor_lengths.append(len(dummies.columns))
    
    if use_missing_data_dummies_as_covars:
        # Find columns with missing values
        cols_with_missing = data.columns[data.isna().any()].tolist()
        
        # Only do something if there are predictors with missingness
        if len(cols_with_missing) > 0:
            # Create missingness indicator matrix
            M = pd.DataFrame(0, index=data.index, columns=[f"M_{col}" for col in cols_with_missing])
            for i in range(len(data)):
                for j, col in enumerate(cols_with_missing):
                    if is_missing(data.iloc[i][col]):
                        M.iloc[i, j] = 1
            
            # Add imputations if provided
            if imputations is not None:
                data = pd.concat([data, imputations], axis=1)
            
            # Append the missing dummy columns to data
            data = pd.concat([data, M], axis=1)
    elif imputations is not None:
        # Add imputations if provided
        data = pd.concat([data, imputations], axis=1)
    
    # Convert to numpy array and return with factor_lengths
    return {
        'data': data.to_numpy(),
        'factor_lengths': factor_lengths
    }

def is_missing(x: Any) -> bool:
    """
    Check if a value is missing.
    
    Args:
        x: The value to check.
    
    Returns:
        True if the value is missing, False otherwise.
    """
    return pd.isna(x) or np.isnan(x) if isinstance(x, (float, np.floating)) else pd.isna(x)

def preprocess_new_data(new_data: pd.DataFrame, bart_machine: Any) -> np.ndarray:
    """
    Preprocess new data for prediction.
    
    Args:
        new_data: The new data to preprocess.
        bart_machine: The BART machine object.
    
    Returns:
        The preprocessed data as a numpy array.
    """
    new_data = pd.DataFrame(new_data)
    n = len(new_data)
    
    imputations = None
    if bart_machine.impute_missingness_with_rf_impute:
        # In Python we would use a different imputation method than missForest
        # For now, we'll use scikit-learn's IterativeImputer as a placeholder
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        combined_data = pd.concat([new_data, bart_machine.X])
        imputer = IterativeImputer(random_state=0)
        imputed_values = imputer.fit_transform(combined_data)
        imputed_df = pd.DataFrame(imputed_values, columns=combined_data.columns)
        imputations = imputed_df.iloc[:n].copy()
        imputations.columns = [f"{col}_imp" for col in imputations.columns]
    
    # Preprocess the new data with the training data to ensure proper dummies
    new_data_and_training_data = pd.concat([new_data, bart_machine.X])
    
    # Reset categorical columns
    categorical_cols = new_data_and_training_data.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        new_data_and_training_data[col] = new_data_and_training_data[col].astype('category')
    
    # Preprocess the combined data
    new_data = preprocess_training_data(
        new_data_and_training_data, 
        bart_machine.use_missing_data_dummies_as_covars, 
        imputations
    )['data']
    
    # Get the appropriate training data features
    if bart_machine.use_missing_data:
        training_data_features = bart_machine.training_data_features_with_missing_features
    else:
        training_data_features = bart_machine.training_data_features
    
    # The new data features has to be a superset of the training data features
    new_data_features_before = list(range(new_data.shape[1]))
    
    # Extract only the first n rows (the new data) and only the columns that match training features
    new_data = new_data[:n, :len(training_data_features)]
    
    # Check for differences in features
    differences = set(new_data_features_before) - set(range(len(training_data_features)))
    if differences:
        print(f"Warning: The following features were found in records for prediction which were not found in the original training data: {differences}. These features will be ignored during prediction.")
    
    # Ensure the new data has the same features as the training data
    new_data_features = list(range(new_data.shape[1]))
    if not all(i == j for i, j in zip(new_data_features, range(len(training_data_features)))):
        print("Warning: Are you sure you have the same feature names in the new record(s) as the training data?")
    
    # Iterate through and ensure feature alignment
    for j in range(len(training_data_features)):
        training_data_feature = training_data_features[j]
        if j >= len(new_data_features) or new_data_features[j] != j:
            # Create a new column of zeros
            new_col = np.zeros((n, 1))
            
            # Insert it at the right position
            if j == 0:
                new_data = np.hstack((new_col, new_data))
            elif j >= new_data.shape[1]:
                new_data = np.hstack((new_data, new_col))
            else:
                new_data = np.hstack((new_data[:, :j], new_col, new_data[:, j:]))
            
            # Update the feature list
            new_data_features = list(range(new_data.shape[1]))
    
    # Ensure the data is a numeric numpy array
    new_data = new_data.astype(float)
    
    return new_data
