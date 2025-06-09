# R-Python Equivalence Testing Guidelines

This document outlines guidelines for testing the equivalence between the R and Python implementations of bartMachine. The goal is to ensure that the Python implementation produces the same results as the R implementation, with the same behavior, numerical results, and API.

## Testing Approach

### 1. Unit Testing

Each Python function should have unit tests that verify its behavior against the expected behavior of the corresponding R function. These tests should cover:

- Basic functionality with simple inputs
- Edge cases and error handling
- Different parameter combinations
- Performance characteristics

### 2. Direct Comparison Testing

For key functions, we should perform direct comparison tests between the R and Python implementations. These tests should:

1. Run the same function with the same inputs in both R and Python
2. Compare the outputs for numerical equivalence
3. Document any differences and their causes

### 3. End-to-End Testing

We should also perform end-to-end tests that compare the entire workflow from data preprocessing to model building, prediction, and evaluation. These tests should:

1. Use the same dataset in both R and Python
2. Follow the same workflow steps
3. Compare the final results for numerical equivalence

## Testing Framework

### R-Python Bridge

To facilitate direct comparison testing, we can use the `rpy2` package to call R functions from Python. This allows us to:

1. Load the same dataset in both R and Python
2. Call the R function and get its result
3. Call the Python function and get its result
4. Compare the results directly

Example:

```python
import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from bartmachine import bart_machine

# Load R packages
pandas2ri.activate()
bartmachine_r = importr('bartMachine')

# Load dataset
X = pd.read_csv('data.csv')
y = X.pop('target')

# Run R function
r_result = bartmachine_r.bartMachine(pandas2ri.py2rpy(X), pandas2ri.py2rpy(y))

# Run Python function
py_result = bart_machine(X, y)

# Compare results
# ...
```

### Numerical Comparison

When comparing numerical results, we should consider:

1. **Precision**: R and Python may use different floating-point representations, so we should allow for small differences in precision.
2. **Random Number Generation**: R and Python use different random number generators, so we should set seeds in both implementations and verify that they produce the same results.
3. **Algorithm Differences**: There may be slight differences in algorithm implementation between R and Python, so we should document these differences and their impact on results.

Example:

```python
def assert_almost_equal(a, b, decimal=7, msg=None):
    """
    Assert that two values are almost equal to a given precision.
    
    Parameters
    ----------
    a : array_like
        First array to compare.
    b : array_like
        Second array to compare.
    decimal : int, optional
        Desired precision, default is 7.
    msg : str, optional
        Error message to be printed in case of failure.
    """
    np.testing.assert_almost_equal(a, b, decimal=decimal, err_msg=msg)
```

## Test Cases

### 1. Basic Functionality

Test basic functionality with simple datasets and default parameters:

```python
def test_bart_machine_basic():
    """Test basic functionality of bart_machine."""
    # Create synthetic data
    np.random.seed(123)
    X = np.random.normal(0, 1, (100, 5))
    y = X[:, 0] + X[:, 1] * X[:, 2] + np.random.normal(0, 0.1, 100)
    
    # Convert to pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(5)])
    y_series = pd.Series(y, name='y')
    
    # Run R function
    r.set_seed(123)
    r_result = bartmachine_r.bartMachine(pandas2ri.py2rpy(X_df), pandas2ri.py2rpy(y_series))
    
    # Run Python function
    np.random.seed(123)
    py_result = bart_machine(X_df, y_series, seed=123)
    
    # Compare results
    r_preds = np.array(r.predict(r_result, pandas2ri.py2rpy(X_df)))
    py_preds = predict(py_result, X_df)
    
    assert_almost_equal(r_preds, py_preds, decimal=5)
```

### 2. Parameter Variations

Test with different parameter combinations:

```python
def test_bart_machine_parameters():
    """Test bart_machine with different parameter combinations."""
    # Create synthetic data
    np.random.seed(123)
    X = np.random.normal(0, 1, (100, 5))
    y = X[:, 0] + X[:, 1] * X[:, 2] + np.random.normal(0, 0.1, 100)
    
    # Convert to pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(5)])
    y_series = pd.Series(y, name='y')
    
    # Parameter combinations to test
    param_combinations = [
        {'num_trees': 10, 'num_burn_in': 100, 'num_iterations_after_burn_in': 500},
        {'num_trees': 50, 'num_burn_in': 250, 'num_iterations_after_burn_in': 1000},
        {'num_trees': 100, 'num_burn_in': 500, 'num_iterations_after_burn_in': 2000},
    ]
    
    for params in param_combinations:
        # Run R function
        r.set_seed(123)
        r_result = bartmachine_r.bartMachine(
            pandas2ri.py2rpy(X_df),
            pandas2ri.py2rpy(y_series),
            num_trees=params['num_trees'],
            num_burn_in=params['num_burn_in'],
            num_iterations_after_burn_in=params['num_iterations_after_burn_in']
        )
        
        # Run Python function
        np.random.seed(123)
        py_result = bart_machine(
            X_df,
            y_series,
            seed=123,
            num_trees=params['num_trees'],
            num_burn_in=params['num_burn_in'],
            num_iterations_after_burn_in=params['num_iterations_after_burn_in']
        )
        
        # Compare results
        r_preds = np.array(r.predict(r_result, pandas2ri.py2rpy(X_df)))
        py_preds = predict(py_result, X_df)
        
        assert_almost_equal(r_preds, py_preds, decimal=5)
```

### 3. Real Datasets

Test with real datasets from the R package:

```python
def test_bart_machine_real_datasets():
    """Test bart_machine with real datasets from the R package."""
    # Load datasets from R
    r('data(boston_housing)')
    r('data(automobile)')
    
    # Boston Housing dataset
    boston_X = pandas2ri.rpy2py(r('boston_housing[, -14]'))
    boston_y = pandas2ri.rpy2py(r('boston_housing[, 14]'))
    
    # Automobile dataset
    auto_X = pandas2ri.rpy2py(r('automobile[, -1]'))
    auto_y = pandas2ri.rpy2py(r('automobile[, 1]'))
    
    # Test with Boston Housing dataset
    r.set_seed(123)
    r_result_boston = bartmachine_r.bartMachine(
        pandas2ri.py2rpy(boston_X),
        pandas2ri.py2rpy(boston_y)
    )
    
    np.random.seed(123)
    py_result_boston = bart_machine(
        boston_X,
        boston_y,
        seed=123
    )
    
    r_preds_boston = np.array(r.predict(r_result_boston, pandas2ri.py2rpy(boston_X)))
    py_preds_boston = predict(py_result_boston, boston_X)
    
    assert_almost_equal(r_preds_boston, py_preds_boston, decimal=5)
    
    # Test with Automobile dataset
    r.set_seed(123)
    r_result_auto = bartmachine_r.bartMachine(
        pandas2ri.py2rpy(auto_X),
        pandas2ri.py2rpy(auto_y)
    )
    
    np.random.seed(123)
    py_result_auto = bart_machine(
        auto_X,
        auto_y,
        seed=123
    )
    
    r_preds_auto = np.array(r.predict(r_result_auto, pandas2ri.py2rpy(auto_X)))
    py_preds_auto = predict(py_result_auto, auto_X)
    
    assert_almost_equal(r_preds_auto, py_preds_auto, decimal=5)
```

### 4. Edge Cases

Test edge cases and error handling:

```python
def test_bart_machine_edge_cases():
    """Test bart_machine with edge cases."""
    # Create synthetic data
    np.random.seed(123)
    X = np.random.normal(0, 1, (100, 5))
    y = X[:, 0] + X[:, 1] * X[:, 2] + np.random.normal(0, 0.1, 100)
    
    # Convert to pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(5)])
    y_series = pd.Series(y, name='y')
    
    # Test with missing values
    X_missing = X_df.copy()
    X_missing.iloc[0, 0] = np.nan
    
    # Run R function
    r.set_seed(123)
    r_result = bartmachine_r.bartMachine(
        pandas2ri.py2rpy(X_missing),
        pandas2ri.py2rpy(y_series),
        use_missing_data=True
    )
    
    # Run Python function
    np.random.seed(123)
    py_result = bart_machine(
        X_missing,
        y_series,
        seed=123,
        use_missing_data=True
    )
    
    # Compare results
    r_preds = np.array(r.predict(r_result, pandas2ri.py2rpy(X_missing)))
    py_preds = predict(py_result, X_missing)
    
    assert_almost_equal(r_preds, py_preds, decimal=5)
    
    # Test with constant columns
    X_const = X_df.copy()
    X_const['X5'] = 1.0
    
    # Run R function
    r.set_seed(123)
    r_result = bartmachine_r.bartMachine(
        pandas2ri.py2rpy(X_const),
        pandas2ri.py2rpy(y_series)
    )
    
    # Run Python function
    np.random.seed(123)
    py_result = bart_machine(
        X_const,
        y_series,
        seed=123
    )
    
    # Compare results
    r_preds = np.array(r.predict(r_result, pandas2ri.py2rpy(X_const)))
    py_preds = predict(py_result, X_const)
    
    assert_almost_equal(r_preds, py_preds, decimal=5)
```

### 5. Classification

Test classification functionality:

```python
def test_bart_machine_classification():
    """Test bart_machine with classification."""
    # Create synthetic data
    np.random.seed(123)
    X = np.random.normal(0, 1, (100, 5))
    p = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] * X[:, 2])))
    y = np.random.binomial(1, p)
    
    # Convert to pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(5)])
    y_series = pd.Series(y, name='y')
    
    # Run R function
    r.set_seed(123)
    r_result = bartmachine_r.bartMachine(
        pandas2ri.py2rpy(X_df),
        pandas2ri.py2rpy(y_series),
        classification=True
    )
    
    # Run Python function
    np.random.seed(123)
    py_result = bart_machine(
        X_df,
        y_series,
        seed=123,
        classification=True
    )
    
    # Compare results
    r_preds = np.array(r.predict(r_result, pandas2ri.py2rpy(X_df)))
    py_preds = predict(py_result, X_df)
    
    assert_almost_equal(r_preds, py_preds, decimal=5)
```

## Documenting Differences

When differences are found between the R and Python implementations, they should be documented in a structured way:

1. **Description**: A clear description of the difference
2. **Cause**: The underlying cause of the difference
3. **Impact**: The impact of the difference on results
4. **Resolution**: How the difference was resolved or why it was accepted

Example:

```
## Difference in Random Number Generation

### Description
The R and Python implementations use different random number generators, which can lead to different results even when the same seed is set.

### Cause
R uses the Mersenne-Twister algorithm with a different implementation than Python's NumPy.

### Impact
Small differences in numerical results, typically within 1e-5 precision.

### Resolution
We accept these small differences as they do not significantly affect the model's performance or interpretation. For testing purposes, we use a precision of 1e-5 when comparing results.
```

## Continuous Integration

To ensure ongoing equivalence between the R and Python implementations, we should set up continuous integration tests that:

1. Run the equivalence tests on each commit
2. Report any differences that exceed the accepted tolerance
3. Maintain a history of test results to track changes over time

## Conclusion

By following these guidelines, we can ensure that the Python implementation of bartMachine is functionally identical to the R implementation, with the same behavior, numerical results, and API. This will provide users with a seamless transition between the two implementations and ensure that the Python implementation can be used as a drop-in replacement for the R implementation.
