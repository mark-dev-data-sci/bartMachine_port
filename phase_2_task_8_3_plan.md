# Phase 2 - Task 8.3: Python API Implementation Plan

## Overview

This document outlines the detailed plan for implementing the Python API for the bartMachine package. The Python API will provide a user-friendly interface to the Java backend through the Java bridge implemented in Task 8.2. The goal is to create a Python implementation that is functionally identical to the R implementation, with the same behavior, numerical results, and API.

## R-to-Python Function Mapping

The following table maps the key R functions to their Python equivalents:

| R Function | Python Function | Description |
|------------|----------------|-------------|
| `bartMachine` | `bart_machine` | Create a BART model |
| `bartMachineCV` | `bart_machine_cv` | Cross-validation for BART models |
| `predict` | `predict` | Make predictions with a BART model |
| `plot_convergence_diagnostics` | `plot_convergence_diagnostics` | Plot convergence diagnostics |
| `plot_y_vs_yhat` | `plot_y_vs_yhat` | Plot actual vs. predicted values |
| `get_var_importance` | `get_var_importance` | Get variable importance measures |
| `get_var_props_over_chain` | `get_var_props_over_chain` | Get variable inclusion proportions |
| `investigate_var_importance` | `investigate_var_importance` | Investigate variable importance |
| `interaction_investigator` | `interaction_investigator` | Investigate variable interactions |
| `cov_importance_test` | `cov_importance_test` | Test variable importance |
| `get_sigsqs` | `get_sigsqs` | Get sigma squared values |
| `get_tree_num` | `get_tree_num` | Get number of trees |
| `extract_raw_node_information` | `extract_raw_node_information` | Extract raw node information |
| `get_projection_weights` | `get_projection_weights` | Get projection weights |
| `get_node_prediction_training_indices` | `get_node_prediction_training_indices` | Get node prediction training indices |

## Implementation Steps

### 1. Core API Functions

1. **bart_machine.py**
   - Implement `bart_machine` function
   - Implement `bart_machine_cv` function
   - Implement `predict` function
   - Implement `get_var_importance` function
   - Implement `get_var_props_over_chain` function
   - Implement other core functions

2. **data_preprocessing.py**
   - Implement functions for data preprocessing
   - Handle missing values, factors, and other data types
   - Ensure compatibility with pandas DataFrames and NumPy arrays

3. **model_building.py**
   - Implement functions for building BART models
   - Handle hyperparameter settings
   - Implement cross-validation

4. **plots.py**
   - Implement `plot_convergence_diagnostics` function
   - Implement `plot_y_vs_yhat` function
   - Implement other plotting functions

5. **summaries.py**
   - Implement functions for model summaries
   - Implement functions for variable importance
   - Implement functions for interaction effects

### 2. Testing

1. **test_basic_functionality.py**
   - Test basic functionality of the Python API
   - Test with simple datasets
   - Test with default parameters

2. **test_r_equivalence.py**
   - Test numerical equivalence with the R implementation
   - Test with real datasets from the R implementation
   - Test with various parameter settings

3. **test_edge_cases.py**
   - Test edge cases and error handling
   - Test with missing values, factors, and other data types
   - Test with invalid inputs

### 3. Documentation

1. **docstrings**
   - Add docstrings to all functions
   - Follow NumPy docstring format
   - Include parameter descriptions, return values, and examples

2. **examples**
   - Create examples for common use cases
   - Create examples for advanced use cases
   - Create examples for integration with other Python libraries

3. **README.md**
   - Update README.md with installation instructions
   - Add usage examples
   - Add API documentation

## Implementation Details

### bart_machine Function

The `bart_machine` function is the main entry point for creating a BART model. It should have the following signature:

```python
def bart_machine(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    num_trees: int = 50,
    num_burn_in: int = 250,
    num_iterations_after_burn_in: int = 1000,
    alpha: float = 0.95,
    beta: float = 2,
    k: float = 2,
    q: float = 0.9,
    nu: float = 3,
    prob_rule_class: float = 0.5,
    mh_prob_steps: List[float] = None,
    debug_log: bool = False,
    run_in_sample: bool = True,
    s_sq_y: float = None,
    sig_sq: float = None,
    seed: int = None,
    serialize: bool = True,
    verbose: bool = True,
    mem_cache_for_speed: bool = True,
    flush_indices_to_save_RAM: bool = False,
    use_missing_data: bool = False,
    use_missing_data_dummies_as_covars: bool = False,
    replace_missing_data_with_x_j_bar: bool = False,
    impute_missingness_with_rf_impute: bool = False,
    impute_missingness_with_x_j_bar_for_lm: bool = False,
    covariates_to_permute: List[str] = None,
    num_rand_samps_in_library: int = 10000,
    use_cov_selector: bool = False,
    cov_selector_alpha: float = 0.05,
    interaction_terms: List[List[str]] = None,
    num_cores: int = None,
    use_missing_data_dummies_as_covars_pct: float = 0.05,
    classification: bool = False,
    rm_const_cols: bool = True,
    model_type: str = "wbart",
    **kwargs
) -> BartMachine:
    """
    Create a BART model.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        The predictor variables.
    y : Union[pd.Series, np.ndarray]
        The response variable.
    num_trees : int, optional
        Number of trees in the ensemble, by default 50
    num_burn_in : int, optional
        Number of burn-in MCMC iterations, by default 250
    num_iterations_after_burn_in : int, optional
        Number of MCMC iterations after burn-in, by default 1000
    alpha : float, optional
        Prior parameter for tree structure, by default 0.95
    beta : float, optional
        Prior parameter for tree structure, by default 2
    k : float, optional
        Prior parameter for leaf values, by default 2
    q : float, optional
        Prior parameter for leaf values, by default 0.9
    nu : float, optional
        Prior parameter for error variance, by default 3
    prob_rule_class : float, optional
        Probability of using a classification rule, by default 0.5
    mh_prob_steps : List[float], optional
        Metropolis-Hastings proposal step probabilities, by default None
    debug_log : bool, optional
        Whether to print debug information, by default False
    run_in_sample : bool, optional
        Whether to run in-sample prediction, by default True
    s_sq_y : float, optional
        Sample variance of y, by default None
    sig_sq : float, optional
        Error variance, by default None
    seed : int, optional
        Random seed, by default None
    serialize : bool, optional
        Whether to serialize the model, by default True
    verbose : bool, optional
        Whether to print verbose output, by default True
    mem_cache_for_speed : bool, optional
        Whether to cache for speed, by default True
    flush_indices_to_save_RAM : bool, optional
        Whether to flush indices to save RAM, by default False
    use_missing_data : bool, optional
        Whether to use missing data, by default False
    use_missing_data_dummies_as_covars : bool, optional
        Whether to use missing data dummies as covariates, by default False
    replace_missing_data_with_x_j_bar : bool, optional
        Whether to replace missing data with column means, by default False
    impute_missingness_with_rf_impute : bool, optional
        Whether to impute missingness with random forest, by default False
    impute_missingness_with_x_j_bar_for_lm : bool, optional
        Whether to impute missingness with column means for linear model, by default False
    covariates_to_permute : List[str], optional
        Covariates to permute, by default None
    num_rand_samps_in_library : int, optional
        Number of random samples in library, by default 10000
    use_cov_selector : bool, optional
        Whether to use covariate selector, by default False
    cov_selector_alpha : float, optional
        Alpha for covariate selector, by default 0.05
    interaction_terms : List[List[str]], optional
        Interaction terms, by default None
    num_cores : int, optional
        Number of cores to use, by default None
    use_missing_data_dummies_as_covars_pct : float, optional
        Percentage for missing data dummies as covariates, by default 0.05
    classification : bool, optional
        Whether to use classification, by default False
    rm_const_cols : bool, optional
        Whether to remove constant columns, by default True
    model_type : str, optional
        Model type, by default "wbart"
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    BartMachine
        The BART model.
    """
    # Implementation here
```

### predict Function

The `predict` function is used to make predictions with a BART model. It should have the following signature:

```python
def predict(
    bart_machine: BartMachine,
    new_data: Union[pd.DataFrame, np.ndarray],
    type: str = "response",
    num_cores: int = None,
    **kwargs
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Make predictions with a BART model.

    Parameters
    ----------
    bart_machine : BartMachine
        The BART model.
    new_data : Union[pd.DataFrame, np.ndarray]
        The new data for prediction.
    type : str, optional
        The type of prediction, by default "response"
    num_cores : int, optional
        Number of cores to use, by default None
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Union[np.ndarray, Dict[str, np.ndarray]]
        The predictions.
    """
    # Implementation here
```

## Validation Plan

To ensure that the Python implementation is functionally identical to the R implementation, we will:

1. **Unit Tests**: Create unit tests for all API functions to verify that they work correctly.

2. **R-Python Comparison Tests**: Create tests that compare the results of the Python implementation with the results of the R implementation using the same data and parameters.

3. **Edge Case Tests**: Create tests for edge cases and error handling to ensure that the Python implementation handles them in the same way as the R implementation.

4. **Performance Tests**: Create tests to measure the performance of the Python implementation compared to the R implementation.

## Timeline

1. **Week 1**: Implement core API functions and data preprocessing
2. **Week 2**: Implement model building, prediction, and visualization
3. **Week 3**: Implement testing and validation
4. **Week 4**: Implement documentation and examples

## Conclusion

This plan outlines the steps for implementing the Python API for the bartMachine package. By following this plan, we will create a Python implementation that is functionally identical to the R implementation, with the same behavior, numerical results, and API.
