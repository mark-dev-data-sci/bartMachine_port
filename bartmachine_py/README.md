# bartMachine Python Package

This package provides a Python implementation of the bartMachine R package, which implements Bayesian Additive Regression Trees (BART).

## Overview

The bartMachine Python package is a direct port of the R package, with the same API and functionality. It uses the same Java backend as the R package, but provides a Python interface.

The package provides a comprehensive set of tools for building, evaluating, and interpreting BART models for regression and classification tasks.

## Key Features

- **BART Models**: Build BART models for regression and classification tasks
- **Cross-Validation**: Perform cross-validation to find optimal hyperparameters
- **Variable Importance**: Assess the importance of variables in the model
- **Visualization**: Visualize model results and diagnostics
- **Prediction**: Make predictions with credible and prediction intervals
- **F-Tests**: Perform statistical tests for variable importance

## Installation

```bash
pip install bartmachine
```

## Quick Start

```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

# Initialize the JVM
initialize_jvm()

# Load data
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

# Build a BART model
model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

# Make predictions
X_test = pd.DataFrame({
    'x1': np.random.normal(0, 1, 10),
    'x2': np.random.normal(0, 1, 10),
    'x3': np.random.normal(0, 1, 10)
})
y_hat = model.predict(X_test)

# Get variable importance
var_importance = model.get_var_importance()
print(var_importance)

# Plot convergence diagnostics
from bartmachine import plot_convergence_diagnostics
plot_convergence_diagnostics(model)

# Plot actual vs. predicted values
from bartmachine import plot_y_vs_yhat
plot_y_vs_yhat(model)
```

## Documentation

For more detailed documentation, see the [API Reference](https://bartmachine.readthedocs.io).

## Implementation Details

The Python API is functionally equivalent to the R API, with the same behavior, numerical results, and user experience. The implementation includes:

1. **Core API Functions**:
   - `bart_machine`: The main function for creating a BART model
   - `bart_machine_cv`: Cross-validation for BART models
   - `predict`: Prediction function for BART models
   - `plot_convergence_diagnostics`: Plotting function for convergence diagnostics
   - `plot_y_vs_yhat`: Plotting function for actual vs. predicted values
   - `get_var_importance`: Function for variable importance
   - `get_var_props_over_chain`: Function for variable inclusion proportions
   - Other utility functions

2. **Data Handling**:
   - Functions for data preprocessing
   - Handling missing values, factors, and other data types
   - Compatibility with pandas DataFrames and NumPy arrays

3. **Model Building**:
   - Functions for building BART models
   - Handling hyperparameter settings
   - Cross-validation

4. **Prediction**:
   - Functions for making predictions
   - Handling different prediction types (point estimates, credible intervals, etc.)
   - Posterior sampling

5. **Visualization**:
   - Plotting functions
   - Matplotlib for visualization
   - Compatibility with Jupyter notebooks

## License

This package is licensed under the MIT License.
