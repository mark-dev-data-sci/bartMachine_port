# bartMachine

A Python implementation of the bartMachine package for Bayesian Additive Regression Trees (BART).

## Description

bartMachine is a Python package that provides an interface to the Java implementation of Bayesian Additive Regression Trees (BART). BART is a Bayesian approach to nonparametric function estimation using regression trees. It is particularly useful for high-dimensional data and can handle both regression and classification tasks.

This package is a Python port of the R package [bartMachine](https://github.com/cran/bartMachine), which was originally developed by Adam Kapelner and Justin Bleich.

## Features

- **BART Models**: Build BART models for regression and classification tasks
- **Cross-Validation**: Perform cross-validation to find optimal hyperparameters
- **Variable Importance**: Assess the importance of variables in the model
- **Visualization**: Visualize model results and diagnostics
- **Prediction**: Make predictions with credible and prediction intervals
- **F-Tests**: Perform statistical tests for variable importance
- **Missing Data Handling**: Automatic handling of missing data
- **Interaction Detection**: Identify important variable interactions
- **Parallelization**: Support for multi-threading to speed up computation

## Installation

### Prerequisites

- Python 3.8 or higher
- Java 8 or higher (JDK must be installed and JAVA_HOME environment variable set)

### Installing from PyPI

```bash
pip install bartmachine
```

### Installing from source

```bash
git clone https://github.com/username/bartmachine.git
cd bartmachine
pip install .
```

### Development Installation

For development purposes, you can install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Regression Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Initialize the JVM
initialize_jvm()

# Load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and build BART model
bart = bart_machine(X=X_train, y=y_train, num_trees=50, num_burn_in=200, num_iterations_after_burn_in=1000)

# Make predictions
y_pred = bart.predict(X_test)

# Evaluate model
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error: {mse:.4f}")

# Get variable importance
var_importance = bart.get_var_importance()
print("Variable Importance:")
print(var_importance)

# Plot convergence diagnostics
bart.plot_convergence_diagnostics()

# Plot actual vs. predicted values
bart.plot_y_vs_yhat(X_test, y_test)
```

### Classification Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Initialize the JVM
initialize_jvm()

# Load data
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and build BART model
bart = bart_machine(X=X_train, y=y_train, num_trees=50, num_burn_in=200, 
                    num_iterations_after_burn_in=1000, classification=True)

# Make predictions
y_pred_prob = bart.predict(X_test, type="prob")
y_pred_class = bart.predict(X_test, type="class")

# Evaluate model
accuracy = np.mean(y_pred_class == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get variable importance
var_importance = bart.get_var_importance()
print("Variable Importance:")
print(var_importance)
```

## API Reference

### Main Functions

- `initialize_jvm()`: Initialize the Java Virtual Machine
- `shutdown_jvm()`: Shutdown the Java Virtual Machine
- `bart_machine()`: Create and build a BART model
- `bart_machine_cv()`: Cross-validation for BART models
- `predict()`: Make predictions with a BART model
- `get_var_importance()`: Get variable importance measures
- `plot_convergence_diagnostics()`: Plot convergence diagnostics
- `plot_y_vs_yhat()`: Plot actual vs. predicted values
- `plot_variable_importance()`: Plot variable importance
- `plot_partial_dependence()`: Plot partial dependence
- `investigate_var_importance()`: Investigate variable importance
- `interaction_investigator()`: Investigate variable interactions
- `cov_importance_test()`: Test covariate importance
- `linearity_test()`: Test linearity of covariates
- `bart_machine_f_test()`: Perform F-test for variable importance

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/bartmachine.git
   cd bartmachine
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

## License

This package is licensed under the MIT License.

## Acknowledgments

- Adam Kapelner and Justin Bleich for the original R package
- The BART authors for the methodology
- The scikit-learn team for inspiration on the API design

## Citation

If you use this package in your research, please cite:

```
@article{kapelner2016bartmachine,
  title={bartMachine: Machine Learning with Bayesian Additive Regression Trees},
  author={Kapelner, Adam and Bleich, Justin},
  journal={Journal of Statistical Software},
  volume={70},
  number={4},
  pages={1--40},
  year={2016}
}
```
