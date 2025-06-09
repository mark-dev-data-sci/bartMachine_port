# bartmachine

Python port of the bartMachine R package for Bayesian Additive Regression Trees (BART).

## Overview

This package provides a Python implementation of Bayesian Additive Regression Trees (BART) based on the original bartMachine R package. It maintains the same functionality and interface as the R package, but with a Python-friendly API.

The implementation uses the original Java backend through a Python-Java bridge, ensuring exact numerical equivalence with the R implementation.

## Installation

```bash
# Clone the repository
git clone https://github.com/username/bartmachine_py.git
cd bartmachine_py

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Requirements

- Python 3.6+
- Java 8+
- NumPy
- pandas
- py4j
- matplotlib
- scikit-learn
- pytest
- rpy2 (for testing equivalence with R implementation)

## Usage

```python
import numpy as np
import pandas as pd
from bartmachine import BartMachine, initialize_jvm, shutdown_jvm

# Initialize the JVM
initialize_jvm()

# Generate synthetic data
np.random.seed(123)
n = 100
p = 5
X = np.random.normal(0, 1, (n, p))
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

# Create and build a BART machine model
bart = BartMachine(
    X_train, y_train,
    num_trees=50,
    num_burn_in=100,
    num_iterations_after_burn_in=500,
    seed=123
)
bart.build()

# Make predictions
y_pred = bart.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"RMSE: {rmse:.4f}")

# Shutdown the JVM when done
shutdown_jvm()
```

## Features

- Regression and classification with BART
- Variable selection and importance
- Hyperparameter tuning
- Cross-validation
- Visualization of trees and variable importance
- Exact numerical equivalence with the R implementation

## Testing

```bash
# Run tests
pytest bartmachine/tests/
```

## License

MIT

## Acknowledgements

This package is a port of the [bartMachine](https://github.com/kapelner/bartMachine) R package by Adam Kapelner and Justin Bleich.
