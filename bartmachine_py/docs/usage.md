# Usage Guide

This document provides a guide for using the bartMachine Python package.

## Table of Contents

1. [Initializing the JVM](#initializing-the-jvm)
2. [Building a BART Model](#building-a-bart-model)
3. [Making Predictions](#making-predictions)
4. [Variable Importance](#variable-importance)
5. [Visualization](#visualization)
6. [Statistical Tests](#statistical-tests)
7. [Cross-Validation](#cross-validation)
8. [Advanced Usage](#advanced-usage)

## Initializing the JVM

Before using bartMachine, you need to initialize the Java Virtual Machine (JVM):

```python
from bartmachine import initialize_jvm

# Initialize the JVM with default settings
initialize_jvm()

# Or with custom settings
initialize_jvm(max_heap_size="4g", classpath="/path/to/additional/jars")
```

You can also check if the JVM is running:

```python
from bartmachine import is_jvm_running

if is_jvm_running():
    print("JVM is running")
else:
    print("JVM is not running")
```

And shut it down when you're done:

```python
from bartmachine import shutdown_jvm

shutdown_jvm()
```

## Building a BART Model

### Regression

```python
import pandas as pd
import numpy as np
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
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000
)

# Print model summary
print(model.summary())
```

### Classification

```python
import pandas as pd
import numpy as np
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
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000, 
    classification=True
)

# Print model summary
print(model.summary())
```

## Making Predictions

### Regression Predictions

```python
# Make point predictions
y_pred = model.predict(X_test)

# Make predictions with credible intervals
y_pred_ci = model.predict(X_test, type="credible", ci_level=0.95)

# Draw posterior samples
y_pred_samples = model.predict(X_test, type="sample", num_samples=1000)
```

### Classification Predictions

```python
# Make probability predictions
y_pred_prob = model.predict(X_test, type="prob")

# Make class predictions
y_pred_class = model.predict(X_test, type="class")

# Draw posterior samples
y_pred_samples = model.predict(X_test, type="sample", num_samples=1000)
```

## Variable Importance

```python
# Get variable importance
var_importance = model.get_var_importance()
print(var_importance)

# Get variable inclusion proportions over the MCMC chain
var_props = model.get_var_props_over_chain()
print(var_props)

# Plot variable importance
model.plot_variable_importance()
```

## Visualization

```python
# Plot convergence diagnostics
model.plot_convergence_diagnostics()

# Plot actual vs. predicted values
model.plot_y_vs_yhat(X_test, y_test)

# Plot partial dependence for a variable
model.plot_partial_dependence('CRIM')
```

## Statistical Tests

```python
# Test the importance of a covariate
test_results = model.cov_importance_test('CRIM', num_permutations=100)
print(test_results)

# Test the linearity of a covariate
test_results = model.linearity_test('CRIM', num_permutations=100)
print(test_results)

# Perform an F-test for variable importance
test_results = model.bart_machine_f_test(['CRIM', 'ZN'], num_permutations=100)
print(test_results)
```

## Cross-Validation

```python
from bartmachine import bart_machine_cv

# Perform cross-validation
cv_results = bart_machine_cv(
    X=X, 
    y=y, 
    k_folds=5, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000
)

# Print cross-validation results
print(cv_results)

# Plot cross-validation results
from bartmachine import plot_cv_results
plot_cv_results(cv_results)
```

## Advanced Usage

### Handling Missing Data

bartMachine automatically handles missing data:

```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

# Initialize the JVM
initialize_jvm()

# Create data with missing values
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
X.loc[0:10, 'x1'] = np.nan  # Set some values to NaN
y = X['x2']**2 + np.random.normal(0, 0.1, 100)

# Build a BART model
model = bart_machine(X, y, num_trees=50, num_burn_in=200, num_iterations_after_burn_in=1000)
```

### Interaction Detection

```python
# Investigate variable interactions
interactions = model.interaction_investigator(num_replicates=5)
print(interactions)
```

### Customizing the Prior

```python
# Build a BART model with custom prior parameters
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    alpha=0.95,  # Prior parameter for tree structure
    beta=2,      # Prior parameter for tree structure
    k=2,         # Prior parameter for tree structure
    q=0.9,       # Prior parameter for tree structure
    nu=3         # Prior parameter for error variance
)
```

### Parallelization

```python
# Build a BART model with multiple threads
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    num_threads=4  # Use 4 threads
)
```

### Memory Caching

```python
# Build a BART model with memory caching
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    mem_cache_for_speed=True  # Use memory caching for speed
)
```

### Setting a Random Seed

```python
# Build a BART model with a specific random seed
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=12345  # Set random seed
)
```

### Verbose Output

```python
# Build a BART model with verbose output
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    verbose=True  # Print verbose output
)
