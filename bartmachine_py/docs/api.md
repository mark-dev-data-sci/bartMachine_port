# API Reference

This document provides a reference for the bartMachine Python API.

## Core Functions

### JVM Management

#### `initialize_jvm()`

Initialize the Java Virtual Machine (JVM) for bartMachine.

**Parameters:**
- `max_heap_size` (str, optional): Maximum heap size for the JVM. Default is "2g".
- `classpath` (str, optional): Additional classpath for the JVM. Default is None.

**Returns:**
- None

**Example:**
```python
from bartmachine import initialize_jvm

initialize_jvm(max_heap_size="4g")
```

#### `shutdown_jvm()`

Shutdown the Java Virtual Machine (JVM).

**Parameters:**
- None

**Returns:**
- None

**Example:**
```python
from bartmachine import shutdown_jvm

shutdown_jvm()
```

#### `is_jvm_running()`

Check if the JVM is running.

**Parameters:**
- None

**Returns:**
- bool: True if the JVM is running, False otherwise.

**Example:**
```python
from bartmachine import is_jvm_running

if is_jvm_running():
    print("JVM is running")
else:
    print("JVM is not running")
```

### Model Building

#### `bart_machine(X, y, **kwargs)`

Create and build a BART model.

**Parameters:**
- `X` (pandas.DataFrame or numpy.ndarray): The predictor variables.
- `y` (pandas.Series or numpy.ndarray): The response variable.
- `num_trees` (int, optional): Number of trees in the ensemble. Default is 50.
- `num_burn_in` (int, optional): Number of burn-in MCMC iterations. Default is 250.
- `num_iterations_after_burn_in` (int, optional): Number of MCMC iterations after burn-in. Default is 1000.
- `alpha` (float, optional): Prior parameter for tree structure. Default is 0.95.
- `beta` (float, optional): Prior parameter for tree structure. Default is 2.
- `k` (float, optional): Prior parameter for tree structure. Default is 2.
- `q` (float, optional): Prior parameter for tree structure. Default is 0.9.
- `nu` (float, optional): Prior parameter for error variance. Default is 3.
- `num_threads` (int, optional): Number of threads to use. Default is 1.
- `mem_cache_for_speed` (bool, optional): Whether to use memory caching for speed. Default is True.
- `verbose` (bool, optional): Whether to print verbose output. Default is False.
- `seed` (int, optional): Random seed. Default is 12345.
- `classification` (bool, optional): Whether to build a classification model. Default is False.

**Returns:**
- BartMachine: A BART model object.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)
```

#### `bart_machine_cv(X, y, k_folds=5, **kwargs)`

Perform cross-validation for a BART model.

**Parameters:**
- `X` (pandas.DataFrame or numpy.ndarray): The predictor variables.
- `y` (pandas.Series or numpy.ndarray): The response variable.
- `k_folds` (int, optional): Number of folds for cross-validation. Default is 5.
- `**kwargs`: Additional arguments to pass to `bart_machine()`.

**Returns:**
- dict: A dictionary containing cross-validation results.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine_cv, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

cv_results = bart_machine_cv(X, y, k_folds=5, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)
```

### Prediction

#### `predict(X, **kwargs)`

Make predictions with a BART model.

**Parameters:**
- `X` (pandas.DataFrame or numpy.ndarray): The predictor variables.
- `type` (str, optional): Type of prediction to make. For regression, options are "mean", "credible", or "sample". For classification, options are "prob", "class", or "sample". Default is "mean" for regression and "prob" for classification.
- `ci_level` (float, optional): Credible interval level. Default is 0.95.
- `num_samples` (int, optional): Number of posterior samples to draw. Default is 1000.

**Returns:**
- numpy.ndarray: Predictions.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

X_test = pd.DataFrame({
    'x1': np.random.normal(0, 1, 10),
    'x2': np.random.normal(0, 1, 10),
    'x3': np.random.normal(0, 1, 10)
})

y_pred = model.predict(X_test)
```

### Variable Importance

#### `get_var_importance()`

Get variable importance measures.

**Parameters:**
- None

**Returns:**
- pandas.DataFrame: Variable importance measures.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

var_importance = model.get_var_importance()
```

#### `get_var_props_over_chain()`

Get variable inclusion proportions over the MCMC chain.

**Parameters:**
- None

**Returns:**
- pandas.DataFrame: Variable inclusion proportions.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

var_props = model.get_var_props_over_chain()
```

### Visualization

#### `plot_convergence_diagnostics()`

Plot convergence diagnostics for a BART model.

**Parameters:**
- None

**Returns:**
- None

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

model.plot_convergence_diagnostics()
```

#### `plot_y_vs_yhat(X=None, y=None)`

Plot actual vs. predicted values.

**Parameters:**
- `X` (pandas.DataFrame or numpy.ndarray, optional): The predictor variables. If None, uses the training data. Default is None.
- `y` (pandas.Series or numpy.ndarray, optional): The response variable. If None, uses the training data. Default is None.

**Returns:**
- None

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

model.plot_y_vs_yhat()
```

#### `plot_variable_importance()`

Plot variable importance measures.

**Parameters:**
- None

**Returns:**
- None

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

model.plot_variable_importance()
```

#### `plot_partial_dependence(var_name, num_points=100)`

Plot partial dependence for a variable.

**Parameters:**
- `var_name` (str): The name of the variable to plot.
- `num_points` (int, optional): Number of points to use for the plot. Default is 100.

**Returns:**
- None

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

model.plot_partial_dependence('x1')
```

### Statistical Tests

#### `cov_importance_test(cov_name, num_permutations=100)`

Test the importance of a covariate.

**Parameters:**
- `cov_name` (str): The name of the covariate to test.
- `num_permutations` (int, optional): Number of permutations to use. Default is 100.

**Returns:**
- dict: Test results.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

test_results = model.cov_importance_test('x1')
```

#### `linearity_test(cov_name, num_permutations=100)`

Test the linearity of a covariate.

**Parameters:**
- `cov_name` (str): The name of the covariate to test.
- `num_permutations` (int, optional): Number of permutations to use. Default is 100.

**Returns:**
- dict: Test results.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

test_results = model.linearity_test('x1')
```

#### `bart_machine_f_test(cov_names, num_permutations=100)`

Perform an F-test for variable importance.

**Parameters:**
- `cov_names` (list): The names of the covariates to test.
- `num_permutations` (int, optional): Number of permutations to use. Default is 100.

**Returns:**
- dict: Test results.

**Example:**
```python
import pandas as pd
import numpy as np
from bartmachine import bart_machine, initialize_jvm

initialize_jvm()

X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
y = X['x1'] + X['x2']**2 + np.random.normal(0, 0.1, 100)

model = bart_machine(X, y, num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000)

test_results = model.bart_machine_f_test(['x1', 'x2'])
