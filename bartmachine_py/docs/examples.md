# Examples

This document provides examples of using the bartMachine Python package for various tasks.

## Basic Regression Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the JVM
initialize_jvm()

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and build BART model
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Print model summary
print(model.summary())

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs. predicted values
model.plot_y_vs_yhat(X_test, y_test)

# Get variable importance
var_importance = model.get_var_importance()
print("Variable Importance:")
print(var_importance)

# Plot variable importance
model.plot_variable_importance()

# Plot partial dependence for the most important variable
most_important_var = var_importance.index[0]
model.plot_partial_dependence(most_important_var)

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Basic Classification Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Initialize the JVM
initialize_jvm()

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and build BART model for classification
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    classification=True,
    seed=42
)

# Print model summary
print(model.summary())

# Make probability predictions
y_pred_prob = model.predict(X_test, type="prob")

# Make class predictions
y_pred_class = model.predict(X_test, type="class")

# Evaluate model
accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred_class)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Get variable importance
var_importance = model.get_var_importance()
print("Variable Importance:")
print(var_importance)

# Plot variable importance
model.plot_variable_importance()

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Cross-Validation Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine_cv, initialize_jvm, plot_cv_results
from sklearn.datasets import load_boston

# Initialize the JVM
initialize_jvm()

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Perform cross-validation to find optimal number of trees
cv_results = bart_machine_cv(
    X=X, 
    y=y, 
    k_folds=5, 
    num_trees_list=[20, 50, 100, 200],
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Print cross-validation results
print("Cross-Validation Results:")
print(cv_results)

# Plot cross-validation results
plot_cv_results(cv_results)

# Get optimal number of trees
optimal_num_trees = cv_results['optimal_parameter']
print(f"Optimal Number of Trees: {optimal_num_trees}")

# Build model with optimal number of trees
model = bart_machine(
    X=X, 
    y=y, 
    num_trees=optimal_num_trees, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Print model summary
print(model.summary())

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Handling Missing Data Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the JVM
initialize_jvm()

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Introduce missing values
np.random.seed(42)
mask = np.random.random(X.shape) < 0.1  # 10% missing values
X_missing = X.copy()
X_missing[mask] = np.nan

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.2, random_state=42)

# Create and build BART model with missing data
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Print model summary
print(model.summary())

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Variable Selection Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm, var_selection_by_permute
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Initialize the JVM
initialize_jvm()

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform variable selection
selected_vars = var_selection_by_permute(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    num_permutations=20,
    alpha=0.05,
    seed=42
)

print("Selected Variables:")
print(selected_vars)

# Build model with selected variables
model = bart_machine(
    X=X_train[selected_vars], 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Print model summary
print(model.summary())

# Make predictions
y_pred = model.predict(X_test[selected_vars])

# Evaluate model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Interaction Detection Example

```python
import numpy as np
import pandas as pd
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Initialize the JVM
initialize_jvm()

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and build BART model
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Investigate interactions
interactions = model.interaction_investigator(num_replicates=5)
print("Interaction Results:")
print(interactions)

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Credible Intervals Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Initialize the JVM
initialize_jvm()

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and build BART model
model = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    seed=42
)

# Make predictions with credible intervals
y_pred_ci = model.predict(X_test, type="credible", ci_level=0.95)

# Extract point predictions and credible intervals
y_pred = y_pred_ci[:, 0]
y_lower = y_pred_ci[:, 1]
y_upper = y_pred_ci[:, 2]

# Plot predictions with credible intervals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.errorbar(y_test, y_pred, yerr=[y_pred - y_lower, y_upper - y_pred], fmt='o', alpha=0.2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted with 95% Credible Intervals')
plt.tight_layout()
plt.show()

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
```

## Parallelization Example

```python
import numpy as np
import pandas as pd
import time
from bartmachine import bart_machine, initialize_jvm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Initialize the JVM
initialize_jvm(max_heap_size="4g")

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Time single-threaded model
start_time = time.time()
model_single = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    num_threads=1,
    seed=42
)
single_time = time.time() - start_time
print(f"Single-threaded time: {single_time:.2f} seconds")

# Time multi-threaded model
start_time = time.time()
model_multi = bart_machine(
    X=X_train, 
    y=y_train, 
    num_trees=50, 
    num_burn_in=200, 
    num_iterations_after_burn_in=1000,
    num_threads=4,  # Use 4 threads
    seed=42
)
multi_time = time.time() - start_time
print(f"Multi-threaded time: {multi_time:.2f} seconds")
print(f"Speedup: {single_time / multi_time:.2f}x")

# Shutdown JVM when done
from bartmachine import shutdown_jvm
shutdown_jvm()
