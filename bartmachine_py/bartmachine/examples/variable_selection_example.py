"""
Variable Selection Example

This example demonstrates how to use the variable selection functions in bartMachine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from bartmachine import (
    initialize_jvm,
    shutdown_jvm,
    bart_machine,
    var_selection_by_permute,
    var_selection_by_permute_cv,
    investigate_var_importance,
    interaction_investigator
)

# Initialize JVM
initialize_jvm(max_heap_size="2g")

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building BART model...")
# Build BART model
bart = bart_machine(
    X=X_train,
    y=y_train,
    num_trees=50,
    num_burn_in=200,
    num_iterations_after_burn_in=1000,
    alpha=0.95,
    beta=2,
    k=2,
    q=0.9,
    nu=3,
    verbose=True
)

# Print model summary
print("\nModel Summary:")
bart.summary()

# Investigate variable importance
print("\nInvestigating variable importance...")
var_importance = investigate_var_importance(
    bart,
    type="splits",
    plot=True,
    num_replicates_for_avg=3,
    num_trees_bottleneck=20
)

print("\nAverage variable proportions:")
print(var_importance["avg_var_props"])

# Variable selection by permutation
print("\nVariable selection by permutation...")
var_selection = var_selection_by_permute(
    bart,
    num_reps_for_avg=3,
    num_permute_samples=20,
    num_trees_for_permute=20,
    alpha=0.05,
    plot=True
)

print("\nImportant variables (local method):")
print(var_selection["important_vars_local_names"])

print("\nImportant variables (global max method):")
print(var_selection["important_vars_global_max_names"])

print("\nImportant variables (global SE method):")
print(var_selection["important_vars_global_se_names"])

# Variable selection by permutation with cross-validation
print("\nVariable selection by permutation with cross-validation...")
var_selection_cv = var_selection_by_permute_cv(
    bart,
    k_folds=3,
    num_reps_for_avg=2,
    num_permute_samples=10,
    num_trees_for_permute=20,
    alpha=0.05,
    num_trees_pred_cv=20
)

print("\nBest method:", var_selection_cv["best_method"])
print("\nImportant variables (CV):")
print(var_selection_cv["important_vars_cv"])

# Investigate interactions
print("\nInvestigating interactions...")
interactions = interaction_investigator(
    bart,
    plot=True,
    num_replicates_for_avg=2,
    num_trees_bottleneck=20,
    num_var_plot=10
)

print("\nTop 5 interactions:")
print(interactions.head(5))

# Make predictions on test data
print("\nMaking predictions on test data...")
y_pred = bart.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nTest RMSE: {rmse:.4f}")

# Plot actual vs. predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted")
plt.tight_layout()
plt.show()

# Shutdown JVM
shutdown_jvm()
