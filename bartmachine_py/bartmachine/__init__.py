"""
bartMachine Python Package

This package provides a Python implementation of the bartMachine R package,
which implements Bayesian Additive Regression Trees (BART).

The package is designed to be a direct port of the R package, with the same
API and functionality. It uses the same Java backend as the R package, but
provides a Python interface.

R Package Correspondence:
    This Python package corresponds to the 'bartMachine' R package.

Role in Port:
    This package provides a Python interface to the BART algorithm, using the
    same Java backend as the R package. It is designed to be a direct port of
    the R package, with the same API and functionality.
"""

__version__ = "0.1.0"

# Import main classes and functions
from .bart_package_builders import BartMachine, bart_machine
from .zzz import initialize_jvm, shutdown_jvm, is_jvm_running
from .bart_package_cross_validation import bart_machine_cv, k_fold_cv, plot_cv_results, get_cv_error, get_cv_predictions
from .bart_package_data_preprocessing import preprocess_training_data, preprocess_new_data, handle_missing_values, create_dummy_variables, standardize_data, normalize_data, check_data_quality
from .bart_package_f_tests import cov_importance_test, linearity_test, bart_machine_f_test
from .bart_package_plots import (
    plot_convergence_diagnostics,
    plot_y_vs_yhat,
    plot_variable_importance,
    plot_partial_dependence,
    plot_tree,
    investigate_var_importance,
    interaction_investigator
)
from .bart_package_summaries import (
    summary,
    get_sigsqs,
    get_var_counts_over_chain,
    get_var_props_over_chain,
    calc_credible_intervals,
    calc_prediction_intervals
)
from .bart_package_variable_selection import (
    get_var_importance,
    get_var_props_over_chain,
    var_selection_by_permute,
    var_selection_by_permute_cv
)

# Define what is exported
__all__ = [
    # Main classes and functions
    "BartMachine",
    "bart_machine",
    
    # JVM management
    "initialize_jvm",
    "shutdown_jvm",
    "is_jvm_running",
    
    # Model building and cross-validation
    "bart_machine_cv",
    "k_fold_cv",
    "plot_cv_results",
    "get_cv_error",
    "get_cv_predictions",
    
    # Data preprocessing
    "preprocess_training_data",
    "preprocess_new_data",
    "handle_missing_values",
    "create_dummy_variables",
    "standardize_data",
    "normalize_data",
    "check_data_quality",
    
    # F-tests
    "cov_importance_test",
    "linearity_test",
    "bart_machine_f_test",
    
    # Plots
    "plot_convergence_diagnostics",
    "plot_y_vs_yhat",
    "plot_variable_importance",
    "plot_partial_dependence",
    "plot_tree",
    
    # Summaries
    "summary",
    "get_sigsqs",
    "get_var_counts_over_chain",
    "get_var_props_over_chain",
    "calc_credible_intervals",
    "calc_prediction_intervals",
    
    # Variable selection
    "get_var_importance",
    "investigate_var_importance",
    "var_selection_by_permute",
    "var_selection_by_permute_cv",
    "interaction_investigator",
    "cov_importance_test"
]
