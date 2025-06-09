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

# PLACEHOLDER MODULE: This module will be fully implemented during the porting process
"""

__version__ = "0.1.0"

# Import main classes and functions
from .bartMachine import BartMachine
from .bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running
from .bart_package_builders import bart_machine
from .bart_package_cross_validation import bart_machine_cv
from .bart_package_data_preprocessing import preprocess_data
from .bart_package_f_tests import bart_machine_f_test, bart_machine_interaction_f_test
from .bart_package_plots import (
    plot_convergence_diagnostics,
    plot_variable_importance,
    plot_partial_dependence,
    plot_tree
)
from .bart_package_summaries import (
    summary,
    get_var_counts_over_chain,
    get_var_props_over_chain,
    get_sigsqs
)
from .bart_package_variable_selection import (
    investigate_var_importance,
    var_selection_by_permute
)
from .bart_package_predicts import (
    predict,
    calc_credible_intervals,
    calc_prediction_intervals,
    calc_quantiles
)

# Initialize the JVM when the package is imported
# This is commented out for now, as it should be explicitly initialized by the user
# initialize_jvm()

# Define what is exported
__all__ = [
    "BartMachine",
    "initialize_jvm",
    "shutdown_jvm",
    "is_jvm_running",
    "bart_machine",
    "bart_machine_cv",
    "preprocess_data",
    "bart_machine_f_test",
    "bart_machine_interaction_f_test",
    "plot_convergence_diagnostics",
    "plot_variable_importance",
    "plot_partial_dependence",
    "plot_tree",
    "summary",
    "get_var_counts_over_chain",
    "get_var_props_over_chain",
    "get_sigsqs",
    "investigate_var_importance",
    "var_selection_by_permute",
    "predict",
    "calc_credible_intervals",
    "calc_prediction_intervals",
    "calc_quantiles"
]
