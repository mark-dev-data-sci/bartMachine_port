"""
Main bartMachine class for BART models.

This module provides the main BartMachine class for BART models.

R File Correspondence:
    This Python module corresponds to 'src/r/bartmachine_cpp_port/bartMachine.R' 
    in the original R package.

Role in Port:
    This module provides the main BartMachine class that serves as the primary interface
    for users of the package. It integrates all the components of the BART algorithm
    and provides a unified API for building and using BART models.
"""

# Import and re-export the BartMachine class and bart_machine function from bart_package_builders.py
from .bart_package_builders import BartMachine, bart_machine

# Import and re-export the bart_machine_cv function from bart_package_cross_validation.py
from .bart_package_cross_validation import bart_machine_cv as bartMachineCV

# Re-export the functions with the same names as in the R package
__all__ = ['BartMachine', 'bart_machine', 'bartMachineCV']
