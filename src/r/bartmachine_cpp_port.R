# bartMachine C++ Port
# This file loads the modified R files that replace Java calls with C++ calls

# Load required libraries
library(Rcpp)

# Source the Rcpp code
sourceCpp("../src/rcpp/bartmachine_rcpp.cpp")

# Source all the modified R files
source("../src/r/bartmachine_cpp_port/bart_arrays.R")
source("../src/r/bartmachine_cpp_port/bart_node_related_methods.R")
source("../src/r/bartmachine_cpp_port/bart_package_builders_cpp.R")
source("../src/r/bartmachine_cpp_port/bart_package_cross_validation.R")
source("../src/r/bartmachine_cpp_port/bart_package_data_preprocessing.R")
source("../src/r/bartmachine_cpp_port/bart_package_f_tests.R")
source("../src/r/bartmachine_cpp_port/bart_package_inits.R")
source("../src/r/bartmachine_cpp_port/bart_package_plots.R")
source("../src/r/bartmachine_cpp_port/bart_package_predicts_cpp.R")
source("../src/r/bartmachine_cpp_port/bart_package_summaries.R")
source("../src/r/bartmachine_cpp_port/bart_package_variable_selection.R")
source("../src/r/bartmachine_cpp_port/bartMachine.R")
source("../src/r/bartmachine_cpp_port/zzz.R")

# Print a message
cat("bartMachine C++ Port loaded successfully.\n")
