# Compile the Rcpp code
library(Rcpp)

# Set the working directory to the project root
# This is necessary to ensure the include paths are correct
setwd("/Users/mark/Documents/Cline/bartMachine_port")

# Compile the Rcpp code
sourceCpp("src/rcpp/bartmachine_rcpp.cpp")

# Print success message
cat("Rcpp code compiled successfully!\n")
