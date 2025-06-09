# Script to check if random samples are being initialized correctly

# Load required libraries
library(Rcpp)

# Compile the Rcpp interface
cat("Compiling the Rcpp interface...\n")
sourceCpp("../src/rcpp/bartmachine_rcpp.cpp")

# Create a simple function to check the random samples
check_random_samples <- function() {
  # Create a small dataset
  X <- matrix(runif(100), nrow = 20)
  y <- runif(20)
  
  # Call the C++ function to create a regression model
  # This should initialize the random samples
  cat("Creating regression model to initialize random samples...\n")
  model <- rcpp_create_regression_model(X, y)
  
  # Print some information about the model
  cat("Model created successfully.\n")
  cat("Number of rows:", model$n, "\n")
  cat("Number of columns:", model$p, "\n")
  
  # Make a prediction to check if the random samples are being used
  cat("Making a prediction...\n")
  pred <- rcpp_regression_predict(model$model_ptr, X[1:5, ])
  cat("Predictions:", pred$predictions, "\n")
  
  # Clean up
  cat("Cleaning up...\n")
  rcpp_cleanup_model(model$model_ptr)
  
  cat("Random samples check completed.\n")
}

# Run the check
check_random_samples()
