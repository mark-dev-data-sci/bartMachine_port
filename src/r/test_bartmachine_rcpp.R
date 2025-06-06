# Test script for the R-to-C++ bridge

# Load the package
# Note: In a real package, this would be done with library(bartMachine)
# But for testing, we'll source the R file directly
source("src/r/bartmachine_rcpp.R")

# Test the basic functions
cat("Testing basic functions...\n")

# Set the seed
set_seed(12345)

# Generate a random number
cat("Random number:", rand(), "\n")

# Sample from a normal distribution
cat("Samples from normal distribution:", sample_from_norm_dist(0, 1, 5), "\n")

# Sample from an inverse gamma distribution
cat("Samples from inverse gamma distribution:", sample_from_inv_gamma(2, 1, 5), "\n")

# Test the regression model
cat("\nTesting regression model...\n")

# Create a simple dataset
n <- 100
p <- 5
X <- matrix(runif(n * p), nrow = n)
y <- 2 * X[, 1] + 1.5 * X[, 2] - 0.5 * X[, 3] + rnorm(n, 0, 0.1)

# Build a regression model
model <- bartmachine_regression(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000)

# Print the model
print(model)

# Make predictions
test_point <- matrix(rep(0.5, p), nrow = 1)
pred <- predict(model, test_point)
cat("Prediction for test point:", pred, "\n")

# Make predictions with intervals
pred_with_intervals <- predict(model, test_point, get_intervals = TRUE)
cat("Prediction with intervals for test point:", pred_with_intervals$predictions, "\n")
cat("Interval:", pred_with_intervals$intervals[[1]], "\n")

# Test the classification model
cat("\nTesting classification model...\n")

# Create a simple dataset
n <- 100
p <- 5
X <- matrix(runif(n * p), nrow = n)
y <- as.integer(X[, 1] + X[, 2] > 1)

# Build a classification model
model <- bartmachine_classification(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000)

# Print the model
print(model)

# Make predictions
test_point <- matrix(rep(0.5, p), nrow = 1)
pred <- predict(model, test_point)
cat("Prediction for test point:", pred, "\n")

# Make probability predictions
prob <- predict(model, test_point, type = "prob")
cat("Probability for test point:", prob, "\n")

cat("\nAll tests completed!\n")
