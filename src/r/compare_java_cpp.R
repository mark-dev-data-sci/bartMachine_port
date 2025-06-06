# Script to compare Java and C++ implementations of bartMachine
# This script runs identical workflows on both implementations and compares the results

# Load required libraries
library(bartMachine)  # Original Java implementation
# We'll need to load our C++ implementation separately

# Set random seed for reproducibility
set.seed(12345)

# Function to create synthetic regression dataset
create_regression_dataset <- function(n = 100, p = 5, noise_level = 0.1) {
  X <- matrix(runif(n * p), nrow = n)
  y <- 2 * X[, 1] + 1.5 * X[, 2] - 0.5 * X[, 3] + rnorm(n, 0, noise_level)
  list(X = X, y = y)
}

# Function to create synthetic classification dataset
create_classification_dataset <- function(n = 100, p = 5, noise_level = 0.1) {
  X <- matrix(runif(n * p), nrow = n)
  prob <- pnorm(2 * X[, 1] + 1.5 * X[, 2] - 0.5 * X[, 3] + rnorm(n, 0, noise_level))
  y <- as.integer(prob > 0.5)
  list(X = X, y = y)
}

# Function to run regression model with Java implementation
run_java_regression <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Temporarily disable Java bartMachine messages
  old_java_options <- options("java.parameters")
  options(java.parameters = "-Xmx2500m")
  old_bart_options <- bartMachine::set_bart_machine_num_cores(1)
  
  # Build model
  start_time <- Sys.time()
  model <- bartMachine(X, y, num_trees = num_trees, num_burn_in = num_burn_in, 
                       num_iterations_after_burn_in = num_iterations_after_burn_in,
                       verbose = FALSE)
  build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions
  start_time <- Sys.time()
  preds <- predict(model, X)
  pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get variable importance
  start_time <- Sys.time()
  var_imp <- investigate_var_importance(model, num_replicates_for_avg = 5)
  var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get intervals
  start_time <- Sys.time()
  intervals <- calc_credible_intervals(model, X[1:5, ])
  interval_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Clean up
  destroy_bart_machine(model)
  options(java.parameters = old_java_options)
  bartMachine::set_bart_machine_num_cores(old_bart_options)
  
  # Return results
  list(
    predictions = preds,
    var_importance = var_imp,
    intervals = intervals,
    times = list(
      build_time = build_time,
      pred_time = pred_time,
      var_imp_time = var_imp_time,
      interval_time = interval_time
    )
  )
}

# Function to run classification model with Java implementation
run_java_classification <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Temporarily disable Java bartMachine messages
  old_java_options <- options("java.parameters")
  options(java.parameters = "-Xmx2500m")
  old_bart_options <- bartMachine::set_bart_machine_num_cores(1)
  
  # Build model
  start_time <- Sys.time()
  model <- bartMachine(X, y, num_trees = num_trees, num_burn_in = num_burn_in, 
                       num_iterations_after_burn_in = num_iterations_after_burn_in,
                       verbose = FALSE)
  build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions
  start_time <- Sys.time()
  preds <- predict(model, X)
  pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get probabilities
  start_time <- Sys.time()
  probs <- predict(model, X, type = "prob")
  prob_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get variable importance
  start_time <- Sys.time()
  var_imp <- investigate_var_importance(model, num_replicates_for_avg = 5)
  var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Clean up
  destroy_bart_machine(model)
  options(java.parameters = old_java_options)
  bartMachine::set_bart_machine_num_cores(old_bart_options)
  
  # Return results
  list(
    predictions = preds,
    probabilities = probs,
    var_importance = var_imp,
    times = list(
      build_time = build_time,
      pred_time = pred_time,
      prob_time = prob_time,
      var_imp_time = var_imp_time
    )
  )
}

# Function to run regression model with C++ implementation
run_cpp_regression <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Load our C++ implementation
  source("src/r/bartmachine_rcpp.R")
  
  # Build model
  start_time <- Sys.time()
  model <- bartmachine_regression(X, y, num_trees = num_trees, num_burn_in = num_burn_in, 
                                 num_iterations_after_burn_in = num_iterations_after_burn_in)
  build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions
  start_time <- Sys.time()
  preds <- predict(model, X)
  pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get variable importance
  start_time <- Sys.time()
  var_imp <- get_variable_importance(model)
  var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get intervals
  start_time <- Sys.time()
  intervals <- predict(model, X[1:5, ], get_intervals = TRUE)
  interval_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Return results
  list(
    predictions = preds,
    var_importance = var_imp,
    intervals = intervals,
    times = list(
      build_time = build_time,
      pred_time = pred_time,
      var_imp_time = var_imp_time,
      interval_time = interval_time
    )
  )
}

# Function to run classification model with C++ implementation
run_cpp_classification <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Load our C++ implementation
  source("src/r/bartmachine_rcpp.R")
  
  # Build model
  start_time <- Sys.time()
  model <- bartmachine_classification(X, y, num_trees = num_trees, num_burn_in = num_burn_in, 
                                     num_iterations_after_burn_in = num_iterations_after_burn_in)
  build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions
  start_time <- Sys.time()
  preds <- predict(model, X)
  pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get probabilities
  start_time <- Sys.time()
  probs <- predict(model, X, type = "prob")
  prob_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Get variable importance
  start_time <- Sys.time()
  var_imp <- get_variable_importance(model)
  var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Return results
  list(
    predictions = preds,
    probabilities = probs,
    var_importance = var_imp,
    times = list(
      build_time = build_time,
      pred_time = pred_time,
      prob_time = prob_time,
      var_imp_time = var_imp_time
    )
  )
}

# Function to compare results
compare_results <- function(java_results, cpp_results, type = "regression") {
  cat("Comparing", type, "results:\n")
  
  # Compare predictions
  pred_corr <- cor(java_results$predictions, cpp_results$predictions)
  pred_rmse <- sqrt(mean((java_results$predictions - cpp_results$predictions)^2))
  cat("Prediction correlation:", pred_corr, "\n")
  cat("Prediction RMSE:", pred_rmse, "\n")
  
  if (type == "classification") {
    # Compare probabilities
    prob_corr <- cor(java_results$probabilities, cpp_results$probabilities)
    prob_rmse <- sqrt(mean((java_results$probabilities - cpp_results$probabilities)^2))
    cat("Probability correlation:", prob_corr, "\n")
    cat("Probability RMSE:", prob_rmse, "\n")
  }
  
  # Compare variable importance
  var_imp_corr <- cor(java_results$var_importance$avg_var_props, cpp_results$var_importance$importance)
  cat("Variable importance correlation:", var_imp_corr, "\n")
  
  # Compare times
  cat("\nPerformance comparison:\n")
  cat("Build time (Java):", java_results$times$build_time, "seconds\n")
  cat("Build time (C++):", cpp_results$times$build_time, "seconds\n")
  cat("Prediction time (Java):", java_results$times$pred_time, "seconds\n")
  cat("Prediction time (C++):", cpp_results$times$pred_time, "seconds\n")
  
  if (type == "regression") {
    cat("Interval time (Java):", java_results$times$interval_time, "seconds\n")
    cat("Interval time (C++):", cpp_results$times$interval_time, "seconds\n")
  }
  
  cat("Variable importance time (Java):", java_results$times$var_imp_time, "seconds\n")
  cat("Variable importance time (C++):", cpp_results$times$var_imp_time, "seconds\n")
  
  # Return comparison metrics
  list(
    pred_corr = pred_corr,
    pred_rmse = pred_rmse,
    var_imp_corr = var_imp_corr,
    prob_corr = if (type == "classification") prob_corr else NULL,
    prob_rmse = if (type == "classification") prob_rmse else NULL,
    time_ratio = list(
      build = as.numeric(cpp_results$times$build_time) / as.numeric(java_results$times$build_time),
      pred = as.numeric(cpp_results$times$pred_time) / as.numeric(java_results$times$pred_time),
      var_imp = as.numeric(cpp_results$times$var_imp_time) / as.numeric(java_results$times$var_imp_time),
      interval = if (type == "regression") 
        as.numeric(cpp_results$times$interval_time) / as.numeric(java_results$times$interval_time) else NULL
    )
  )
}

# Main function to run comparison
run_comparison <- function() {
  cat("Starting comparison between Java and C++ implementations...\n")
  
  # Create datasets
  cat("Creating datasets...\n")
  reg_data <- create_regression_dataset()
  class_data <- create_classification_dataset()
  
  # Run regression models
  cat("\nRunning regression models...\n")
  cat("Java implementation...\n")
  java_reg <- run_java_regression(reg_data$X, reg_data$y)
  cat("C++ implementation...\n")
  cpp_reg <- run_cpp_regression(reg_data$X, reg_data$y)
  
  # Run classification models
  cat("\nRunning classification models...\n")
  cat("Java implementation...\n")
  java_class <- run_java_classification(class_data$X, class_data$y)
  cat("C++ implementation...\n")
  cpp_class <- run_cpp_classification(class_data$X, class_data$y)
  
  # Compare results
  cat("\n=== Regression Comparison ===\n")
  reg_comparison <- compare_results(java_reg, cpp_reg, "regression")
  
  cat("\n=== Classification Comparison ===\n")
  class_comparison <- compare_results(java_class, cpp_class, "classification")
  
  # Return all results
  list(
    regression = list(
      java = java_reg,
      cpp = cpp_reg,
      comparison = reg_comparison
    ),
    classification = list(
      java = java_class,
      cpp = cpp_class,
      comparison = class_comparison
    )
  )
}

# Run the comparison
results <- run_comparison()

# Save results
save(results, file = "comparison_results.RData")
cat("\nResults saved to comparison_results.RData\n")
