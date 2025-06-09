# Script to compare Java and C++ implementations of bartMachine
# This script uses the original Java implementation and our C++ port on the same datasets

# Load required libraries
library(bartMachine)  # Original Java implementation

# Source our C++ port
source("../src/r/bartmachine_cpp_port.R")

# Set random seed for reproducibility
set.seed(12345)

# Function to create synthetic regression dataset
create_regression_dataset <- function(n = 100, p = 5, noise_level = 0.1) {
  X <- as.data.frame(matrix(runif(n * p), nrow = n))
  colnames(X) <- paste0("X", 1:p)
  y <- 2 * X[, 1] + 1.5 * X[, 2] - 0.5 * X[, 3] + rnorm(n, 0, noise_level)
  list(X = X, y = y)
}

# Function to create synthetic classification dataset
create_classification_dataset <- function(n = 100, p = 5, noise_level = 0.1) {
  X <- as.data.frame(matrix(runif(n * p), nrow = n))
  colnames(X) <- paste0("X", 1:p)
  prob <- pnorm(2 * X[, 1] + 1.5 * X[, 2] - 0.5 * X[, 3] + rnorm(n, 0, noise_level))
  y <- as.factor(ifelse(prob > 0.5, 1, 0))
  list(X = X, y = y)
}

# Function to compare regression models
compare_regression <- function(dataset_name, X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  cat("Comparing regression models on dataset:", dataset_name, "\n")
  
  # Split data into training and testing sets
  n <- nrow(X)
  train_idx <- sample(1:n, 0.8 * n)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Train Java model
  cat("Training Java model...\n")
  start_time <- Sys.time()
  java_model <- bartMachine(X_train, y_train, 
                           num_trees = num_trees, 
                           num_burn_in = num_burn_in,
                           num_iterations_after_burn_in = num_iterations_after_burn_in,
                           verbose = FALSE)
  java_build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Train C++ model
  cat("Training C++ model...\n")
  start_time <- Sys.time()
  cpp_model <- build_bart_machine(X_train, y_train, 
                                 num_trees = num_trees, 
                                 num_burn_in = num_burn_in,
                                 num_iterations_after_burn_in = num_iterations_after_burn_in,
                                 verbose = FALSE)
  cpp_build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions with Java model
  cat("Making predictions with Java model...\n")
  start_time <- Sys.time()
  java_preds <- predict(java_model, X_test)
  java_pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions with C++ model
  cat("Making predictions with C++ model...\n")
  start_time <- Sys.time()
  cpp_preds <- predict(cpp_model, X_test)
  cpp_pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Calculate intervals with Java model
  cat("Calculating intervals with Java model...\n")
  start_time <- Sys.time()
  java_intervals <- calc_credible_intervals(java_model, X_test)
  java_interval_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Calculate intervals with C++ model
  cat("Calculating intervals with C++ model...\n")
  start_time <- Sys.time()
  cpp_intervals <- calc_credible_intervals(cpp_model, X_test)
  cpp_interval_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Calculate variable importance with Java model
  cat("Calculating variable importance with Java model...\n")
  start_time <- Sys.time()
  java_var_imp <- investigate_var_importance(java_model, num_replicates = 5)
  java_var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Calculate variable importance with C++ model (if available)
  cat("Calculating variable importance with C++ model...\n")
  start_time <- Sys.time()
  cpp_var_imp_time <- 0
  cpp_var_imp <- NULL
  tryCatch({
    cpp_var_imp <- investigate_var_importance(cpp_model, num_replicates = 5)
    cpp_var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  }, error = function(e) {
    cat("Variable importance not implemented in C++ model\n")
  })
  
  # Compare predictions
  pred_cor <- cor(java_preds, cpp_preds)
  pred_rmse <- sqrt(mean((java_preds - cpp_preds)^2))
  
  # Compare variable importance
  var_imp_cor <- NA
  if (!is.null(cpp_var_imp)) {
    var_imp_cor <- cor(java_var_imp$avg_var_props, cpp_var_imp$avg_var_props)
  }
  
  # Compare performance
  build_time_ratio <- as.numeric(cpp_build_time / java_build_time)
  pred_time_ratio <- as.numeric(cpp_pred_time / java_pred_time)
  var_imp_time_ratio <- if (java_var_imp_time > 0) as.numeric(cpp_var_imp_time / java_var_imp_time) else 0
  interval_time_ratio <- as.numeric(cpp_interval_time / java_interval_time)
  
  # Return results
  list(
    dataset = dataset_name,
    pred_cor = pred_cor,
    pred_rmse = pred_rmse,
    var_imp_cor = var_imp_cor,
    build_time_ratio = build_time_ratio,
    pred_time_ratio = pred_time_ratio,
    var_imp_time_ratio = var_imp_time_ratio,
    interval_time_ratio = interval_time_ratio,
    java_preds = java_preds,
    cpp_preds = cpp_preds,
    java_intervals = java_intervals,
    cpp_intervals = cpp_intervals,
    java_var_imp = java_var_imp,
    cpp_var_imp = cpp_var_imp
  )
}

# Function to compare classification models
compare_classification <- function(dataset_name, X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  cat("Comparing classification models on dataset:", dataset_name, "\n")
  
  # Split data into training and testing sets
  n <- nrow(X)
  train_idx <- sample(1:n, 0.8 * n)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Train Java model
  cat("Training Java model...\n")
  start_time <- Sys.time()
  java_model <- bartMachine(X_train, y_train, 
                           num_trees = num_trees, 
                           num_burn_in = num_burn_in,
                           num_iterations_after_burn_in = num_iterations_after_burn_in,
                           verbose = FALSE)
  java_build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Train C++ model
  cat("Training C++ model...\n")
  start_time <- Sys.time()
  cpp_model <- build_bart_machine(X_train, y_train, 
                                 num_trees = num_trees, 
                                 num_burn_in = num_burn_in,
                                 num_iterations_after_burn_in = num_iterations_after_burn_in,
                                 verbose = FALSE)
  cpp_build_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions with Java model (probabilities)
  cat("Making predictions with Java model...\n")
  start_time <- Sys.time()
  java_probs <- predict(java_model, X_test, type = "prob")
  java_preds <- predict(java_model, X_test, type = "class")
  java_pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Make predictions with C++ model (probabilities)
  cat("Making predictions with C++ model...\n")
  start_time <- Sys.time()
  cpp_probs <- predict(cpp_model, X_test, type = "prob")
  cpp_preds <- predict(cpp_model, X_test, type = "class")
  cpp_pred_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Calculate variable importance with Java model
  cat("Calculating variable importance with Java model...\n")
  start_time <- Sys.time()
  java_var_imp <- investigate_var_importance(java_model, num_replicates = 5)
  java_var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Calculate variable importance with C++ model (if available)
  cat("Calculating variable importance with C++ model...\n")
  start_time <- Sys.time()
  cpp_var_imp_time <- 0
  cpp_var_imp <- NULL
  tryCatch({
    cpp_var_imp <- investigate_var_importance(cpp_model, num_replicates = 5)
    cpp_var_imp_time <- difftime(Sys.time(), start_time, units = "secs")
  }, error = function(e) {
    cat("Variable importance not implemented in C++ model\n")
  })
  
  # Compare predictions
  pred_cor <- NA
  pred_rmse <- mean(java_preds != cpp_preds)
  
  # Compare probabilities
  prob_cor <- cor(java_probs, cpp_probs)
  prob_rmse <- sqrt(mean((java_probs - cpp_probs)^2))
  
  # Compare variable importance
  var_imp_cor <- NA
  if (!is.null(cpp_var_imp)) {
    var_imp_cor <- cor(java_var_imp$avg_var_props, cpp_var_imp$avg_var_props)
  }
  
  # Compare performance
  build_time_ratio <- as.numeric(cpp_build_time / java_build_time)
  pred_time_ratio <- as.numeric(cpp_pred_time / java_pred_time)
  var_imp_time_ratio <- if (java_var_imp_time > 0) as.numeric(cpp_var_imp_time / java_var_imp_time) else 0
  
  # Return results
  list(
    dataset = dataset_name,
    pred_cor = pred_cor,
    pred_rmse = pred_rmse,
    prob_cor = prob_cor,
    prob_rmse = prob_rmse,
    var_imp_cor = var_imp_cor,
    build_time_ratio = build_time_ratio,
    pred_time_ratio = pred_time_ratio,
    var_imp_time_ratio = var_imp_time_ratio,
    java_probs = java_probs,
    cpp_probs = cpp_probs,
    java_preds = java_preds,
    cpp_preds = cpp_preds,
    java_var_imp = java_var_imp,
    cpp_var_imp = cpp_var_imp
  )
}

# Function to generate a comparison report
generate_comparison_report <- function(regression_results, classification_results) {
  report <- "Comparison Report: Java vs C++ Implementation of bartMachine\n\n"
  report <- paste0(report, "## Overview\n\n")
  report <- paste0(report, "This report compares the Java and C++ implementations of bartMachine on various datasets.\n\n")
  
  # Regression results
  report <- paste0(report, "## Regression Results\n\n")
  for (result in regression_results) {
    report <- paste0(report, "### Dataset: ", result$dataset, "\n\n")
    
    report <- paste0(report, "#### Numerical Equivalence\n\n")
    report <- paste0(report, "- Prediction correlation: ", round(result$pred_cor, 4), "\n")
    report <- paste0(report, "- Prediction RMSE: ", round(result$pred_rmse, 4), "\n")
    report <- paste0(report, "- Variable importance correlation: ", round(result$var_imp_cor, 4), "\n\n")
    
    report <- paste0(report, "#### Performance Comparison\n\n")
    report <- paste0(report, "- Build time ratio (C++/Java): ", round(result$build_time_ratio, 2), "\n")
    report <- paste0(report, "- Prediction time ratio (C++/Java): ", round(result$pred_time_ratio, 2), "\n")
    report <- paste0(report, "- Variable importance time ratio (C++/Java): ", round(result$var_imp_time_ratio, 2), "\n")
    report <- paste0(report, "- Interval time ratio (C++/Java): ", round(result$interval_time_ratio, 2), "\n\n")
    
    report <- paste0(report, "#### Interpretation\n\n")
    if (result$pred_rmse < 0.01) {
      report <- paste0(report, "The predictions from the C++ and Java implementations are numerically equivalent.\n\n")
    } else {
      report <- paste0(report, "There are significant differences between the predictions from the C++ and Java implementations.\n\n")
    }
    
    if (result$build_time_ratio < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for model building.\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for model building.\n")
    }
    
    if (result$pred_time_ratio < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for prediction.\n\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for prediction.\n\n")
    }
  }
  
  # Classification results
  report <- paste0(report, "## Classification Results\n\n")
  for (result in classification_results) {
    report <- paste0(report, "### Dataset: ", result$dataset, "\n\n")
    
    report <- paste0(report, "#### Numerical Equivalence\n\n")
    report <- paste0(report, "- Prediction correlation: ", round(result$pred_cor, 4), "\n")
    report <- paste0(report, "- Prediction RMSE: ", round(result$pred_rmse, 4), "\n")
    report <- paste0(report, "- Probability correlation: ", round(result$prob_cor, 4), "\n")
    report <- paste0(report, "- Probability RMSE: ", round(result$prob_rmse, 4), "\n")
    report <- paste0(report, "- Variable importance correlation: ", round(result$var_imp_cor, 4), "\n\n")
    
    report <- paste0(report, "#### Performance Comparison\n\n")
    report <- paste0(report, "- Build time ratio (C++/Java): ", round(result$build_time_ratio, 2), "\n")
    report <- paste0(report, "- Prediction time ratio (C++/Java): ", round(result$pred_time_ratio, 2), "\n")
    report <- paste0(report, "- Variable importance time ratio (C++/Java): ", round(result$var_imp_time_ratio, 2), "\n\n")
    
    report <- paste0(report, "#### Interpretation\n\n")
    if (result$pred_rmse < 0.01 && result$prob_rmse < 0.01) {
      report <- paste0(report, "The predictions from the C++ and Java implementations are numerically equivalent.\n\n")
    } else {
      report <- paste0(report, "There are significant differences between the predictions from the C++ and Java implementations.\n\n")
    }
    
    if (result$build_time_ratio < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for model building.\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for model building.\n")
    }
    
    if (result$pred_time_ratio < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for prediction.\n\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for prediction.\n\n")
    }
  }
  
  # Conclusion
  report <- paste0(report, "## Conclusion\n\n")
  report <- paste0(report, "Based on the comparison results, we can conclude that:\n\n")
  
  # Check if predictions are numerically equivalent
  all_pred_rmse <- c(sapply(regression_results, function(x) x$pred_rmse), 
                    sapply(classification_results, function(x) x$pred_rmse))
  if (all(all_pred_rmse < 0.01)) {
    report <- paste0(report, "1. The predictions from the C++ and Java implementations are numerically equivalent.\n")
  } else {
    report <- paste0(report, "1. There are significant differences between the results from the C++ and Java implementations.\n")
  }
  
  # Compare build times
  all_build_time_ratios <- c(sapply(regression_results, function(x) x$build_time_ratio), 
                           sapply(classification_results, function(x) x$build_time_ratio))
  avg_build_time_ratio <- mean(all_build_time_ratios)
  if (avg_build_time_ratio < 1) {
    report <- paste0(report, "2. The C++ implementation is faster than the Java implementation for model building (", 
                    round((1 - avg_build_time_ratio) * 100), "% faster on average).\n")
  } else {
    report <- paste0(report, "2. The Java implementation is faster than the C++ implementation for model building (", 
                    round((avg_build_time_ratio - 1) * 100), "% faster on average).\n")
  }
  
  # Compare prediction times
  all_pred_time_ratios <- c(sapply(regression_results, function(x) x$pred_time_ratio), 
                          sapply(classification_results, function(x) x$pred_time_ratio))
  avg_pred_time_ratio <- mean(all_pred_time_ratios)
  if (avg_pred_time_ratio < 1) {
    report <- paste0(report, "3. The C++ implementation is faster than the Java implementation for prediction (", 
                    round((1 - avg_pred_time_ratio) * 100), "% faster on average)")
  } else {
    report <- paste0(report, "3. The Java implementation is faster than the C++ implementation for prediction (", 
                    round((avg_pred_time_ratio - 1) * 100), "% faster on average)")
  }
  
  return(report)
}

# Run comparison on synthetic datasets
run_synthetic_comparison <- function() {
  # Create synthetic datasets
  reg_data <- create_regression_dataset(n = 1000, p = 5, noise_level = 0.1)
  class_data <- create_classification_dataset(n = 1000, p = 5, noise_level = 0.1)
  
  # Run comparisons
  reg_result <- compare_regression("synthetic", reg_data$X, reg_data$y, 
                                  num_trees = 50, 
                                  num_burn_in = 50, 
                                  num_iterations_after_burn_in = 100)
  
  class_result <- compare_classification("synthetic", class_data$X, class_data$y, 
                                       num_trees = 50, 
                                       num_burn_in = 50, 
                                       num_iterations_after_burn_in = 100)
  
  # Generate report
  report <- generate_comparison_report(list(reg_result), list(class_result))
  
  # Save results
  save(reg_result, class_result, file = "../build/comparison_results_port.RData")
  writeLines(report, "../build/comparison_report_port.md")
  
  cat("Comparison completed. Results saved to build/comparison_results_port.RData and build/comparison_report_port.md\n")
}

# Run comparison
run_synthetic_comparison()
