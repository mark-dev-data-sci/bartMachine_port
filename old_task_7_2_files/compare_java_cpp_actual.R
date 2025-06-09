# Script to compare Java and C++ implementations of bartMachine
# This script uses the actual C++ implementation via the validation_runner executable

# Load required libraries
library(bartMachine)  # Original Java implementation

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
  y <- as.integer(prob > 0.5)
  list(X = X, y = y)
}

# Function to save dataset to CSV
save_dataset <- function(dataset, filename) {
  data <- cbind(dataset$X, y = dataset$y)
  write.csv(data, filename, row.names = FALSE)
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
  # The model will be garbage collected automatically
  options(java.parameters = old_java_options)
  old_cores <- bartMachine::set_bart_machine_num_cores(old_bart_options)
  cat("bartMachine now using", old_cores, "cores.\n")
  
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
  # The model will be garbage collected automatically
  options(java.parameters = old_java_options)
  old_cores <- bartMachine::set_bart_machine_num_cores(old_bart_options)
  cat("bartMachine now using", old_cores, "cores.\n")
  
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
run_cpp_regression <- function(dataset_file, output_file) {
  # Run the validation_runner executable
  cmd <- paste("../build/validation_runner regression", dataset_file, output_file)
  system(cmd)
  
  # Read the results
  results <- list()
  
  # Read predictions
  lines <- readLines(output_file)
  
  # Find the start of each section
  pred_start <- which(lines == "predictions") + 1
  var_imp_start <- which(lines == "variable_importance") + 1
  intervals_start <- which(lines == "intervals") + 1
  times_start <- which(lines == "times") + 1
  
  # Read predictions
  pred_end <- var_imp_start - 2
  predictions <- as.numeric(lines[pred_start:pred_end])
  
  # Read variable importance
  var_imp_end <- intervals_start - 2
  var_importance <- as.numeric(lines[var_imp_start:var_imp_end])
  
  # Read intervals
  times_end <- times_start - 2
  intervals_raw <- lines[intervals_start:times_end]
  intervals <- list()
  intervals$predictions <- numeric(length(intervals_raw))
  intervals$intervals <- matrix(0, nrow = length(intervals_raw), ncol = 2)
  for (i in 1:length(intervals_raw)) {
    parts <- strsplit(intervals_raw[i], ",")[[1]]
    intervals$predictions[i] <- (as.numeric(parts[1]) + as.numeric(parts[2])) / 2
    intervals$intervals[i, 1] <- as.numeric(parts[1])
    intervals$intervals[i, 2] <- as.numeric(parts[2])
  }
  
  # Read times
  times <- list()
  for (i in (times_start):length(lines)) {
    if (lines[i] != "") {
      parts <- strsplit(lines[i], ",")[[1]]
      times[[parts[1]]] <- as.numeric(parts[2])
    }
  }
  
  # Return results
  list(
    predictions = predictions,
    var_importance = list(importance = var_importance),
    intervals = intervals,
    times = times
  )
}

# Function to run classification model with C++ implementation
run_cpp_classification <- function(dataset_file, output_file) {
  # Run the validation_runner executable
  cmd <- paste("../build/validation_runner classification", dataset_file, output_file)
  system(cmd)
  
  # Read the results
  results <- list()
  
  # Read predictions
  lines <- readLines(output_file)
  
  # Find the start of each section
  pred_start <- which(lines == "predictions") + 1
  prob_start <- which(lines == "probabilities") + 1
  var_imp_start <- which(lines == "variable_importance") + 1
  times_start <- which(lines == "times") + 1
  
  # Read predictions
  pred_end <- prob_start - 2
  predictions <- as.integer(lines[pred_start:pred_end])
  
  # Read probabilities
  prob_end <- var_imp_start - 2
  probabilities <- as.numeric(lines[prob_start:prob_end])
  
  # Read variable importance
  var_imp_end <- times_start - 2
  var_importance <- as.numeric(lines[var_imp_start:var_imp_end])
  
  # Read times
  times <- list()
  for (i in (times_start):length(lines)) {
    if (lines[i] != "") {
      parts <- strsplit(lines[i], ",")[[1]]
      times[[parts[1]]] <- as.numeric(parts[2])
    }
  }
  
  # Return results
  list(
    predictions = predictions,
    probabilities = probabilities,
    var_importance = list(importance = var_importance),
    times = times
  )
}

# Function to compare results
compare_results <- function(java_results, cpp_results, type = "regression") {
  cat("Comparing", type, "results:\n")
  
  # Compare predictions
  pred_corr <- tryCatch({
    cor(java_results$predictions, cpp_results$predictions)
  }, error = function(e) {
    cat("Error calculating prediction correlation:", e$message, "\n")
    return(NA)
  })
  
  pred_rmse <- sqrt(mean((java_results$predictions - cpp_results$predictions)^2))
  cat("Prediction correlation:", pred_corr, "\n")
  cat("Prediction RMSE:", pred_rmse, "\n")
  
  if (type == "classification") {
    # Compare probabilities
    prob_corr <- tryCatch({
      cor(java_results$probabilities, cpp_results$probabilities)
    }, error = function(e) {
      cat("Error calculating probability correlation:", e$message, "\n")
      return(NA)
    })
    
    prob_rmse <- sqrt(mean((java_results$probabilities - cpp_results$probabilities)^2))
    cat("Probability correlation:", prob_corr, "\n")
    cat("Probability RMSE:", prob_rmse, "\n")
  }
  
  # Compare variable importance
  var_imp_corr <- tryCatch({
    cor(java_results$var_importance$avg_var_props, cpp_results$var_importance$importance)
  }, error = function(e) {
    cat("Error calculating variable importance correlation:", e$message, "\n")
    return(NA)
  })
  
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

# Function to load datasets from the original bartMachine repository
load_original_datasets <- function() {
  # Path to the original repository datasets
  datasets_path <- "/Users/mark/Documents/Cline/bartMachine/datasets"
  
  # List of datasets to use
  datasets <- list()
  datasets$regression <- list()
  datasets$classification <- list()
  
  # Check if the datasets directory exists
  if (dir.exists(datasets_path)) {
    cat("Loading datasets from original bartMachine repository...\n")
    
    # Load regression datasets
    if (file.exists(file.path(datasets_path, "friedman1.RData"))) {
      load(file.path(datasets_path, "friedman1.RData"))
      # Convert X to data.frame if it's not already
      if (!is.data.frame(X)) {
        X <- as.data.frame(X)
        colnames(X) <- paste0("X", 1:ncol(X))
      }
      datasets$regression$friedman1 <- list(X = X, y = y)
      cat("Loaded friedman1 dataset\n")
    }
    
    # Load classification datasets
    if (file.exists(file.path(datasets_path, "heart.RData"))) {
      load(file.path(datasets_path, "heart.RData"))
      # Convert X to data.frame if it's not already
      if (!is.data.frame(X)) {
        X <- as.data.frame(X)
        colnames(X) <- paste0("X", 1:ncol(X))
      }
      datasets$classification$heart <- list(X = X, y = y)
      cat("Loaded heart dataset\n")
    }
    
    # Add more datasets as needed
  } else {
    cat("Original datasets directory not found. Using synthetic datasets instead.\n")
  }
  
  # If no datasets were loaded, create synthetic ones
  if (length(datasets$regression) == 0) {
    cat("Creating synthetic regression dataset...\n")
    datasets$regression$synthetic <- create_regression_dataset()
  }
  
  if (length(datasets$classification) == 0) {
    cat("Creating synthetic classification dataset...\n")
    datasets$classification$synthetic <- create_classification_dataset()
  }
  
  return(datasets)
}

# Main function to run comparison
run_comparison <- function() {
  cat("Starting comparison between Java and C++ implementations...\n")
  
  # Load or create datasets
  datasets <- load_original_datasets()
  
  # Results to store all comparisons
  all_results <- list()
  
  # Run regression models on all regression datasets
  cat("\nRunning regression models...\n")
  for (dataset_name in names(datasets$regression)) {
    cat("\nDataset:", dataset_name, "\n")
    dataset <- datasets$regression[[dataset_name]]
    
    # Save dataset to CSV for C++ implementation
    dataset_file <- paste0(dataset_name, "_regression.csv")
    save_dataset(dataset, dataset_file)
    
    cat("Java implementation...\n")
    java_reg <- run_java_regression(dataset$X, dataset$y)
    
    cat("C++ implementation...\n")
    cpp_output_file <- paste0(dataset_name, "_regression_cpp_output.txt")
    cpp_reg <- run_cpp_regression(dataset_file, cpp_output_file)
    
    # Compare results
    cat("\n=== Regression Comparison for", dataset_name, "===\n")
    reg_comparison <- compare_results(java_reg, cpp_reg, "regression")
    
    # Store results
    all_results$regression[[dataset_name]] <- list(
      java = java_reg,
      cpp = cpp_reg,
      comparison = reg_comparison
    )
  }
  
  # Run classification models on all classification datasets
  cat("\nRunning classification models...\n")
  for (dataset_name in names(datasets$classification)) {
    cat("\nDataset:", dataset_name, "\n")
    dataset <- datasets$classification[[dataset_name]]
    
    # Save dataset to CSV for C++ implementation
    dataset_file <- paste0(dataset_name, "_classification.csv")
    save_dataset(dataset, dataset_file)
    
    cat("Java implementation...\n")
    java_class <- run_java_classification(dataset$X, dataset$y)
    
    cat("C++ implementation...\n")
    cpp_output_file <- paste0(dataset_name, "_classification_cpp_output.txt")
    cpp_class <- run_cpp_classification(dataset_file, cpp_output_file)
    
    # Compare results
    cat("\n=== Classification Comparison for", dataset_name, "===\n")
    class_comparison <- compare_results(java_class, cpp_class, "classification")
    
    # Store results
    all_results$classification[[dataset_name]] <- list(
      java = java_class,
      cpp = cpp_class,
      comparison = class_comparison
    )
  }
  
  # Return all results
  return(all_results)
}

# Generate a detailed report
generate_report <- function(results) {
  report <- "# Comparison Report: Java vs C++ Implementation of bartMachine\n\n"
  report <- paste0(report, "## Overview\n\n")
  report <- paste0(report, "This report compares the Java and C++ implementations of bartMachine on various datasets.\n\n")
  
  # Add regression results
  report <- paste0(report, "## Regression Results\n\n")
  for (dataset_name in names(results$regression)) {
    report <- paste0(report, "### Dataset: ", dataset_name, "\n\n")
    
    comparison <- results$regression[[dataset_name]]$comparison
    
    report <- paste0(report, "#### Numerical Equivalence\n\n")
    report <- paste0(report, "- Prediction correlation: ", round(comparison$pred_corr, 4), "\n")
    report <- paste0(report, "- Prediction RMSE: ", round(comparison$pred_rmse, 4), "\n")
    report <- paste0(report, "- Variable importance correlation: ", round(comparison$var_imp_corr, 4), "\n\n")
    
    report <- paste0(report, "#### Performance Comparison\n\n")
    report <- paste0(report, "- Build time ratio (C++/Java): ", round(comparison$time_ratio$build, 2), "\n")
    report <- paste0(report, "- Prediction time ratio (C++/Java): ", round(comparison$time_ratio$pred, 2), "\n")
    report <- paste0(report, "- Variable importance time ratio (C++/Java): ", round(comparison$time_ratio$var_imp, 2), "\n")
    report <- paste0(report, "- Interval time ratio (C++/Java): ", round(comparison$time_ratio$interval, 2), "\n\n")
    
    # Add interpretation
    report <- paste0(report, "#### Interpretation\n\n")
    
    # Numerical equivalence interpretation
    if (!is.na(comparison$pred_corr) && comparison$pred_corr > 0.99 && comparison$pred_rmse < 0.01) {
      report <- paste0(report, "The predictions from the C++ implementation are numerically equivalent to the Java implementation.\n\n")
    } else if (!is.na(comparison$pred_corr) && comparison$pred_corr > 0.95) {
      report <- paste0(report, "The predictions from the C++ implementation are very similar to the Java implementation, but not exactly equivalent.\n\n")
    } else {
      report <- paste0(report, "There are significant differences between the predictions from the C++ and Java implementations.\n\n")
    }
    
    # Performance interpretation
    if (comparison$time_ratio$build < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for model building.\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for model building.\n")
    }
    
    if (comparison$time_ratio$pred < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for prediction.\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for prediction.\n")
    }
    
    report <- paste0(report, "\n")
  }
  
  # Add classification results
  report <- paste0(report, "## Classification Results\n\n")
  for (dataset_name in names(results$classification)) {
    report <- paste0(report, "### Dataset: ", dataset_name, "\n\n")
    
    comparison <- results$classification[[dataset_name]]$comparison
    
    report <- paste0(report, "#### Numerical Equivalence\n\n")
    report <- paste0(report, "- Prediction correlation: ", round(comparison$pred_corr, 4), "\n")
    report <- paste0(report, "- Prediction RMSE: ", round(comparison$pred_rmse, 4), "\n")
    report <- paste0(report, "- Probability correlation: ", round(comparison$prob_corr, 4), "\n")
    report <- paste0(report, "- Probability RMSE: ", round(comparison$prob_rmse, 4), "\n")
    report <- paste0(report, "- Variable importance correlation: ", round(comparison$var_imp_corr, 4), "\n\n")
    
    report <- paste0(report, "#### Performance Comparison\n\n")
    report <- paste0(report, "- Build time ratio (C++/Java): ", round(comparison$time_ratio$build, 2), "\n")
    report <- paste0(report, "- Prediction time ratio (C++/Java): ", round(comparison$time_ratio$pred, 2), "\n")
    report <- paste0(report, "- Variable importance time ratio (C++/Java): ", round(comparison$time_ratio$var_imp, 2), "\n\n")
    
    # Add interpretation
    report <- paste0(report, "#### Interpretation\n\n")
    
    # Numerical equivalence interpretation
    if (!is.na(comparison$pred_corr) && !is.na(comparison$prob_corr) && 
        comparison$pred_corr > 0.99 && comparison$pred_rmse < 0.01 && 
        comparison$prob_corr > 0.99 && comparison$prob_rmse < 0.01) {
      report <- paste0(report, "The predictions from the C++ implementation are numerically equivalent to the Java implementation.\n\n")
    } else if (!is.na(comparison$pred_corr) && !is.na(comparison$prob_corr) && 
               comparison$pred_corr > 0.95 && comparison$prob_corr > 0.95) {
      report <- paste0(report, "The predictions from the C++ implementation are very similar to the Java implementation, but not exactly equivalent.\n\n")
    } else {
      report <- paste0(report, "There are significant differences between the predictions from the C++ and Java implementations.\n\n")
    }
    
    # Performance interpretation
    if (comparison$time_ratio$build < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for model building.\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for model building.\n")
    }
    
    if (comparison$time_ratio$pred < 1) {
      report <- paste0(report, "The C++ implementation is faster than the Java implementation for prediction.\n")
    } else {
      report <- paste0(report, "The Java implementation is faster than the C++ implementation for prediction.\n")
    }
    
    report <- paste0(report, "\n")
  }
  
  # Add conclusion
  report <- paste0(report, "## Conclusion\n\n")
  report <- paste0(report, "Based on the comparison results, we can conclude that:\n\n")
  
  # Calculate average correlation and RMSE across all datasets
  pred_corrs <- sapply(c(results$regression, results$classification), function(x) x$comparison$pred_corr)
  pred_rmses <- sapply(c(results$regression, results$classification), function(x) x$comparison$pred_rmse)
  avg_pred_corr <- mean(pred_corrs, na.rm = TRUE)
  avg_pred_rmse <- mean(pred_rmses, na.rm = TRUE)
  
  # Calculate average performance ratios
  avg_build_ratio <- mean(sapply(c(results$regression, results$classification), function(x) x$comparison$time_ratio$build))
  avg_pred_ratio <- mean(sapply(c(results$regression, results$classification), function(x) x$comparison$time_ratio$pred))
  
  if (!is.na(avg_pred_corr) && avg_pred_corr > 0.99 && avg_pred_rmse < 0.01) {
    report <- paste0(report, "1. The C++ implementation produces numerically equivalent results to the Java implementation.\n")
  } else if (!is.na(avg_pred_corr) && avg_pred_corr > 0.95) {
    report <- paste0(report, "1. The C++ implementation produces very similar results to the Java implementation, but not exactly equivalent.\n")
  } else {
    report <- paste0(report, "1. There are significant differences between the results from the C++ and Java implementations.\n")
  }
  
  if (avg_build_ratio < 1) {
    report <- paste0(report, "2. The C++ implementation is faster than the Java implementation for model building (", round(100 * (1 - avg_build_ratio)), "% faster on average).\n")
  } else {
    report <- paste0(report, "2. The Java implementation is faster than the C++ implementation for model building (", round(100 * (avg_build_ratio - 1)), "% faster on average).\n")
  }
  
  if (avg_pred_ratio < 1) {
    report <- paste0(report, "3. The C++ implementation is faster than the Java implementation for prediction (", round(100 * (1 - avg_pred_ratio)), "% faster on average).\n")
  } else {
    report <- paste0(report, "3. The Java implementation is faster than the C++ implementation for prediction (", round(100 * (avg_pred_ratio - 1)), "% faster on average).\n")
  }
  
  return(report)
}

# Run the comparison
results <- run_comparison()

# Save results
save(results, file = "comparison_results.RData")
cat("\nResults saved to comparison_results.RData\n")

# Generate the report
report <- generate_report(results)

# Write the report to a file
writeLines(report, "comparison_report.md")
cat("Report saved to comparison_report.md\n")
