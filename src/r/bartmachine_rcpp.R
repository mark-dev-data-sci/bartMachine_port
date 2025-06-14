#' @title bartMachine R-to-C++ Bridge
#' @description R functions that interface with the C++ implementation of bartMachine
#' @author Mark Huber (original R-to-Java bridge)
#' @author C++ Port Team (R-to-C++ bridge)

#' @useDynLib bartMachine
#' @importFrom Rcpp sourceCpp
NULL

#' @title Set the random seed for the C++ implementation
#' @description Sets the random seed for the C++ implementation of bartMachine
#' @param seed An integer seed value
#' @export
set_seed <- function(seed) {
  rcpp_set_seed(as.integer(seed))
}

#' @title Generate a random number
#' @description Generates a random number using the C++ implementation
#' @return A random number between 0 and 1
#' @export
rand <- function() {
  rcpp_rand()
}

#' @title Sample from a normal distribution
#' @description Samples from a normal distribution using the C++ implementation
#' @param mu The mean of the normal distribution
#' @param sigsq The variance of the normal distribution
#' @param n The number of samples to generate
#' @return A vector of n samples from the normal distribution
#' @export
sample_from_norm_dist <- function(mu, sigsq, n = 1) {
  rcpp_sample_from_norm_dist(mu, sigsq, as.integer(n))
}

#' @title Sample from an inverse gamma distribution
#' @description Samples from an inverse gamma distribution using the C++ implementation
#' @param k The shape parameter of the inverse gamma distribution
#' @param theta The scale parameter of the inverse gamma distribution
#' @param n The number of samples to generate
#' @return A vector of n samples from the inverse gamma distribution
#' @export
sample_from_inv_gamma <- function(k, theta, n = 1) {
  rcpp_sample_from_inv_gamma(k, theta, as.integer(n))
}

#' @title Build a bartMachine regression model
#' @description Builds a bartMachine regression model using the C++ implementation
#' @param X A matrix of predictors
#' @param y A vector of responses
#' @param num_trees The number of trees to use
#' @param num_burn_in The number of burn-in iterations
#' @param num_iterations_after_burn_in The number of iterations after burn-in
#' @return A bartMachine regression model
#' @export
bartmachine_regression <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Convert inputs to the right format
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  # Create the model using Rcpp
  model_data <- rcpp_create_regression_model(X, y, num_trees, num_burn_in, num_iterations_after_burn_in)
  
  # Create a new environment to store the model
  model <- new.env()
  
  # Store the model parameters and pointer
  model$model_ptr <- model_data$model_ptr
  model$n <- model_data$n
  model$p <- model_data$p
  model$num_trees <- num_trees
  model$num_burn_in <- num_burn_in
  model$num_iterations_after_burn_in <- num_iterations_after_burn_in
  
  # Set the class
  class(model) <- "bartmachine_regression"
  
  # Register finalizer to clean up C++ resources when R object is garbage collected
  reg.finalizer(model, function(e) {
    if (!is.null(e$model_ptr)) {
      rcpp_cleanup_model(e$model_ptr, FALSE)
      e$model_ptr <- NULL
    }
  }, onexit = TRUE)
  
  return(model)
}

#' @title Predict using a bartMachine regression model
#' @description Predicts using a bartMachine regression model
#' @param model A bartMachine regression model
#' @param newdata A matrix of new data to predict
#' @param get_intervals Whether to get prediction intervals
#' @return A list with predictions and optionally prediction intervals
#' @export
predict.bartmachine_regression <- function(model, newdata, get_intervals = FALSE) {
  # Convert inputs to the right format
  newdata <- as.matrix(newdata)
  
  # Check if model pointer is valid
  if (is.null(model$model_ptr)) {
    stop("Model has been cleaned up or is invalid")
  }
  
  # Make predictions using Rcpp
  result <- rcpp_regression_predict(model$model_ptr, newdata, get_intervals)
  
  # Format the output
  if (get_intervals) {
    return(list(
      predictions = result$predictions,
      intervals = result$intervals
    ))
  } else {
    return(result$predictions)
  }
}

#' @title Get variable importance from a bartMachine regression model
#' @description Gets variable importance from a bartMachine regression model
#' @param model A bartMachine regression model
#' @return A vector of variable importance scores
#' @export
get_variable_importance.bartmachine_regression <- function(model) {
  # Check if model pointer is valid
  if (is.null(model$model_ptr)) {
    stop("Model has been cleaned up or is invalid")
  }
  
  # Get variable importance using Rcpp
  result <- rcpp_get_variable_importance(model$model_ptr, FALSE)
  
  return(result$importance)
}

#' @title Build a bartMachine classification model
#' @description Builds a bartMachine classification model using the C++ implementation
#' @param X A matrix of predictors
#' @param y A vector of binary responses (0 or 1)
#' @param num_trees The number of trees to use
#' @param num_burn_in The number of burn-in iterations
#' @param num_iterations_after_burn_in The number of iterations after burn-in
#' @return A bartMachine classification model
#' @export
bartmachine_classification <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Convert inputs to the right format
  X <- as.matrix(X)
  y <- as.integer(y)
  
  # Create the model using Rcpp
  model_data <- rcpp_create_classification_model(X, y, num_trees, num_burn_in, num_iterations_after_burn_in)
  
  # Create a new environment to store the model
  model <- new.env()
  
  # Store the model parameters and pointer
  model$model_ptr <- model_data$model_ptr
  model$n <- model_data$n
  model$p <- model_data$p
  model$num_trees <- num_trees
  model$num_burn_in <- num_burn_in
  model$num_iterations_after_burn_in <- num_iterations_after_burn_in
  
  # Set the class
  class(model) <- "bartmachine_classification"
  
  # Register finalizer to clean up C++ resources when R object is garbage collected
  reg.finalizer(model, function(e) {
    if (!is.null(e$model_ptr)) {
      rcpp_cleanup_model(e$model_ptr, TRUE)
      e$model_ptr <- NULL
    }
  }, onexit = TRUE)
  
  return(model)
}

#' @title Predict using a bartMachine classification model
#' @description Predicts using a bartMachine classification model
#' @param model A bartMachine classification model
#' @param newdata A matrix of new data to predict
#' @param type Whether to return "class" or "prob"
#' @return A vector of predictions
#' @export
predict.bartmachine_classification <- function(model, newdata, type = c("class", "prob")) {
  # Convert inputs to the right format
  newdata <- as.matrix(newdata)
  type <- match.arg(type)
  
  # Check if model pointer is valid
  if (is.null(model$model_ptr)) {
    stop("Model has been cleaned up or is invalid")
  }
  
  # Make predictions using Rcpp
  result <- rcpp_classification_predict(model$model_ptr, newdata, type)
  
  # Return the predictions
  return(result$predictions)
}

#' @title Get variable importance from a bartMachine classification model
#' @description Gets variable importance from a bartMachine classification model
#' @param model A bartMachine classification model
#' @return A vector of variable importance scores
#' @export
get_variable_importance.bartmachine_classification <- function(model) {
  # Check if model pointer is valid
  if (is.null(model$model_ptr)) {
    stop("Model has been cleaned up or is invalid")
  }
  
  # Get variable importance using Rcpp
  result <- rcpp_get_variable_importance(model$model_ptr, TRUE)
  
  return(result$importance)
}

#' @title Print a bartMachine regression model
#' @description Prints a summary of a bartMachine regression model
#' @param x A bartMachine regression model
#' @param ... Additional arguments
#' @export
print.bartmachine_regression <- function(x, ...) {
  cat("bartMachine Regression Model\n")
  cat("Number of trees:", x$num_trees, "\n")
  cat("Number of burn-in iterations:", x$num_burn_in, "\n")
  cat("Number of iterations after burn-in:", x$num_iterations_after_burn_in, "\n")
  cat("Number of predictors:", x$p, "\n")
  cat("Number of observations:", x$n, "\n")
  cat("Model status:", ifelse(is.null(x$model_ptr), "Invalid (cleaned up)", "Valid"), "\n")
}

#' @title Print a bartMachine classification model
#' @description Prints a summary of a bartMachine classification model
#' @param x A bartMachine classification model
#' @param ... Additional arguments
#' @export
print.bartmachine_classification <- function(x, ...) {
  cat("bartMachine Classification Model\n")
  cat("Number of trees:", x$num_trees, "\n")
  cat("Number of burn-in iterations:", x$num_burn_in, "\n")
  cat("Number of iterations after burn-in:", x$num_iterations_after_burn_in, "\n")
  cat("Number of predictors:", x$p, "\n")
  cat("Number of observations:", x$n, "\n")
  cat("Model status:", ifelse(is.null(x$model_ptr), "Invalid (cleaned up)", "Valid"), "\n")
}
