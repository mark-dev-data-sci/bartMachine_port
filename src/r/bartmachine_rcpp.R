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
  
  # Create a new environment to store the model
  model <- new.env()
  
  # Store the model parameters
  model$X <- X
  model$y <- y
  model$num_trees <- num_trees
  model$num_burn_in <- num_burn_in
  model$num_iterations_after_burn_in <- num_iterations_after_burn_in
  
  # Set the class
  class(model) <- "bartmachine_regression"
  
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
  
  # Make predictions for each row of newdata
  predictions <- list()
  for (i in 1:nrow(newdata)) {
    result <- rcpp_bartmachine_regression(model$X, model$y, newdata[i, ])
    predictions[[i]] <- result
  }
  
  # Format the output
  if (get_intervals) {
    return(list(
      predictions = sapply(predictions, function(x) x$prediction),
      intervals = lapply(predictions, function(x) c(x$lower, x$upper))
    ))
  } else {
    return(sapply(predictions, function(x) x$prediction))
  }
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
  
  # Create a new environment to store the model
  model <- new.env()
  
  # Store the model parameters
  model$X <- X
  model$y <- y
  model$num_trees <- num_trees
  model$num_burn_in <- num_burn_in
  model$num_iterations_after_burn_in <- num_iterations_after_burn_in
  
  # Set the class
  class(model) <- "bartmachine_classification"
  
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
  
  # Make predictions for each row of newdata
  predictions <- list()
  for (i in 1:nrow(newdata)) {
    result <- rcpp_bartmachine_classification(model$X, model$y, newdata[i, ])
    predictions[[i]] <- result
  }
  
  # Format the output
  if (type == "class") {
    return(sapply(predictions, function(x) x$prediction))
  } else {
    return(sapply(predictions, function(x) x$probability))
  }
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
  cat("Number of predictors:", ncol(x$X), "\n")
  cat("Number of observations:", nrow(x$X), "\n")
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
  cat("Number of predictors:", ncol(x$X), "\n")
  cat("Number of observations:", nrow(x$X), "\n")
}
