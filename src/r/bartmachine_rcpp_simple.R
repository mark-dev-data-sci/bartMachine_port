# R wrapper functions for the C++ implementation of bartMachine

# Function to create a regression model
bartmachine_regression <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Call the C++ function
  model <- rcpp_create_regression_model(X, y, num_trees, num_burn_in, num_iterations_after_burn_in)
  
  # Set the class
  class(model) <- "bartmachine_regression"
  
  return(model)
}

# Function to create a classification model
bartmachine_classification <- function(X, y, num_trees = 50, num_burn_in = 250, num_iterations_after_burn_in = 1000) {
  # Call the C++ function
  model <- rcpp_create_classification_model(X, y, num_trees, num_burn_in, num_iterations_after_burn_in)
  
  # Set the class
  class(model) <- "bartmachine_classification"
  
  return(model)
}

# Predict method for regression models
predict.bartmachine_regression <- function(object, newdata, get_intervals = FALSE, ...) {
  # Call the C++ function
  result <- rcpp_regression_predict(object$model_ptr, newdata, get_intervals)
  
  # Make sure the predictions are numeric
  if (is.list(result)) {
    predictions <- as.numeric(result$predictions)
    if (get_intervals) {
      return(list(predictions = predictions, intervals = result$intervals))
    } else {
      return(predictions)
    }
  } else {
    return(as.numeric(result))
  }
}

# Predict method for classification models
predict.bartmachine_classification <- function(object, newdata, type = "class", ...) {
  # Call the C++ function
  result <- rcpp_classification_predict(object$model_ptr, newdata, type)
  
  # Make sure the predictions are numeric
  if (type == "prob") {
    return(as.numeric(result$predictions))
  } else {
    return(as.integer(result$predictions))
  }
}

# Function to get variable importance
get_variable_importance <- function(model) {
  # Check if the model is a classification model
  is_classification <- inherits(model, "bartmachine_classification")
  
  # Call the C++ function
  result <- rcpp_get_variable_importance(model$model_ptr, is_classification)
  
  return(result)
}

# Function to clean up the model
cleanup_model <- function(model) {
  # Check if the model is a classification model
  is_classification <- inherits(model, "bartmachine_classification")
  
  # Call the C++ function
  rcpp_cleanup_model(model$model_ptr, is_classification)
}
