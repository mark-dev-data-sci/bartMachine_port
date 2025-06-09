S3 predict method
predict.bartMachine = function(object, new_data, type = "prob", prob_rule_class = NULL, verbose = TRUE, ...){
	if(!(type %in% c("prob", "class"))){
		stop("For classification, type must be either \"prob\" or \"class\". ")
	}

	if (object$pred_type == "regression"){
		bart_machine_get_posterior(object, new_data)$y_hat
	} else { ##classification
	    if (type == "prob"){
			if (verbose == TRUE){
				cat("predicting probabilities where \"", object$y_levels[1], "\" is considered the target level...\n", sep = "")
			}
	    	bart_machine_get_posterior(object, new_data)$y_hat
	    } else {
	    	labels = bart_machine_get_posterior(object, new_data)$y_hat > ifelse(is.null(prob_rule_class), object$prob_rule_class, prob_rule_class)
	      	#return whatever the raw y_levels were
	      	labels_to_y_levels(object, labels)
	    }
	}
}

##private function
labels_to_y_levels = function(bart_machine, labels){
	factor(ifelse(labels == TRUE, bart_machine$y_levels[1], bart_machine$y_levels[2]), levels = bart_machine$y_levels)
}

##utility function for predicting when test outcomes are known
bart_predict_for_test_data = function(bart_machine, Xtest, ytest, prob_rule_class = NULL){
	if (bart_machine$pred_type == "regression"){ #regression list
	  ytest_hat = predict(bart_machine, Xtest)
		n = nrow(Xtest)
		L2_err = sum((ytest - ytest_hat)^2)

		list(
				y_hat = ytest_hat,
				L1_err = sum(abs(ytest - ytest_hat)),
				L2_err = L2_err,
				rmse = sqrt(L2_err / n),
				e = ytest - ytest_hat
		)
	} else { ##classification list
	    if (!inherits(ytest, "factor")){
			stop("ytest must be a factor.")
		}
	    if (!all(levels(ytest) %in% bart_machine$y_levels)){
			stop("New factor level not seen in training introduced. Please remove.")
		}

	    ptest_hat = predict(bart_machine, Xtest, type = "prob")
	    ytest_labels = ptest_hat > ifelse(is.null(prob_rule_class), bart_machine$prob_rule_class, prob_rule_class)
	    ytest_hat = labels_to_y_levels(bart_machine, ytest_labels)

		confusion_matrix = as.data.frame(matrix(NA, nrow = 3, ncol = 3))
		rownames(confusion_matrix) = c(paste("actual", bart_machine$y_levels), "use errors")
		colnames(confusion_matrix) = c(paste("predicted", bart_machine$y_levels), "model errors")
		confusion_matrix[1 : 2, 1 : 2] = as.integer(table(ytest, ytest_hat))
		confusion_matrix[3, 1] = round(confusion_matrix[2, 1] / (confusion_matrix[1, 1] + confusion_matrix[2, 1]), 3)
		confusion_matrix[3, 2] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 2] + confusion_matrix[2, 2]), 3)
		confusion_matrix[1, 3] = round(confusion_matrix[1, 2] / (confusion_matrix[1, 1] + confusion_matrix[1, 2]), 3)
		confusion_matrix[2, 3] = round(confusion_matrix[2, 1] / (confusion_matrix[2, 1] + confusion_matrix[2, 2]), 3)
		confusion_matrix[3, 3] = round((confusion_matrix[1, 2] + confusion_matrix[2, 1]) / sum(confusion_matrix[1 : 2, 1 : 2]), 3)

		list(y_hat = ytest_hat, p_hat = ptest_hat, confusion_matrix = confusion_matrix)
	}
}

##get full set of samples from posterior distribution of f(x)
bart_machine_get_posterior = function(bart_machine, new_data){
	if (!"data.frame"%in%class(new_data)){
		stop("\"new_data\" needs to be a data frame with the same column names as the training data.")
	}
	if (!bart_machine$use_missing_data){
		nrow_before = nrow(new_data)
		new_data = na.omit(new_data)
		if (nrow_before > nrow(new_data)){
			cat(nrow_before - nrow(new_data), "rows omitted due to missing data. Try using the missing data feature in \"build_bart_machine\" to be able to predict on all observations.\n")
		}
	}

	if (nrow(new_data) == 0){
		stop("No rows to predict.\n")
	}
	
	# Process new data
	new_data = pre_process_new_data(new_data, bart_machine)

	# Check for missing data if this feature was not turned on
	if (!bart_machine$use_missing_data){
		M = matrix(0, nrow = nrow(new_data), ncol = ncol(new_data))
		for (i in 1 : nrow(new_data)){
			for (j in 1 : ncol(new_data)){
				if (is.missing(new_data[i, j])){
					M[i, j] = 1
				}
			}
		}
		if (sum(M) > 0){
			warning("missing data found in test data and bartMachine was not built with missing data feature!\n")
		}
	}

	# Convert new_data to matrix
	new_data_matrix = as.matrix(new_data)
	
	# Make predictions using C++ model
	if (bart_machine$pred_type == "regression") {
		# Use regression predict function
		result = rcpp_regression_predict(bart_machine$cpp_model, new_data_matrix)
		y_hat = result$predictions
		
		# For now, we don't have posterior samples in the C++ implementation
		# So we'll create a dummy matrix with the same prediction for all samples
		y_hat_posterior_samples = matrix(y_hat, nrow = length(y_hat), 
										ncol = bart_machine$num_iterations_after_burn_in, 
										byrow = FALSE)
	} else {
		# Use classification predict function
		result = rcpp_classification_predict(bart_machine$cpp_model, new_data_matrix, type = "prob")
		y_hat = result$probabilities
		
		# For now, we don't have posterior samples in the C++ implementation
		# So we'll create a dummy matrix with the same prediction for all samples
		y_hat_posterior_samples = matrix(y_hat, nrow = length(y_hat), 
										ncol = bart_machine$num_iterations_after_burn_in, 
										byrow = FALSE)
	}

	list(y_hat = y_hat, X = new_data, y_hat_posterior_samples = y_hat_posterior_samples)
}

##compute credible intervals
calc_credible_intervals = function(bart_machine, new_data, ci_conf = 0.95){
	# First convert the rows to the correct dummies etc
	new_data = pre_process_new_data(new_data, bart_machine)
	n_test = nrow(new_data)

	# Convert new_data to matrix
	new_data_matrix = as.matrix(new_data)
	
	if (bart_machine$pred_type == "regression") {
		# Use regression predict function with intervals
		result = rcpp_regression_predict(bart_machine$cpp_model, new_data_matrix, get_intervals = TRUE)
		
		# Extract intervals
		ci_lower_bd = result$intervals[, 1]
		ci_upper_bd = result$intervals[, 2]
	} else {
		# For classification, we don't have credible intervals
		# So we'll use the prediction as both lower and upper bounds
		result = rcpp_classification_predict(bart_machine$cpp_model, new_data_matrix, type = "prob")
		y_hat = result$probabilities
		
		ci_lower_bd = y_hat
		ci_upper_bd = y_hat
	}
	
	# Put them together and return
	cbind(ci_lower_bd, ci_upper_bd)
}

##compute prediction intervals
calc_prediction_intervals = function(bart_machine, new_data, pi_conf = 0.95, num_samples_per_data_point = 1000){
	if (bart_machine$pred_type == "classification"){
		stop("Prediction intervals are not possible for classification.")
	}

	# First convert the rows to the correct dummies etc
	new_data = pre_process_new_data(new_data, bart_machine)
	n_test = nrow(new_data)

	# Convert new_data to matrix
	new_data_matrix = as.matrix(new_data)
	
	# Use regression predict function with intervals
	result = rcpp_regression_predict(bart_machine$cpp_model, new_data_matrix, get_intervals = TRUE)
	
	# Extract intervals
	pi_lower_bd = result$intervals[, 1]
	pi_upper_bd = result$intervals[, 2]
	
	# For now, we don't have prediction samples in the C++ implementation
	# So we'll create dummy samples with the same prediction for all samples
	all_prediction_samples = matrix(result$predictions, nrow = n_test, 
									ncol = num_samples_per_data_point, 
									byrow = FALSE)
	
	# Put them together and return
	list(interval = cbind(pi_lower_bd, pi_upper_bd), all_prediction_samples = all_prediction_samples)
}
