BART_MAX_MEM_MB_DEFAULT = 1100 #1.1GB is the most a 32bit machine can give without throwing an error or crashing
BART_NUM_CORES_DEFAULT = 1 #Stay conservative as a default

##build a BART model
build_bart_machine = function(X = NULL, y = NULL, Xy = NULL,
		num_trees = 50, #found many times to not get better after this value... so let it be the default, it's faster too
		num_burn_in = 250,
		num_iterations_after_burn_in = 1000,
		alpha = 0.95,
		beta = 2,
		k = 2,
		q = 0.9,
		nu = 3.0,
		prob_rule_class = 0.5,
		mh_prob_steps = c(2.5, 2.5, 4) / 9, #only the first two matter
		debug_log = FALSE,
		run_in_sample = TRUE,
		s_sq_y = "mse", # "mse" or "var"
		sig_sq_est = NULL, #you can pass this in to speed things up if you have an idea about what you want to use a priori
		print_tree_illustrations = FALSE, #POWER USERS ONLY
		cov_prior_vec = NULL,
		interaction_constraints = NULL,
		use_missing_data = FALSE,
		covariates_to_permute = NULL, #PRIVATE
		num_rand_samps_in_library = 10000, #give the user the option to make a bigger library of random samples of normals and inv-gammas
		use_missing_data_dummies_as_covars = FALSE,
		replace_missing_data_with_x_j_bar = FALSE,
		impute_missingness_with_rf_impute = FALSE,
		impute_missingness_with_x_j_bar_for_lm = TRUE,
		mem_cache_for_speed = TRUE,
		flush_indices_to_save_RAM = TRUE,
		serialize = FALSE,
		seed = NULL,
		verbose = TRUE){

	if (verbose){
		cat("bartMachine initializing with", num_trees, "trees...\n")
	}
	t0 = Sys.time()

	if (use_missing_data_dummies_as_covars && replace_missing_data_with_x_j_bar){
		stop("You cannot impute by averages and use missing data as dummies simultaneously.")
	}

	if ((is.null(X) && is.null(Xy)) || is.null(y) && is.null(Xy)){
		stop("You need to give bartMachine a training set either by specifying X and y or by specifying a matrix Xy which contains the response named \"y.\"\n")
	} else if (!is.null(X) && !is.null(y) && !is.null(Xy)){
		stop("You cannot specify both X,y and Xy simultaneously.")
	} else if (is.null(X) && is.null(y)){ #they specified Xy, so now just pull out X,y
		#first ensure it's a dataframe
		if (!inherits(Xy, "data.frame")){
			stop(paste("The training data Xy must be a data frame."), call. = FALSE)
		}
		y = Xy[, ncol(Xy)]
		for (cov in 1 : (ncol(Xy) - 1)){
			if (colnames(Xy)[cov] == ""){
				colnames(Xy)[cov] = paste("V", cov, sep = "")
			}
		}
		X = as.data.frame(Xy[, 1 : (ncol(Xy) - 1)])
		colnames(X) = colnames(Xy)[1 : (ncol(Xy) - 1)]
	}

	#make sure it's a data frame
	if (!inherits(X, "data.frame")){
		stop(paste("The training data X must be a data frame."), call. = FALSE)
	}
	if (verbose){
		cat("bartMachine vars checked...\n")
	}
	#we are about to construct a bartMachine object. First, let R garbage collect
	#to clean up previous bartMachine objects that are no longer in use. This is important
	#because R's garbage collection system does not "see" the size of Java objects. Thus,
	#you are at risk of running out of memory without this invocation.
	gc() #Delete at your own risk!

	#now take care of classification or regression
	y_levels = levels(y)
	if (inherits(y, "numeric") || inherits(y, "integer")){ #if y is numeric, then it's a regression problem
		if (inherits(y, "integer")){
			cat("Warning: The response y is integer, bartMachine will run regression.\n")
		}
		#java expects doubles, not ints, so we need to cast this now to avoid errors later
		if (inherits(y, "integer")){
			y = as.numeric(y)
		}
		# MODIFIED: Use C++ implementation instead of Java
		# java_bart_machine = .jnew("bartMachine.bartMachineRegressionMultThread")
		
		# Set random seed if provided
		if (!is.null(seed)) {
			rcpp_set_seed(seed)
		}
		
		# Convert X to matrix
		X_matrix = as.matrix(X)
		
		# Create the C++ model
		cpp_model = rcpp_create_regression_model(X_matrix, y, 
												num_trees = num_trees,
												num_burn_in = num_burn_in,
												num_iterations_after_burn_in = num_iterations_after_burn_in)
		
		y_remaining = y
		pred_type = "regression"
	} else if (inherits(y, "factor") & length(y_levels) == 2){ #if y is a factor and binary
		#convenience for users that use 0/1 variables to ensure positive category is first as a level and a label (i.e. the naive expectation)
		if (all(sort(levels(factor(y))) == c("0", "1"))){
			y = factor(y, levels = c(1, 0), labels = c(1, 0))
			y_levels = levels(y)
		}
		# MODIFIED: Use C++ implementation instead of Java
		# java_bart_machine = .jnew("bartMachine.bartMachineClassificationMultThread")
		
		# Set random seed if provided
		if (!is.null(seed)) {
			rcpp_set_seed(seed)
		}
		
		# Convert X to matrix
		X_matrix = as.matrix(X)
		
		# Convert y to integer vector (0/1)
		y_int = as.integer(ifelse(y == y_levels[1], 1, 0))
		
		# Create the C++ model
		cpp_model = rcpp_create_classification_model(X_matrix, y_int, 
													num_trees = num_trees,
													num_burn_in = num_burn_in,
													num_iterations_after_burn_in = num_iterations_after_burn_in)
		
		y_remaining = ifelse(y == y_levels[1], 1, 0)
		pred_type = "classification"
	} else { #otherwise throw an error
		stop("Your response must be either numeric, an integer or a factor with two levels.\n")
	}

	# Create a bartMachine object with the C++ model
	bart_machine = list(
		cpp_model = cpp_model,
		X = X,
		y = y,
		y_levels = y_levels,
		pred_type = pred_type,
		num_trees = num_trees,
		num_burn_in = num_burn_in,
		num_iterations_after_burn_in = num_iterations_after_burn_in,
		alpha = alpha,
		beta = beta,
		k = k,
		q = q,
		nu = nu,
		prob_rule_class = prob_rule_class,
		mh_prob_steps = mh_prob_steps,
		verbose = verbose,
		mem_cache_for_speed = mem_cache_for_speed,
		flush_indices_to_save_RAM = flush_indices_to_save_RAM
	)
	
	class(bart_machine) = "bartMachine"
	
	if (verbose){
		cat("bartMachine training completed.\n")
	}
	
	return(bart_machine)
}
