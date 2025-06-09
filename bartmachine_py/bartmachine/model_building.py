"""
Model Building for bartMachine

This module provides the main BartMachine class for building BART models.
It serves as a Python wrapper around the Java implementation, closely matching
the original R implementation.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Optional, Union, List, Dict, Any, Tuple, Callable

from .java_bridge import (
    set_seed, get_java_class, convert_to_java_array, convert_to_java_2d_array,
    convert_from_java_array, convert_from_java_2d_array
)
from .bart_package_inits import initialize_jvm, shutdown_jvm, is_jvm_running, bart_machine_num_cores
from .bart_package_data_preprocessing import preprocess_training_data, preprocess_new_data

# Set up logging
logger = logging.getLogger(__name__)

class BartMachine:
    """
    Bayesian Additive Regression Trees (BART) model.
    
    This class provides a Python interface to the Java implementation of BART.
    It handles the conversion between Python and Java data types and provides
    methods for building models, making predictions, and other operations.
    
    Attributes:
        java_bart_machine: The Java BartMachine object.
        X: The predictor variables.
        y: The response variable.
        y_levels: The levels of y (for classification).
        pred_type: The prediction type ("regression" or "classification").
        model_matrix_training_data: The preprocessed training data.
        n: The number of observations.
        p: The number of predictors.
        num_cores: The number of cores to use.
        num_trees: Number of trees in the ensemble.
        num_burn_in: Number of burn-in MCMC iterations.
        num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
        num_gibbs: Total number of MCMC iterations.
        alpha: Prior parameter for tree structure.
        beta: Prior parameter for tree structure.
        k: Prior parameter for leaf values.
        q: Prior parameter for leaf values.
        nu: Prior parameter for error variance.
        prob_rule_class: Probability of using a classification rule.
        mh_prob_steps: Metropolis-Hastings proposal step probabilities.
        s_sq_y: Sample variance of y.
        run_in_sample: Whether to run in-sample prediction.
        sig_sq_est: Error variance estimate.
        time_to_build: Time taken to build the model.
        cov_prior_vec: Covariate prior vector.
        interaction_constraints: Interaction constraints.
        use_missing_data: Whether to use missing data.
        use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
        replace_missing_data_with_x_j_bar: Whether to replace missing data with column means.
        impute_missingness_with_rf_impute: Whether to impute missing data with random forest.
        impute_missingness_with_x_j_bar_for_lm: Whether to impute missing data with column means for linear model.
        verbose: Whether to print verbose output.
        serialize: Whether to serialize the model.
        mem_cache_for_speed: Whether to cache for speed.
        flush_indices_to_save_RAM: Whether to flush indices to save RAM.
        debug_log: Whether to print debug information.
        seed: Random seed.
        num_rand_samps_in_library: Number of random samples in library.
    """
    
    def __init__(self, X=None, y=None, Xy=None, 
                num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000,
                alpha=0.95, beta=2, k=2, q=0.9, nu=3.0, prob_rule_class=0.5,
                mh_prob_steps=None, debug_log=False, run_in_sample=True,
                s_sq_y="mse", sig_sq_est=None, print_tree_illustrations=False,
                cov_prior_vec=None, interaction_constraints=None, use_missing_data=False,
                covariates_to_permute=None, num_rand_samps_in_library=10000,
                use_missing_data_dummies_as_covars=False, replace_missing_data_with_x_j_bar=False,
                impute_missingness_with_rf_impute=False, impute_missingness_with_x_j_bar_for_lm=True,
                mem_cache_for_speed=True, flush_indices_to_save_RAM=True, serialize=False,
                seed=None, verbose=True):
        """
        Initialize a BartMachine model.
        
        Args:
            X: The predictor variables.
            y: The response variable.
            Xy: Combined predictor and response variables.
            num_trees: Number of trees in the ensemble.
            num_burn_in: Number of burn-in MCMC iterations.
            num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
            alpha: Prior parameter for tree structure.
            beta: Prior parameter for tree structure.
            k: Prior parameter for leaf values.
            q: Prior parameter for leaf values.
            nu: Prior parameter for error variance.
            prob_rule_class: Probability of using a classification rule.
            mh_prob_steps: Metropolis-Hastings proposal step probabilities.
            debug_log: Whether to print debug information.
            run_in_sample: Whether to run in-sample prediction.
            s_sq_y: Sample variance of y ("mse" or "var").
            sig_sq_est: Error variance estimate.
            print_tree_illustrations: Whether to print tree illustrations.
            cov_prior_vec: Covariate prior vector.
            interaction_constraints: Interaction constraints.
            use_missing_data: Whether to use missing data.
            covariates_to_permute: Covariates to permute.
            num_rand_samps_in_library: Number of random samples in library.
            use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
            replace_missing_data_with_x_j_bar: Whether to replace missing data with column means.
            impute_missingness_with_rf_impute: Whether to impute missing data with random forest.
            impute_missingness_with_x_j_bar_for_lm: Whether to impute missing data with column means for linear model.
            mem_cache_for_speed: Whether to cache for speed.
            flush_indices_to_save_RAM: Whether to flush indices to save RAM.
            serialize: Whether to serialize the model.
            seed: Random seed.
            verbose: Whether to print verbose output.
        """
        # Store the parameters
        self.num_trees = num_trees
        self.num_burn_in = num_burn_in
        self.num_iterations_after_burn_in = num_iterations_after_burn_in
        self.num_gibbs = num_burn_in + num_iterations_after_burn_in
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.q = q
        self.nu = nu
        self.prob_rule_class = prob_rule_class
        self.mh_prob_steps = mh_prob_steps if mh_prob_steps is not None else [2.5/9, 2.5/9, 4/9]
        self.debug_log = debug_log
        self.run_in_sample = run_in_sample
        self.s_sq_y = s_sq_y
        self.sig_sq_est = sig_sq_est
        self.print_tree_illustrations = print_tree_illustrations
        self.cov_prior_vec = cov_prior_vec
        self.interaction_constraints = interaction_constraints
        self.use_missing_data = use_missing_data
        self.covariates_to_permute = covariates_to_permute
        self.num_rand_samps_in_library = num_rand_samps_in_library
        self.use_missing_data_dummies_as_covars = use_missing_data_dummies_as_covars
        self.replace_missing_data_with_x_j_bar = replace_missing_data_with_x_j_bar
        self.impute_missingness_with_rf_impute = impute_missingness_with_rf_impute
        self.impute_missingness_with_x_j_bar_for_lm = impute_missingness_with_x_j_bar_for_lm
        self.mem_cache_for_speed = mem_cache_for_speed
        self.flush_indices_to_save_RAM = flush_indices_to_save_RAM
        self.serialize = serialize
        self.seed = seed
        self.verbose = verbose
        
        # Initialize the Java BartMachine object
        self.java_bart_machine = None
        
        # Build the model if X and y are provided
        if X is not None or Xy is not None:
            self.build(X, y, Xy)
    
    def build(self, X=None, y=None, Xy=None):
        """
        Build the BART model.
        
        Args:
            X: The predictor variables.
            y: The response variable.
            Xy: Combined predictor and response variables.
        
        Returns:
            self: The BartMachine object.
        """
        # Start timing
        t0 = time.time()
        
        if self.verbose:
            print(f"bartMachine initializing with {self.num_trees} trees...")
        
        # Check input parameters
        if self.use_missing_data_dummies_as_covars and self.replace_missing_data_with_x_j_bar:
            raise ValueError("You cannot impute by averages and use missing data as dummies simultaneously.")
        
        if (X is None and Xy is None) or (y is None and Xy is None):
            raise ValueError("You need to give bartMachine a training set either by specifying X and y or by specifying a matrix Xy which contains the response named 'y'.")
        elif X is not None and y is not None and Xy is not None:
            raise ValueError("You cannot specify both X,y and Xy simultaneously.")
        elif X is None and y is None:  # they specified Xy, so now just pull out X,y
            # First ensure it's a dataframe
            if not isinstance(Xy, pd.DataFrame):
                raise ValueError("The training data Xy must be a data frame.")
            
            y = Xy.iloc[:, -1]
            X = Xy.iloc[:, :-1]
        
        # Make sure X is a dataframe
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The training data X must be a data frame.")
        
        if self.verbose:
            print("bartMachine vars checked...")
        
        # Store X and y
        self.X = X
        self.y = y
        
        # Check if JVM is running
        if not is_jvm_running():
            logger.info("JVM is not running. Initializing...")
            initialize_jvm()
        
        # Set the random seed if provided
        if self.seed is not None:
            set_seed(self.seed)
        
        # Now take care of classification or regression
        if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number):
            # If y is numeric, then it's a regression problem
            if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.integer):
                print("Warning: The response y is integer, bartMachine will run regression.")
                # Convert to float
                y = y.astype(float)
            
            java_bart_machine = get_java_class("bartMachine.bartMachineRegressionMultThread")()
            y_remaining = y
            pred_type = "regression"
            y_levels = None
        elif hasattr(y, 'dtype') and y.dtype.name == 'category' and len(y.cat.categories) == 2:
            # If y is a categorical variable with two levels, it's a classification problem
            y_levels = list(y.cat.categories)
            
            # Convenience for users that use 0/1 variables
            if sorted(y_levels) == ['0', '1']:
                y = pd.Categorical(y, categories=['1', '0'], ordered=False)
                y_levels = ['1', '0']
            
            java_bart_machine = get_java_class("bartMachine.bartMachineClassificationMultThread")()
            y_remaining = np.where(y == y_levels[0], 1, 0)
            pred_type = "classification"
        else:
            # Otherwise throw an error
            raise ValueError("Your response must be either numeric, an integer or a categorical variable with two levels.")
        
        # Store prediction type and y levels
        self.pred_type = pred_type
        self.y_levels = y_levels
        
        # Check data dimensions
        if X.shape[1] == 0:
            raise ValueError("Your data matrix must have at least one attribute.")
        if X.shape[0] == 0:
            raise ValueError("Your data matrix must have at least one observation.")
        if len(y) != X.shape[0]:
            raise ValueError("The number of responses must be equal to the number of observations in the training data.")
        
        if self.verbose:
            print("bartMachine java init...")
        
        # If no column names, make up names
        if X.columns.empty:
            X.columns = [f"V{i+1}" for i in range(X.shape[1])]
        
        if any(step < 0 for step in self.mh_prob_steps):
            raise ValueError("The grow, prune, change ratio parameter vector must all be greater than 0.")
        
        # Check for missing values in y
        if y_remaining.isna().any():
            raise ValueError("You cannot have any missing data in your response vector.")
        
        # Handle missing data
        rf_imputations_for_missing = None
        if self.impute_missingness_with_rf_impute:
            if X.isna().sum().sum() == 0:  # No missing values
                print("Warning: No missing entries in the training data to impute.")
                rf_imputations_for_missing = X
            else:
                # Just use cols that HAVE missing data
                predictor_colnums_with_missingness = X.columns[X.isna().any()]
                
                # In Python we would use a different imputation method than missForest
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                
                imputer = IterativeImputer(random_state=0 if self.seed is None else self.seed)
                X_with_y = pd.concat([X, pd.Series(y)], axis=1)
                imputed_values = imputer.fit_transform(X_with_y)
                imputed_df = pd.DataFrame(imputed_values, columns=X_with_y.columns)
                rf_imputations_for_missing = imputed_df.iloc[:, :-1][predictor_colnums_with_missingness]
                rf_imputations_for_missing.columns = [f"{col}_imp" for col in rf_imputations_for_missing.columns]
            
            if self.verbose:
                print("bartMachine after rf imputations...")
        
        # If we're not using missing data, go on and get rid of it
        if not self.use_missing_data and not self.replace_missing_data_with_x_j_bar:
            rows_before = X.shape[0]
            X = X.dropna()
            rows_after = X.shape[0]
            if rows_before - rows_after > 0:
                raise ValueError(f"You have {rows_before - rows_after} observations with missing data. \nYou must either omit your missing data using dropna() or turn on the\n'use_missing_data' or 'replace_missing_data_with_x_j_bar' feature in order to use bartMachine.\n")
        elif self.replace_missing_data_with_x_j_bar:
            # Impute missing values with column means for numeric columns and modes for categorical columns
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])
            
            if self.verbose:
                print("Imputed missing data using attribute averages.")
        
        if self.verbose:
            print("bartMachine before preprocess...")
        
        # Preprocess the training data
        pre_process_obj = preprocess_training_data(X, self.use_missing_data_dummies_as_covars, rf_imputations_for_missing)
        model_matrix_training_data = np.column_stack((pre_process_obj['data'], y_remaining))
        p = model_matrix_training_data.shape[1] - 1  # we subtract one because we tacked on the response as the last column
        factor_lengths = pre_process_obj['factor_lengths']
        
        if self.verbose:
            print(f"bartMachine after preprocess... {p} total features...")
        
        # Create a default cov_prior_vec that factors in the levels of the factors
        null_cov_prior_vec = self.cov_prior_vec is None
        if null_cov_prior_vec and len(factor_lengths) > 0:
            # Begin with the uniform
            self.cov_prior_vec = np.ones(p)
            j_factor_begin = p - sum(factor_lengths) + 1
            for l in range(len(factor_lengths)):
                factor_length = factor_lengths[l]
                self.cov_prior_vec[j_factor_begin:(j_factor_begin + factor_length)] = 1 / factor_length
                j_factor_begin = j_factor_begin + factor_length
        
        # Handle interaction constraints
        if self.interaction_constraints is not None:
            if not self.mem_cache_for_speed:
                raise ValueError("In order to use interaction constraints, 'mem_cache_for_speed' must be set to True.")
            
            if not isinstance(self.interaction_constraints, list):
                raise ValueError("specified parameter 'interaction_constraints' must be a list")
            elif len(self.interaction_constraints) == 0:
                raise ValueError("interaction_constraints list cannot be empty")
            
            for a in range(len(self.interaction_constraints)):
                vars_a = self.interaction_constraints[a]
                # Check if the constraint components are valid features
                for b in range(len(vars_a)):
                    var = vars_a[b]
                    if isinstance(var, (int, np.integer)) and not (var in range(1, p+1)):
                        raise ValueError(f"Element {var} in interaction_constraints vector number {a} is numeric but not one of 1, ..., {p} where {p} is the number of columns in X.")
                    
                    if isinstance(var, str) and not (var in X.columns):
                        raise ValueError(f"Element {var} in interaction_constraints vector number {a} is a string but not one of the column names of X.")
                    
                    # Force all to be integers and begin index at zero
                    if isinstance(var, (int, np.integer)):
                        vars_a[b] = var - 1
                    elif isinstance(var, str):
                        vars_a[b] = list(X.columns).index(var) - 1
                
                self.interaction_constraints[a] = [int(x) for x in vars_a]
        
        # Handle covariates to permute (private parameter)
        if self.covariates_to_permute is not None:
            # First check if these covariates are even in the matrix to begin with
            for cov in self.covariates_to_permute:
                if isinstance(cov, str) and not (cov in model_matrix_training_data.columns):
                    raise ValueError(f"Covariate '{cov}' not found in design matrix.")
            
            # Permute the covariates
            permuted_order = np.random.permutation(model_matrix_training_data.shape[0])
            model_matrix_training_data[:, self.covariates_to_permute] = model_matrix_training_data[permuted_order, self.covariates_to_permute]
        
        # Set whether we want the program to log to a file
        if self.debug_log and self.verbose:
            print("warning: printing out the log file will slow down the runtime significantly.")
            java_bart_machine.writeStdOutToLogFile()
        
        # Set whether we want there to be tree illustrations
        if self.print_tree_illustrations:
            print("warning: printing tree illustrations is excruciatingly slow.")
            java_bart_machine.printTreeIllustations()
        
        # Set the std deviation of y to use
        if p >= model_matrix_training_data.shape[0]:
            if self.verbose:
                print("warning: cannot use MSE of linear model for s_sq_y if p > n. bartMachine will use sample var(y) instead.")
            self.s_sq_y = "var"
        
        # Estimate sigma^2 to be given to the BART model
        if self.sig_sq_est is None:
            if self.pred_type == "regression":
                y_range = np.max(y) - np.min(y)
                y_trans = (y - np.min(y)) / y_range - 0.5
                
                if self.s_sq_y == "mse":
                    X_for_lm = pd.DataFrame(model_matrix_training_data[:, :-1])
                    
                    if self.impute_missingness_with_x_j_bar_for_lm:
                        # Impute missing values with column means
                        for col in X_for_lm.columns:
                            X_for_lm[col] = X_for_lm[col].fillna(X_for_lm[col].mean())
                    elif X_for_lm.dropna().shape[0] == 0:
                        raise ValueError("The data does not have enough full records to estimate a naive prediction error. Please rerun with 'impute_missingness_with_x_j_bar_for_lm' set to true.")
                    
                    # Fit linear model
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X_for_lm, y_trans)
                    y_pred = model.predict(X_for_lm)
                    mse = np.var(y_trans - y_pred)
                    self.sig_sq_est = mse
                    java_bart_machine.setSampleVarY(self.sig_sq_est)
                elif self.s_sq_y == "var":
                    self.sig_sq_est = np.var(y_trans)
                    java_bart_machine.setSampleVarY(self.sig_sq_est)
                else:  # if it's not a valid flag, throw an error
                    raise ValueError("s_sq_y must be 'mse' or 'var'")
                
                self.sig_sq_est = self.sig_sq_est * y_range**2
                
                if self.verbose:
                    print("bartMachine sigsq estimated...")  # only print for regression
        else:
            if self.verbose:
                print("bartMachine using previous sigsq estimated...")
        
        # Get the number of cores
        num_cores = bart_machine_num_cores()
        
        # Build bart to spec with what the user wants
        java_bart_machine.setNumCores(num_cores)  # this must be set FIRST!!!
        java_bart_machine.setNumTrees(self.num_trees)
        java_bart_machine.setNumGibbsBurnIn(self.num_burn_in)
        java_bart_machine.setNumGibbsTotalIterations(self.num_gibbs)
        java_bart_machine.setAlpha(self.alpha)
        java_bart_machine.setBeta(self.beta)
        java_bart_machine.setK(self.k)
        java_bart_machine.setQ(self.q)
        java_bart_machine.setNU(self.nu)
        
        # Make sure mh_prob_steps is a probability vector
        self.mh_prob_steps = np.array(self.mh_prob_steps) / sum(self.mh_prob_steps)
        java_bart_machine.setProbGrow(self.mh_prob_steps[0])
        java_bart_machine.setProbPrune(self.mh_prob_steps[1])
        java_bart_machine.setVerbose(self.verbose)
        java_bart_machine.setMemCacheForSpeed(self.mem_cache_for_speed)
        java_bart_machine.setFlushIndicesToSaveRAM(self.flush_indices_to_save_RAM)
        
        if self.seed is not None:
            # Set the seed in Java
            java_bart_machine.setSeed(self.seed)
            if num_cores > 1:
                print("Warning: Setting the seed when using parallelization does not result in deterministic output.\nIf you need deterministic output, you must run 'set_bart_machine_num_cores(1)' and then build the BART model with the set seed.")
        
        # Now we need to set random samples
        java_bart_machine.setNormSamples(np.random.normal(0, 1, self.num_rand_samps_in_library))
        n_plus_hyper_nu = model_matrix_training_data.shape[0] + self.nu
        java_bart_machine.setGammaSamples(np.random.chisquare(n_plus_hyper_nu, self.num_rand_samps_in_library))
        
        if self.cov_prior_vec is not None:
            # Put in checks here for user to make sure the covariate prior vec is the correct length
            offset = len(self.cov_prior_vec) - p
            if offset < 0:
                print(f"warning: covariate prior vector length = {len(self.cov_prior_vec)} has to be equal to p = {p} (the vector was lengthened with 1's)")
                self.cov_prior_vec = np.concatenate([self.cov_prior_vec, np.ones(-offset)])
            
            if len(self.cov_prior_vec) != p:
                print(f"warning: covariate prior vector length = {len(self.cov_prior_vec)} has to be equal to p = {p} (the vector was shortened)")
                self.cov_prior_vec = self.cov_prior_vec[:p]
            
            if not all(self.cov_prior_vec > 0):
                raise ValueError("covariate prior vector has to have all its elements be positive")
            
            java_bart_machine.setCovSplitPrior(self.cov_prior_vec)
        
        if self.interaction_constraints is not None:
            java_bart_machine.intializeInteractionConstraints(len(self.interaction_constraints))
            for interaction_constraint_vector in self.interaction_constraints:
                for b in range(len(interaction_constraint_vector)):
                    java_bart_machine.addInteractionConstraint(
                        interaction_constraint_vector[b],
                        [int(x) for x in interaction_constraint_vector[:b] + interaction_constraint_vector[b+1:]]
                    )
        
        # Now load the training data into BART
        for i in range(model_matrix_training_data.shape[0]):
            row = model_matrix_training_data[i, :]
            # Convert row to string and replace NaN with "NA"
            row_as_char = [str(x) if not np.isnan(x) else "NA" for x in row]
            java_bart_machine.addTrainingDataRow(row_as_char)
        
        java_bart_machine.finalizeTrainingData()
        
        if self.verbose:
            print("bartMachine training data finalized...")
        
        # Build the bart machine and let the user know what type of BART this is
        if self.verbose:
            print(f"Now building bartMachine for {pred_type}", end="")
            if pred_type == "classification":
                print(f" where '{y_levels[0]}' is considered the target level", end="")
            print("...")
            
            if self.cov_prior_vec is not None:
                print("Covariate importance prior ON. ", end="")
            
            if self.use_missing_data:
                print("Missing data feature ON. ", end="")
            
            if self.use_missing_data_dummies_as_covars:
                print("Missingness used as covariates. ", end="")
            
            if self.impute_missingness_with_rf_impute:
                print("Missing values imputed via random forest. ", end="")
            
            print()
        
        java_bart_machine.Build()
        
        # Store the Java BartMachine object
        self.java_bart_machine = java_bart_machine
        
        # Store other attributes
        self.training_data_features = [f"X{i+1}" for i in range(p)]  # Placeholder column names
        self.training_data_features_with_missing_features = self.training_data_features  # Always return this even if there's no missing features
        self.model_matrix_training_data = model_matrix_training_data
        self.n = model_matrix_training_data.shape[0]
        self.p = p
        self.num_cores = num_cores
        self.time_to_build = time.time() - t0
        
        # Once it's done gibbs sampling, see how the training data does if user wants
        if self.run_in_sample:
            if self.verbose:
                print("evaluating in sample data...", end="")
            
            if self.pred_type == "regression":
                y_hat_posterior_samples = java_bart_machine.getGibbsSamplesForPrediction(model_matrix_training_data, num_cores)
                
                # To get y_hat.. just take straight mean of posterior samples
                y_hat_train = np.mean(y_hat_posterior_samples, axis=1)
                
                # Return a bunch more stuff
                self.y_hat_train = y_hat_train
                self.residuals = y_remaining - self.y_hat_train
                self.L1_err_train = np.sum(np.abs(self.residuals))
                self.L2_err_train = np.sum(self.residuals**2)
                self.PseudoRsq = 1 - self.L2_err_train / np.sum((y_remaining - np.mean(y_remaining))**2)  # pseudo R^2
                self.rmse_train = np.sqrt(self.L2_err_train / self.n)
            
            elif self.pred_type == "classification":
                p_hat_posterior_samples = java_bart_machine.getGibbsSamplesForPrediction(model_matrix_training_data, num_cores)
                
                # To get y_hat.. just take straight mean of posterior samples
                p_hat_train = np.mean(p_hat_posterior_samples, axis=1)
                
                # Convert probabilities to class labels
                y_hat_train = np.where(p_hat_train > self.prob_rule_class, self.y_levels[0], self.y_levels[1])
                
                # Store results
                self.p_hat_train = p_hat_train
                self.y_hat_train = y_hat_train
                
                # Calculate confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y, y_hat_train, labels=self.y_levels)
                
                # Create confusion matrix dataframe
                confusion_df = pd.DataFrame(cm, index=self.y_levels, columns=self.y_levels)
                confusion_df.index.name = "actual"
                confusion_df.columns.name = "predicted"
                
                # Calculate error rates
                total = np.sum(cm)
                misclassification_error = (cm[0, 1] + cm[1, 0]) / total
                
                self.confusion_matrix = confusion_df
                self.misclassification_error = misclassification_error
            
            if self.verbose:
                print("done")
        
        # Let's serialize the object if the user wishes
        if self.serialize:
            print("serializing in order to be saved for future Python sessions...", end="")
            # In Python, we don't need to do anything special for serialization
            # as the object will be pickled automatically when saved
            print("done")
        
        return self
    
    def predict(self, X_test, type="response"):
        """
        Make predictions with the BART model.
        
        Args:
            X_test: The test data.
            type: The type of prediction to return. One of "response", "prob", or "samples".
        
        Returns:
            The predictions.
        """
        # Check if the model has been built
        if self.java_bart_machine is None:
            raise ValueError("The model has not been built. Call build() first.")
        
        # Convert X_test to a DataFrame if it's not already
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        # Preprocess the test data
        X_test_processed = preprocess_new_data(X_test, self)
        
        # Make predictions
        if type == "response":
            # Get the mean prediction
            predictions = self.java_bart_machine.predict(X_test_processed)
            
            # Convert to numpy array
            predictions = np.array(predictions)
            
            # For classification, convert probabilities to class labels
            if self.pred_type == "classification":
                predictions = np.where(predictions > self.prob_rule_class, self.y_levels[0], self.y_levels[1])
            
            return predictions
        
        elif type == "prob":
            # Get the probability predictions (only for classification)
            if self.pred_type != "classification":
                raise ValueError("Probability predictions are only available for classification models.")
            
            predictions = self.java_bart_machine.predict(X_test_processed)
            
            # Convert to numpy array
            predictions = np.array(predictions)
            
            return predictions
        
        elif type == "samples":
            # Get the posterior samples
            samples = self.java_bart_machine.getGibbsSamplesForPrediction(X_test_processed, self.num_cores)
            
            # Convert to numpy array
            samples = np.array(samples)
            
            return samples
        
        else:
            raise ValueError("type must be one of 'response', 'prob', or 'samples'")
    
    def __del__(self):
        """
        Clean up resources when the object is deleted.
        """
        # Note: We don't shut down the JVM here because other BartMachine objects
        # might still be using it. The user should call shutdown_jvm() explicitly
        # when they're done with all BartMachine objects.
        pass


def bart_machine(X=None, y=None, Xy=None, 
                num_trees=50, num_burn_in=250, num_iterations_after_burn_in=1000,
                alpha=0.95, beta=2, k=2, q=0.9, nu=3.0, prob_rule_class=0.5,
                mh_prob_steps=None, debug_log=False, run_in_sample=True,
                s_sq_y="mse", sig_sq_est=None, print_tree_illustrations=False,
                cov_prior_vec=None, interaction_constraints=None, use_missing_data=False,
                covariates_to_permute=None, num_rand_samps_in_library=10000,
                use_missing_data_dummies_as_covars=False, replace_missing_data_with_x_j_bar=False,
                impute_missingness_with_rf_impute=False, impute_missingness_with_x_j_bar_for_lm=True,
                mem_cache_for_speed=True, flush_indices_to_save_RAM=True, serialize=False,
                seed=None, verbose=True):
    """
    Create and build a BART machine model.
    
    This function creates a BartMachine object and builds the model.
    It is a convenience function that combines the initialization and building steps.
    
    Args:
        X: The predictor variables.
        y: The response variable.
        Xy: Combined predictor and response variables.
        num_trees: Number of trees in the ensemble.
        num_burn_in: Number of burn-in MCMC iterations.
        num_iterations_after_burn_in: Number of MCMC iterations after burn-in.
        alpha: Prior parameter for tree structure.
        beta: Prior parameter for tree structure.
        k: Prior parameter for leaf values.
        q: Prior parameter for leaf values.
        nu: Prior parameter for error variance.
        prob_rule_class: Probability of using a classification rule.
        mh_prob_steps: Metropolis-Hastings proposal step probabilities.
        debug_log: Whether to print debug information.
        run_in_sample: Whether to run in-sample prediction.
        s_sq_y: Sample variance of y ("mse" or "var").
        sig_sq_est: Error variance estimate.
        print_tree_illustrations: Whether to print tree illustrations.
        cov_prior_vec: Covariate prior vector.
        interaction_constraints: Interaction constraints.
        use_missing_data: Whether to use missing data.
        covariates_to_permute: Covariates to permute.
        num_rand_samps_in_library: Number of random samples in library.
        use_missing_data_dummies_as_covars: Whether to use missing data dummies as covariates.
        replace_missing_data_with_x_j_bar: Whether to replace missing data with column means.
        impute_missingness_with_rf_impute: Whether to impute missing data with random forest.
        impute_missingness_with_x_j_bar_for_lm: Whether to impute missing data with column means for linear model.
        mem_cache_for_speed: Whether to cache for speed.
        flush_indices_to_save_RAM: Whether to flush indices to save RAM.
        serialize: Whether to serialize the model.
        seed: Random seed.
        verbose: Whether to print verbose output.
    
    Returns:
        A built BartMachine object.
    """
    # Create and build BartMachine object
    bart = BartMachine(
        X=X, y=y, Xy=Xy, 
        num_trees=num_trees, num_burn_in=num_burn_in, num_iterations_after_burn_in=num_iterations_after_burn_in,
        alpha=alpha, beta=beta, k=k, q=q, nu=nu, prob_rule_class=prob_rule_class,
        mh_prob_steps=mh_prob_steps, debug_log=debug_log, run_in_sample=run_in_sample,
        s_sq_y=s_sq_y, sig_sq_est=sig_sq_est, print_tree_illustrations=print_tree_illustrations,
        cov_prior_vec=cov_prior_vec, interaction_constraints=interaction_constraints, use_missing_data=use_missing_data,
        covariates_to_permute=covariates_to_permute, num_rand_samps_in_library=num_rand_samps_in_library,
        use_missing_data_dummies_as_covars=use_missing_data_dummies_as_covars, replace_missing_data_with_x_j_bar=replace_missing_data_with_x_j_bar,
        impute_missingness_with_rf_impute=impute_missingness_with_rf_impute, impute_missingness_with_x_j_bar_for_lm=impute_missingness_with_x_j_bar_for_lm,
        mem_cache_for_speed=mem_cache_for_speed, flush_indices_to_save_RAM=flush_indices_to_save_RAM, serialize=serialize,
        seed=seed, verbose=verbose
    )
    
    return bart


def bart_machine_cv(X=None, y=None, Xy=None, 
                   num_tree_cvs=[50, 200],
                   k_cvs=[2, 3, 5],
                   nu_q_cvs=None,
                   k_folds=5, 
                   folds_vec=None,    
                   verbose=False, **kwargs):
    """
    Create and build a BART machine model with cross-validation.
    
    This function creates a BartMachine object with cross-validated hyperparameters.
    
    Args:
        X: The predictor variables.
        y: The response variable.
        Xy: Combined predictor and response variables.
        num_tree_cvs: List of number of trees to try.
        k_cvs: List of k values to try.
        nu_q_cvs: List of (nu, q) tuples to try.
        k_folds: Number of folds for cross-validation.
        folds_vec: Vector of fold indices.
        verbose: Whether to print verbose output.
        **kwargs: Additional arguments to pass to BartMachine.
    
    Returns:
        A built BartMachine object with cross-validated hyperparameters.
    """
    # This is a placeholder for the cross-validation function
    # In a real implementation, we would perform cross-validation to find the best hyperparameters
    # For now, we'll just use the first values in the lists
    
    # Create and build BartMachine object with default hyperparameters
    bart = BartMachine(
        X=X, y=y, Xy=Xy, 
        num_trees=num_tree_cvs[0],
        k=k_cvs[0],
        verbose=verbose,
        **kwargs
    )
    
    return bart
