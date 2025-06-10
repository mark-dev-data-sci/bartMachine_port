# R script to run bartMachine on Boston housing dataset with longer MCMC chains

# Load required packages
library(bartMachine)

# Set seed for reproducibility
set.seed(123)

# Load Boston housing dataset
boston_data <- read.csv("/Users/mark/Documents/Cline/bartMachine/datasets/r_boston.csv")

# Load the same train/test split used in Python
train_idx <- read.csv("py_train_idx.csv")$train_idx
test_idx <- read.csv("py_test_idx.csv")$test_idx

# Split into features and target
X <- boston_data[, -which(names(boston_data) == "y")]
y <- boston_data$y

# Use the same train/test split as Python
X_train <- X[train_idx + 1, ]  # R is 1-indexed, Python is 0-indexed
y_train <- y[train_idx + 1]
X_test <- X[test_idx + 1, ]
y_test <- y[test_idx + 1]

cat("Training data shape:", dim(X_train)[1], "x", dim(X_train)[2], "\n")
cat("Testing data shape:", dim(X_test)[1], "x", dim(X_test)[2], "\n")

# Set model parameters - run MCMC for longer
num_trees <- 50
num_burn_in <- 500  # 5x more burn-in iterations
num_iterations_after_burn_in <- 1000  # 5x more iterations after burn-in
alpha <- 0.95
beta <- 2.0  # Use floating-point values to match Python
k <- 2.0
q <- 0.9
nu <- 3.0

# Build the model
cat("\nRunning R implementation with longer MCMC chains...\n")
start_time <- Sys.time()
r_bart <- bartMachine(
  X = X_train,
  y = y_train,
  num_trees = num_trees,
  num_burn_in = num_burn_in,
  num_iterations_after_burn_in = num_iterations_after_burn_in,
  alpha = alpha,
  beta = beta,
  k = k,
  q = q,
  nu = nu,
  verbose = FALSE
)
end_time <- Sys.time()
build_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat("R build time:", round(build_time, 2), "seconds\n")

# Make predictions
pred_start_time <- Sys.time()
r_pred <- predict(r_bart, X_test)
pred_end_time <- Sys.time()
pred_time <- as.numeric(difftime(pred_end_time, pred_start_time, units = "secs"))
cat("R prediction time:", round(pred_time, 2), "seconds\n")

# Get variable importance using a simpler method
var_imp_start_time <- Sys.time()
# Get the proportion of times each variable is used in the trees
r_var_props <- get_var_props_over_chain(r_bart)
var_imp_end_time <- Sys.time()
var_imp_time <- as.numeric(difftime(var_imp_end_time, var_imp_start_time, units = "secs"))
cat("R variable importance time:", round(var_imp_time, 2), "seconds\n")

# Save results to CSV files
write.csv(data.frame(
  index = test_idx,
  prediction = r_pred
), "r_bart_predictions_long.csv", row.names = FALSE)

# Convert variable proportions to a data frame
var_names <- names(r_var_props)
importance <- as.numeric(r_var_props)

write.csv(data.frame(
  variable = var_names,
  importance = importance
), "r_bart_var_importance_long.csv", row.names = FALSE)

# Plot predictions vs actual
png("r_actual_vs_predicted_long.png", width = 800, height = 600)
plot(y_test, r_pred, main = "R BART (Long MCMC): Actual vs Predicted", 
     xlab = "Actual Values", ylab = "Predicted Values", pch = 16, col = "blue")
abline(0, 1, col = "red", lty = 2)
dev.off()

# Plot variable importance
png("r_var_importance_long.png", width = 1000, height = 800)
barplot(importance, names.arg = var_names, main = "R BART (Long MCMC): Variable Importance", 
        xlab = "Variables", ylab = "Importance (Proportion)", col = "blue")
dev.off()

cat("\nR implementation with longer MCMC chains completed. Results saved to:\n")
cat("- r_bart_predictions_long.csv\n")
cat("- r_bart_var_importance_long.csv\n")
cat("- r_actual_vs_predicted_long.png\n")
cat("- r_var_importance_long.png\n")

# No cleanup needed - R will handle garbage collection
