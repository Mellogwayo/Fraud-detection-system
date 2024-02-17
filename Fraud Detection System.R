# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)

# Set the working directory if needed
# setwd("path_to_directory")

# Load the CSV file
df <- read_csv("/Users/mellogwayo/Desktop/Fraud-detection-system/creditcard.csv")

# Display the first few rows of the DataFrame
head(df)

# Display basic information about the dataset
str(df)

# Display descriptive statistics for numerical variables
summary(df)

# Count missing values in each column
missing_values <- colSums(is.na(df))

# Display the count of missing values for each column
missing_values

# Define a function to find outliers using IQR
find_outliers_iqr <- function(data) {
  outliers <- data.frame(matrix(NA, nrow = nrow(data), ncol = ncol(data)))
  
  for (column in colnames(data)) {
    if (is.numeric(data[[column]])) {
      Q1 <- quantile(data[[column]], 0.25)
      Q3 <- quantile(data[[column]], 0.75)
      IQR <- Q3 - Q1
      
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      outliers[[column]] <- (data[[column]] < lower_bound) | (data[[column]] > upper_bound)
    }
  }
  
  return(outliers)
}

# Find outliers
outliers <- find_outliers_iqr(df)

# Count the number of outliers for each variable
outliers_count <- colSums(outliers)
outliers_count

# Variables to include in each box plot
variables <- list(c('V1', 'V2', 'V3', 'V4', 'V5'),
                  c('V6', 'V7', 'V8', 'V9', 'V10'),
                  c('V11', 'V12', 'V13', 'V14', 'V15'),
                  c('V16', 'V17', 'V18', 'V19', 'V20'),
                  c('V21', 'V22', 'V23', 'V24', 'V25'),
                  c('V26', 'V27', 'V28', 'Amount', 'Class'))

# Create box plots
par(mfrow=c(2,3))
for (i in 1:length(variables)) {
  boxplot(df[,variables[[i]]], main=paste("Boxplot", i), xlab="Variables", ylab="Values")
}

# Check the distribution of features
par(mfrow=c(6,5))
for (i in 1:ncol(df)) {
  hist(df[,i], main=paste("Histogram of", names(df)[i]), xlab="Values", ylab="Frequency", col="lightblue")
}

# Check the distribution of features
par(mfrow=c(6, 5))
for (i in 1:(ncol(df) - 1)) {  # Exclude the last column "Class"
  if (is.numeric(df[, i])) {
    hist(df[, i], main = paste("Histogram of", names(df)[i]), xlab = "Values", ylab = "Frequency", col = "lightblue")
  } else {
    cat("Column", names(df)[i], "is not numeric. Skipping histogram.\n")
  }
}

# Bar plot for the categorical variable "Class"
barplot(table(df$Class), main = "Bar Plot of Class", xlab = "Class", ylab = "Frequency", col = "lightblue")

# Check the distribution of features
par(mfrow=c(6, 5))
for (i in 1:(ncol(df) - 1)) {  # Exclude the last column "Class"
  if (is.numeric(df[, i])) {
    hist(df[, i], main = paste("Histogram of", names(df)[i]), xlab = "Values", ylab = "Frequency", col = "lightblue")
  } else {
    cat("Column", names(df)[i], "is not numeric. Skipping histogram.\n")
  }
}

# Bar plot for the categorical variable "Class"
barplot(table(df$Class), main = "Bar Plot of Class", xlab = "Class", ylab = "Frequency", col = "lightblue")

# Plot correlation matrix
correlation_matrix <- cor(df)
heatmap(correlation_matrix, Colv=NA, Rowv=NA, col=bluered(20), symm=TRUE)

# Assuming 'Time' is the non-numerical variable and 'Class' is the target variable
non_numerical_variable <- 'Time'
class_variable <- 'Class'
numerical_variables <- colnames(df)[!colnames(df) %in% c(non_numerical_variable, class_variable)]

# Scatter Plots for each pair of numerical variables against 'Time' and 'Class'
for (feature in numerical_variables) {
  # Scatter Plot against 'Time'
  plot(df[,non_numerical_variable], df[,feature], col=df[,class_variable], pch=19, main=paste("Scatter Plot of", feature, "vs", non_numerical_variable, "with", class_variable, "Coloring"), xlab=non_numerical_variable, ylab=feature)
  legend("topleft", legend=levels(df[,class_variable]), col=1:length(levels(df[,class_variable])), pch=19, title=class_variable)
  
  # Scatter Plot against 'Class'
  plot(df[,feature], df[,class_variable], col=df[,non_numerical_variable], pch=19, main=paste("Scatter Plot of", feature, "vs", class_variable, "with", non_numerical_variable, "Coloring"), xlab=feature, ylab=class_variable)
  legend("topleft", legend=unique(df[,non_numerical_variable]), col=1:length(unique(df[,non_numerical_variable])), pch=19, title=non_numerical_variable)
}

# Splitting the dataset into features and target variable
X <- df %>% select(-Class)
y <- df$Class

# Splitting the dataset into training and testing sets
set.seed(42)
train_indices <- createDataPartition(y, p=0.8, list=FALSE)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# Initialize and train models
log_reg <- glm(Class ~ ., data = df, family = binomial)
decision_tree <- rpart(Class ~ ., data = df, method = "class")
random_forest <- randomForest(Class ~ ., data = df)

# Making predictions
y_pred_log_reg <- predict(log_reg, type="response")
y_pred_decision_tree <- predict(decision_tree, type="class")
y_pred_random_forest <- predict(random_forest, type="response")

# Evaluating model performance
accuracy_log_reg <- mean(y_pred_log_reg == y_test)
accuracy_decision_tree <- mean(y_pred_decision_tree == y_test)
accuracy_random_forest <- mean(y_pred_random_forest == y_test)

precision_log_reg <- sum(y_pred_log_reg == 1 & y_test == 1) / sum(y_pred_log_reg == 1)
precision_decision_tree <- sum(y_pred_decision_tree == 1 & y_test == 1) / sum(y_pred_decision_tree == 1)
precision_random_forest <- sum(y_pred_random_forest == 1 & y_test == 1) / sum(y_pred_random_forest == 1)

recall_log_reg <- sum(y_pred_log_reg == 1 & y_test == 1) / sum(y_test == 1)
recall_decision_tree <- sum(y_pred_decision_tree == 1 & y_test == 1) / sum(y_test == 1)
recall_random_forest <- sum(y_pred_random_forest == 1 & y_test == 1) / sum(y_test == 1)

f1_log_reg <- 2 * precision_log_reg * recall_log_reg / (precision_log_reg + recall_log_reg)
f1_decision_tree <- 2 * precision_decision_tree * recall_decision_tree / (precision_decision_tree + recall_decision_tree)
f1_random_forest <- 2 * precision_random_forest * recall_random_forest / (precision_random_forest + recall_random_forest)

# Print the performance metrics
cat("Logistic Regression Performance:\n")
cat("Accuracy:", accuracy_log_reg, "\n")
cat("Precision:", precision_log_reg, "\n")
cat("Recall:", recall_log_reg, "\n")
cat("F1-score:", f1_log_reg, "\n\n")

cat("Decision Tree Performance:\n")
cat("Accuracy:", accuracy_decision_tree, "\n")
cat("Precision:", precision_decision_tree, "\n")
cat("Recall:", recall_decision_tree, "\n")
cat("F1-score:", f1_decision_tree, "\n\n")

cat("Random Forest Performance:\n")
cat("Accuracy:", accuracy_random_forest, "\n")
cat("Precision:", precision_random_forest, "\n")
cat("Recall:", recall_random_forest, "\n")
cat("F1-score:", f1_random_forest, "\n\n")

# Create confusion matrices for all models
conf_matrix_log_reg <- confusionMatrix(factor(y_pred_log_reg, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))
conf_matrix_decision_tree <- confusionMatrix(factor(y_pred_decision_tree, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))
conf_matrix_random_forest <- confusionMatrix(factor(y_pred_random_forest, levels = c(0, 1)), factor(y_test, levels = c(0, 1)))

conf_matrix_log_reg
conf_matrix_decision_tree
conf_matrix_random_forest

# Define parameter grids for each model
param_grid_log_reg <- list(C = c(0.001, 0.01, 0.1, 1, 10, 100))
param_grid_decision_tree <- list(maxdepth = c(5, 10, 15), minsplit = c(2, 5, 10))
param_grid_random_forest <- list(n_estimators = c(50, 100, 200), max_depth = c(NA, 10, 20), min_samples_split = c(2, 5, 10))

# Perform GridSearchCV or RandomizedSearchCV for each model
grid_search_log_reg <- train(Class ~ ., data = df, method = "glm", trControl = trainControl(method = "cv", number = 5), tuneGrid = param_grid_log_reg)
grid_search_decision_tree <- train(Class ~ ., data = df, method = "rpart", trControl = trainControl(method = "cv", number = 5), tuneGrid = param_grid_decision_tree)
random_search_random_forest <- train(Class ~ ., data = df, method = "rf", trControl = trainControl(method = "cv", number = 5), tuneGrid = param_grid_random_forest)

# Evaluate models using cross-validation
cv_accuracy_log_reg <- mean(predict(grid_search_log_reg, newdata = df) == df$Class)
cv_accuracy_decision_tree <- mean(predict(grid_search_decision_tree, newdata = df) == df$Class)
cv_accuracy_random_forest <- mean(predict(random_search_random_forest, newdata = df) == df$Class)

# Print mean cross-validation accuracy
cat("Mean CV Accuracy - Logistic Regression:", cv_accuracy_log_reg, "\n")
cat("Mean CV Accuracy - Decision Tree:", cv_accuracy_decision_tree, "\n")
cat("Mean CV Accuracy - Random Forest:", cv_accuracy_random_forest, "\n")

# Serialize the trained models
saveRDS(log_reg, "logistic_regression_model.rds")
saveRDS(decision_tree, "decision_tree_model.rds")
saveRDS(random_forest, "random_forest_model.rds")

