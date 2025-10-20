# =====================================================
# Description: This script predicts household poverty in Colombia
# using household and individual-level data with CART models
# (Classification and Regression Trees implemented with caret package).
# =====================================================

# ------------------------
# 1. Load required libraries
# ------------------------
require("pacman")
p_load(tidyverse, caret, rpart, rpart.plot, MLmetrics, Metrics, DMwR)


# ------------------------
# 2. Load data
# ------------------------
# setwd("c:/Users/Asuar/OneDrive/Escritorio/Libros Clases/Economía/Big Data/Problem-Set-2-Predicting-Poverty")
# TODO: Update paths to your actual data location
train_hogares  <- read.csv("data/train_hogares.csv")
train_personas <- read.csv(unz("data/train_personas.csv.zip", "train_personas.csv"))  # Read from zip file
test_hogares   <- read.csv("data/test_hogares.csv")
test_personas  <- read.csv(unz("data/test_personas.csv.zip", "test_personas.csv"))   # Read from zip file


# =====================================================
# 3. Function to calculate classification metrics
# =====================================================
# Note: We use the complete training set (train) and test set (test) as provided
# Model evaluation will be done through k-fold cross-validation on the training set
# =====================================================
calculate_classification_metrics <- function(y_true, y_pred, model_name) {
  
  # Convert factors to numeric if needed
  if(is.factor(y_true)) {
    y_true_num <- as.numeric(y_true) - 1  # Convert "No"/"Yes" to 0/1
  } else {
    y_true_num <- y_true
  }
  
  if(is.factor(y_pred)) {
    y_pred_num <- as.numeric(y_pred) - 1  # Convert "No"/"Yes" to 0/1
  } else {
    y_pred_num <- y_pred
  }
  
  # Confusion matrix
  cm <- table(Predicted = y_pred_num, Actual = y_true_num)
  
  # Calculate metrics
  accuracy <- sum(diag(cm)) / sum(cm)
  
  # Precision, Recall, F1 for class 1 (Poor = Yes)
  if(length(unique(y_pred_num)) > 1 && sum(y_pred_num == 1) > 0) {
    precision <- cm[2,2] / sum(cm[2,])  # TP / (TP + FP)
    recall <- cm[2,2] / sum(cm[,2])     # TP / (TP + FN)
    f1_score <- 2 * (precision * recall) / (precision + recall)
  } else {
    precision <- 0
    recall <- 0
    f1_score <- 0
  }
  
  metrics <- tibble(
    modelo = model_name,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  )
  
  return(metrics)
}

# =====================================================
# 4. CART Model Training
# =====================================================

# Control de entrenamiento con validación cruzada
fitControl <- trainControl(
  method = "cv", 
  number = 10,
  classProbs = TRUE,
  summaryFunction = prSummary,  # Use precision-recall summary for imbalanced data
  savePredictions = TRUE,
  sampling = "smote"
)

# =====================================================
# 4.1) CART Model optimized by complexity parameter (cp)
# =====================================================
set.seed(2025)
cart_complexity <- train(
  Pobre ~ .,
  data = train,  # Use complete training set with k-fold CV
  method = "rpart",
  metric = "F",  # Optimize F1 score (better for imbalanced classes)
  trControl = fitControl,
  tuneGrid = expand.grid(cp = seq(0.00001, 0.001, 0.00005)),  # Test complexity parameters
  weights = ifelse(train$Pobre == "Yes",
                1/mean(train$Pobre == "Yes"),
                1/mean(train$Pobre == "No"))
)

print("=== CART MODEL OPTIMIZED BY COMPLEXITY PARAMETER ===")
print(cart_complexity$bestTune)
print("Best model results:")
print(cart_complexity$results[cart_complexity$results$cp == cart_complexity$bestTune$cp, ])

# =====================================================
# 4.2) CART Model optimized by maximum depth
# =====================================================
set.seed(2025)
cart_depth <- train(
  Pobre ~ .,
  data = train,  # Use complete training set with k-fold CV
  method = "rpart2",
  metric = "F",  # Optimize F1 score
  trControl = fitControl,
  tuneGrid = expand.grid(maxdepth = seq(1, 15, 1))  # Test depths from 1 to 15
    weights = ifelse(train$Pobre == "Yes", 
                  1/mean(train$Pobre == "Yes"),
                  1/mean(train$Pobre == "No"))
)

print("=== CART MODEL OPTIMIZED BY MAXIMUM DEPTH ===")
print(cart_depth$bestTune)
print("Best model results:")
print(cart_depth$results[cart_depth$results$maxdepth == cart_depth$bestTune$maxdepth, ])

# =====================================================
# 5. Model Comparison using Cross-Validation Results
# =====================================================

# Extract CV metrics from both models
complexity_results <- cart_complexity$results[cart_complexity$results$cp == cart_complexity$bestTune$cp, ]
depth_results <- cart_depth$results[cart_depth$results$maxdepth == cart_depth$bestTune$maxdepth, ]

# Create comparison table with CV metrics
metricas_cart <- tibble(
  modelo = c("CART (Complexity)", "CART (Depth)"),
  Precision = c(complexity_results$Precision, depth_results$Precision),
  Recall = c(complexity_results$Recall, depth_results$Recall),
  F1_Score = c(complexity_results$F, depth_results$F),
  AUC = c(complexity_results$AUC, depth_results$AUC)
)

print("=== CART MODELS COMPARISON (Cross-Validation Metrics) ===")
print(metricas_cart)

# Find best CART model
mejor_cart <- metricas_cart[which.max(metricas_cart$F1_Score), ]
cat("Best CART model:", mejor_cart$modelo, "with F1 Score =", mejor_cart$F1_Score, "\n")

# =====================================================
# 6. Visualize the best CART models
# =====================================================

# Create views directory if it doesn't exist
if (!dir.exists("views")) {
  dir.create("views")
}

print("=== CART MODEL OPTIMIZED BY COMPLEXITY ===")

# Save plot for complexity-optimized tree
png("views/cart_complexity_poverty.png", width = 1200, height = 800, res = 150)
prp(cart_complexity$finalModel, 
    under = TRUE,              # Show additional info below nodes
    branch.lty = 2,            # Dotted line style for branches
    yesno = 2,                 # Show "yes/no" at bifurcations
    faclen = 0,                # Show complete factor labels
    varlen = 15,               # Maximum variable name length
    tweak = 1.2,               # Adjust text size
    clip.facs = TRUE,          # Clip long factor levels
    box.palette = "Greens",    # Color palette for boxes
    compress = TRUE,           # Compress tree vertically
    ycompress = TRUE,          # Compress y-axis too
    main = "CART for Poverty Prediction - Optimized by Complexity (cp)",
    digits = 3                 # Show 3 decimals
)
dev.off()

print("=== CART MODEL OPTIMIZED BY DEPTH ===")

# Save plot for depth-optimized tree
png("views/cart_depth_poverty.png", width = 1200, height = 800, res = 150)
prp(cart_depth$finalModel, 
    under = TRUE,              # Show additional info below nodes
    branch.lty = 2,            # Dotted line style for branches
    yesno = 2,                 # Show "yes/no" at bifurcations
    faclen = 0,                # Show complete factor labels
    varlen = 15,               # Maximum variable name length
    tweak = 1.2,               # Adjust text size
    clip.facs = TRUE,          # Clip long factor levels
    box.palette = "Greens",  # Different color palette
    compress = TRUE,           # Compress tree vertically
    ycompress = TRUE,          # Compress y-axis too
    main = "CART for Poverty Prediction - Optimized by Depth",
    digits = 3                 # Show 3 decimals
)
dev.off()

cat("\n✅ CART tree plots saved in:\n")
cat("   - views/cart_complexity_poverty.png\n") 
cat("   - views/cart_depth_poverty.png\n")

# =====================================================
# 7. Best vs 1SE Rule Analysis (Cross-Validation Results)
# =====================================================

# Find best cp and 1SE cp from complexity model results
all_results <- cart_complexity$results
best_cp_value <- all_results$cp[which.max(all_results$F)]

# Calculate 1SE rule: find simplest model within 1 standard error of best
best_f1 <- max(all_results$F)
best_f1_se <- all_results$F_SD[which.max(all_results$F)]  
f1_threshold <- best_f1 - best_f1_se  # 1SE below best

cat("DEBUG: Best F1 =", best_f1, ", SE =", best_f1_se, ", Threshold =", f1_threshold, "\n")

# Find simplest model (highest cp) that is within 1SE of best
valid_models <- all_results[all_results$F >= f1_threshold, ]

# Check if valid_models is empty and provide fallback
if(nrow(valid_models) == 0) {
  cat("⚠️  No models within 1SE threshold. Using alternative approach...\n")
  # Alternative: find model with second-best F1 or use best model
  sorted_results <- all_results[order(-all_results$F), ]
  if(nrow(sorted_results) >= 2) {
    # Use second best model
    onese_cp_value <- sorted_results$cp[2]
    cat("Using second-best F1 model as 1SE alternative\n")
  } else {
    # Fallback to best model
    onese_cp_value <- best_cp_value
    cat("Using best model as 1SE fallback\n")
  }
} else {
  onese_cp_value <- max(valid_models$cp)  # Highest cp (simplest) within threshold
}

# Final validation to ensure onese_cp_value is valid
if(is.infinite(onese_cp_value) || is.na(onese_cp_value)) {
  cat("⚠️  Invalid 1SE cp value detected. Using best cp as fallback.\n")
  onese_cp_value <- best_cp_value
}

cat("=== BEST vs 1SE RULE COMPARISON ===\n")
cat("Best Rule  - cp:", best_cp_value, ", CV F1 =", best_f1, "\n")

# Get F1 for the selected 1SE cp
onese_f1 <- all_results$F[all_results$cp == onese_cp_value]
cat("1SE Rule   - cp:", onese_cp_value, ", CV F1 =", onese_f1, "\n")

if(onese_cp_value > best_cp_value) {
  cat("→ 1SE rule chooses SIMPLER model (higher cp = less complex tree)\n")
  cat("→ Trade-off: Slightly lower F1 but better generalization\n")
} else if(onese_cp_value < best_cp_value) {
  cat("→ 1SE rule chooses MORE COMPLEX model (lower cp)\n")
} else {
  cat("→ Both rules choose the same cp value\n")
}

cat("Recommendation: Use BEST rule cp =", best_cp_value, "for maximum F1-Score\n")

# =====================================================
# 8. Generate predictions with 1SE rule
# =====================================================

# Create CART model with 1SE cp value
library(rpart)
set.seed(2025)
cart_1se_model <- rpart(Pobre ~ ., data = train, cp = onese_cp_value)

# Generate predictions on test set with 1SE model
predictSample_1se <- test |> 
  mutate(pobre_lab = predict(cart_1se_model, newdata = test, type = "class")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

# Create filename with 1SE cp value
cp_1se_str <- gsub("\\.", "_", as.character(onese_cp_value))
filename_1se <- paste0("CART_cp_", cp_1se_str, ".csv")

# Save 1SE predictions
write.csv(predictSample_1se, filename_1se, row.names = FALSE)
cat("✅ 1SE predictions saved as:", filename_1se, "\n")

# Show preview of 1SE predictions
cat("Preview of 1SE predictions:\n")
print(head(predictSample_1se))

# =====================================================
# 9. Generate predictions with BEST rule (original)
# =====================================================

# Choose best model from CV results for final predictions
if(metricas_cart$F1_Score[1] > metricas_cart$F1_Score[2]) {  # Complexity vs Depth
  best_cart_model <- cart_complexity
  model_name <- "CART_Complexity"
  best_cp <- cart_complexity$bestTune$cp
  best_maxdepth <- NA
  cat("Using CART Complexity model for final predictions\n")
} else {
  best_cart_model <- cart_depth
  model_name <- "CART_Depth"
  best_cp <- NA
  best_maxdepth <- cart_depth$bestTune$maxdepth
  cat("Using CART Depth model for final predictions\n")
}

# =====================================================
# 10. Generate final predictions on test set (BEST rule)
# =====================================================

# Generate predictions on test set
predictSample <- test |> 
  mutate(pobre_lab = predict(best_cart_model, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

head(predictSample)

# =====================================================
# 11. Save BEST rule predictions
# =====================================================

# Create filename based on best model and hyperparameters
if(!is.na(best_cp)) {
  cp_str <- gsub("\\.", "_", as.character(best_cp))
  filename <- paste0("CART_cp_", cp_str, ".csv")
} else {
  filename <- paste0("CART_maxdepth_", best_maxdepth, ".csv")
}

write.csv(predictSample, filename, row.names = FALSE)
cat("Predictions saved as:", filename, "\n")

cat("\n=== SCRIPT COMPLETED SUCCESSFULLY ===\n")
cat("Final model:", model_name, "\n")
if(!is.na(best_cp)) {
  cat("Best cp:", best_cp, "\n")
} else {
  cat("Best maxdepth:", best_maxdepth, "\n")
}
cat("BEST rule predictions saved as:", filename, "\n")
cat("Performance: F1 =", round(mejor_cart$F1_Score, 4), 
    ", Precision =", round(mejor_cart$Precision, 4), 
    ", Recall =", round(mejor_cart$Recall, 4), "\n")

cat("\n=== SUMMARY OF GENERATED FILES ===\n")
cat("📁 1SE Rule predictions : ", filename_1se, " (cp = ", onese_cp_value, ")\n")
cat("📁 BEST Rule predictions: ", filename, " (cp = ", best_cp_value, ")\n")
cat("\n💡 Compare both files to see differences between simpler (1SE) vs optimal (BEST) models\n")

// ...existing code...

# =====================================================
# 12. Save Depth-based predictions
# =====================================================

# Generate predictions using depth-optimized model
predictSample_depth <- test |> 
  mutate(pobre_lab = predict(cart_depth, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

# Save depth-based predictions
write.csv(predictSample_depth, "CART_depth.csv", row.names = FALSE)

# Update final summary
cat("\n=== SUMMARY OF GENERATED FILES ===\n")
cat("📁 1SE Rule predictions : ", filename_1se, " (cp = ", onese_cp_value, ")\n")
cat("📁 BEST Rule predictions: ", filename, " (cp = ", best_cp_value, ")\n")
cat("📁 Depth predictions    : CART_depth.csv (maxdepth = ", cart_depth$bestTune$maxdepth, ")\n")
cat("\n💡 Compare files to see differences between models\n")


# =====================================================
# Load libraries
library(tidyverse)

# Load predictions and show initial structure
my_cart <- read.csv("CART_depth.csv")
other_pred <- read.csv("LOGIT_cv10_thresh_0_32.csv")

# Print structure to debug
print("=== DATA STRUCTURE ===")
print("CART predictions:")
str(my_cart)
print("\nOther predictions:")
str(other_pred)

# Ensure both dataframes have the same structure
my_cart <- my_cart %>% 
  select(id, pobre) %>%
  rename(pobre_cart = pobre)

other_pred <- other_pred %>%
  select(id, pobre) %>%
  rename(pobre_other = pobre)

# Merge and compare
comparison <- my_cart %>%
  inner_join(other_pred, by = "id") %>%
  mutate(match = pobre_cart == pobre_other)

# Results
print("\n=== COMPARACIÓN DE PREDICCIONES ===")
print(paste("Total observaciones:", nrow(comparison)))
print(paste("Predicciones iguales:", sum(comparison$match)))
print(paste("Porcentaje de coincidencia:", round(mean(comparison$match) * 100, 2), "%"))

# Contingency table
print("\nTabla de contingencia:")
print(table(CART = comparison$pobre_cart, LOGIT = comparison$pobre_other))

# Save comparison results
write.csv(comparison, "predictions_comparison.csv", row.names = FALSE)