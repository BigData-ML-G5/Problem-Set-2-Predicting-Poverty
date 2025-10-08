# =====================================================
# Description: This script predicts household poverty in Colombia
# using household and individual-level data with CART models
# (Classification and Regression Trees implemented with caret package).
# =====================================================

# ------------------------
# 1. Load required libraries
# ------------------------
require("pacman")
p_load(tidyverse, caret, rpart, rpart.plot, MLmetrics, Metrics)


# ------------------------
# 2. Load data
# ------------------------
setwd("c:/Users/Asuar/OneDrive/Escritorio/Libros Clases/Economía/Big Data/Problem-Set-2-Predicting-Poverty")
# TODO: Update paths to your actual data location
train_hogares  <- read.csv("data/train_hogares.csv")
train_personas <- read.csv(unz("data/train_personas.csv.zip", "train_personas.csv"))  # Read from zip file
test_hogares   <- read.csv("data/test_hogares.csv")
test_personas  <- read.csv(unz("data/test_personas.csv.zip", "test_personas.csv"))   # Read from zip file

# ------------------------
# 3. Poverty variables
# ------------------------
# The poverty line (Lp) is used to construct a binary poverty indicator.
train_hogares <- train_hogares |> 
  mutate(Pobre_hand = ifelse(Ingpcug < Lp, 1, 0),
         Pobre_hand_2 = ifelse(Ingtotugarr < Lp*Npersug, 1, 0))

# Compare official poverty variable (DANE) vs. manually computed one
table(train_hogares$Pobre, train_hogares$Pobre_hand)
table(train_hogares$Pobre, train_hogares$Pobre_hand_2)

# =====================================================
# 4. Individual-level preprocessing
# =====================================================
# This function creates binary and categorical variables at the individual level:
# - bin_woman: 1 if female (P6020 == 2)
# - bin_head: 1 if head of household (P6050 == 1)
# - bin_minor: 1 if child under 6 years old (P6040 <= 6)
# - cat_educ: education level (replaces 9 = missing with 0)
# - bin_occupied: 1 if employed (variable Oc is not NA)

pre_process_personas <- function(data) {
  data |> 
    mutate(
      bin_woman   = ifelse(P6020 == 2, 1, 0),
      bin_head    = ifelse(P6050 == 1, 1, 0),
      bin_minor   = ifelse(P6040 <= 6, 1, 0),
      cat_educ    = ifelse(P6210 == 9, 0, P6210),
      bin_occupied = ifelse(is.na(Oc), 0, 1)
    ) |> 
    select(id, Orden, bin_woman, bin_head, bin_minor, cat_educ, bin_occupied)
}

train_personas <- pre_process_personas(train_personas)
test_personas  <- pre_process_personas(test_personas)

# =====================================================
# 5. Household-level variables derived from individuals
# =====================================================
# Aggregated characteristics from individuals at the household level:
#   - num_women: number of women
#   - num_minors: number of children under 6
#   - cat_maxEduc: maximum education level in the household
#   - num_occupied: number of employed persons
#   - mean_educ: average education level
#   - dep_ratio: economic dependency ratio (total persons / employed)

# ----- Training -----
train_personas_nivel_hogar <- train_personas |> 
  group_by(id) |>
  summarize(
    num_women    = sum(bin_woman, na.rm = TRUE),
    num_minors   = sum(bin_minor, na.rm = TRUE),
    cat_maxEduc  = max(cat_educ, na.rm = TRUE),
    num_occupied = sum(bin_occupied, na.rm = TRUE),
    mean_educ    = mean(cat_educ, na.rm = TRUE),
    dep_ratio    = ifelse(num_occupied > 0, n() / num_occupied, n())
  ) |> 
  ungroup()

train_personas_hogar <- train_personas |> 
  filter(bin_head == 1) |>
  select(id, bin_woman, cat_educ, bin_occupied) |>
  rename(
    bin_headWoman    = bin_woman,
    cat_educHead     = cat_educ,
    bin_occupiedHead = bin_occupied
  ) |>
  left_join(train_personas_nivel_hogar, by = "id")

# ----- Test -----
test_personas_nivel_hogar <- test_personas |> 
  group_by(id) |>
  summarize(
    num_women    = sum(bin_woman, na.rm = TRUE),
    num_minors   = sum(bin_minor, na.rm = TRUE),
    cat_maxEduc  = max(cat_educ, na.rm = TRUE),
    num_occupied = sum(bin_occupied, na.rm = TRUE),
    mean_educ    = mean(cat_educ, na.rm = TRUE),
    dep_ratio    = ifelse(num_occupied > 0, n() / num_occupied, n())
  ) |> 
  ungroup()

test_personas_hogar <- test_personas |> 
  filter(bin_head == 1) |>
  select(id, bin_woman, cat_educ, bin_occupied) |>
  rename(
    bin_headWoman    = bin_woman,
    cat_educHead     = cat_educ,
    bin_occupiedHead = bin_occupied
  ) |>
  left_join(test_personas_nivel_hogar, by = "id")

# =====================================================
# 6. Household-level economic variables
# =====================================================
# Economic characteristics of the household:
#   - bin_rent: 1 if household rents the dwelling (P5090 == 3)
#   - Ingpcug: per capita household income (P5000 / Npersug)
#   - IPR: income-to-poverty-line ratio (economic capacity)

train_hogares <- train_hogares |> 
  mutate(
    bin_rent = ifelse(P5090 == 3, 1, 0),
    Ingpcug  = P5000 / Npersug,
    IPR      = Ingpcug / Lp
  ) |> 
  select(id, Dominio, bin_rent, Ingpcug, IPR, Pobre)

test_hogares <- test_hogares |> 
  mutate(
    bin_rent = ifelse(P5090 == 3, 1, 0),
    Ingpcug  = P5000 / Npersug,
    IPR      = Ingpcug / Lp
  ) |> 
  select(id, Dominio, bin_rent, Ingpcug, IPR)

# =====================================================
# 7. Merge household and individual data
# =====================================================
# Merge household-level data with person-level aggregates.
# Format categorical variables.

train <- train_hogares |> 
  left_join(train_personas_hogar, by = "id") |>
  select(-id) |> 
  mutate(
    Pobre   = factor(Pobre, levels = c(0, 1), labels = c("No", "Yes")),
    Dominio = factor(Dominio),
    cat_educHead = factor(cat_educHead, levels = c(0:6),
                          labels = c("No information", "None", "Preschool", "Primary",
                                     "Secondary", "High school", "University"))
  )

test <- test_hogares |> 
  left_join(test_personas_hogar, by = "id") |> 
  mutate(
    Dominio = factor(Dominio),
    cat_educHead = factor(cat_educHead, levels = c(0:6),
                          labels = c("No information", "None", "Preschool", "Primary",
                                     "Secondary", "High school", "University"))
  )

# =====================================================
# 7. Function to calculate classification metrics
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
# 8. CART Model Training
# =====================================================

# Control de entrenamiento con validación cruzada
fitControl <- trainControl(
  method = "cv", 
  number = 10,
  classProbs = TRUE,
  summaryFunction = prSummary,  # Use precision-recall summary for imbalanced data
  savePredictions = TRUE
)

# =====================================================
# 8.1) CART Model optimized by complexity parameter (cp)
# =====================================================
set.seed(2025)
cart_complexity <- train(
  Pobre ~ .,
  data = train,  # Use complete training set with k-fold CV
  method = "rpart",
  metric = "F",  # Optimize F1 score (better for imbalanced classes)
  trControl = fitControl,
  tuneGrid = expand.grid(cp = seq(0.00001, 0.001, 0.00005))  # Test complexity parameters
)

print("=== CART MODEL OPTIMIZED BY COMPLEXITY PARAMETER ===")
print(cart_complexity$bestTune)
print("Best model results:")
print(cart_complexity$results[cart_complexity$results$cp == cart_complexity$bestTune$cp, ])

# =====================================================
# 8.2) CART Model optimized by maximum depth
# =====================================================
set.seed(2025)
cart_depth <- train(
  Pobre ~ .,
  data = train,  # Use complete training set with k-fold CV
  method = "rpart2",
  metric = "F",  # Optimize F1 score
  trControl = fitControl,
  tuneGrid = expand.grid(maxdepth = seq(1, 15, 1))  # Test depths from 1 to 15
)

print("=== CART MODEL OPTIMIZED BY MAXIMUM DEPTH ===")
print(cart_depth$bestTune)
print("Best model results:")
print(cart_depth$results[cart_depth$results$maxdepth == cart_depth$bestTune$maxdepth, ])

# =====================================================
# 9. Model Comparison using Cross-Validation Results
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
# 10. Visualize the best CART models
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
# 11. Best vs 1SE Rule Analysis (Cross-Validation Results)
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
# 13. Generate predictions with 1SE rule
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
# 14. Generate predictions with BEST rule (original)
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
# 15. Generate final predictions on test set (BEST rule)
# =====================================================

# Generate predictions on test set
predictSample <- test |> 
  mutate(pobre_lab = predict(best_cart_model, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

head(predictSample)

# =====================================================
# 16. Save BEST rule predictions with dynamic filename
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