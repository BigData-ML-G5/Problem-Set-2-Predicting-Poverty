# =====================================================
# Random Forest for Poverty Prediction
# =====================================================

# ------------------------
# 1. Load libraries
# ------------------------
require("pacman")
p_load(tidyverse, 
       caret,
       randomForest,
       MLmetrics,
       DMwR
)

# ------------------------
# 2. Random Forest Training
# ------------------------
print("Starting Random Forest training...")

# Control
rf_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE,
  sampling = "smote"
)

# Hyperparameter grid
rf_grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6)  # Number of variables at each split
)

# Entrenar Random Forest
rf_model <- train(
  Pobre ~ .,
  data = train,
  method = "rf",
  metric = "F",
  trControl = rf_control,
  tuneGrid = rf_grid,
  ntree = 200,      # Número de árboles
  importance = TRUE,  # Calcular importancia de variables
  weights = ifelse(train$Pobre == "Yes",
              1/mean(train$Pobre == "Yes"),
              1/mean(train$Pobre == "No"))  # Ajustar pesos para clases desbalanceadas
)

print("=== RANDOM FOREST RESULTS ===")
print(rf_model$bestTune)
print("Best performance:")
best_results <- rf_model$results[rf_model$results$mtry == rf_model$bestTune$mtry, ]
print(best_results)

# ------------------------
# 3. Variable Importance
# ------------------------
print("=== VARIABLE IMPORTANCE ===")
importance_rf <- varImp(rf_model)
print(importance_rf)

# ------------------------
# 4. Test Predictions
# ------------------------
print("Generating predictions on test set...")

rf_predictions <- predict(rf_model, newdata = test, type = "raw")
rf_probs <- predict(rf_model, newdata = test, type = "prob")

# Create final predictions
predictSample <- test |> 
  mutate(
    pobre_lab = rf_predictions,
    pobre = ifelse(pobre_lab == "Yes", 1, 0),
    prob_pobre = rf_probs$Yes
  ) |> 
  select(id, pobre, prob_pobre)

head(predictSample)

# ------------------------
# 5. Save predictions
# ------------------------
# Save binary predictions (same format as other models)
predictSample_binary <- predictSample |> select(id, pobre)
write.csv(predictSample_binary, "RF_poverty_predictions.csv", row.names = FALSE)

# Save with probabilities (bonus)
write.csv(predictSample, "RF_poverty_predictions_with_probs.csv", row.names = FALSE)

print("=== COMPLETED ===")
print(paste("Best mtry:", rf_model$bestTune$mtry))
print(paste("CV F1-Score:", round(best_results$F, 4)))
print(paste("CV Precision:", round(best_results$Precision, 4)))
print(paste("CV Recall:", round(best_results$Recall, 4)))
print("Predictions saved:")
print("- RF_poverty_predictions.csv (binary predictions)")
print("- RF_poverty_predictions_with_probs.csv (with probabilities)")