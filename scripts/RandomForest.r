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
       MLmetrics
)

# ------------------------
# 2. Load data
# ------------------------
setwd("c:/Users/Asuar/OneDrive/Escritorio/Libros Clases/Economía/Big Data/Problem-Set-2-Predicting-Poverty")

train_hogares  <- read.csv("data/train_hogares.csv")
train_personas <- read.csv(unz("data/train_personas.csv.zip", "train_personas.csv"))
test_hogares   <- read.csv("data/test_hogares.csv")
test_personas  <- read.csv(unz("data/test_personas.csv.zip", "test_personas.csv"))

# ------------------------
# 3. Poverty variables
# ------------------------
train_hogares <- train_hogares |> 
  mutate(Pobre_hand = ifelse(Ingpcug < Lp, 1, 0),
         Pobre_hand_2 = ifelse(Ingtotugarr < Lp*Npersug, 1, 0))

# ------------------------
# 4. Individual preprocessing
# ------------------------
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

# ------------------------
# 5. Household aggregates
# ------------------------
# Training
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

# Test
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

# ------------------------
# 6. Economic variables
# ------------------------
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

# ------------------------
# 7. Merge data
# ------------------------
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

# ------------------------
# 8. Random Forest Training
# ------------------------
print("Starting Random Forest training...")

# Control
rf_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE
)

# Hyperparameter grid
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10, 12)  # Number of variables at each split
)

# Entrenar Random Forest
rf_model <- train(
  Pobre ~ .,
  data = train,
  method = "rf",
  metric = "F",
  trControl = rf_control,
  tuneGrid = rf_grid,
  ntree = 1000,      # Número de árboles
  importance = TRUE # Calcular importancia de variables
)

print("=== RANDOM FOREST RESULTS ===")
print(rf_model$bestTune)
print("Best performance:")
best_results <- rf_model$results[rf_model$results$mtry == rf_model$bestTune$mtry, ]
print(best_results)

# ------------------------
# 9. Variable Importance
# ------------------------
print("=== VARIABLE IMPORTANCE ===")
importance_rf <- varImp(rf_model)
print(importance_rf)

# ------------------------
# 10. Test Predictions
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
# 11. Save predictions
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