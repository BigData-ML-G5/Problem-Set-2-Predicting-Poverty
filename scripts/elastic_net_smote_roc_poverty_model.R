# =====================================================
# Description: This script predicts household poverty in Colombia
# using household and individual-level data with Machine Learning
# models (Elastic Net implemented with the caret package).
# =====================================================

# ------------------------
# 1. Load required libraries
# ------------------------
require("pacman")
p_load(tidyverse, 
       glmnet,
       caret,
       MLmetrics, 
       Metrics,
       ggplot2,
       DMwR)  # para SMOTE

# ------------------------
# 2. Load data
# ------------------------
read_csv_from_zip <- function(zip_path) {
  files <- unzip(zip_path, list = TRUE)$Name
  csv_file <- files[grepl("\\.csv$", files)][1]  # Usa el primer .csv que encuentre
  read.csv(unz(zip_path, csv_file))
}

train_personas <- read_csv_from_zip("data/train_personas.csv.zip")
test_personas  <- read_csv_from_zip("data/test_personas.csv.zip")

train_hogares  <- read.csv("data/train_hogares.csv")
test_hogares   <- read.csv("data/test_hogares.csv")


# ------------------------
# 3. Model training (Elastic Net with SMOTE and ROC optimization)
# ------------------------

# trainControl con SMOTE y optimización para ROC
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "smote",  # importante: SMOTE en entrenamiento
  savePredictions = "final",
  verboseIter = TRUE
)

# tuneGrid alrededor del lambda y alpha fijos
tune_grid <- expand.grid(
  alpha = seq(0.6, 1.0, by = 0.1),
  lambda = seq(0.002, 0.008, length.out = 10) 
)

set.seed(2025)

model2 <- train(
  Pobre ~ .,
  data = train,
  method = "glmnet",
  metric = "ROC",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)

print(model2)

# Matriz de confusión para la clase positiva "Yes"
confusion <- confusionMatrix(model2$pred$pred, model2$pred$obs, positive = "Yes")
print(confusion)

# ------------------------
# 4. Generate predictions on the test set
# ------------------------
predictSample <- test |> 
  mutate(pobre_lab = predict(model2, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

head(predictSample)

# ------------------------
# 5. Save predictions with dynamic filename
# ------------------------
lambda_str <- gsub("\\.", "_", as.character(round(model2$bestTune$lambda, 4)))
alpha_str  <- gsub("\\.", "_", as.character(model2$bestTune$alpha))

name <- paste0("EN_lambda_", lambda_str,
               "_alpha_", alpha_str, ".csv")

write.csv(predictSample, name, row.names = FALSE)
cat("Predictions saved as:", name, "\n")

# ------------------------
# End of script
# ------------------------
