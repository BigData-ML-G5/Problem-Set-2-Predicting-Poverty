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
# 3. Poverty variables
# ------------------------
train_hogares <- train_hogares |> 
  mutate(Pobre_hand = ifelse(Ingpcug < Lp, 1, 0),
         Pobre_hand_2 = ifelse(Ingtotugarr < Lp*Npersug, 1, 0))

# =====================================================
# 4. Individual-level preprocessing
# =====================================================
pre_process_personas <- function(data) {
  data |> 
    mutate(
      bin_woman    = ifelse(P6020 == 2, 1, 0),
      bin_head     = ifelse(P6050 == 1, 1, 0),
      bin_minor    = ifelse(P6040 <= 6, 1, 0),
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
train <- train_hogares |> 
  left_join(train_personas_hogar, by = "id") |>
  select(-id) |> 
  mutate(
    Pobre   = factor(Pobre, levels = c(0, 1), labels = c("No", "Yes")),
    Pobre   = relevel(Pobre, ref = "Yes"),  # positive class
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
# 8. Model training (Elastic Net with SMOTE and ROC optimization)
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
# 9. Generate predictions on the test set
# ------------------------
predictSample <- test |> 
  mutate(pobre_lab = predict(model2, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

head(predictSample)

# ------------------------
# 10. Save predictions with dynamic filename
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
