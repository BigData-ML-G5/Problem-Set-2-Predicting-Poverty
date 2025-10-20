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
       Metrics
)

table(train_hogares$Dominio)
# ------------------------
# 2. Load data
# ------------------------
train_hogares  <- read.csv("/Users/selene/Desktop/MECA/BD&ML/GitHub/Problem-Set-2-Predicting-Poverty/data/train_hogares.csv")
train_personas <- read.csv("/Users/selene/Desktop/MECA/BD&ML/GitHub/Problem-Set-2-Predicting-Poverty/data/train_personas.csv")
test_hogares   <- read.csv("/Users/selene/Desktop/MECA/BD&ML/GitHub/Problem-Set-2-Predicting-Poverty/data/test_hogares.csv")
test_personas  <- read.csv("/Users/selene/Desktop/MECA/BD&ML/GitHub/Problem-Set-2-Predicting-Poverty/data/test_personas.csv")
s_nivel_hogar, by = "id")


# ------------------------
# 3. Model training (Elastic Net)
# ------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE
)

set.seed(202424503)
model1 <- train(
  Pobre ~ .,
  data = train,
  metric = "F",
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  tuneGrid = expand.grid(
    alpha  = seq(0, 2, by = 0.1),
    lambda = 10^seq(-3, 3, length = 20)
  )   
)

print(model1)

# ------------------------
# 4. Generate predictions on the test set
# ------------------------
predictSample <- test |> 
  mutate(pobre_lab = predict(model1, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

head(predictSample)

# ------------------------
# 5. Save predictions with dynamic filename
# ------------------------
lambda_str <- gsub("\\.", "_", as.character(round(model1$bestTune$lambda, 4)))
alpha_str  <- gsub("\\.", "_", as.character(model1$bestTune$alpha))

name <- paste0("EN_lambda_", lambda_str,
               "_alpha_", alpha_str, ".csv")

write.csv(predictSample, name, row.names = FALSE)
cat("Predictions saved as:", name, "\n")

# ------------------------
# End of script
# ------------------------
