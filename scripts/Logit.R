# =======================================================
# Poverty Prediction in Colombia — LOGIT Model
# Date: 2025-10-04
# Description: This script predicts household poverty status
# in Colombia using household and individual-level data 
# with a Logistic Regression (Logit) model.
# =======================================================

# ------------------------
# 1. Load libraries
# ------------------------
require("pacman")
p_load(
  tidyverse,   # includes dplyr, tidyr, readr, ggplot2, forcats
  caret,
  MLmetrics,
  Metrics
)

# ------------------------
# 2. Load data
# ------------------------
train_households  <- read.csv("data/train_hogares.csv")
train_individuals <- read.csv("data/train_personas.csv")                  
test_households   <- read.csv("data/test_hogares.csv")
test_individuals  <- read.csv("data/test_personas.csv")                   

# ------------------------
# 3. Poverty variables
# ------------------------
if (all(c("Ingpcug","Lp","Ingtotugarr","Npersug","Pobre") %in% names(train_households))) {
  train_households <- train_households |>
    mutate(
      Pobre_hand   = ifelse(Ingpcug < Lp, 1, 0),
      Pobre_hand_2 = ifelse(Ingtotugarr < Lp * Npersug, 1, 0)
    )
  print(table(Official = train_households$Pobre, Hand = train_households$Pobre_hand))
  print(table(Official = train_households$Pobre, Hand2 = train_households$Pobre_hand_2))
}

# ------------------------
# 4. Individual-level preprocessing
# ------------------------
pre_process_individuals <- function(data) {
  data |>
    mutate(
      bin_woman    = ifelse(P6020 == 2, 1, 0),       # 1 = female
      bin_head     = ifelse(P6050 == 1, 1, 0),       # 1 = household head
      bin_minor    = ifelse(P6040 <= 6, 1, 0),       # 1 = child (≤ 6 years)
      cat_educ     = ifelse(P6210 == 9, 0, P6210),   # 9 = missing category → 0
      bin_occupied = ifelse(is.na(Oc), 0, 1)         # 1 = employed/occupied
    ) |>
    select(id, Orden, bin_woman, bin_head, bin_minor, cat_educ, bin_occupied)
}

train_individuals <- pre_process_individuals(train_individuals)
test_individuals  <- pre_process_individuals(test_individuals)

# ------------------------
# 5. Aggregate to household level
# ------------------------
agg_to_household <- function(dfp) {
  level <- dfp |>
    group_by(id) |>
    summarize(
      num_women    = sum(bin_woman, na.rm = TRUE),
      num_minors   = sum(bin_minor, na.rm = TRUE),
      cat_maxEduc  = suppressWarnings(max(cat_educ, na.rm = TRUE)),
      num_occupied = sum(bin_occupied, na.rm = TRUE),
      .groups = "drop"
    )
  
  head_info <- dfp |>
    filter(bin_head == 1) |>
    select(id, bin_woman, cat_educ, bin_occupied) |>
    rename(
      bin_headWoman    = bin_woman,
      cat_educHead     = cat_educ,
      bin_occupiedHead = bin_occupied
    )
  
  left_join(head_info, level, by = "id")
}

train_individuals_household <- agg_to_household(train_individuals)
test_individuals_household  <- agg_to_household(test_individuals)

# ------------------------
# 6. Household-level variables
# ------------------------
mk_household_vars <- function(dh) {
  dh |>
    mutate(
      bin_rent = ifelse(P5090 == 3, 1, 0)   # 1 = renting
    ) |>
    select(id, Dominio, bin_rent, any_of("Pobre"))
}

train_households2 <- mk_household_vars(train_households)
test_households2  <- mk_household_vars(test_households)

# ------------------------
# 7. Merge individuals and households
# ------------------------
train <- train_households2 |>
  left_join(train_individuals_household, by = "id") |>
  mutate(
    Pobre   = factor(Pobre, levels = c(0, 1), labels = c("No", "Yes")),
    Dominio = factor(Dominio),
    cat_educHead = factor(
      cat_educHead,
      levels = c(0:6),
      labels = c("Unknown", "None", "Preschool", "Primary",
                 "Secondary", "HighSchool", "University")
    )
  ) |>
  droplevels()   # remove empty factor levels

test <- test_households2 |>
  left_join(test_individuals_household, by = "id") |>
  mutate(
    Dominio = factor(Dominio, levels = levels(train$Dominio)),
    cat_educHead = factor(
      cat_educHead,
      levels = c(0:6),
      labels = c("Unknown", "None", "Preschool", "Primary",
                 "Secondary", "HighSchool", "University")
    )
  )

# ------------------------
# 8. Light data cleaning
# ------------------------

# 8.1 Replace missing levels in factors with "Missing" to avoid dropped cases
to_factor <- names(train %>% select(where(is.factor)))
for (v in to_factor) {
  train[[v]] <- forcats::fct_explicit_na(train[[v]], na_level = "Missing")
  if (v %in% names(test)) test[[v]] <- forcats::fct_explicit_na(test[[v]], na_level = "Missing")
}

# 8.2 Remove ID column and handle near-zero variance (NZV) predictors
X_train <- train %>% select(-id)
nzv_cols <- nearZeroVar(X_train, saveMetrics = TRUE)
drop_cols <- rownames(nzv_cols)[nzv_cols$nzv]
if (length(drop_cols)) {
  message("Removing NZV variables: ", paste(drop_cols, collapse = ", "))
  X_train <- X_train %>% select(-all_of(drop_cols))
  test    <- test    %>% select(-all_of(drop_cols))   # keeps 'id' since it was not in X_train
}

# ------------------------
# 9. Train the LOGIT model
# ------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = 10,            # 10-fold cross-validation
  classProbs      = TRUE,
  savePredictions = TRUE,
  verbose         = TRUE,
  summaryFunction = prSummary      # computes AUC, Precision, Recall, and F1
)

set.seed(2025)
model_logit <- train(
  Pobre ~ .,
  data      = X_train,
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "F"                  # optimize F1 metric
)

print(model_logit)

# (Optional for reporting): mean accuracy across folds
acc_cv <- mean(model_logit$resample$Accuracy)
cat(sprintf("Cross-Validation Accuracy (10-fold): %.3f\n", acc_cv))

# ------------------------
# 10. Out-of-Fold (OOF) predictions and threshold tuning
# ------------------------
oof <- model_logit$pred
oof$obs <- factor(oof$obs, levels = c("No", "Yes"))
stopifnot("Yes" %in% names(oof))
probs <- oof$Yes

grid_t <- seq(0.10, 0.90, by = 0.01)
f1_vec <- sapply(grid_t, function(t) {
  pred_lab <- factor(ifelse(probs >= t, "Yes", "No"), levels = c("No", "Yes"))
  MLmetrics::F1_Score(y_pred = pred_lab, y_true = oof$obs, positive = "Yes")
})
best_t  <- grid_t[which.max(f1_vec)]
best_f1 <- max(f1_vec, na.rm = TRUE)
cat(sprintf("Optimal OOF threshold (F1): t* = %.2f | F1 = %.3f\n", best_t, best_f1))

# ------------------------
# 11. Predictions on test data
# ------------------------
test_probs <- predict(model_logit, newdata = test, type = "prob")[,"Yes"]

predictClass <- tibble(
  id    = test$id,
  poor  = as.integer(test_probs >= best_t)   # binary output for Kaggle (F1-based)
)

predictProba <- tibble(
  id    = test$id,
  poor  = test_probs                         # continuous probabilities
)

print(head(predictClass, 10))

# ------------------------
# 12. Save results with dynamic filenames
# ------------------------
tstr <- gsub("\\.", "_", sprintf("%.2f", best_t))
name_cls  <- paste0("LOGIT_cv10_thresh_", tstr, ".csv")   # algorithm + threshold
name_prob <- "LOGIT_cv10_probs.csv"

# Kaggle format
write.csv(predictClass, name_cls,  row.names = FALSE, fileEncoding = "UTF-8")
write.csv(predictProba, name_prob, row.names = FALSE, fileEncoding = "UTF-8")
cat("Predictions saved to:\n - ", name_cls,  "\n - ", name_prob, "\n")

# ------------------------
# End of script
# ------------------------


