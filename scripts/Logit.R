# =======================================================
# Poverty Prediction in Colombia — LOGIT Model (caret)
# Date: 2025-10-19
# Description: Logistic regression with 10-fold CV optimizing F1
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,   # dplyr, tidyr, readr, ggplot2, forcats
    caret,
    MLmetrics,
    Metrics
  )
})

# ------------------------
# 2) Load data
# ------------------------
# Expect train_clean.csv and test_clean.csv in the current working directory.
# (If needed, setwd(...) before running.)
train <- read.csv("train_clean.csv", na.strings = c("", "NA", "NaN"))
test  <- read.csv("test_clean.csv",  na.strings = c("", "NA", "NaN"))

# Basic checks
stopifnot("id"    %in% names(test))
stopifnot("Pobre" %in% names(train))

# ------------------------
# 3) Target encoding & basic cleaning
# ------------------------
# Ensure a binary factor compatible with caret + prSummary.
# (Assume 0 = No, 1 = Yes if numeric; otherwise normalize common strings.)
if (is.numeric(train$Pobre)) {
  train$Pobre <- factor(ifelse(train$Pobre == 1, "Yes", "No"), levels = c("No","Yes"))
} else {
  train$Pobre <- tolower(as.character(train$Pobre))
  train$Pobre <- ifelse(train$Pobre %in% c("1","yes","si","sí","y","true"), "Yes", "No")
  train$Pobre <- factor(train$Pobre, levels = c("No","Yes"))
}
# If you prefer CV metrics with "Yes" as the positive class, use:
# train$Pobre <- factor(train$Pobre, levels = c("Yes","No"))

# Remove identifiers/leakage columns from predictors
cols_leak <- c("id", "Pobre")
x_cols <- setdiff(names(train), cols_leak)

# Align columns between TRAIN and TEST 
common_cols <- intersect(x_cols, setdiff(names(test), "id"))
if (length(common_cols) == 0) stop("No columns in common between train and test (except id). Check preprocessing.")
x_cols <- common_cols

# Final modeling frames
train_df <- train %>% select(all_of(c("Pobre", x_cols)))
test_df  <- test  %>% select(all_of(c("id", x_cols)))

# ------------------------
# 4) Handle NAs and types (simple, deterministic)
# ------------------------
# Numerics: median imputation
# Factors/characters: cast to factor; impute mode; align test levels to train
impute_mode <- function(v) {
  tb <- table(v, useNA = "no")
  names(tb)[which.max(tb)]
}
for (nm in names(train_df)) {
  if (nm == "Pobre") next
  if (is.numeric(train_df[[nm]])) {
    med <- median(train_df[[nm]], na.rm = TRUE)
    train_df[[nm]][is.na(train_df[[nm]])] <- med
    if (nm %in% names(test_df)) {
      test_df[[nm]][is.na(test_df[[nm]])] <- med
    }
  } else {
    train_df[[nm]] <- as.factor(train_df[[nm]])
    mode_val <- impute_mode(train_df[[nm]])
    train_df[[nm]][is.na(train_df[[nm]])] <- mode_val
    
    if (nm %in% names(test_df)) {
      test_df[[nm]] <- as.factor(test_df[[nm]])
      # Keep only levels seen in train; map others to "other"/NA safely
      test_df[[nm]] <- forcats::fct_explicit_na(test_df[[nm]], na_level = mode_val)
      test_df[[nm]] <- forcats::fct_other(test_df[[nm]], keep = levels(train_df[[nm]]))
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
      test_df[[nm]][is.na(test_df[[nm]])] <- mode_val
    }
  }
}

# ------------------------
# 5) Training control (10-fold CV, PR metrics)
# ------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = 10,            # 10-fold cross-validation
  classProbs      = TRUE,
  savePredictions = "final",       # keep OOF predictions
  verboseIter     = TRUE,
  summaryFunction = prSummary      # computes AUC-PR, Precision, Recall, and F
)

# ------------------------
# 6) Train LOGIT
# ------------------------
set.seed(2025)
model_logit <- caret::train(
  Pobre ~ .,
  data      = train_df,
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "F"                  # optimize F1 (per prSummary's positive class)
)

print(model_logit)

# (Optional) Cross-validated summary
if ("Accuracy" %in% names(model_logit$resample)) {
  acc_cv <- mean(model_logit$resample$Accuracy)
  cat(sprintf("Cross-Validation Accuracy (10-fold): %.3f\n", acc_cv))
}
if ("F" %in% names(model_logit$resample)) {
  f1_cv <- mean(model_logit$resample$F, na.rm = TRUE)
  cat(sprintf("Cross-Validation F1 (10-fold): %.3f\n", f1_cv))
}

# ------------------------
# 7) OOF-based threshold tuning (maximize F1)
# ------------------------
# Use OOF probabilities to choose a decision threshold that maximizes F1 for "Yes".
oof <- model_logit$pred
oof$obs <- factor(oof$obs, levels = c("No","Yes"))  # align to training levels
stopifnot("Yes" %in% names(oof))                    # caret stores prob cols by class name
probs <- oof$Yes

grid_t <- seq(0.10, 0.90, by = 0.01)
f1_vec <- sapply(grid_t, function(t) {
  pred_lab <- factor(ifelse(probs >= t, "Yes", "No"), levels = c("No","Yes"))
  MLmetrics::F1_Score(y_pred = pred_lab, y_true = oof$obs, positive = "Yes")
})
best_t  <- grid_t[which.max(f1_vec)]
best_f1 <- max(f1_vec, na.rm = TRUE)
cat(sprintf("Optimal OOF threshold (F1): t* = %.2f | F1 = %.3f\n", best_t, best_f1))

# ------------------------
# 8) Test predictions
# ------------------------
# Predict probabilities for "Yes" on the test set, then apply the tuned threshold.
test_probs <- predict(model_logit, newdata = test_df %>% select(all_of(x_cols)), type = "prob")[,"Yes"]

predictClass <- tibble(
  id   = test_df$id,
  poor = as.integer(test_probs >= best_t)  # binary labels for Kaggle (1 = poor)
)

predictProba <- tibble(
  id   = test_df$id,
  poor = test_probs                         # continuous probabilities 
)

print(head(predictClass, 10))

# ------------------------
# 9) Save results
# ------------------------
tstr <- gsub("\\.", "_", sprintf("%.2f", best_t))
name_cls  <- paste0("LOGIT_cv10_thresh_", tstr, ".csv")  # Kaggle submission (id, poor)
name_prob <- "LOGIT_cv10_probs.csv"                      

write.csv(predictClass, name_cls,  row.names = FALSE, fileEncoding = "UTF-8")
write.csv(predictProba, name_prob, row.names = FALSE, fileEncoding = "UTF-8")
cat("Predictions saved to:\n - ", name_cls,  "\n - ", name_prob, "\n")

# ------------------------
# End of script
# ------------------------



