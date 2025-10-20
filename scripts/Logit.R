# =======================================================
# Poverty Prediction in Colombia — LOGIT Model (caret)
# Date: 2025-10-19
# =======================================================

# ------------------------
# 1) Load libraries
# ------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
suppressPackageStartupMessages({
  pacman::p_load(
    tidyverse,   # dplyr, tidyr, readr, ggplot2, forcats
    caret,
    MLmetrics    # F1, Precision, Recall helpers
    # Metrics     # (not used below; keep commented unless needed)
  )
})

# ------------------------
# 2) Load data
# ------------------------
# Expect train_clean.csv and test_clean.csv in the working directory.
train <- read.csv("train_clean.csv", na.strings = c("", "NA", "NaN"))
test  <- read.csv("test_clean.csv",  na.strings = c("", "NA", "NaN"))

# Basic checks
stopifnot("id"    %in% names(test))
stopifnot("Pobre" %in% names(train))

# ------------------------
# 3) Target encoding & basic cleaning
# ------------------------
# Ensure a binary factor compatible with caret + prSummary.
# IMPORTANT: put "Yes" FIRST so prSummary treats it as the positive class.
if (is.numeric(train$Pobre)) {
  train$Pobre <- factor(ifelse(train$Pobre == 1, "Yes", "No"), levels = c("Yes","No"))
} else {
  train$Pobre <- tolower(as.character(train$Pobre))
  train$Pobre <- ifelse(train$Pobre %in% c("1","yes","si","sí","y","true"), "Yes", "No")
  train$Pobre <- factor(train$Pobre, levels = c("Yes","No"))
}

# Remove identifiers / leakage columns from predictors
cols_leak <- c("id", "Pobre")
x_cols <- setdiff(names(train), cols_leak)

# Align columns between TRAIN and TEST
common_cols <- intersect(x_cols, setdiff(names(test), "id"))
if (length(common_cols) == 0) stop("No common columns between train and test (except id). Check preprocessing.")
x_cols <- common_cols

# Final modeling frames
train_df <- train %>% dplyr::select(all_of(c("Pobre", x_cols)))
test_df  <- test  %>% dplyr::select(all_of(c("id", x_cols)))

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
      # Keep only levels seen in train; map unseen to mode to avoid NAs at predict-time
      test_df[[nm]] <- forcats::fct_explicit_na(test_df[[nm]], na_level = mode_val)
      test_df[[nm]] <- forcats::fct_other(test_df[[nm]],
                                          keep = levels(train_df[[nm]]),
                                          other_level = mode_val)
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
      test_df[[nm]][is.na(test_df[[nm]])] <- mode_val
    }
  }
}

# ------------------------
# 5) Cross-validation control (stratified 10-fold, PR metrics)
# ------------------------
set.seed(2025)  # reproducible folds for all models
folds <- createFolds(train_df$Pobre, k = 10, returnTrain = TRUE)  # stratified

ctrl <- trainControl(
  method          = "cv",
  number          = 10,            # 10-fold CV
  classProbs      = TRUE,
  savePredictions = "final",       # keep out-of-fold predictions
  verboseIter     = TRUE,
  summaryFunction = prSummary,     # PR-AUC (AUC), Precision, Recall, and F
  index           = folds          # ensure same folds across models for comparability
)

# ------------------------
# 6) Train LOGIT (unweighted baseline)
# ------------------------
set.seed(2025)
model_logit <- caret::train(
  Pobre ~ .,
  data      = train_df,
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "F"                  # optimize F1 (positive class = "Yes")
)

print(model_logit)

# ------------------------
# 7) OOF-based threshold tuning for baseline (maximize F1)
# ------------------------
# Use out-of-fold probabilities to choose a decision threshold that maximizes F1.
oof <- model_logit$pred
# Ensure "Yes" (positive) is first level here as well
oof$obs <- factor(as.character(oof$obs), levels = c("Yes","No"))
stopifnot("Yes" %in% names(oof))    # caret stores prob columns by class name
probs <- oof$Yes

grid_t <- seq(0.10, 0.90, by = 0.01)
f1_vec <- sapply(grid_t, function(t) {
  pred_lab <- factor(ifelse(probs >= t, "Yes", "No"), levels = c("Yes","No"))
  MLmetrics::F1_Score(y_pred = pred_lab, y_true = oof$obs, positive = "Yes")
})
best_t  <- grid_t[which.max(f1_vec)]
best_f1 <- max(f1_vec, na.rm = TRUE)
cat(sprintf("Baseline LOGIT — optimal OOF threshold (F1): t* = %.2f | F1 = %.3f\n", best_t, best_f1))

# ------------------------
# 8) Test predictions (baseline)
# ------------------------
# Predict probabilities for "Yes" on the test set, then apply the tuned threshold.
test_probs <- predict(model_logit, newdata = test_df %>% dplyr::select(all_of(x_cols)), type = "prob")[,"Yes"]

predictClass <- tibble(
  id   = test_df$id,
  poor = as.integer(test_probs >= best_t)  # Kaggle: 1 = poor
)

predictProba <- tibble(
  id   = test_df$id,
  poor = test_probs                         # continuous probabilities
)

print(head(predictClass, 10))

# Save baseline predictions
tstr <- gsub("\\.", "_", sprintf("%.2f", best_t))
name_cls  <- paste0("LOGIT_cv10_thresh_", tstr, ".csv")
name_prob <- "LOGIT_cv10_probs.csv"
write.csv(predictClass, name_cls,  row.names = FALSE, fileEncoding = "UTF-8")
write.csv(predictProba, name_prob, row.names = FALSE, fileEncoding = "UTF-8")
cat("Baseline predictions saved to:\n - ", name_cls,  "\n - ", name_prob, "\n")

# ------------------------
# 9) Summaries for Section 3.2 (baseline)
# ------------------------
# (1) Variables used
vars_used <- x_cols
n_vars    <- length(vars_used)
cat(sprintf("Variables used: %d\n", n_vars))
write.csv(data.frame(var = vars_used), "logit_vars_used.csv", row.names = FALSE)

# (2) Class balance in training
class_bal <- prop.table(table(train_df$Pobre))
print(class_bal)
write.csv(data.frame(class = names(class_bal), share = as.numeric(class_bal)),
          "logit_class_balance.csv", row.names = FALSE)

# (3) CV(10) mean metrics from prSummary (guidance)
cv_means <- model_logit$resample %>%
  summarize(
    F1        = mean(F,         na.rm = TRUE),
    Precision = mean(Precision, na.rm = TRUE),
    Recall    = mean(Recall,    na.rm = TRUE),
    PRAUC     = mean(AUC,       na.rm = TRUE)   # in prSummary, AUC = PR-AUC
  )
print(cv_means)
write.csv(cv_means, "logit_cv10_means.csv", row.names = FALSE)

# (4–6) OOF threshold tuning outputs: best row and comparison vs t=0.50
score_at <- function(t) {
  pred_lab <- factor(ifelse(probs >= t, "Yes", "No"), levels = c("Yes","No"))
  data.frame(
    Threshold = t,
    Precision = MLmetrics::Precision(y_pred = pred_lab, y_true = oof$obs, positive = "Yes"),
    Recall    = MLmetrics::Recall(   y_pred = pred_lab, y_true = oof$obs, positive = "Yes"),
    F1        = MLmetrics::F1_Score( y_pred = pred_lab, y_true = oof$obs, positive = "Yes")
  )
}
best_row <- score_at(best_t)
tab_compare <- rbind(
  cbind(Model = "Logit (t=0.50)", score_at(0.50)),
  cbind(Model = sprintf("Logit (t*=%.2f)", best_t), best_row)
)
write.csv(best_row,    "logit_best_threshold_metrics.csv", row.names = FALSE)
write.csv(tab_compare, "logit_threshold_compare.csv",      row.names = FALSE)

# Also store an RDS bundle if needed later
summary_3_2 <- list(
  n_vars = n_vars,
  vars_used = vars_used,
  class_balance = class_bal,
  cv_means = cv_means,
  grid = list(min = 0.10, max = 0.90, step = 0.01, strategy = "OOF grid search"),
  best_threshold_row = best_row,
  compare_table = tab_compare
)
saveRDS(summary_3_2, "logit_3_2_summary.rds")
cat("Baseline 3.2 bundles written (CSV + RDS).\n")

# =======================================================
# 10) LOGIT with class-imbalance correction (weights) + tuning
#     Goal: maximize OOF F1 by tuning positive-class weight ratio and threshold
# =======================================================

# A) Use the SAME folds as baseline (object 'folds') for apples-to-apples comparison
ctrl_w <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  savePredictions = "final",
  verboseIter     = TRUE,
  summaryFunction = prSummary,
  index           = folds
)

# B) Balanced base weights: each class contributes ~0.5 of total weight
p_yes <- mean(train_df$Pobre == "Yes")
w_equal <- ifelse(train_df$Pobre == "Yes", 0.5 / p_yes, 0.5 / (1 - p_yes))
w_equal <- w_equal / mean(w_equal)  # normalize so mean weight ≈ 1

# C) Grid for additional relative weight on "Yes"
ratio_grid <- c(0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00, 5.00)

score_ratio <- function(ratio) {
  # compose per-observation weights for this ratio
  w <- ifelse(train_df$Pobre == "Yes", w_equal * ratio, w_equal)
  
  set.seed(2025)
  fit <- caret::train(
    Pobre ~ ., data = train_df,
    method = "glm", family = binomial(link = "logit"),
    trControl = ctrl_w, metric = "F",
    weights = w
  )
  
  # OOF probabilities to tune decision threshold t
  oof_w <- fit$pred
  oof_w$obs <- factor(as.character(oof_w$obs), levels = c("Yes","No"))
  stopifnot("Yes" %in% names(oof_w))
  probs_w <- oof_w$Yes
  
  grid_t <- seq(0.10, 0.90, by = 0.01)
  score_at_w <- function(t) {
    pred <- factor(ifelse(probs_w >= t, "Yes", "No"), levels = c("Yes","No"))
    c(
      Precision = MLmetrics::Precision(pred, oof_w$obs, positive = "Yes"),
      Recall    = MLmetrics::Recall(   pred, oof_w$obs, positive = "Yes"),
      F1        = MLmetrics::F1_Score( pred, oof_w$obs, positive = "Yes")
    )
  }
  mat <- sapply(grid_t, score_at_w)
  j   <- which.max(mat["F1", ])
  best_t_w <- grid_t[j]
  
  list(
    ratio    = ratio,
    fit      = fit,
    best_t   = best_t_w,
    prec     = unname(mat["Precision", j]),
    rec      = unname(mat["Recall", j]),
    f1       = unname(mat["F1", j]),
    cv_means = summarize(fit$resample,
                         F1 = mean(F, na.rm=TRUE),
                         Precision = mean(Precision, na.rm=TRUE),
                         Recall = mean(Recall, na.rm=TRUE),
                         PRAUC = mean(AUC, na.rm=TRUE))
  )
}

# D) Run the ratio grid and pick the best by OOF F1 (with tuned threshold)
grid_results <- lapply(ratio_grid, score_ratio)
best_idx <- which.max(sapply(grid_results, function(z) z$f1))
best     <- grid_results[[best_idx]]

cat(sprintf("\n>> WEIGHTED LOGIT — best ratio*: %.2f | best t*: %.2f | F1_OOF=%.3f | P=%.3f | R=%.3f\n",
            best$ratio, best$best_t, best$f1, best$prec, best$rec))

# E) Summary table across ratios (ranked by OOF F1)
ratio_table <- do.call(rbind, lapply(grid_results, function(z)
  data.frame(
    ratio = z$ratio,
    best_t = round(z$best_t, 2),
    Precision = round(z$prec, 3),
    Recall = round(z$rec, 3),
    F1 = round(z$f1, 3)
  )))
ratio_table <- ratio_table[order(-ratio_table$F1), ]
write.csv(ratio_table, "logit_weight_tuning.csv", row.names = FALSE)
print(ratio_table)

# F) Final weighted model and tuned threshold
model_logit_w <- best$fit
best_t_w      <- best$best_t

# G) Test predictions for the weighted model
test_probs_w <- predict(model_logit_w, newdata = test_df, type = "prob")[,"Yes"]
predClass_w  <- tibble(id = test_df$id, poor = as.integer(test_probs_w >= best_t_w))
predProba_w  <- tibble(id = test_df$id, poor = test_probs_w)

fname_cls_w  <- sprintf("LOGIT_w_ratio_%.2f_t_%.2f.csv", best$ratio, best_t_w)
fname_prob_w <- sprintf("LOGIT_w_ratio_%.2f_probs.csv", best$ratio)

write.csv(predClass_w, fname_cls_w,  row.names = FALSE, fileEncoding = "UTF-8")
write.csv(predProba_w, fname_prob_w, row.names = FALSE, fileEncoding = "UTF-8")
cat("Weighted predictions saved to:\n - ", fname_cls_w, "\n - ", fname_prob_w, "\n")

# H) Threshold comparison (t=0.50 vs tuned t*) for the selected weighted model
oof_w <- model_logit_w$pred
oof_w$obs <- factor(as.character(oof_w$obs), levels = c("Yes","No"))
probs_w <- oof_w$Yes
score_row_w <- function(t){
  pred <- factor(ifelse(probs_w >= t, "Yes","No"), levels=c("Yes","No"))
  data.frame(
    Model     = sprintf("Logit w (t=%.2f)", t),
    Threshold = t,
    Precision = MLmetrics::Precision(pred, oof_w$obs, positive="Yes"),
    Recall    = MLmetrics::Recall(   pred, oof_w$obs, positive="Yes"),
    F1        = MLmetrics::F1_Score( pred, oof_w$obs, positive="Yes")
  )
}
tab_compare_w <- rbind(
  score_row_w(0.50),
  data.frame(Model=sprintf("Logit w (t* = %.2f)", best_t_w),
             Threshold=best_t_w,
             Precision=best$prec, Recall=best$rec, F1=best$f1)
)
write.csv(tab_compare_w, "logit_w_threshold_compare.csv", row.names = FALSE)
print(tab_compare_w)

# ------------------------
# 9) Summaries for Section 3.2 (exports)
# ------------------------
# (1) Variables used
vars_used <- x_cols
n_vars    <- length(vars_used)
cat(sprintf("Variables used: %d\n", n_vars))
write.csv(data.frame(var = vars_used), "logit_vars_used.csv", row.names = FALSE)

# (2) Class balance in training
class_bal <- prop.table(table(train_df$Pobre))
print(class_bal)
write.csv(data.frame(class = names(class_bal), share = as.numeric(class_bal)),
          "logit_class_balance.csv", row.names = FALSE)

# (3) CV(10) mean metrics from prSummary (guidance)
cv_means <- model_logit$resample %>%
  summarize(
    F1        = mean(F,         na.rm = TRUE),
    Precision = mean(Precision, na.rm = TRUE),
    Recall    = mean(Recall,    na.rm = TRUE),
    PRAUC     = mean(AUC,       na.rm = TRUE)   # in prSummary, AUC = PR-AUC
  )
print(cv_means)
write.csv(cv_means, "logit_cv10_means.csv", row.names = FALSE)

# (4) OOF threshold tuning comparison: t=0.50 vs t*
tab_compare <- rbind(
  cbind(Model = "Logit (t=0.50)", score_at(0.50)),
  cbind(Model = sprintf("Logit (t*=%.2f)", best_t), best_row)
)
write.csv(best_row,    "logit_best_threshold_metrics.csv", row.names = FALSE)
write.csv(tab_compare, "logit_threshold_compare.csv",      row.names = FALSE)

# Bundle (optional)
summary_3_2 <- list(
  n_vars = n_vars,
  vars_used = vars_used,
  class_balance = class_bal,
  cv_means = cv_means,
  grid = list(min = min(grid_t), max = max(grid_t), step = 0.01, strategy = "OOF grid search"),
  best_threshold_row = best_row,
  compare_table = tab_compare
)
saveRDS(summary_3_2, "logit_3_2_summary.rds")
cat("Section 3.2 bundles written (CSV + RDS).\n")

# =======================================================
# 10) Feature Importance / Influence (WEIGHTED)
# =======================================================

# Tidy GLM coefficients (exclude intercept), compute |z| and ORs with 95% CI
tidy_glm_terms <- function(fit) {
  tt <- broom::tidy(fit$finalModel)
  tt <- tt[tt$term != "(Intercept)", , drop = FALSE]
  if (!"statistic" %in% names(tt)) stop("No 'statistic' (z-value) in tidy output; check model/glm summary.")
  tt$abs_z <- abs(tt$statistic)
  tt$OR    <- exp(tt$estimate)
  tt$OR_lo <- exp(tt$estimate - 1.96 * tt$std.error)
  tt$OR_hi <- exp(tt$estimate + 1.96 * tt$std.error)
  rownames(tt) <- NULL
  tt
}

# Safe helper: get factor variable NAMES (character) from a glm fitted by caret
get_factor_names <- function(fit) {
  xl <- fit$finalModel$xlevels
  if (is.null(xl)) return(character(0))
  if (is.list(xl)) {
    nn <- names(xl)
    if (is.null(nn)) character(0) else as.character(nn)
  } else {
    as.character(xl)  # fallback
  }
}

# Map coefficient term back to original variable group (robust, strips backticks)
group_of_term <- function(term, factor_names) {
  if (grepl(":", term, fixed = TRUE)) return(term)  # keep interactions as their own groups
  t2 <- gsub("`", "", term, fixed = TRUE)
  idx <- if (length(factor_names)) vapply(factor_names, function(v) startsWith(t2, v), logical(1)) else logical(0)
  hits <- if (length(idx)) factor_names[idx] else character(0)
  if (length(hits) >= 1) return(hits[1])
  # Fallback: take alphanumeric prefix (works for numeric vars and most engineered terms)
  sub("^([A-Za-z0-9_\\.]+).*$", "\\1", t2)
}

# Aggregate importance by original variable (sum |z|; direction by signed |z|)
group_varimp <- function(tt, fit) {
  factor_names <- get_factor_names(fit)
  grp <- vapply(tt$term, group_of_term, character(1), factor_names = factor_names)
  tt$group <- grp
  agg <- tt %>%
    dplyr::group_by(group) %>%
    dplyr::summarize(
      importance_raw  = sum(abs_z, na.rm = TRUE),
      direction_score = sum(sign(estimate) * abs_z, na.rm = TRUE),
      top_term        = term[which.max(abs_z)],
      .groups = "drop"
    )
  mx <- max(agg$importance_raw, na.rm = TRUE)
  agg$Overall   <- if (mx > 0) 100 * agg$importance_raw / mx else 0
  agg$Direction <- ifelse(agg$direction_score >= 0, "positive", "negative")
  agg <- agg[order(-agg$Overall), ]
  agg
}

# Build and export variable-importance artifacts (UNWEIGHTED)
cat("\n[Feature importance] Computing grouped and term-level importance...\n")
tt_base  <- tidy_glm_terms(model_logit)
agg_base <- group_varimp(tt_base, model_logit)

# Export CSVs
readr::write_csv(
  tt_base %>% dplyr::select(term, estimate, std.error, statistic, abs_z, OR, OR_lo, OR_hi),
  "logit_varimp_terms.csv"
)
readr::write_csv(
  agg_base %>% dplyr::select(group, Overall, Direction, top_term, importance_raw, direction_score),
  "logit_varimp_grouped.csv"
)

# Optional: Top-20 barplot (needs ggplot2 from tidyverse)
top_k <- head(agg_base, 20)
p <- ggplot2::ggplot(top_k, ggplot2::aes(x = reorder(group, Overall), y = Overall, fill = Direction)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Variable (grouped)", y = "Relative importance (0–100)",
                title = "Top-20 variable importance — weighted Logit") +
  ggplot2::guides(fill = ggplot2::guide_legend(title = "Direction")) +
  ggplot2::theme_minimal(base_size = 12)
ggplot2::ggsave("logit_varimp_top20.png", p, width = 8, height = 6, dpi = 300)

cat(">> Exported: logit_varimp_terms.csv, logit_varimp_grouped.csv, logit_varimp_top20.png\n")

# ------------------------
# End of script
# ------------------------