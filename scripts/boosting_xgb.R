# =====================================================
# Boosting for Poverty Prediction 
# =====================================================
# -------------------------------
# -------------------------------
# 0) Setup: Install and Load Packages
# -------------------------------

# -- Basic package management
if (!require("pacman")) install.packages("pacman")

# -- Use pacman to load (and install if missing) all required libraries
pacman::p_load(
  tidyverse,    # Data manipulation and visualization
  caret,        # Machine learning framework
  gbm,          # Gradient Boosting
  adabag,       # AdaBoost
  xgboost,      # XGBoost
  PRROC         # Precision-Recall curves and F1 metrics
)

# 1) -- Ensure rgl is installed (needed for adabag)
if (!require("rgl")) install.packages("rgl", dependencies = TRUE)

# -------------------------------
# 2) Paths and data loading
# -------------------------------
# NOTE: Change this to your local path if needed.
# setwd("c:/Users/Asuar/OneDrive/Escritorio/Libros Clases/Economía/Big Data/Problem-Set-2-Predicting-Poverty")

train_households  <- read.csv("data/train_hogares.csv")
train_persons     <- read.csv(unz("data/train_personas.csv.zip", "train_personas.csv"))
test_households   <- read.csv("data/test_hogares.csv")
test_persons      <- read.csv(unz("data/test_personas.csv.zip", "test_personas.csv"))

cat("Data loaded. Rows (train households/persons/test households/persons):",
    nrow(train_households), nrow(train_persons), nrow(test_households), nrow(test_persons), "\n")

# -------------------------------
# 3) (Optional) Reference poverty variables (kept for reproducibility)
# -------------------------------
# These are auxiliary references; the official target used below is train_households$Pobre
train_households <- train_households |>
  mutate(
    Pobre_hand   = ifelse(Ingpcug < Lp, 1, 0),
    Pobre_hand_2 = ifelse(Ingtotugarr < Lp * Npersug, 1, 0)
  )

# -------------------------------
# 4) Person-level preprocessing
# -------------------------------
# Keep original DANE column names; build English-named derived features where possible.
preprocess_persons <- function(data) {
  data |>
    mutate(
      is_woman       = ifelse(P6020 == 2, 1, 0),
      is_head        = ifelse(P6050 == 1, 1, 0),
      is_minor       = ifelse(P6040 <= 6, 1, 0),
      educ_category  = ifelse(P6210 == 9, 0, P6210),
      is_occupied    = ifelse(is.na(Oc), 0, 1)
    ) |>
    select(id, Orden, is_woman, is_head, is_minor, educ_category, is_occupied)
}

train_persons <- preprocess_persons(train_persons)
test_persons  <- preprocess_persons(test_persons)

# -------------------------------
# 5) Household-level aggregates from persons
# -------------------------------
aggregate_by_household <- function(df) {
  out <- df |>
    group_by(id) |>
    summarize(
      n_women        = sum(is_woman, na.rm = TRUE),
      n_minors       = sum(is_minor, na.rm = TRUE),
      max_educ_cat   = suppressWarnings(max(educ_category, na.rm = TRUE)),
      n_occupied     = sum(is_occupied, na.rm = TRUE),
      mean_educ_cat  = mean(educ_category, na.rm = TRUE),
      dep_ratio      = ifelse(n_occupied > 0, n() / n_occupied, n())
    ) |>
    ungroup()
  out$max_educ_cat[!is.finite(out$max_educ_cat)] <- NA_real_
  out
}

train_persons_hh_agg <- aggregate_by_household(train_persons)
test_persons_hh_agg  <- aggregate_by_household(test_persons)

# Extract head-of-household features and merge with aggregates
pick_head_and_merge <- function(df_people, df_agg) {
  df_people |>
    filter(is_head == 1) |>
    select(id, is_woman, educ_category, is_occupied) |>
    rename(
      head_is_woman   = is_woman,
      head_educ_cat   = educ_category,
      head_is_occupied= is_occupied
    ) |>
    left_join(df_agg, by = "id")
}

train_persons_hh <- pick_head_and_merge(train_persons, train_persons_hh_agg)
test_persons_hh  <- pick_head_and_merge(test_persons,  test_persons_hh_agg)

# -------------------------------
# 6) Household-level economic variables
# -------------------------------
train_households <- train_households |>
  mutate(
    is_renter = ifelse(P5090 == 3, 1, 0),
    Ingpcug   = P5000 / Npersug,
    IPR       = Ingpcug / Lp
  ) |>
  select(id, Dominio, is_renter, Ingpcug, IPR, Pobre)

test_households <- test_households |>
  mutate(
    is_renter = ifelse(P5090 == 3, 1, 0),
    Ingpcug   = P5000 / Npersug,
    IPR       = Ingpcug / Lp
  ) |>
  select(id, Dominio, is_renter, Ingpcug, IPR)

# -------------------------------
# 7) Final merge and factor handling
# -------------------------------
train <- train_households |>
  left_join(train_persons_hh, by = "id") |>
  mutate(
    # Outcome must be a factor with the POSITIVE class first for caret's prSummary
    Pobre   = factor(Pobre, levels = c(1, 0), labels = c("Yes", "No")),
    Dominio = factor(Dominio),
    head_educ_cat = factor(
      head_educ_cat, levels = c(0:6),
      labels = c("No information","None","Preschool","Primary",
                 "Secondary","High school","University")
    )
  ) |>
  select(-id)

test <- test_households |>
  left_join(test_persons_hh, by = "id") |>
  mutate(
    Dominio = factor(Dominio, levels = levels(train$Dominio)),
    head_educ_cat = factor(head_educ_cat, levels = levels(train$head_educ_cat))
  )

cat("Final train/test shapes:", nrow(train), "x", ncol(train), " / ",
    nrow(test), "x", ncol(test), "\n")

# -------------------------------
# 8) Training control (cross-validated F1)
# -------------------------------
set.seed(91519)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = prSummary,   # Returns AUC, Precision, Recall, and F
  classProbs = TRUE,
  savePredictions = "final",     # Keeps OOF predictions for the winning tune
  verboseIter = TRUE
)

# Helper: compute F1 from probabilities at a given threshold
f1_from_probs <- function(obs_factor, prob_yes, thr = 0.5) {
  pred_yes <- ifelse(prob_yes >= thr, "Yes", "No")
  TP <- sum(pred_yes == "Yes" & obs_factor == "Yes")
  FP <- sum(pred_yes == "Yes" & obs_factor == "No")
  FN <- sum(pred_yes == "No"  & obs_factor == "Yes")
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  if ((precision + recall) == 0) return(0)
  2 * precision * recall / (precision + recall)
}

# Helper: find the F1-maximizing threshold using OOF probabilities
find_best_threshold <- function(model, positive_col = "Yes",
                                grid = seq(0.05, 0.95, by = 0.01)) {
  stopifnot(!is.null(model$pred))
  pred <- model$pred
  obs  <- factor(pred$obs, levels = c("Yes","No"))
  prob <- pred[[positive_col]]
  scores <- sapply(grid, function(t) f1_from_probs(obs, prob, t))
  best_i <- which.max(scores)
  list(thresh = grid[best_i], F1 = scores[best_i])
}

# Helpers: formatting for file names
fmt_dec <- function(x) gsub("\\.", "", sprintf("%.2f", x))
mkfile  <- function(prefix, parts) paste0(prefix, "_", paste(parts, collapse = "_"), ".csv")

# -------------------------------
# 9) GBM (gbm) via caret
# -------------------------------
grid_gbm <- expand.grid(
  n.trees = c(50, 100, 150),
  interaction.depth = c(1, 2),
  shrinkage = 0.01,
  n.minobsinnode = c(5, 10)
)

set.seed(91519)
cat("\n[Training] GBM ...\n")

grid_gbm <- expand.grid(
  n.trees = c(50, 100, 150),
  interaction.depth = c(1, 2),
  shrinkage = 0.01,
  n.minobsinnode = c(5, 10)
)
exists("grid_gbm")

GBM_model <- train(
  Pobre ~ .,
  data     = train,
  method   = "gbm",
  trControl= ctrl,
  tuneGrid = grid_gbm,
  metric   = "F",
  verbose  = FALSE
)

gbm_thr <- find_best_threshold(GBM_model)
cat(sprintf("[GBM] Best CV-F1 = %.4f at threshold = %.2f\n", gbm_thr$F1, gbm_thr$thresh))

# -------------------------------
# 10) XGBoost (xgbTree) via caret
# -------------------------------
grid_xbgoost <- expand.grid(
  nrounds = c(250, 500),
  max_depth = c(1, 2),
  eta = c(0.10, 0.01),
  gamma = c(0, 1),
  min_child_weight = c(10, 25),
  colsample_bytree = c(0.4, 0.7),
  subsample = c(0.7)
)

set.seed(91519)
cat("\n[Training] XGBoost ...\n")
XGB_model <- train(
  Pobre ~ .,
  data     = train,
  method   = "xgbTree",
  trControl= ctrl,
  tuneGrid = grid_xbgoost,
  metric   = "F",
  verbosity= 0
)
xgb_thr <- find_best_threshold(XGB_model)
cat(sprintf("[XGBoost] Best CV-F1 = %.4f at threshold = %.2f\n", xgb_thr$F1, xgb_thr$thresh))

# -------------------------------
# 11) Keep ONLY the best model and write a single Kaggle CSV
# -------------------------------

cleanup_previous <- TRUE
if (cleanup_previous) {
  old <- list.files(pattern = "^(GBM|XGB)_.*\\.csv$")
  if (length(old)) file.remove(old)
}

# Show thresholds for both models
print(gbm_thr)
print(xgb_thr)

# Compare F1 scores and select best model
cv_scores <- tibble(
  alg = c("GBM", "XGB"),
  F1  = c(gbm_thr$F1, xgb_thr$F1)
) |> arrange(desc(F1))

best_alg <- cv_scores$alg[1]
best_F1  <- cv_scores$F1[1]
cat(sprintf("\n>>> BEST MODEL by CV-F1: %s (F1 = %.4f)\n", best_alg, best_F1))

# -------------------------------
# Prepare test data exactly like training
# -------------------------------
categorical_vars <- c("Dominio", "head_educ_cat")
train_levels <- lapply(XGB_model$trainingData[, categorical_vars], levels)

for (v in categorical_vars) {
  test[[v]] <- as.character(test[[v]])
  test[[v]][is.na(test[[v]])] <- "Unknown"
  test[[v]] <- factor(test[[v]], levels = train_levels[[v]])
}

# Create dummy variables
dummies <- model.matrix(~ Dominio + head_educ_cat - 1, data = test)

# Numeric variables
numeric_vars <- setdiff(names(test), categorical_vars)
test_numeric <- test[, numeric_vars, drop = FALSE]

# Combine numeric and dummy variables
test_x <- cbind(test_numeric, dummies)

# -------------------------------
# Align test columns with model features
# -------------------------------
model_features <- XGB_model$finalModel$feature_names
missing <- setdiff(model_features, colnames(test_x))
if (length(missing)) {
  cat("⚠️ Adding", length(missing), "missing columns to test data.\n")
  for (m in missing) test_x[[m]] <- 0
}

# Reorder columns to match model
test_x <- test_x[, model_features, drop = FALSE]

# Convert to matrix for XGBoost
test_x_mat <- as.matrix(test_x)

# -------------------------------
# Generate predictions for the best model
# -------------------------------
if (best_alg == "GBM") {
  bt <- GBM_model$bestTune
  prob <- predict(GBM_model$finalModel, newdata = test_x, n.trees = bt$n.trees, type = "response")
  thr <- gbm_thr$thresh
  fname <- mkfile("GBM", c(
    paste0("ntrees_", bt$n.trees),
    paste0("depth_",  bt$interaction.depth),
    paste0("sh_",     fmt_dec(bt$shrinkage)),
    paste0("minobs_", bt$n.minobsinnode),
    paste0("th_",     fmt_dec(thr))
  ))
} else { # XGB
  bt <- XGB_model$bestTune
  prob <- predict(XGB_model$finalModel, newdata = test_x_mat)
  thr <- xgb_thr$thresh
  fname <- mkfile("XGB", c(
    paste0("nrounds_",    bt$nrounds),
    paste0("maxdepth_",   bt$max_depth),
    paste0("eta_",        fmt_dec(bt$eta)),
    paste0("gamma_",      bt$gamma),
    paste0("minchild_",   bt$min_child_weight),
    paste0("colsample_",  fmt_dec(bt$colsample_bytree)),
    paste0("subsample_",  fmt_dec(bt$subsample)),
    paste0("th_",         fmt_dec(thr))
  ))
}

# -------------------------------
# Apply threshold and export CSV
# -------------------------------
labels <- ifelse(prob > thr, 1, 0)
stopifnot(length(labels) == nrow(test))

write.csv(tibble(id = test$id, pobre = labels), fname, row.names = FALSE)
cat("Submission CSV created. Upload to Kaggle ->", fname, "\n")

