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
# 5.1. Household size and composition variables
# =====================================================

# Variables adicionales basadas en la cantidad y estructura del hogar:
#   - n_personas: total de personas en el hogar
#   - n_menores18: número de personas menores de 18 años
#   - n_mayores65: número de personas mayores de 65 años
#   - prop_menores: proporción de menores de edad
#   - prop_mayores: proporción de adultos mayores
#   - adult_equiv: escala de equivalencia (ajuste demográfico)
#   - densidad_personas: número de personas por dormitorio (si hay variable de cuartos)
#   - adultos_depend: proporción de dependientes (menores+mayores) sobre adultos

# ----- Entrenamiento -----
train_personas_hogar_extra <- train_personas |> 
  group_by(id) |> 
  summarize(
    n_personas     = n(),
    n_menores18    = sum(P6040 < 18, na.rm = TRUE),
    n_mayores65    = sum(P6040 >= 65, na.rm = TRUE),
    prop_menores   = n_menores18 / n_personas,
    prop_mayores   = n_mayores65 / n_personas,
    adult_equiv    = n_menores18 * 0.5 + (n_personas - n_menores18) * 1,
    adultos_depend = (n_menores18 + n_mayores65) / pmax(n_personas - n_menores18 - n_mayores65, 1)
  ) |> 
  ungroup()

# ----- Test -----
test_personas_hogar_extra <- test_personas |> 
  group_by(id) |> 
  summarize(
    n_personas     = n(),
    n_menores18    = sum(P6040 < 18, na.rm = TRUE),
    n_mayores65    = sum(P6040 >= 65, na.rm = TRUE),
    prop_menores   = n_menores18 / n_personas,
    prop_mayores   = n_mayores65 / n_personas,
    adult_equiv    = n_menores18 * 0.5 + (n_personas - n_menores18) * 1,
    adultos_depend = (n_menores18 + n_mayores65) / pmax(n_personas - n_menores18 - n_mayores65, 1)
  ) |> 
  ungroup()

# Merge con las variables de nivel hogar ya creadas
train_personas_hogar <- left_join(train_personas_hogar, train_personas_hogar_extra, by = "id")
test_personas_hogar  <- left_join(test_personas_hogar,  test_personas_hogar_extra,  by = "id")


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
    Pobre   = factor(Pobre, levels = c(1, 0), labels = c("Yes", "No")),
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
# 7.1. Create interaction variables
# =====================================================
# Create meaningful socioeconomic interactions to capture nonlinear relationships:
#   - educ_ipr: interaction between household education and economic capacity
#   - headwoman_dep: female head of household × dependency ratio
#   - rent_ipr: rental status × economic capacity
#   - educ_dep: average education × dependency ratio

train <- train |> 
  mutate(
    educ_ipr      = mean_educ * IPR,
    headwoman_dep = bin_headWoman * dep_ratio,
    rent_ipr      = bin_rent * IPR,
    educ_dep      = mean_educ * dep_ratio
  )

test <- test |> 
  mutate(
    educ_ipr      = mean_educ * IPR,
    headwoman_dep = bin_headWoman * dep_ratio,
    rent_ipr      = bin_rent * IPR,
    educ_dep      = mean_educ * dep_ratio
  )

# ------------------------
# 8. Model training (Elastic Net)
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
# 9. Generate predictions on the test set
# ------------------------
predictSample <- test |> 
  mutate(pobre_lab = predict(model1, newdata = test, type = "raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes", 1, 0)) |>
  select(id, pobre)

head(predictSample)

# ------------------------
# 10. Save predictions with dynamic filename
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
