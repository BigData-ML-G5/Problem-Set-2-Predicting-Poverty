# =====================================================
# Predicción de la Pobreza en Colombia
# Autor: Andrez Guerrero
# Fecha: 2025-10-01
# Descripción: Este script predice la pobreza en Colombia
# utilizando información de hogares y personas con modelos
# de Machine Learning (Elastic Net con caret).
# =====================================================

# ------------------------
# 1. Cargar librerías
# ------------------------
require("pacman")
p_load(tidyverse, 
       glmnet,
       caret,
       MLmetrics, 
       Metrics
)

# ------------------------
# 2. Cargar datos
# ------------------------
train_hogares  <- read.csv("train_hogares.csv")
train_personas <- read.csv("train_personas.csv")
test_hogares   <- read.csv("test_hogares.csv")
test_personas  <- read.csv("test_personas.csv")

# ------------------------
# 3. Variables de pobreza
# ------------------------
# Usamos línea de pobreza (Lp) para construir variable binaria
train_hogares <- train_hogares |> 
  mutate(Pobre_hand = ifelse(Ingpcug < Lp, 1, 0),
         Pobre_hand_2 = ifelse(Ingtotugarr < Lp*Npersug,1,0))

# Variable oficial (DANE) vs manual
table(train_hogares$Pobre, train_hogares$Pobre_hand)
table(train_hogares$Pobre, train_hogares$Pobre_hand_2)

# ------------------------
# 4. Preprocesamiento de personas
# ------------------------
pre_process_personas <- function(data){
  data |> 
    mutate(bin_woman = ifelse(P6020==2,1,0), 
           bin_head  = ifelse(P6050== 1, 1, 0),
           bin_minor = ifelse(P6040<=6,1,0), 
           cat_educ  = ifelse(P6210==9,0,P6210), # 9 = categoría sin valor
           bin_occupied = ifelse(is.na(Oc),0,1)) |> 
    select(id, Orden, bin_woman, bin_head, bin_minor, cat_educ, bin_occupied)
}

train_personas <- pre_process_personas(train_personas)
test_personas  <- pre_process_personas(test_personas)

# ------------------------
# 5. Variables a nivel hogar desde personas
# ------------------------
train_personas_nivel_hogar <- train_personas |> 
  group_by(id) |>
  summarize(num_women    = sum(bin_woman, na.rm=TRUE),
            num_minors   = sum(bin_minor, na.rm=TRUE),
            cat_maxEduc  = max(cat_educ, na.rm=TRUE),
            num_occupied = sum(bin_occupied, na.rm=TRUE)) |> 
  ungroup()

train_personas_hogar <- train_personas |> 
  filter(bin_head == 1) |>
  select(id, bin_woman, cat_educ, bin_occupied) |>
  rename(bin_headWoman   = bin_woman,
         cat_educHead    = cat_educ,
         bin_occupiedHead= bin_occupied) |>
  left_join(train_personas_nivel_hogar)

test_personas_nivel_hogar <- test_personas |> 
  group_by(id) |>
  summarize(num_women    = sum(bin_woman, na.rm=TRUE),
            num_minors   = sum(bin_minor, na.rm=TRUE),
            cat_maxEduc  = max(cat_educ, na.rm=TRUE),
            num_occupied = sum(bin_occupied, na.rm=TRUE)) |> 
  ungroup()

test_personas_hogar <- test_personas |> 
  filter(bin_head == 1) |>
  select(id, bin_woman, cat_educ, bin_occupied) |>
  rename(bin_headWoman   = bin_woman,
         cat_educHead    = cat_educ,
         bin_occupiedHead= bin_occupied) |>
  left_join(test_personas_nivel_hogar)

# ------------------------
# 6. Variables de hogares
# ------------------------
train_hogares <- train_hogares |> 
  mutate(bin_rent = ifelse(P5090 == 3,1,0)) |> 
  select(id, Dominio, bin_rent, Pobre)

test_hogares <- test_hogares |> 
  mutate(bin_rent = ifelse(P5090 == 3,1,0)) |> 
  select(id, Dominio, bin_rent)

# ------------------------
# 7. Unir personas y hogares
# ------------------------
train <- train_hogares |> 
  left_join(train_personas_hogar) |>
  select(-id) |> 
  mutate(Pobre   = factor(Pobre, levels=c(0,1), labels=c("No","Yes")),
         Dominio = factor(Dominio),
         cat_educHead = factor(cat_educHead, levels = c(0:6),
           labels=c("No sabe",'Ninguno', 'Preescolar', 'Primaria',
                    'Secundaria', 'Media', 'Universitaria')))

test <- test_hogares |> 
  left_join(test_personas_hogar) |> 
  mutate(Dominio = factor(Dominio),
         cat_educHead = factor(cat_educHead, levels = c(0:6),
           labels=c("No sabe",'Ninguno', 'Preescolar', 'Primaria',
                    'Secundaria', 'Media', 'Universitaria')))

# ------------------------
# 8. Entrenamiento del modelo (Elastic Net)
# ------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = prSummary,
  savePredictions = TRUE
)

set.seed(2025)
model1 <- train(
  Pobre ~ .,
  data = train,
  metric = "F",
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  tuneGrid = expand.grid(
    alpha  = seq(0, 1, by=0.1),
    lambda = 10^seq(-3, 3, length = 10)
  )   
)

print(model1)

# ------------------------
# 9. Predicciones en test
# ------------------------
predictSample <- test |> 
  mutate(pobre_lab = predict(model1, newdata=test, type="raw")) |>
  mutate(pobre = ifelse(pobre_lab == "Yes",1,0)) |>
  select(id, pobre)

head(predictSample)

# ------------------------
# 10. Guardar resultados con nombre dinámico
# ------------------------
lambda_str <- gsub("\\.", "_", as.character(round(model1$bestTune$lambda, 4)))
alpha_str  <- gsub("\\.", "_", as.character(model1$bestTune$alpha))

name <- paste0("EN_lambda_", lambda_str,
               "_alpha_", alpha_str, ".csv")

write.csv(predictSample, name, row.names = FALSE)
cat("Predicciones guardadas en:", name, "\n")

# ------------------------
# Fin del script
# ------------------------
