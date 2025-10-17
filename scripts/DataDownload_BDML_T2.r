# -----------------------------------------------------
# 0) Good practices, clean variables and libraries
# -----------------------------------------------------
# Clean variables and Libraries
rm(list = ls())

setwd("~/Desktop/GitHub/DATA BDSMT2")

require("pacman")
p_load(tidyverse, 
       glmnet,
       caret,
       readr,
       dplyr,
       skimr
)

## -----------------------------------------------------
## 1) Import Databases
## -----------------------------------------------------
train_personas <- read.csv("train_personas.csv")
test_hogares  <- read.csv("test_hogares.csv")
test_personas <- read.csv("test_personas.csv")
train_hogares <- read.csv("train_hogares.csv")

## -----------------------------------------------------
## 2) Rename Database
## -----------------------------------------------------
#Check the existing names
names(test_hogares)
names(test_personas)
names(train_hogares)
names(train_personas)

# Rename function: applies dictionary mapping to column names. 
# Only renames columns that exist in the dataset.

dictionary <- list(
  "sex" = "hombre",
  "P6050" = "parentesco_jefe_hogar",
  "P6090" = "afiliado_seguridad_social",
  "P6100" = "regimen_salud",
  "P6210" = "nivel_educativo",
  "P6210s1" = "grado_aprobado",
  "P6240" = "actividad_semana_pasada",
  "P6426" = "tiempo_empresa_actual",
  "P6500" = "ingreso_trabajo_principal",
  "P6510" = "recibio_horas_extras",
  "P6510s1" = "valor_horas_extras",
  "P6510s2" = "incluyo_horas_extras",
  "P6545" = "recibio_primas",
  "P6545s1" = "valor_primas",
  "P6545s2" = "incluyo_primas",
  "P6580" = "recibio_bonificaciones",
  "P6580s1" = "valor_bonificaciones",
  "P6580s2" = "incluyo_bonificaciones",
  "P6585s1" = "subsidio_alimentacion",
  "P6585s1a1" = "valor_subsidio_alimentacion",
  "P6585s1a2" = "incluyo_subsidio_alimentacion",
  "P6585s2" = "subsidio_transporte",
  "P6585s2a1" = "valor_subsidio_transporte",
  "P6585s2a2" = "incluyo_subsidio_transporte",
  "P6585s3" = "subsidio_familiar",
  "P6585s3a1" = "valor_subsidio_familiar",
  "P6585s3a2" = "incluyo_subsidio_familiar",
  "P6585s4" = "subsidio_educativo",
  "P6585s4a1" = "valor_subsidio_educativo",
  "P6585s4a2" = "incluyo_subsidio_educativo",
  "P6590" = "recibio_alimentos_trabajo",
  "P6590s1" = "valor_alimentos_trabajo",
  "P6600" = "recibio_vivienda_trabajo",
  "P6600s1" = "valor_vivienda_trabajo",
  "P6610" = "transporte_empresa",
  "P6610s1" = "valor_transporte_empresa",
  "P6620" = "otros_ingresos_especie",
  "P6620s1" = "valor_otros_especie",
  "P6630s1" = "prima_servicios",
  "P6630s1a1" = "valor_prima_servicios",
  "P6630s2" = "prima_navidad",
  "P6630s2a1" = "valor_prima_navidad",
  "P6630s3" = "prima_vacaciones",
  "P6630s3a1" = "valor_prima_vacaciones",
  "P6630s4" = "viaticos_permanentes",
  "P6630s4a1" = "valor_viaticos_permanentes",
  "P6630s6" = "bonificaciones_anuales",
  "P6630s6a1" = "valor_bonificaciones_anuales",
  "P6750" = "ganancia_neta_mes",
  "P6760" = "meses_ganancia_corresponde",
  "P550" = "ganancia_neta_12_meses",
  "P6870" = "tamano_empresa",
  "P6920" = "cotiza_pension_actual",
  "P7040" = "tiene_segundo_trabajo",
  "P7050" = "posicion_segundo_trabajo",
  "P7070" = "ingreso_segundo_trabajo",
  "P7090" = "quiere_trabajar_mas_horas",
  "P7110" = "hizo_diligencias_mas_horas",
  "P7120" = "disponible_mas_horas",
  "P7140s1" = "cambiar_por_capacidades",
  "P7140s2" = "cambiar_por_ingresos",
  "P7150" = "diligencias_cambiar_trabajo",
  "P7160" = "disponible_nuevo_trabajo",
  "P7310" = "primera_vez_o_trabajo_antes",
  "P7350" = "posicion_ultimo_trabajo",
  "P7422" = "ingresos_trabajo_desocupado",
  "P7422s1" = "valor_ingresos_desocupado",
  "P7472" = "ingresos_trabajo_inactivo",
  "P7472s1" = "valor_ingresos_inactivo",
  "P7495" = "recibio_arriendos_pensiones",
  "P7500s1" = "arriendos_propiedades",
  "P7500s1a1" = "valor_arriendos",
  "P7500s2" = "pensiones_jubilaciones",
  "P7500s2a1" = "valor_pensiones",
  "P7500s3" = "pension_alimenticia",
  "P7500s3a1" = "valor_pension_alimenticia",
  "P7505" = "recibio_transferencias_12m",
  "P7510s1" = "dinero_hogares_pais",
  "P7510s1a1" = "valor_dinero_hogares_pais",
  "P7510s2" = "dinero_hogares_exterior",
  "P7510s2a1" = "valor_dinero_exterior",
  "P7510s3" = "ayudas_instituciones",
  "P7510s3a1" = "valor_ayudas_instituciones",
  "P7510s5" = "intereses_dividendos",
  "P7510s5a1" = "valor_intereses_dividendos",
  "P7510s6" = "cesantias_intereses",
  "P7510s6a1" = "valor_cesantias",
  "P7510s7" = "otros_ingresos_fuentes",
  "P7510s7a1" = "valor_otros_ingresos",
  "pet" = "poblacion_edad_trabajar",
  "ina" = "inactivo",
  "ocu" = "ocupado",
  "dsi" = "desocupado",
  "pea" = "poblacion_economicamente_activa",
  "inac" = "inactivos",
  "wap" = "working_age_population",
  "sizeFirm" = "tamano_firma",
  "totalHoursWorked" = "total_horas_trabajadas",
  "impa" = "ingreso_actividad_principal_obs",
  "isa" = "ingreso_segunda_actividad_obs",
  "ie" = "ingreso_especie_obs",
  "imdi" = "ingreso_desocupados_inactivos_obs",
  "iof1" = "ingreso_intereses_dividendos_obs",
  "iof2" = "ingreso_pensiones_obs",
  "iof3h" = "ingreso_ayudas_hogares_obs",
  "iof3i" = "ingreso_ayudas_instituciones_obs",
  "iof6" = "ingreso_arriendos_obs",
  "cclasnr2" = "estado_imputacion_actividad_principal",
  "cclasnr3" = "estado_imputacion_segunda_actividad",
  "cclasnr4" = "estado_imputacion_especie",
  "cclasnr5" = "estado_imputacion_desocupados_inactivos",
  "cclasnr6" = "estado_imputacion_intereses",
  "cclasnr7" = "estado_imputacion_pensiones",
  "cclasnr8" = "estado_imputacion_ayudas",
  "cclasnr11" = "estado_imputacion_arriendos",
  "impaes" = "ingreso_actividad_principal_imp",
  "isaes" = "ingreso_segunda_actividad_imp",
  "iees" = "ingreso_especie_imp",
  "imdies" = "ingreso_desocupados_inactivos_imp",
  "iof1es" = "ingreso_intereses_dividendos_imp",
  "iof2es" = "ingreso_pensiones_imp",
  "iof3hes" = "ingreso_ayudas_hogares_imp",
  "iof3ies" = "ingreso_ayudas_instituciones_imp",
  "iof6es" = "ingreso_arriendos_imp",
  "ingtotob" = "ingreso_total_observado",
  "ingtotes" = "ingreso_total_imputado",
  "ingtot" = "ingreso_total_final",
  "y_salary_m" = "salario_mensual",
  "y_salary_m_hu" = "salario_mensual_horas_usuales",
  "y_ingLab_m" = "ingreso_laboral_mensual",
  "y_horasExtras_m" = "ingresos_horas_extras_m",
  "y_especie_m" = "ingresos_especie_m",
  "y_vivienda_m" = "ingresos_vivienda_m",
  "y_otros_m" = "otros_ingresos_m",
  "y_auxilioAliment_m" = "auxilio_alimentacion_m",
  "y_auxilioTransp_m" = "auxilio_transporte_m",
  "y_subFamiliar_m" = "subsidio_familiar_m",
  "y_subEducativo_m" = "subsidio_educativo_m",
  "y_primas_m" = "primas_mensuales",
  "y_bonificaciones_m" = "bonificaciones_mensuales",
  "y_primaServicios_m" = "prima_servicios_m",
  "y_primaNavidad_m" = "prima_navidad_m",
  "y_primaVacaciones_m" = "prima_vacaciones_m",
  "y_viaticos_m" = "viaticos_m",
  "y_accidentes_m" = "seguros_accidentes_m",
  "y_salarySec_m" = "salario_segundo_trabajo_m",
  "y_ingLab_m_ha" = "ingreso_laboral_horas_actuales",
  "y_gananciaNeta_m" = "ganancia_neta_mensual",
  "y_gananciaNetaAgro_m" = "ganancia_neta_agro_m",
  "y_gananciaIndep_m" = "ganancia_independiente_m",
  "y_gananciaIndep_m_hu" = "ganancia_independiente_horas_usuales",
  "y_total_m" = "ingreso_total_mensual",
  "y_total_m_ha" = "ingreso_total_horas_actuales",
  "fex_c" = "factor_expansion_anual",
  "depto" = "departamento",
  "fex_dpto" = "factor_expansion_departamental",
  "fweight" = "factor_ponderacion",
  "maxEducLevel" = "maximo_nivel_educativo",
  "college" = "educacion_superior",
  "regSalud" = "registrado_salud",
  "cotPension" = "cotiza_pension",
  "relab" = "relacion_laboral",
  "P6020" = "sexo",
  "P6040" = "Edad",
  "P6430" = "tipo_ocupado",
  "P6800" = "horas_trabajadas_semanalmente",
  "P5000" = "numero_cuartos_hogar",
  "P5010" = "numero_personas_en_cuarto",
  "P5090" = "tipo_propiedad",
  "P5100" = "pago_amortizacion",
  "P5130" = "estimado_arriendo",
  "P5140" = "valor_real_arriendo"
)

# Function to rename columns
dict_vec <- unlist(dictionary, use.names = TRUE)

renombrar_columnas <- function(df, dictionary) {
  # validamos nombres de columnas
  valid_cols <- !is.na(names(df)) & names(df) != ""
  df <- df[, valid_cols, drop = FALSE]
  
  dict_vec <- unlist(dictionary, use.names = TRUE)
  
  # encontrar correspondencias
  idx <- match(names(df), names(dict_vec))  # NA si no hay match
  to_change <- which(!is.na(idx))
  if (length(to_change) == 0) return(df)  # nada que renombrar
  
  new_names <- names(df)
  new_names[to_change] <- dict_vec[idx[to_change]]
  
  # chequeo de duplicados resultantes
  if (any(duplicated(new_names))) {
    warning("Algunos nombres resultantes están duplicados. Revisa el diccionario.")
  }
  
  names(df) <- new_names
  df
}

#Apply the funtion to the different databases
train_personas <- renombrar_columnas(train_personas, dictionary)
test_personas <- renombrar_columnas(test_personas, dictionary)
train_hogares <- renombrar_columnas(train_hogares, dictionary)
test_hogares <- renombrar_columnas(test_hogares, dictionary)

## -----------------------------------------------------
## 3) Create new variables
## -----------------------------------------------------
#Remove Bogotá
# Para train_personas
if ("Dominio" %in% names(train_personas)) {
  train_personas <- train_personas[train_personas$Dominio != "BOGOTA", ]
}

# Para test_hogares
if ("Dominio" %in% names(test_hogares)) {
  test_hogares <- test_hogares[test_hogares$Dominio != "BOGOTA", ]
}

# Para test_personas
if ("Dominio" %in% names(test_personas)) {
  test_personas <- test_personas[test_personas$Dominio != "BOGOTA", ]
}

# Para train_hogares
if ("Dominio" %in% names(train_hogares)) {
  train_hogares <- train_hogares[train_hogares$Dominio != "BOGOTA", ]
}

#Age is only present in the "personas" database
#Create a variable number of minors
train_personas <- train_personas %>%
  mutate(bin_minor = ifelse(test = Edad <= 18, yes = 1, no = 0))
test_personas <- test_personas %>%
  mutate(bin_minor = ifelse(test = Edad <= 18, yes = 1, no = 0))

#Create a variable to identify the household head
train_personas <- train_personas %>% mutate(bin_head = ifelse(test = parentesco_jefe_hogar == 1, yes = 1, no = 0))
test_personas <- train_personas %>% mutate(bin_head = ifelse(test = parentesco_jefe_hogar == 1, yes = 1, no = 0))

#Create a variable to identify if the household head is female
train_personas <- train_personas %>% mutate(bin_headFemale = bin_head*(1-sexo))
test_personas <- test_personas %>% mutate(bin_headFemale = bin_head*(1-sexo))

#Create age squared
train_personas <- train_personas %>% mutate(Edad2 = Edad*Edad)
test_personas <- train_personas %>% mutate(Edad2 = Edad*Edad)

# Apply filters to the first base
db <- db %>% filter(total_horas_trabajadas>0) %>% filter(age>17)

## -----------------------------------------------------
## 4) Missing Values
## -----------------------------------------------------
#skim the number of missing values 1
db_miss_personas_train <- skim(train_personas) %>% select(skim_variable, n_missing)
#view missing values as percentage
nobs <- nrow(db_miss_personas_train) # number of observations
db_miss_personas_train<- db_miss_personas_train %>% mutate(p_missing= n_missing/nobs) # new variable of number of NA
db_miss_personas_train <- db_miss_personas_train %>% arrange(-n_missing) # descendant order
db_miss_personas_train<- db_miss_personas_train %>% filter(n_missing!= 0) # keep only NA
head(db_miss_personas_train, 10) # Show the 10 first observations

#skim the number of missing values 2
db_miss_personas_test <- skim(test_personas) %>% select(skim_variable, n_missing)
nobs_test <- nrow(test_personas)
db_miss_personas_test <- db_miss_personas_test %>% 
  mutate(p_missing = n_missing/nobs_test) %>% 
  arrange(-n_missing) %>% 
  filter(n_missing != 0)
head(db_miss_personas_test, 10)

#skim the number of missing values 3
db_miss_hogares_train <- skim(train_hogares) %>% select(skim_variable, n_missing)
nobs_train_hogares <- nrow(train_hogares)
db_miss_hogares_train <- db_miss_hogares_train %>% 
  mutate(p_missing = n_missing/nobs_train_hogares) %>% 
  arrange(-n_missing) %>% 
  filter(n_missing != 0)
head(db_miss_hogares_train, 10)

#skim the number of missing values 4
db_miss_hogares_test <- skim(test_hogares) %>% select(skim_variable, n_missing)
nobs_test_hogares <- nrow(test_hogares)
db_miss_hogares_test <- db_miss_hogares_test %>% 
  mutate(p_missing = n_missing/nobs_test_hogares) %>% 
  arrange(-n_missing) %>% 
  filter(n_missing != 0)
head(db_miss_hogares_test, 10)

# Variables as.factor
test_personas <- test_personas %>%
  mutate(Estrato1 = as.factor(Estrato1)) %>%
  mutate(sexo = as.factor(sexo)) %>%
  mutate(nivel_educativo = as.factor(nivel_educativo)) %>%
  mutate(Oficio = as.factor(Oficio)) %>%
  mutate(log_ingreso_trabajo_principal = log(ingreso_trabajo_principal))%>%
  mutate(log_Ingtot = log(Ingtot))  %>%
  mutate(log_Ingtotes = log(Ingtotes)) %>%
  mutate(log_Ingtotob = log(Ingtotob)) 




