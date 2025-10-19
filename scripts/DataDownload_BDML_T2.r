# -----------------------------------------------------
# 0) Good practices, clean variables and libraries
# -----------------------------------------------------
# Clean variables and Libraries
rm(list = ls())

#setwd("~/Desktop/GitHub/DATA BDSMT2")

require("pacman")
p_load(tidyverse, 
       glmnet,
       caret,
       readr,
       dplyr,
       skimr,
       caret
)

## -----------------------------------------------------
## 1) Import Databases
## -----------------------------------------------------
train_hogares  <- read.csv("data/train_hogares.csv")
train_personas <- read.csv(unz("data/train_personas.csv.zip", "train_personas.csv"))
test_hogares   <- read.csv("data/test_hogares.csv")
test_personas  <- read.csv(unz("data/test_personas.csv.zip", "test_personas.csv"))

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
# Remove Bogotá
if ("Dominio" %in% names(train_personas)) {
  train_personas <- train_personas[train_personas$Dominio != "BOGOTA", ]
}
if ("Dominio" %in% names(test_hogares)) {
  test_hogares <- test_hogares[test_hogares$Dominio != "BOGOTA", ]
}
if ("Dominio" %in% names(test_personas)) {
  test_personas <- test_personas[test_personas$Dominio != "BOGOTA", ]
}
if ("Dominio" %in% names(train_hogares)) {
  train_hogares <- train_hogares[train_hogares$Dominio != "BOGOTA", ]
}

train_hogares <- train_hogares |> 
  mutate(Pobre_hand = ifelse(Ingpcug < Lp, 1, 0),
         Pobre_hand_2 = ifelse(Ingtotugarr < Lp*Npersug, 1, 0))

pre_process_personas <- function(data) {
  data |> 
    mutate(
      bin_woman   = ifelse(sexo == 2, 1, 0),
      bin_head    = ifelse(parentesco_jefe_hogar == 1, 1, 0),
      bin_minor   = ifelse(Edad <= 6, 1, 0),
      bin_minor18 = ifelse(Edad <= 18, 1, 0),
      cat_educ    = ifelse(is.na(nivel_educativo) | nivel_educativo == 9, 0, nivel_educativo),
      
      bin_occupied = case_when(
        !is.na(Oc) & Oc == 1 ~ 1,
        !is.na(Oc) & Oc != 1 ~ 0,
        !is.na(actividad_semana_pasada) & actividad_semana_pasada == 1 ~ 1,
        TRUE ~ 0
      ),
      
      Edad2 = Edad * Edad,
      bin_edad_productiva = ifelse(Edad >= 15 & Edad <= 64, 1, 0),
      bin_adulto_mayor = ifelse(Edad >= 65, 1, 0),
      bin_educ_superior = ifelse(!is.na(nivel_educativo) & nivel_educativo >= 5, 1, 0),
      experiencia_potencial = pmax(0, Edad - cat_educ - 6),
      
      bin_trabajador_formal = case_when(
        bin_occupied == 1 & !is.na(afiliado_seguridad_social) & afiliado_seguridad_social == 1 ~ 1,
        TRUE ~ 0
      ),
      
      bin_jefa_hogar = bin_head * bin_woman,
      
      bin_subempleo_horas = case_when(
        bin_occupied == 1 & 
          !is.na(horas_trabajadas_semanalmente) & 
          horas_trabajadas_semanalmente < 30 ~ 1,
        TRUE ~ 0
      ),
      
      bin_quiere_mas_horas = case_when(
        !is.na(quiere_trabajar_mas_horas) & quiere_trabajar_mas_horas == 1 ~ 1,
        TRUE ~ 0
      ),
      
      intensidad_laboral = case_when(
        !is.na(horas_trabajadas_semanalmente) ~ pmin(horas_trabajadas_semanalmente / 48, 1),
        TRUE ~ 0
      ),
      
      bin_segundo_trabajo = case_when(
        !is.na(tiene_segundo_trabajo) & tiene_segundo_trabajo == 1 ~ 1,
        TRUE ~ 0
      ),
      
      bin_empresa_grande = case_when(
        !is.na(tamano_empresa) & tamano_empresa >= 3 ~ 1,
        TRUE ~ 0
      )
    )
}

train_personas <- pre_process_personas(train_personas)
test_personas  <- pre_process_personas(test_personas)

train_personas_nivel_hogar <- train_personas |> 
  group_by(id) |>
  summarize(
    num_personas = n(),
    num_women    = sum(bin_woman, na.rm = TRUE),
    num_minors   = sum(bin_minor, na.rm = TRUE),
    num_minors18 = sum(bin_minor18, na.rm = TRUE),
    num_adultos_mayores = sum(bin_adulto_mayor, na.rm = TRUE),
    num_edad_productiva = sum(bin_edad_productiva, na.rm = TRUE),
    cat_maxEduc  = max(cat_educ, na.rm = TRUE),
    num_occupied = sum(bin_occupied, na.rm = TRUE),
    num_formal   = sum(bin_trabajador_formal, na.rm = TRUE),
    num_educ_superior = sum(bin_educ_superior, na.rm = TRUE),
    num_subempleo = sum(bin_subempleo_horas, na.rm = TRUE),
    num_quiere_mas_horas = sum(bin_quiere_mas_horas, na.rm = TRUE),
    num_segundo_trabajo = sum(bin_segundo_trabajo, na.rm = TRUE),
    num_empresa_grande = sum(bin_empresa_grande, na.rm = TRUE),
    mean_educ    = mean(cat_educ, na.rm = TRUE),
    mean_experiencia = mean(experiencia_potencial, na.rm = TRUE),
    mean_intensidad_laboral = mean(intensidad_laboral, na.rm = TRUE),
    .groups = 'drop'
  )

train_personas_hogar <- train_personas |> 
  filter(bin_head == 1) |>
  select(id, bin_woman, cat_educ, bin_occupied, Edad, Edad2,
         bin_trabajador_formal, bin_educ_superior, experiencia_potencial,
         bin_jefa_hogar, bin_adulto_mayor, bin_edad_productiva,
         bin_subempleo_horas, bin_quiere_mas_horas, intensidad_laboral,
         bin_segundo_trabajo, bin_empresa_grande) |>
  rename(
    bin_headWoman = bin_woman,
    cat_educHead = cat_educ,
    bin_occupiedHead = bin_occupied,
    edad_head = Edad,
    edad2_head = Edad2,
    bin_formalHead = bin_trabajador_formal,
    bin_educSuperiorHead = bin_educ_superior,
    experiencia_head = experiencia_potencial,
    bin_adulto_mayor_head = bin_adulto_mayor,
    bin_edad_productiva_head = bin_edad_productiva,
    bin_subempleo_head = bin_subempleo_horas,
    bin_quiere_mas_horas_head = bin_quiere_mas_horas,
    intensidad_laboral_head = intensidad_laboral,
    bin_segundo_trabajo_head = bin_segundo_trabajo,
    bin_empresa_grande_head = bin_empresa_grande
  ) |>
  left_join(train_personas_nivel_hogar, by = "id")

test_personas_nivel_hogar <- test_personas |> 
  group_by(id) |>
  summarize(
    num_personas = n(),
    num_women    = sum(bin_woman, na.rm = TRUE),
    num_minors   = sum(bin_minor, na.rm = TRUE),
    num_minors18 = sum(bin_minor18, na.rm = TRUE),
    num_adultos_mayores = sum(bin_adulto_mayor, na.rm = TRUE),
    num_edad_productiva = sum(bin_edad_productiva, na.rm = TRUE),
    cat_maxEduc  = max(cat_educ, na.rm = TRUE),
    num_occupied = sum(bin_occupied, na.rm = TRUE),
    num_formal   = sum(bin_trabajador_formal, na.rm = TRUE),
    num_educ_superior = sum(bin_educ_superior, na.rm = TRUE),
    num_subempleo = sum(bin_subempleo_horas, na.rm = TRUE),
    num_quiere_mas_horas = sum(bin_quiere_mas_horas, na.rm = TRUE),
    num_segundo_trabajo = sum(bin_segundo_trabajo, na.rm = TRUE),
    num_empresa_grande = sum(bin_empresa_grande, na.rm = TRUE),
    mean_educ    = mean(cat_educ, na.rm = TRUE),
    mean_experiencia = mean(experiencia_potencial, na.rm = TRUE),
    mean_intensidad_laboral = mean(intensidad_laboral, na.rm = TRUE),
    .groups = 'drop'
  )

test_personas_hogar <- test_personas |> 
  filter(bin_head == 1) |>
  select(id, bin_woman, cat_educ, bin_occupied, Edad, Edad2,
         bin_trabajador_formal, bin_educ_superior, experiencia_potencial,
         bin_jefa_hogar, bin_adulto_mayor, bin_edad_productiva,
         bin_subempleo_horas, bin_quiere_mas_horas, intensidad_laboral,
         bin_segundo_trabajo, bin_empresa_grande) |>
  rename(
    bin_headWoman = bin_woman,
    cat_educHead = cat_educ,
    bin_occupiedHead = bin_occupied,
    edad_head = Edad,
    edad2_head = Edad2,
    bin_formalHead = bin_trabajador_formal,
    bin_educSuperiorHead = bin_educ_superior,
    experiencia_head = experiencia_potencial,
    bin_adulto_mayor_head = bin_adulto_mayor,
    bin_edad_productiva_head = bin_edad_productiva,
    bin_subempleo_head = bin_subempleo_horas,
    bin_quiere_mas_horas_head = bin_quiere_mas_horas,
    intensidad_laboral_head = intensidad_laboral,
    bin_segundo_trabajo_head = bin_segundo_trabajo,
    bin_empresa_grande_head = bin_empresa_grande
  ) |>
  left_join(test_personas_nivel_hogar, by = "id")

train_hogares <- train_hogares |> 
  mutate(
    bin_rent = ifelse(tipo_propiedad == 3, 1, 0),
    Ingpcug  = numero_cuartos_hogar / Npersug,
    IPR      = Ingpcug / Lp
  ) |> 
  select(id, Dominio, bin_rent, Ingpcug, IPR, Pobre)

test_hogares <- test_hogares |> 
  mutate(
    bin_rent = ifelse(tipo_propiedad == 3, 1, 0),
    Ingpcug  = numero_cuartos_hogar / Npersug,
    IPR      = Ingpcug / Lp
  ) |> 
  select(id, Dominio, bin_rent, Ingpcug, IPR)

# Variables demográficas:
# bin_minor18: Persona menor de 18 años (vs bin_minor que era ≤6)
# bin_edad_productiva: Persona entre 15-64 años (edad laboral activa)
# bin_adulto_mayor: Persona ≥65 años 

# Variables laborales:
# bin_segundo_trabajo: Persona tiene segundo trabajo (diversificación de ingresos)
# bin_empresa_grande: Trabaja en empresa grande (≥3 empleados, mejor calidad empleo)
# bin_subempleo_horas: Trabaja <30 horas semanales
# bin_quiere_mas_horas: Quiere trabajar más horas 
# intensidad_laboral: Ratio horas trabajadas/48 horas (dedicación laboral)

# VARIABLES AGREGADAS A NIVEL HOGAR (por cada hogar)
# num_personas: Total personas en el hogar
# num_adultos_mayores: Cuántos adultos mayores (dependientes)
# num_edad_productiva: Cuántos en edad de trabajar
# num_subempleo: Cuántos en subempleo por horas
# num_quiere_mas_horas: Cuántos quieren más horas
# num_segundo_trabajo: Cuántos tienen segundo trabajo
# num_empresa_grande: Cuántos trabajan en empresas grandes

# Promedios del hogar:
# mean_experiencia: Experiencia laboral promedio del hogar
# mean_intensidad_laboral: Intensidad laboral promedio

# VARIABLES DEL JEFE DE HOGAR NUEVAS
# Demográficas del jefe:
# edad_head: Edad del jefe de hogar
# edad2_head: Edad al cuadrado (efectos no lineales)

# Laborales del jefe:
# bin_subempleo_head: Jefe en subempleo por horas
# bin_quiere_mas_horas_head: Jefe quiere más horas
# intensidad_laboral_head: Intensidad laboral del jefe
# bin_segundo_trabajo_head: Jefe tiene segundo trabajo
# bin_empresa_grande_head: Jefe trabaja en empresa grande

# INTERACCIONES ECONÓMICAS PARA PREDICCIÓN DE POBREZA:
# dep_burden: Carga total de dependientes (menores + adultos mayores)
# minors_per_worker: Menores por trabajador (carga económica específica)
# dependents_per_worker: Dependientes totales por trabajador
# head_female_with_minors: Jefa de hogar con menores (doble vulnerabilidad)
# head_educ_times_workers: Educación del jefe multiplicada por trabajadores
# subempleo_household_size: Subempleo amplificado por tamaño del hogar
# head_subempleo_with_minors: Jefe en subempleo con menores a cargo
# need_hours_household_size: Necesidad de horas amplificada por tamaño
# formal_employment_depth: Profundidad del empleo formal
# head_educ_formal: Jefe educado y formal (protección)
# head_age_with_minors: Edad del jefe con responsabilidad de menores
# vulnerable_head: Jefa en subempleo (máxima vulnerabilidad)
# household_productivity: Productividad del hogar (educados trabajando)
# elderly_burden_workers: Carga de adultos mayores por trabajador
# diversification_strength: Fortaleza por diversificación laboral
# quality_employment: Empleo de calidad (empresa grande y formal)

train <- train_hogares |> 
  left_join(train_personas_hogar, by = "id") |>
  select(-id) |> 
  mutate(
    Pobre   = factor(Pobre, levels = c(0, 1), labels = c("No", "Yes")),
    Dominio = factor(Dominio),
    cat_educHead = factor(cat_educHead, levels = c(0:6),
                          labels = c("No information", "None", "Preschool", "Primary",
                                     "Secondary", "High school", "University")),
    
    dep_burden = num_minors + num_adultos_mayores,
    minors_per_worker = ifelse(num_occupied > 0, num_minors / num_occupied, num_minors),
    dependents_per_worker = ifelse(num_occupied > 0, dep_burden / num_occupied, dep_burden),
    head_female_with_minors = bin_headWoman * num_minors,
    head_educ_times_workers = as.numeric(cat_educHead) * num_occupied,
    subempleo_household_size = num_subempleo * num_personas,
    head_subempleo_with_minors = bin_subempleo_head * num_minors,
    need_hours_household_size = num_quiere_mas_horas * num_personas,
    formal_employment_depth = num_formal * num_occupied,
    head_educ_formal = bin_educSuperiorHead * bin_formalHead,
    head_age_with_minors = edad_head * num_minors,
    vulnerable_head = bin_headWoman * bin_subempleo_head,
    household_productivity = num_educ_superior * num_occupied,
    elderly_burden_workers = num_adultos_mayores * ifelse(num_occupied > 0, 1/num_occupied, 1),
    diversification_strength = num_segundo_trabajo * num_occupied,
    quality_employment = num_empresa_grande * num_formal
  )

test <- test_hogares |> 
  left_join(test_personas_hogar, by = "id") |> 
  mutate(
    Dominio = factor(Dominio),
    cat_educHead = factor(cat_educHead, levels = c(0:6),
                          labels = c("No information", "None", "Preschool", "Primary",
                                     "Secondary", "High school", "University")),
    
    dep_burden = num_minors + num_adultos_mayores,
    minors_per_worker = ifelse(num_occupied > 0, num_minors / num_occupied, num_minors),
    dependents_per_worker = ifelse(num_occupied > 0, dep_burden / num_occupied, dep_burden),
    head_female_with_minors = bin_headWoman * num_minors,
    head_educ_times_workers = as.numeric(cat_educHead) * num_occupied,
    subempleo_household_size = num_subempleo * num_personas,
    head_subempleo_with_minors = bin_subempleo_head * num_minors,
    need_hours_household_size = num_quiere_mas_horas * num_personas,
    formal_employment_depth = num_formal * num_occupied,
    head_educ_formal = bin_educSuperiorHead * bin_formalHead,
    head_age_with_minors = edad_head * num_minors,
    vulnerable_head = bin_headWoman * bin_subempleo_head,
    household_productivity = num_educ_superior * num_occupied,
    elderly_burden_workers = num_adultos_mayores * ifelse(num_occupied > 0, 1/num_occupied, 1),
    diversification_strength = num_segundo_trabajo * num_occupied,
    quality_employment = num_empresa_grande * num_formal
  )

colSums(is.na(train))
colSums(is.na(test))

## -----------------------------------------------------
## 5) As Factor
## -----------------------------------------------------
# Variables as.factor
train <- train |> 
  mutate(
    bin_rent = as.factor(bin_rent),
    bin_headWoman = as.factor(bin_headWoman), 
    bin_occupiedHead = as.factor(bin_occupiedHead),
    bin_formalHead = as.factor(bin_formalHead),
    bin_educSuperiorHead = as.factor(bin_educSuperiorHead),
    bin_jefa_hogar = as.factor(bin_jefa_hogar),
    bin_adulto_mayor_head = as.factor(bin_adulto_mayor_head),
    bin_edad_productiva_head = as.factor(bin_edad_productiva_head),
    bin_subempleo_head = as.factor(bin_subempleo_head),
    bin_quiere_mas_horas_head = as.factor(bin_quiere_mas_horas_head),
    bin_segundo_trabajo_head = as.factor(bin_segundo_trabajo_head),
    bin_empresa_grande_head = as.factor(bin_empresa_grande_head),
    head_educ_formal = as.factor(head_educ_formal),
    vulnerable_head = as.factor(vulnerable_head)
  )

test <- test |> 
  mutate(
    bin_rent = as.factor(bin_rent),
    bin_headWoman = as.factor(bin_headWoman), 
    bin_occupiedHead = as.factor(bin_occupiedHead),
    bin_formalHead = as.factor(bin_formalHead),
    bin_educSuperiorHead = as.factor(bin_educSuperiorHead),
    bin_jefa_hogar = as.factor(bin_jefa_hogar),
    bin_adulto_mayor_head = as.factor(bin_adulto_mayor_head),
    bin_edad_productiva_head = as.factor(bin_edad_productiva_head),
    bin_subempleo_head = as.factor(bin_subempleo_head),
    bin_quiere_mas_horas_head = as.factor(bin_quiere_mas_horas_head),
    bin_segundo_trabajo_head = as.factor(bin_segundo_trabajo_head),
    bin_empresa_grande_head = as.factor(bin_empresa_grande_head),
    head_educ_formal = as.factor(head_educ_formal),
    vulnerable_head = as.factor(vulnerable_head)
  )
## -----------------------------------------------------
## 6) Standirize
## -----------------------------------------------------
# Variables to standardize as they are not categorical
variables_to_standardize <- c(
  "Ingpcug", 
  "IPR",
  "edad_head", 
  "edad2_head",
  "experiencia_head",
  "num_personas", "num_women", "num_minors", "num_minors18",
  "num_adultos_mayores", "num_edad_productiva", "num_occupied", 
  "num_formal", "num_educ_superior", "num_subempleo",
  "num_quiere_mas_horas", "num_segundo_trabajo", "num_empresa_grande",
  "mean_experiencia",
  "dep_burden",
  "head_female_with_minors", "head_educ_times_workers",
  "subempleo_household_size", "head_subempleo_with_minors",
  "need_hours_household_size", "formal_employment_depth",
  "head_age_with_minors", "household_productivity",
  "diversification_strength", "quality_employment",
  "minors_per_worker", "dependents_per_worker", "elderly_burden_workers"
)

preprocess_params <- preProcess(train[variables_to_standardize], method = c("center", "scale"))
train[variables_to_standardize] <- predict(preprocess_params, train[variables_to_standardize])
test[variables_to_standardize] <- predict(preprocess_params, test[variables_to_standardize])

## -----------------------------------------------------
## 6) Export CSV
## -----------------------------------------------------
setwd("~/Desktop/GitHub/Problem-Set-2-Predicting-Poverty/data")

# Exportar cada dataset
write.csv(train, "train_clean.csv", row.names = FALSE)
write.csv(test, "test_clean.csv", row.names = FALSE)

