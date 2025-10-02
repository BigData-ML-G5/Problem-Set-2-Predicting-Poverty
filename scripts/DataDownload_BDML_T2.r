require("pacman")
p_load(tidyverse, 
       glmnet,
       caret,
       readr
)

train_personas <- read.csv("data/train_personas.csv")
test_hogares  <- read.csv("data/test_hogares.csv")
test_personas <- read.csv("data/test_personas.csv")

# O si prefieres más directo:
train_hogares <- read.csv("https://github.com/BigData-ML-G5/Problem-Set-2-Predicting-Poverty/raw/refs/heads/main/data/train_hogares.csv")
test_hogares  <- read.csv("https://github.com/BigData-ML-G5/Problem-Set-2-Predicting-Poverty/raw/refs/heads/main/data/test_hogares.csv")
