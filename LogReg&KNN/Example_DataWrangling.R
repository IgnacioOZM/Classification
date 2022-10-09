####################################################
########### Data wrangling example #################
####################################################

## Install and load libraries --------------------------------------------------
library(dplyr)        # Contains functions to manipulate datasets
library(naniar)       # Contains functions to visualize missing values
library(simputation)  # Contains functions to fill missing values

## Set working directory and clean workspace
x <-dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(x)
rm(list = ls())
cat("\014")

## Load Titanic dataset --------------------------------------------------------
# read.table is a function that allows to read multiple kinds of text files. It admits the following arguments:
Titanic <- read.table(file = "Titanic.csv", #Name of text file.
           sep = ",",                       #Separation character.
           header = TRUE,                   #If column names are in the first row.
           na.strings = "NA",               #Character to be marked as missing value.
           stringsAsFactors = FALSE)        #convert string to factors?
#See structure 
str(Titanic)
#Print first few rows
head(Titanic)

## FACTORS AND CONVERSIONS --------------------------------------------------------
## Identify each type of variable in the dataset. Make the necessary conversions.

# summarize the data read to explore range of values and missing data.
summary(Titanic)
#"Name" is character, but should not be considered as categorical data.
#"PClass" and "Sex" are character variables, but they only contain a small number of classes
unique(Titanic$PClass) #unique function returns a vector with duplicate elements removed  
table(Titanic$Sex) #table counts unique elements.
#Therefore, they should be considered as categorical data.
Titanic <- Titanic %>% 
  mutate(PClass = as.factor(PClass),
         Sex = as.factor(Sex))
str(Titanic) #Labels are created
summary(Titanic) #Summary gives the number of cases
#Contingency table
table(Titanic$PClass, Titanic$Sex)
#"Survived" and "SexCode" are numerical variables, but they only contain a small number of classes
table(Titanic$Survived)
table(Titanic$SexCode)
#Therefore, they should be considered as categorical data.
Titanic <- Titanic %>% mutate(Survived = as.factor(Survived))
str(Titanic)
#some functions do not admit names of factors starting with numbers. We can change the factor names:
levels(Titanic$Survived) <- c("NO","YES")
str(Titanic)
#Another useful function when having a variable with 0 and 1 is ifelse()
Titanic <- Titanic %>% mutate(SexCode = as.factor(ifelse(SexCode, "F","M")))
str(Titanic)

## MISSING VALUES --------------------------------------------------------
## Identify missing values and eliminate those observations from the dataset.
# The summary function provides the number of missing values (if any) found for each variable.
summary(Titanic)
# Variable PClass contains an odd value: "*"
# Lets remove that row
Titanic <- Titanic %>% filter(PClass != "*")

# Functions gg_miss_var and vis_miss can be used to visualize the number of empty values for each variable.
gg_miss_var(Titanic)
vis_miss(Titanic)
# Define a new dataset only with correctly observed data
Titanic_noNA <- Titanic %>% filter(!is.na(Age))
# In addition, the function na.omit() directly obtains the dataset without NA in all variables.
Titanic_noNA <- Titanic %>% na.omit()
summary(Titanic_noNA)


# However, we can fill the empty values of the dataset using some statistical tool

# 1. Empty values can be replaced with the mean value of the data
Titanic_noNA_2 <- Titanic %>% mutate(Age = ifelse(is.na(Age),
                                                  mean(Age,na.rm = T),
                                                  Age))
# Remember that the mean can be sensible to outliers.

# 2. Empty values can be replaced with the median of the data, which is more robust than the mean
Titanic_noNA_3 <- Titanic %>% mutate(Age = ifelse(is.na(Age),
                                                  median(Age,na.rm = T),
                                                  Age))


# 3. A knn model can be used to fill empty values
#    The inputs for the knn model will be variables PClass and Sex
Titanic_noNA_4 <- Titanic %>%
  impute_knn(Age ~ PClass + Sex, # Formula: Variables PClass and Sex will be used to fill variable Age empty values
             k = 9)              # Number of nearest neighbours used by the model


## BASIC DATA MANIPULATION --------------------------------------------------------
## Create two datasets, one for each passenger sex,
## only with Age, Sex and Survived variables
Titanic_F <- Titanic %>% 
  filter(Sex == "female") %>%
  select(Age,Sex,Survived)

Titanic_M <- Titanic %>% 
  filter(Sex == "male") %>%
  select(Age,Sex,Survived)


## Obtain the number of survivors 
## grouping them by passenger sex.
Titanic %>% 
  filter(Survived == "YES") %>%
  group_by(Sex) %>% 
  summarise(proportion = n())



