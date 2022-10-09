## Load libraries ----
library(caret)          # La biblia de los paquetes
library(ggplot2)        # La biblia para figuras
library(naniar)         # Contains functions to visualize missing values
library(ROCR)           # For plotting ROC curves.
library(pROC)           # For a comparative analysis of roc curve
library(MLTools)        # Paquete propio de ICAI

# XGBOOST
library(xgboost)        # xgboost library

# Decision trees
library(rpart)          # Decisions trees 1
library(rpart.plot)     # Decisions trees 2
library(partykit)       # Decisions trees 3

# Bagging & Random Forest
library(randomForest)   # Random forest and bagging library

# MLP
library(NeuralNetTools) # Useful tools for plotting and analyzing neural networks
library(nnet)           # Neuronal netword
library(NeuralSens)     # Statistical sensitivity analysis


# Set working directory and clean workspace
x <-dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(x)
rm(list = ls())
cat("\014")


## Load data ----
cat("
##############################################################################
#########################    Data Analysis    ################################
##############################################################################",
    "\n\n")
fdata <- read.table("TrainingSet.dat", sep = ";", header = TRUE, stringsAsFactors = FALSE)

fdata$Y <- as.factor(fdata$Y)
levels(fdata$Y) <- c("NO", "YES")

summary(fdata)#The two clases in cardio are leveled
cat("Data:\n")
print(head(fdata))

if (file.exists("MapaCorrelacion.png") == FALSE){
  pm <- ggpairs(fdata, aes(color = Y, alpha = 0.3), progress = TRUE)
  ggsave("MapaCorrelacion.png", plot = pm)
  PlotDataframe(fdata, output.name = "Y")
}


## Análisis outliers ----
# X1
ggplot(fdata, aes(x=X1)) + geom_boxplot(outlier.colour="red",
                                         outlier.shape=3, outlier.size=2)
outliers_X1 <- boxplot(fdata$X1, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X1, fill = Y),bins = 15)

# X2
ggplot(fdata, aes(x=X2)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X2 <- boxplot(fdata$X2, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X2, fill = Y),bins = 15)


# X3
ggplot(fdata, aes(x=X3)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X3 <- boxplot(fdata$X3, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X3, fill = Y),bins = 15)

# X4
ggplot(fdata, aes(x=X4)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X4 <- boxplot(fdata$X4, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X4, fill = Y),bins = 15)

# X5
ggplot(fdata, aes(x=X5)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X5 <- boxplot(fdata$X5, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X5, fill = Y),bins = 15)

# X6
ggplot(fdata, aes(x=X6)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X6 <- boxplot(fdata$X6, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X6, fill = Y),bins = 15)
fdata<- fdata[-which(fdata$X6 %in% outliers_X6),]

# X7
ggplot(fdata, aes(x=X7)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X7 <- boxplot(fdata$X7, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X7, fill = Y),bins = 15)

# X8
ggplot(fdata, aes(x=X8)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X8 <- boxplot(fdata$X8, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X8, fill = Y),bins = 15)

# X9
ggplot(fdata, aes(x=X9)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X9 <- boxplot(fdata$X9, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X9, fill = Y),bins = 15)

# X10
ggplot(fdata, aes(x=X10)) + geom_boxplot(outlier.colour="red",
                                        outlier.shape=3, outlier.size=2)
outliers_X10 <- boxplot(fdata$X10, plot=FALSE)$out
ggplot(fdata) + geom_histogram(aes(x = X10, fill = Y),bins = 15)

## Preparación para entrenamiento ----
# Create random 80/20 % split
set.seed(150) #For replication
trainIndex <- createDataPartition(fdata$Y,
                                  p = 0.8,
                                  list = FALSE,
                                  times = 1)
# Obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]
# The overall class distribution of the data is preserved
table(fTR$Y) 
table(fTS$Y)

# Create dataset to include model predictions
fTR_eval <- fTR
fTS_eval <- fTS

## XGBoost ----
cat("\n\n
##############################################################################
############################    XGBoost    ###################################
##############################################################################",
    "\n\n")

## training control
ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = defaultSummary,
                     classProbs = TRUE)

## Train XGBoost
set.seed(200) #For replication
xgb_grid <- expand.grid(
  nrounds = 300, #Boosting Iterations
  eta = 0.3,     #Shrinkage
  max_depth = 10, #Max Tree Depth
  gamma = 0,    #Minimum Loss Reduction
  colsample_bytree=1, #Subsample Ratio of Columns
  min_child_weight=1, #Minimum Sum of Instance Weight
  subsample = 0.5    #Subsample Percentage
)

# train
training_data = fTR[,c("X5","X6","X9","X10","Y")]
test_data = fTS[,c("X5","X6","X9","X10", "Y")]

xgb.fit = train(
  x = training_data[-ncol(training_data)],   #Input variables
  y = training_data$Y,   #Output variables 
  tuneGrid = xgb_grid, #Uncomment to use values previously defined
  tuneLength = 4, #Use caret tuning
  method = "xgbTree",
  trControl = ctrl,
  metric="Accuracy"
)

cat("\nTrain summary:\n\n")
print(summary(xgb.fit))

#Measure for variable importance
varImp(xgb.fit,scale = FALSE)
plot(varImp(xgb.fit,scale = FALSE))


## Evaluate model
#training
fTR_eval$xgb_prob <- predict(xgb.fit, type="prob",
                             newdata = training_data)
fTR_eval$xgb_pred <- predict(xgb.fit, type="raw",
                             newdata = training_data)
#Test
fTS_eval$xgb_prob <- predict(xgb.fit, type="prob",
                             newdata = test_data)
fTS_eval$xgb_pred <- predict(xgb.fit, type="raw",
                             newdata = test_data)



#Plot classification in a 2 dimensional space
Plot2DClass(test_data, #Input variables
            test_data$Y,     #Output variable
            xgb.fit,   #Fitted model with caret
            var1 = "X5", var2 = "X6", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 



## Performance measures

## Confusion matices
cat("\nTraining Accuracy:\n")
print(confusionMatrix(fTR_eval$xgb_pred, 
                      fTR_eval$Y, 
                      positive = "YES"))

cat("\nValidation Accuracy:\n")
print(confusionMatrix(fTS_eval$xgb_pred, 
                fTS_eval$Y, 
                positive = "YES"))

## Classification performance plots 
cat("\nROC validation data:\n")
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$xgb_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)



## Evaluación test total
TestData <- read.table("TestSet.dat", sep = ";", header = TRUE, stringsAsFactors = FALSE)
ValForecast <- predict(xgb.fit, type="raw" , newdata = TestData) # predict classes
write.table(ValForecast ,"T10.csv", col.names = FALSE, row.names = FALSE)





