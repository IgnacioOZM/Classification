################################################################################
##############    Classification: Logistic Regression - kNN    ######################
################################################################################

## Load libraries --------------------------------------------------------------------------------
library(caret)
library(ggplot2)
library(ROCR) #for plotting ROC curves.
#########################################################
## load MLTools package. It contains useful functions for Machine Learning Course.
#install the package only the first time to be used. Set working directory to library folder
# install.packages("MLTools_0.0.30.tar.gz", repos = NULL, dep = TRUE)
#load package
library(MLTools)
#From now on, only library(MLTools) should be called to load the package
#########################################################


## Set working directory and clean workspace
x <-dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(x)
rm(list = ls())
cat("\014")

## Load file -------------------------------------------------------------------------------------
fdata <- read.table("SimData.dat", sep = "", header = TRUE, stringsAsFactors = FALSE)
str(fdata)
#Convert output variable to factor
fdata$Y <- as.factor(fdata$Y)
str(fdata)


## Exploratory analysis -------------------------------------------------------------------------------------
ggplot(fdata) + geom_point(aes(x = X1, y = X2, color = Y))


#For datasets with more than two inputs
library(GGally)
ggpairs(fdata,aes(color = Y, alpha = 0.3))
#Function for plotting multiple plots between the output of a data frame
#and the predictors of this output.
PlotDataframe(fdata = fdata, 
              output.name = "Y")


## Divide the data into training and test sets ---------------------------------------------------
# see http://topepo.github.io/caret/data-splitting.html for details
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(fdata$Y,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]
#the overall class distribution of the data is preserved
table(fTR$Y) 
table(fTS$Y)

#Create dataset to include model predictions
fTR_eval <- fTR
fTS_eval <- fTS


#plot training and test sets. 
#Try changing the value inside set.seed(). How does it affect the selected data?
ggplot(fTR) + geom_point(aes(x = X1, y = X2, color = Y))
ggplot(fTS) + geom_point(aes(x = X1, y = X2, color = Y))


## Initialize trainControl -----------------------------------------------------------------------
ctrl <- trainControl(method = "cv",                        #k-fold cross-validation
                     number = 10,                          #Number of folds
                     summaryFunction = defaultSummary,     #Performance summary for comparing models in hold-out samples.
                     classProbs = TRUE)                    #Compute class probs in Hold-out samples
 


#-------------------------------------------------------------------------------------------------
#---------------------------- LOGISTIC REGRESSION MODEL ----------------------------------------------
#-------------------------------------------------------------------------------------------------
## Train model -----------------------------------------------------------------------------------
set.seed(150) #For replication
#Train model using training data
LogReg.fit <- train(form = Y ~ ., #formula for specifying inputs and outputs.
                   data = fTR,               #Training dataset 
                   method = "glm",                   #Train logistic regression
                   preProcess = c("center","scale"), #Center an scale inputs
                   trControl = ctrl,                 #trainControl Object
                   metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg.fit          #information about the resampling
summary(LogReg.fit) #detailed information about the fit of the final model
str(LogReg.fit)     #all information stored in the model


## Understanding resampling methods -------------------------------------------
str(LogReg.fit$control$index)       #Training indexes
str(LogReg.fit$control$indexOut)    #Validation indexes
LogReg.fit$resample                 #Resample test results
boxplot(LogReg.fit$resample$Accuracy, xlab = "Accuracy", main="Boxplot for summary metrics of test samples")


## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTS) # predict classes 

head(fTR_eval)

#Plot predictions of the model
ggplot(fTR_eval) + geom_point(aes(x = X1, y = X2, color = LRpred)) + labs(title = "Predictions for training data")
ggplot(fTS_eval) + geom_point(aes(x = X1, y = X2, color = LRpred)) + labs(title = "Predictions for test data")


#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Dataframe with input variables
            fTR$Y,     #Output variable
            LogReg.fit,#Fitted model with Caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 

Plot2DClass(fTR, #Dataframe with input variables
            fTR$Y,     #Output variable
            LogReg.fit,#Fitted model with Caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$LRpred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$LRpred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$LRprob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)




###################################################################################
## include X1 variable squared ####################################################
###################################################################################
fdata$X1sq <- fdata$X1^2
#obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]
## Train model
set.seed(150) #For replication
#Train model using training data
LogReg2.fit <- train(form = Y ~ X1 + X1sq + X2, #formula for specifying inputs and outputs.
                    data = fTR,               #Training dataset 
                    method = "glm",                   #Train logistic regression
                    preProcess = c("center","scale"), #Center an scale inputs
                    trControl = ctrl,                 #trainControl Object
                    metric = "Accuracy")              #summary metric used for selecting hyperparameters
LogReg2.fit          #information about the resampling
summary(LogReg2.fit) #detailed information about the fit of the final model

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$LRprob2 <- predict(LogReg2.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$LRpred2 <- predict(LogReg2.fit, type="raw" , newdata = fTR) # predict classes 
#test
fTS_eval$LRprob2 <- predict(LogReg2.fit, type="prob" , newdata = fTS) # predict probabilities
fTS_eval$LRpred2 <- predict(LogReg2.fit, type="raw" , newdata = fTS) # predict classes 


#Plot classification results in input variables
Plot2DClass(fTR, #Dataframe with input variables 
            fTR$Y,     #Output variable
            LogReg2.fit,#Fitted model with caret
            var1 = "X1sq", var2 = "X2", #variables to represent the plot
            selClass = "YES")     #Class output to be analyzed 


#plot in X1 space
#Plot classification results in input variables
Plot2DClass(fTR, #Dataframe with input variables 
            fTR$Y,     #Output variable
            LogReg2.fit,#Fitted model with caret
            var1 = "X1", var2 = "X2", #variables to represent the plot
            selClass = "YES")     #Class output to be analyzed 


## Performance measures --------------------------------------------------------------------------------

#######confusion matrices
# Training
confusionMatrix(data = fTR_eval$LRpred2, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$LRpred2, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$LRprob2,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$LRprob2,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)






#-------------------------------------------------------------------------------------------------
#---------------------------- KNN MODEL  ----------------------------------------------
#-------------------------------------------------------------------------------------------------
set.seed(150) #For replication
#Train knn model model.
#Knn contains 1 tuning parameter k (number of neigbors). Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(k = 5),
#  - Try with a range of values specified in tuneGrid: tuneGrid = data.frame(k = seq(2,120,4)),
#  - Caret chooses 10 values: tuneLength = 10,
knn.fit = train(form = Y ~ X1 + X2, #formula for specifying inputs and outputs.
                data = fTR,   #Training dataset 
                method = "knn",
                preProcess = c("center","scale"),
                #tuneGrid = data.frame(k = 5),
                tuneGrid = data.frame(k = seq(3,115,4)),
                #tuneLength = 10,
                trControl = ctrl, 
                metric = "Accuracy")
knn.fit #information about the settings
ggplot(knn.fit) #plot the summary metric as a function of the tuning parameter
knn.fit$finalModel #information about final model trained


## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTR) # predict probabilities
fTR_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTR) # predict classes 
#test
fTS_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTS) # predict probabilities
fTS_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTS) # predict classes 


#Plot classification in a 2 dimensional space
Plot2DClass(fTR, #Dataframe with input variables of the model
            fTR$Y,     #Output variable
            knn.fit,#Fitted model with caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 



## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$knn_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$knn_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$knn_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)






#-------------------------------------------------------------------------------------------------
#---------------------- COMPARATIVE ANALYSIS ----------------------------------------------
#-------------------------------------------------------------------------------------------------

## comparison of models in training and test set --------------------------------------------------------
#TRAINING
#resampling summary metric
transformResults <- resamples(list(lr = LogReg.fit, lr2 = LogReg2.fit, knn = knn.fit ))
summary(transformResults)
dotplot(transformResults)

#Overall accuracy
confusionMatrix(fTR_eval$LRpred, fTR_eval$Y, positive = "YES")$overall[1]
confusionMatrix(fTR_eval$LRpred2, fTR_eval$Y, positive = "YES")$overall[1]
confusionMatrix(fTR_eval$knn_pred, fTR_eval$Y, positive = "YES")$overall[1]


#ROC curve
library(pROC)
reducedRoc <- roc(response = fTR_eval$Y, fTR_eval$LRprob$YES)
plot(reducedRoc, col="black")
auc(reducedRoc)
reducedRoc <- roc(response = fTR_eval$Y, fTR_eval$LRprob2$YES)
plot(reducedRoc, add=TRUE, col="red")
auc(reducedRoc)
reducedRoc <- roc(response = fTR_eval$Y, fTR_eval$knn_prob$YES)
plot(reducedRoc, add=TRUE, col="green")
auc(reducedRoc)
legend("bottomright", legend=c("LR", "LR2","knn"), col=c("black", "red","green"), lwd=2)



#TEST
#Overall accuracy
confusionMatrix(fTS_eval$LRpred, fTS_eval$Y, positive = "YES")$overall[1]
confusionMatrix(fTS_eval$LRpred2, fTS_eval$Y, positive = "YES")$overall[1]
confusionMatrix(fTS_eval$knn_pred, fTS_eval$Y, positive = "YES")$overall[1]

#Calibration curve     
calPlotData <- calibration(Y ~ LRprob$YES+LRprob2$YES+knn_prob$YES, data =fTS_eval, class = "YES",cuts = 6)
xyplot(calPlotData, auto.key = list(columns = 2))

#ROC curve
library(pROC)
reducedRoc <- roc(response = fTS_eval$Y, fTS_eval$LRprob$YES)
plot(reducedRoc, col="black")
auc(reducedRoc)
reducedRoc <- roc(response = fTS_eval$Y, fTS_eval$LRprob2$YES)
plot(reducedRoc, add=TRUE, col="red")
auc(reducedRoc)
reducedRoc <- roc(response = fTS_eval$Y, fTS_eval$knn_prob$YES)
plot(reducedRoc, add=TRUE, col="green")
auc(reducedRoc)
legend("bottomright", legend=c("LR", "LR2","knn"), col=c("black", "red","green"), lwd=2)

