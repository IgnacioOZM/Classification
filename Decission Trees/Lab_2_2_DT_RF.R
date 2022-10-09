###############################################################
#####   Classification:  Decision trees                    ####
###############################################################

## Set working directory and clean workspace
x <-dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(x)
rm(list = ls())
cat("\014")


## Load libraries --------------------------------------------------------------------------------
library(caret)
library(ggplot2)
library(ROCR) #for plotting ROC curves.
library(MLTools)


## Load file -------------------------------------------------------------------------------------
fdata <- read.table("SimData.dat", sep = "", header = TRUE, stringsAsFactors = FALSE)
str(fdata); head(fdata)
#Convert output variable to factor
fdata$Y <- as.factor(fdata$Y)
str(fdata)


## Exploratory analysis -------------------------------------------------------------------------------------
ggplot(fdata) + geom_point(aes(x = X1, y = X2, color = Y))



## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(fdata$Y,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]

#Create datasets to store evaluations
fTR_eval <- fTR
fTS_eval <- fTS


## Initialize trainControl -----------------------------------------------------------------------
ctrl <- trainControl(method = "cv",                        #k-fold cross-validation
                     number = 10,                          #Number of folds
                     summaryFunction = defaultSummary,     #Performance summary for comparing models in hold-out samples.
                     classProbs = TRUE)                    #Compute class probs in Hold-out samples


#-------------------------------------------------------------------------------------------------
#---------------------------- DECISION TREE ------------------------------------------------------
#-------------------------------------------------------------------------------------------------
library(rpart)
library(rpart.plot)
library(partykit)
set.seed(150) #For replication
#Train decision tree
#rpart contains 1 tuning parameter cp (Complexity parameter). Three options:
#  - Train with a fixed parameter: tuneGrid = data.frame(cp = 0.1),
#  - Try with a range of values specified in tuneGrid: tuneGrid = data.frame(cp = seq(0,0.4,0.05))),
#  - Caret chooses 10 values: tuneLength = 10,

#NOTE: Formula method could be used, but it will automatically create dummy variables. 
# Decision trees can work with categorical variables as theey are. Then, x and y arguments are used
tree.fit <- train(x = fTR[,c(1,2)],  #Input variables.
                 y = fTR$Y,   #Output variable
                 method = "rpart",   #Decision tree with cp as tuning parameter
                 control = rpart.control(minsplit = 5,  # Minimum number of obs in node to keep cutting
                                        minbucket = 5), # Minimum number of obs in a terminal node
                 parms = list(split = "gini"),          # impuriry measure
                 #tuneGrid = data.frame(cp = 0.025), # TRY this: tuneGrid = data.frame(cp = 0.25),
                 #tuneLength = 10,
                 tuneGrid = data.frame(cp = seq(0,0.05,0.0005)),
                 trControl = ctrl, 
                 metric = "Accuracy")
tree.fit #information about the resampling settings
ggplot(tree.fit) #plot the summary metric as a function of the tuning parameter
summary(tree.fit)  #information about the model trained
tree.fit$finalModel #Cuts performed and nodes. Also shows the number and percentage of cases in each node.
#Basic plot of the tree:
plot(tree.fit$finalModel, uniform = TRUE, margin = 0)
text(tree.fit$finalModel, use.n = TRUE, all = TRUE, cex = .8)
#Advanced plots
rpart.plot(tree.fit$finalModel, type = 2, fallen.leaves = FALSE, box.palette = "Oranges")
tree.fit.party <- as.party(tree.fit$finalModel)
plot(tree.fit.party)

#Measure for variable importance
varImp(tree.fit,scale = FALSE)
plot(varImp(tree.fit,scale = FALSE))

## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval$tree_prob <- predict(tree.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$tree_pred <- predict(tree.fit, type="raw", newdata = fTR) # predict classes 
#test
fTS_eval$tree_prob <- predict(tree.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$tree_pred <- predict(tree.fit, type="raw", newdata = fTS) # predict classes 



#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            tree.fit,#Fitted model with caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$tree_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$tree_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# test
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$tree_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)



#-------------------------------------------------------------------------------------------------
#---------------------------- RANDOM FOREST ------------------------------------------------------
#-------------------------------------------------------------------------------------------------
library(randomForest)
set.seed(150) #For replication
#Train decision tree
#rf contains one tuning parameter mtry: 
#   the number of variables randomly sampled as candidates at each split.
#   The ntree argument can be used to specify the number of trees to grow.
rf.fit <- train(  x = fTR[,1:2],   #Input variables
                  y = fTR$Y,   #Output variables 
                  method = "rf", #Random forest
                  ntree = 200,  #Number of trees to grow
                  tuneGrid = data.frame(mtry = seq(1,ncol(fTR)-1)),           
                  tuneLength = 4,
                  trControl = ctrl, #Resampling settings 
                  metric = "Accuracy") #Summary metrics
rf.fit #information about the resampling settings
ggplot(rf.fit)   

#Measure for variable importance
varImp(rf.fit,scale = FALSE)
plot(varImp(rf.fit,scale = FALSE))

## Evaluate model --------------------------------------------------------------------------------
#training
fTR_eval$rf_prob <- predict(rf.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$rf_pred <- predict(rf.fit, type="raw", newdata = fTR) # predict classes 
#Test
fTS_eval$rf_prob <- predict(rf.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$rf_pred <- predict(rf.fit, type="raw", newdata = fTS) # predict classes 



#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            rf.fit,#Fitted model with caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 



## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$rf_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# Validation
confusionMatrix(fTS_eval$rf_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$rf_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# Validation
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$rf_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)


#-------------------------------------------------------------------------------------------------
#---------------------------- XGBoost ------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# See the following for some examples 
# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret/code


library(xgboost)
set.seed(150) #For replication
#Parameter tuning
xgb_grid <- expand.grid(
  nrounds = 150, #Boosting Iterations
  eta = 0.3,     #Shrinkage
  max_depth = 5, #Max Tree Depth
  gamma = 0,    #Minimum Loss Reduction
  colsample_bytree=1, #Subsample Ratio of Columns
  min_child_weight=1, #Minimum Sum of Instance Weight
  subsample = 0.5    #Subsample Percentage
)

# train
xgb.fit = train(
  x = fTR[,1:2],   #Input variables
  y = fTR$Y,   #Output variables 
  #tuneGrid = xgb_grid, #Uncomment to use values previously defined
  tuneLength = 4, #Use caret tuning
  method = "xgbTree",
  trControl = ctrl,
  metric="Accuracy"
)

#plot grid
# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}
tuneplot(xgb.fit)

#Measure for variable importance
varImp(xgb.fit,scale = FALSE)
plot(varImp(xgb.fit,scale = FALSE))


## Evaluate model --------------------------------------------------------------------------------
#training
fTR_eval$xgb_prob <- predict(xgb.fit, type="prob", newdata = fTR) # predict probabilities
fTR_eval$xgb_pred <- predict(xgb.fit, type="raw", newdata = fTR) # predict classes 
#Test
fTS_eval$xgb_prob <- predict(xgb.fit, type="prob", newdata = fTS) # predict probabilities
fTS_eval$xgb_pred <- predict(xgb.fit, type="raw", newdata = fTS) # predict classes 



#Plot classification in a 2 dimensional space
Plot2DClass(fTR[,1:2], #Input variables of the model
            fTR$Y,     #Output variable
            xgb.fit,#Fitted model with caret
            var1 = "X1", var2 = "X2", #variables that define x and y axis
            selClass = "YES")     #Class output to be analyzed 



## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$xgb_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# Validation
confusionMatrix(fTS_eval$xgb_pred, 
                fTS_eval$Y, 
                positive = "YES")

#######Classification performance plots 
# Training
PlotClassPerformance(fTR_eval$Y,       #Real observations
                     fTR_eval$xgb_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed
# Validation
PlotClassPerformance(fTS_eval$Y,       #Real observations
                     fTS_eval$xgb_prob,  #predicted probabilities
                     selClass = "YES") #Class to be analyzed)

#-------------------------------------------------------------------------------------------------
#---------------------- COMPARATIVE ANALYSIS ----------------------------------------------
#-------------------------------------------------------------------------------------------------

## comparison of models in training and validation set --------------------------------------------------------
#resampling summary metric
transformResults <- resamples(list(tree = tree.fit, rf = rf.fit, xgb = xgb.fit))
summary(transformResults)
dotplot(transformResults)





