#################################################################################
##############     Example MLP for Classification   ############################
#################################################################################

## Set working directory -------------------------------------------------------------------------
x <-dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(x)
rm(list = ls())
cat("\014")

## Load libraries --------------------------------------------------------------------------------
library(caret)
library(ggplot2)
library(NeuralNetTools) ##Useful tools for plotting and analyzing neural networks
library(nnet)
library(NeuralSens)


## Read an example dataset
fdata <- read.table("SimData.dat", sep = "", header = TRUE, stringsAsFactors = FALSE)
## Add a noisy input variable
fdata$X3 <- rnorm(nrow(fdata))

## Convert the output into a factor variable with levels "YES" and "NO"
fdata$Y <- as.factor(fdata$Y)
levels(fdata$Y) <- c("YES", "NO")
str(fdata)

## Exploratory analysis -------------------------------------------------------------------------------------
ggplot(fdata) + geom_point(aes(x = X1, y = X2, color = Y))
summary(fdata$Y)

## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
# create random split
trainIndex <- createDataPartition(fdata$Y,     #output variable. createDataPartition creates proportional partitions
                                  p = 0.7, #split probability
                                  list = FALSE, #Avoid output as a list
                                  times = 1) #only one partition
# obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]

# Grid for evaluating the model 
np_grid <- 150 #number of discretization points in each dimension
np.X1 <- seq(from = min(fdata$X1), to = max(fdata$X1), length.out = np_grid)
np.X2 <- seq(from = min(fdata$X2), to = max(fdata$X2), length.out = np_grid)
grid_X1_X2 <- expand.grid(X1 = np.X1, X2 = np.X2)
grid_X1_X2$X3 <- rnorm(nrow(grid_X1_X2))


## Initialize trainControl as no CV --------------------------------------------------------------
ctrl <- trainControl(method = "none",                      #No CV
                     summaryFunction = defaultSummary,     #Performance summary for comparing models in hold-out samples.
                     classProbs = TRUE)                    #Compute class probs in Hold-out samples

#-------------------------------------------------------------------------------------------------
#-------------------------------- MLP ------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Try decay = 0 and 1 neuron: what do you expect?
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1", "X2", "X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                tuneGrid = data.frame(size = 1, decay = 0),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred = predict(mlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + 
  geom_point(aes(x = X1, y = X2, color = pred), show.legend = TRUE) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1) 
#accuracy measures
#fTR$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
#onfusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES") 
#fTS$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
#confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 

# Try decay = 0 and 2 neurons: are 2 neurons enough?
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1", "X2", "X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                tuneGrid = data.frame(size = 2, decay = 0),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(mlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) +geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)
#accuracy measures
fTR$mlp_pred = predict(mlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred = predict(mlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 

# Sensitivity analysis: how do you interpret the results? next step?
plotnet(mlp.fit$finalModel) #Plot the network
SensAnalysisMLP(mlp.fit) #Statistical sensitivity analysis

# Try decay = 0 and 10 neurons: a reasonable number of neurons 
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                tuneGrid = data.frame(size = 10, decay = 0),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(mlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)

#SensAnalysisMLP(mlp.fit) #Statistical sensitivity analysis

#accuracy measures
fTR$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 


# Try decay = 3 and 10 neurons: too much weight decay
set.seed(150) #For replication
mlp.fit = train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                tuneGrid = data.frame(size = 10, decay = 3),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(mlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)

#accuracy measures
fTR$mlp_pred <- predict(mlp.fit, type="raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred <- predict(mlp.fit, type="raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 

# Try decay = 0 and 50 neurons: too complex model
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 2500,    # Maximum number of iterations
                tuneGrid = data.frame(size = 50, decay = 0),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(mlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)

#accuracy measures
fTR$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 

# Try decay = 0.1 and 50 neurons: the excess of complexity is corrected by weight decay
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 2500,    # Maximum number of iterations
                tuneGrid = data.frame(size = 50, decay = 0.1),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(mlp.fit, type = "raw" , newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)

#SensAnalysisMLP(mlp.fit) #Statistical sensitivity analysis
#accuracy measures
fTR$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 

# Try MAXIT=25, decay = 0 and 50 neurons: test the effect of early stopping
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 25,    # Maximum number of iterations
                tuneGrid = data.frame(size = 50, decay = 0),
                trControl = ctrl, 
                metric = "Accuracy")

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(mlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)

#accuracy measures
fTR$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred <- predict(mlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 


## OPTIMIZATION OF THE #NEURONS AND WEIGHT DECAY PARAMETER
## Initialize trainControl -----------------------------------------------------------------------
ctrl <- trainControl(method = "cv",                        #k-fold cross-validation
                     number = 8,                           #Number of folds
                     summaryFunction = defaultSummary,     #Performance summary for comparing models in hold-out samples.
                     classProbs = TRUE)                    #Compute class probs in Hold-out samples

#----
# Try tuneGrid = 4: th eoptimal values are not reached
set.seed(150) #For replication
mlp.fit <- train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                #tuneGrid = data.frame(size =5, decay = 0),
                #tuneGrid = expand.grid(size = seq(5,25,length.out = 5), decay=c(10^(-9),0.0001,0.001,0.01,0.1,1,10)),
                tuneLength = 4,
                trControl = ctrl, 
                metric = "Accuracy")
mlp.fit #information about the resampling settings
ggplot(mlp.fit)


# Expand the grid
set.seed(150) #For replication
Gmlp.fit = train(fTR[,c("X1","X2","X3")],
                y = fTR$Y, 
                method = "nnet",
                preProcess = c("center","scale"),
                maxit = 250,    # Maximum number of iterations
                #tuneGrid = data.frame(size =5, decay = 0),
                tuneGrid = expand.grid(size = seq(5,25,length.out = 5), decay=c(10^(-9),0.0001,0.001,0.01,0.1,1,10)),
                #tuneLength = 4,
                trControl = ctrl, 
                metric = "Accuracy")
Gmlp.fit #information about the resampling settings
ggplot(Gmlp.fit) + scale_x_log10()

Gmlp.fit$finalModel #information about the model trained
#summary(Gmlp.fit$finalModel) #information about the network and weights
plotnet(Gmlp.fit$finalModel) #Plot the network
SensAnalysisMLP(Gmlp.fit) #Statistical sensitivity analysis

## Plot a 2D graph with the results of the model -------------------------------------------------
grid_X1_X2$pred <- predict(Gmlp.fit, type = "raw", newdata = grid_X1_X2) # predicted probabilities for class YES
ggplot(grid_X1_X2) + geom_point(aes(x = X1, y = X2, color = pred)) +
  geom_point(data = fdata[fdata$Y == "YES",], aes(x = X1, y = X2), color = "black", size = 1) +
  geom_point(data = fdata[fdata$Y == "NO",], aes(x = X1, y = X2), color = "white", size = 1)
#accuracy measures
fTR$mlp_pred <- predict(Gmlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")])
confusionMatrix(data = fTR$mlp_pred, reference = fTR$Y, positive = "YES")$overall[1] 
fTS$mlp_pred <- predict(Gmlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")])
confusionMatrix(data = fTS$mlp_pred, reference = fTS$Y, positive = "YES")$overall[1] 



## Evaluate model --------------------------------------------------------------------------------
#Evaluate the model with training and test sets
#training
fTR_eval <- fTR
fTR_eval$mlp_prob <- predict(Gmlp.fit, type = "prob", newdata = fTR[,c("X1","X2","X3")]) # predict probabilities
fTR_eval$mlp_pred <- predict(Gmlp.fit, type = "raw", newdata = fTR[,c("X1","X2","X3")]) # predict classes 
#test
fTS_eval <- fTS
fTS_eval$mlp_prob <- predict(Gmlp.fit, type = "prob", newdata = fTS[,c("X1","X2","X3")]) # predict probabilities
fTS_eval$mlp_pred <- predict(Gmlp.fit, type = "raw", newdata = fTS[,c("X1","X2","X3")]) # predict classes 


## Performance measures --------------------------------------------------------------------------------

#######confusion matices
# Training
confusionMatrix(data = fTR_eval$mlp_pred, #Predicted classes
                reference = fTR_eval$Y, #Real observations
                positive = "YES") #Class labeled as Positive
# test
confusionMatrix(fTS_eval$mlp_pred, 
                fTS_eval$Y, 
                positive = "YES")



