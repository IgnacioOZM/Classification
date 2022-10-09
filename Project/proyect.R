################################################################################
####################    Classification: Proyect    #############################
################################################################################

## Load libraries
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


## Set working directory and clean workspace
  x <-dirname(rstudioapi::getActiveDocumentContext()$path)
  setwd(x)
  rm(list = ls())
  cat("\014")


################################################################################
##########################    Data analysis   ##################################
################################################################################
cat("
##############################################################################
##########################    Data Analysis   ################################
##############################################################################",
    "\n")

# Load of data
  fdata <- read.table(file = "cardioMod4.csv",
                      sep = ";",
                      header = TRUE,
                      na.strings = "NA",
                      stringsAsFactors = FALSE)


# Analizando el .csv observamos que hay múltiples datos que se deben de 
  # considerar como factores (Gender, Cholesterol, Gluc, ...)
  fdata$gender <- as.factor(fdata$gender)
  levels(fdata$gender) <- c("Female", "Male")
  
  fdata$cholesterol <- as.factor(fdata$cholesterol)
  levels(fdata$cholesterol) <- c("Normal", "High", "Very High")
  
  fdata$gluc <- as.factor(fdata$gluc)
  levels(fdata$gluc) <- c("Normal", "High", "Very High")
  
  fdata$smoke <- as.factor(fdata$smoke)
  levels(fdata$smoke) <- c("No", "Yes")
  
  fdata$alco <- as.factor(fdata$alco)
  levels(fdata$alco) <- c("No", "Yes")
  
  fdata$active <- as.factor(fdata$active)
  levels(fdata$active) <- c("No", "Yes")
  
  fdata$cardio <- as.factor(fdata$cardio)
  levels(fdata$cardio) <- c("No", "Yes")

## Análisis detallado de cada categoría para detectar outliers
  # Age  -  Se han elminidao 39 outliers con valor -1
  ggplot(fdata, aes(x=age)) + geom_boxplot(outlier.colour="red",
                 outlier.shape=3, outlier.size=2)
  fdata <- fdata[which(fdata$age > 0),]
  outliers_age <- c()
  ggplot(fdata) + geom_histogram(aes(x = age, fill = cardio),bins = 15)
  
  # Gender - No presenta outliers por ser factor y no faltan datos
  unique(fdata$gender)
  ggplot(fdata) + geom_bar(aes(x = gender, fill = cardio))

  
  # Height  -  Eliminamos todos los datos que esten por denajo de 140
  ggplot(fdata, aes(x=height)) + geom_boxplot(outlier.colour="red",
                                           outlier.shape=3, outlier.size=2)
  fdata <- fdata[fdata$height > 140,]
  outliers_height <- boxplot(fdata$height, plot=FALSE)$out
  ggplot(fdata) + geom_histogram(aes(x = height, fill = cardio),bins = 15)
  
  # Weight  -  Consideramos que este puede ser un factor importante y los
  # outliers pueden afectar. Pesos extremos presentan un riesgo para la salud.
  # Por ello dejamos los outliers.
  ggplot(fdata, aes(x=weight)) + geom_boxplot(outlier.colour="red",
                                              outlier.shape=3, outlier.size=2)
  outliers_weight <- boxplot(fdata$weight, plot=FALSE)$out
  ggplot(fdata) + geom_histogram(aes(x = weight, fill = cardio),bins = 10)
  
  # ap_hi  -  Consideramos que este puede ser un factor importante y los
  # outliers pueden afectar. Pesos extremos presentan un riesgo para la salud.
  # Por ello dejamos los outliers.
  ggplot(fdata, aes(x=ap_hi)) + geom_boxplot(outlier.colour="red",
                                             outlier.shape=3, outlier.size=2)
  outliers_ap_hi <- boxplot(fdata$ap_hi, plot=FALSE)$out
  ggplot(fdata) + geom_histogram(aes(x = ap_hi, fill = cardio),bins = 10)
  
  # ap_lo  -  Consideramos que este puede ser un factor importante y los
  # outliers pueden afectar. Pesos extremos presentan un riesgo para la salud.
  # Por ello dejamos los outliers. No tengo del todo claro que 140 sea posible.
  ggplot(fdata, aes(x=ap_lo)) + geom_boxplot(outlier.colour="red",
                                             outlier.shape=3, outlier.size=2)
  outliers_ap_lo <- boxplot(fdata$ap_lo, plot=FALSE)$out
  ggplot(fdata) + geom_histogram(aes(x = ap_lo, fill = cardio),bins = 10)
  
  # cholesterol - No presenta outliers por ser factor y no faltan datos
  unique(fdata$cholesterol)
  ggplot(fdata) + geom_bar(aes(x = cholesterol, fill = cardio))
  
  # gluc - Presenta outliers por ser factor y faltan datos
  unique(fdata$gluc)
  ggplot(fdata) + geom_bar(aes(x = gluc, fill = cardio))
  
  # smoke - No presenta outliers por ser factor y no faltan datos
  unique(fdata$smoke)
  ggplot(fdata) + geom_bar(aes(x = smoke, fill = cardio))
  
  # alco - No presenta outliers por ser factor y no faltan datos
  unique(fdata$alco)
  ggplot(fdata) + geom_bar(aes(x = alco, fill = cardio))
  
  # active - No presenta outliers por ser factor y no faltan datos
  unique(fdata$active)
  ggplot(fdata) + geom_bar(aes(x = active, fill = cardio))

  
## Detección de valores NA  -  Como no se trata de demasiados datos y
  # son factores se ha decidido omitirlos.  
  gg_miss_var(fdata)
  vis_miss(data.frame(fdata$smoke, fdata$gluc, fdata$active))
  fdata <- fdata %>% na.omit()

## Resumen de los datos de trabajo
  if (file.exists("MapaCorrelacion.png") == FALSE){
    library(GGally)     # For ggpairs
    pm <- ggpairs(fdata, aes(color = cardio, alpha = 0.3), progress = TRUE)
    ggsave("MapaCorrelacion.png", plot = pm)
    PlotDataframe(fdata, output.name = "cardio")
  }
  cat("\nSummary of data:", "\n\n")
  print(summary(fdata))
  cat("\nFirst rows of data:", "\n\n")
  print(head(fdata))
  
## Preparación para entrenamiento
  # Create random 80/20 % split
  set.seed(150) #For replication
  trainIndex <- createDataPartition(fdata$cardio,
                                    p = 0.8,
                                    list = FALSE,
                                    times = 1)
  # Obtain training and test sets
  fTR <- fdata[trainIndex,]
  fTS <- fdata[-trainIndex,]
  # The overall class distribution of the data is preserved
  table(fTR$cardio) 
  table(fTS$cardio)
  
  # Create dataset to include model predictions
  fTR_eval <- fTR
  fTS_eval <- fTS


################################################################################
#####################    LOGISTIC REGRESSION MODEL   ###########################
################################################################################
cat("\n\n
##############################################################################
####################    LOGISTIC REGRESSION MODEL   ##########################
##############################################################################",
    "\n\n")
  
  ## Initialize trainControl
  ctrl <- trainControl(method = "cv",
                       number = 10,
                       summaryFunction = defaultSummary,
                       classProbs = TRUE)
  
  ## Train model using training data
  set.seed(150) #For replication
  LogReg.fit <- train(form = cardio ~ .,
                      data = fTR,
                      method = "glm",
                      preProcess = c("center","scale"),
                      trControl = ctrl,
                      metric = "Accuracy")
  LogReg.fit          #information about the resampling
  # str(LogReg.fit)     #all information stored in the model
  
  # Los p-valores son demasiados bajos. Por ello reducimos dimensión.
  set.seed(150)
  LogReg.fit <- train(form = cardio ~ age + weight + ap_hi + cholesterol,
                      data = fTR,
                      method = "glm",
                      preProcess = c("center","scale"),
                      trControl = ctrl,
                      metric = "Accuracy")
  cat("Summary of training:","\n\n")
  print(LogReg.fit)
  print(summary(LogReg.fit))
  
  # LogReg.fit$resample                 #Resample test results
  boxplot(LogReg.fit$resample$Accuracy, xlab = "Accuracy",
          main="Boxplot for summary metrics of test samples")
  
  
  ## Evaluate model
  #Evaluate the model with training and test sets
  #training
  fTR_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTR)
  fTR_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTR)
  #test
  fTS_eval$LRprob <- predict(LogReg.fit, type="prob", newdata = fTS)
  fTS_eval$LRpred <- predict(LogReg.fit, type="raw", newdata = fTS)
  
  #Plot predictions of the model
  ggplot(fTR_eval) + geom_point(aes(x = weight, y = ap_hi, color = LRpred)) + 
    labs(title = "Predictions for training data")
  ggplot(fTS_eval) + geom_point(aes(x = age, y = ap_hi, color = LRpred)) + 
    labs(title = "Predictions for test data")
  
  ## Performance measures
  # Confusion matices
  # Training
  confusionMatrix(data = fTR_eval$LRpred, #Predicted classes
                  reference = fTR_eval$cardio, #Real observations
                  positive = "Yes") #Class labeled as Positive
  # test
  confusionMatrix(fTS_eval$LRpred, 
                  fTS_eval$cardio, 
                  positive = "Yes")
  
  # Classification performance plots 
  # Training
  cat("Training data:\n")
  PlotClassPerformance(fTR_eval$cardio,       #Real observations
                       fTR_eval$LRprob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed
  # test
  cat("\nTest data:\n")
  PlotClassPerformance(fTS_eval$cardio,       #Real observations
                       fTS_eval$LRprob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed)
  

################################################################################
#############################    KNN Model   ###################################
################################################################################
  cat("\n\n
##############################################################################
############################    KNN Model   ##################################
##############################################################################",
      "\n\n")
  
  ## Initialize trainControl
  ctrl <- trainControl(method = "cv",
                       number = 10,
                       summaryFunction = defaultSummary,
                       classProbs = TRUE)
  
  ## Train knn model model. El modelo funciona mejor con todos los datos.
  set.seed(150)
  knn.fit = train(form = cardio ~ ., #formula for specifying inputs and outputs.
                  data = fTR,   #Training dataset 
                  method = "knn",
                  preProcess = c("center","scale"),
                  tuneGrid = data.frame(k = seq(10,30,1)),
                  tuneLength = 10,
                  trControl = ctrl, 
                  metric = "Accuracy")
  knn.fit #information about the settings
  ggplot(knn.fit) #plot the summary metric as a function of the tuning parameter
  
  cat("Summary of training:\n\n")
  print(summary(knn.fit))
  cat("\n")
  print(knn.fit$finalModel) #information about final model trained

  ## Evaluate model
  #Evaluate the model with training and test sets
  #training
  fTR_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTR)
  fTR_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTR)
  #test
  fTS_eval$knn_prob <- predict(knn.fit, type="prob" , newdata = fTS)
  fTS_eval$knn_pred <- predict(knn.fit, type="raw" , newdata = fTS)

  ## Performance measures
  
  # Confusion matices
  # Training
  confusionMatrix(data = fTR_eval$knn_pred, #Predicted classes
                  reference = fTR_eval$cardio, #Real observations
                  positive = "Yes") #Class labeled as Positive
  # test
  confusionMatrix(fTS_eval$knn_pred, 
                  fTS_eval$cardio, 
                  positive = "Yes")
  
  # Classification performance plots 
  # Training
  cat("Training data:\n")
  PlotClassPerformance(fTR_eval$cardio,       #Real observations
                       fTR_eval$knn_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed
  # test
  cat("\nTest data:\n")
  PlotClassPerformance(fTS_eval$cardio,       #Real observations
                       fTS_eval$knn_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed)
  
  
################################################################################
##########################    Decision Trees   #################################
################################################################################
  cat("\n\n
##############################################################################
#########################    Decision Trees   ################################
##############################################################################",
      "\n\n")
  
  ## training control
  ctrl <- trainControl(method = "cv",
                       number = 10,
                       summaryFunction = defaultSummary,
                       classProbs = TRUE)

  ## Train decision tree  -  Excluimos las variables con menor importancia
  tree.fit <- train(x = fTR[,c("ap_hi","ap_lo","age","cholesterol","weight")],
                    y = fTR$cardio,   #Output variable
                    method = "rpart",
                    control = rpart.control(minsplit = 5,
                                            minbucket = 5),
                    parms = list(split = "gini"),
                    tuneGrid = data.frame(cp = seq(0,0.01,0.0001)),
                    trControl = ctrl, 
                    metric = "Accuracy")
  print(tree.fit)
  ggplot(tree.fit)
  #summary(tree.fit)
  tree.fit$finalModel
  # Plot of the tree:
  tree.fit.party <- as.party(tree.fit$finalModel)
  plot(tree.fit.party)
  
  #Measure for variable importance
  varImp(tree.fit,scale = FALSE)
  plot(varImp(tree.fit,scale = FALSE))
  
  ## Evaluate model
  #Evaluate the model with training and test sets
  #training
  fTR_eval$tree_prob <- predict(tree.fit, type="prob", newdata = fTR) 
  fTR_eval$tree_pred <- predict(tree.fit, type="raw", newdata = fTR) 
  #test
  fTS_eval$tree_prob <- predict(tree.fit, type="prob", newdata = fTS)
  fTS_eval$tree_pred <- predict(tree.fit, type="raw", newdata = fTS)
  
  
  ## Performance measures 
  
  ## Confusion matices
  # Training
  confusionMatrix(data = fTR_eval$tree_pred, #Predicted classes
                  reference = fTR_eval$cardio, #Real observations
                  positive = "Yes") #Class labeled as Positive
  # test
  confusionMatrix(fTS_eval$tree_pred, 
                  fTS_eval$cardio, 
                  positive = "Yes")
  
  ## Classification performance plots 
  # Training
  cat("\nTraining data:\n")
  PlotClassPerformance(fTR_eval$cardio,       #Real observations
                       fTR_eval$tree_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed
  # test
  cat("\nTest data:\n")
  PlotClassPerformance(fTS_eval$cardio,       #Real observations
                       fTS_eval$tree_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed)
  
  
  
################################################################################
####################    Bagging & Random Forests    ############################
################################################################################
  cat("\n\n
##############################################################################
####################    Bagging & Random Forests    ##########################
##############################################################################",
      "\n\n")
  
  ## training control
  ctrl <- trainControl(method = "cv",
                       number = 10,
                       summaryFunction = defaultSummary,
                       classProbs = TRUE)
  
  ## Train decision tree
  set.seed(150) #For replication
  rf.fit <- train(  x = fTR[,seq(1,11,1)],   #Input variables
                    y = fTR$cardio,   #Output variables 
                    method = "rf", #Random forest
                    ntree = 100,  #Number of trees to grow
                    tuneGrid = data.frame(mtry = seq(1,ncol(fTR)-1)),           
                    tuneLength = 4,
                    trControl = ctrl, #Resampling settings 
                    metric = "Accuracy") #Summary metrics
  rf.fit #information about the resampling settings
  ggplot(rf.fit)
  cat("Summary of training:\n\n")
  print(summary(rf.fit))
  
  #Plot classification in a 2 dimensional space
  Plot2DClass(fTR[,seq(1,11,1)], #Input variables of the model
              fTR$cardio,     #Output variable
              rf.fit,#Fitted model with caret
              var1 = "age", var2 = "ap_hi", #variables that define x and y axis
              selClass = "Yes")     #Class output to be analyzed 
  
  #Measure for variable importance
  varImp(rf.fit,scale = FALSE)
  plot(varImp(rf.fit,scale = FALSE))
  
  ## Evaluate model
  #training
  fTR_eval$rf_prob <- predict(rf.fit, type="prob", newdata = fTR) # predict probabilities
  fTR_eval$rf_pred <- predict(rf.fit, type="raw", newdata = fTR) # predict classes 
  #Test
  fTS_eval$rf_prob <- predict(rf.fit, type="prob", newdata = fTS) # predict probabilities
  fTS_eval$rf_pred <- predict(rf.fit, type="raw", newdata = fTS) # predict classes 
  
  ## Performance measures
  
  ## Confusion matices
  # Training
  confusionMatrix(data = fTR_eval$rf_pred, #Predicted classes
                  reference = fTR_eval$cardio, #Real observations
                  positive = "Yes") #Class labeled as Positive
  # Validation
  confusionMatrix(fTS_eval$rf_pred, 
                  fTS_eval$cardio, 
                  positive = "Yes")
  
  #######Classification performance plots 
  # Training
  cat("\nTraining data:\n")
  PlotClassPerformance(fTR_eval$cardio,       #Real observations
                       fTR_eval$rf_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed
  # Validation
  cat("\nTest data:\n")
  PlotClassPerformance(fTS_eval$cardio,       #Real observations
                       fTS_eval$rf_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed)
  
  
  
################################################################################
############################    XGBoost    ####################################
################################################################################
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
  set.seed(150) #For replication
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
    x = fTR[,c("age","weight","height","ap_hi","ap_lo")],   #Input variables
    y = fTR$cardio,   #Output variables 
    tuneGrid = xgb_grid, #Uncomment to use values previously defined
    tuneLength = 4, #Use caret tuning
    method = "xgbTree",
    trControl = ctrl,
    metric="Accuracy"
  )
  
  cat("\nSummary of training:\n\n")
  print(summary(xgb.fit))
  
  #Measure for variable importance
  varImp(xgb.fit,scale = FALSE)
  plot(varImp(xgb.fit,scale = FALSE))
  
  
  ## Evaluate model
  #training
  fTR_eval$xgb_prob <- predict(xgb.fit, type="prob",
                     newdata = fTR[,c("age","weight","height","ap_hi","ap_lo")])
  fTR_eval$xgb_pred <- predict(xgb.fit, type="raw",
                     newdata = fTR[,c("age","weight","height","ap_hi","ap_lo")])
  #Test
  fTS_eval$xgb_prob <- predict(xgb.fit, type="prob",
                     newdata = fTS[,c("age","weight","height","ap_hi","ap_lo")])
  fTS_eval$xgb_pred <- predict(xgb.fit, type="raw",
                     newdata = fTS[,c("age","weight","height","ap_hi","ap_lo")])
  
  
  
  #Plot classification in a 2 dimensional space
  Plot2DClass(fTR[,c("age","weight","height","ap_hi","ap_lo")], #Input variables
              fTR$cardio,     #Output variable
              xgb.fit,#Fitted model with caret
              var1 = "age", var2 = "ap_hi", #variables that define x and y axis
              selClass = "Yes")     #Class output to be analyzed 
  
  
  
  ## Performance measures
  
  ## Confusion matices
  # Training
  confusionMatrix(data = fTR_eval$xgb_pred, #Predicted classes
                  reference = fTR_eval$cardio, #Real observations
                  positive = "Yes") #Class labeled as Positive
  # Validation
  confusionMatrix(fTS_eval$xgb_pred, 
                  fTS_eval$cardio, 
                  positive = "Yes")
  
  ## Classification performance plots 
  # Training
  cat("\nTraining data:\n")
  PlotClassPerformance(fTR_eval$cardio,       #Real observations
                       fTR_eval$xgb_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed
  # Validation
  cat("\nTest data:\n")
  PlotClassPerformance(fTS_eval$cardio,       #Real observations
                       fTS_eval$xgb_prob,  #predicted probabilities
                       selClass = "Yes") #Class to be analyzed)
  
  
  
################################################################################
##############################    MLP    #######################################
################################################################################
  cat("\n\n
##############################################################################
##############################    MLP    #####################################
##############################################################################",
      "\n\n")
  
  ## Initialize trainControl
  ctrl <- trainControl(method = "cv",
                       number = 10,
                       summaryFunction = defaultSummary,
                       classProbs = TRUE)
  
  ## Training of Network
  set.seed(150) #For replication
  mlp.fit = train(form = cardio ~ age + weight + ap_hi + alco + cholesterol,
                  data = fTR,
                  method = "nnet",
                  preProcess = c("center","scale"),
                  maxit = 200,    # Maximum number of iterations
                  #tuneGrid = data.frame(size =5, decay = 0),
                  tuneGrid = expand.grid(size = seq(5,70,by = 5),
                                         decay=c(10^(-9),0.0001,0.001,0.01,0.1,1)),
                  trControl = ctrl,
                  metric = "Accuracy")
  
  mlp.fit #information about the resampling settings
  ggplot(mlp.fit)+scale_x_log10()
  
  mlp.fit$finalModel #information about the model trained
  #summary(mlp.fit$finalModel) #information about the network and weights
  plotnet(mlp.fit$finalModel) #Plot the network
  
  #Statistical sensitivity analysis
  SensAnalysisMLP(mlp.fit) 
  
  
  
################################################################################
##########################    Comparativa    ###################################
################################################################################

  ## Curvas ROC
  # Linear Regression
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$LRprob$Yes)
  plot(reducedRoc, col="black")
  auc(reducedRoc)
  # KNN
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$knn_prob$Yes)
  plot(reducedRoc, add=TRUE, col="red")
  auc(reducedRoc)
  # Decision trees
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$tree_prob$Yes)
  plot(reducedRoc, add=TRUE, col="green")
  auc(reducedRoc)
  # Random Forest
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$rf_prob$Yes)
  plot(reducedRoc, add=TRUE, col="blue")
  auc(reducedRoc)
  # XGBoost
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$xgb_prob$Yes)
  plot(reducedRoc, add=TRUE, col="orange")
  auc(reducedRoc)
  # SVM
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$svm_prob$Yes)
  plot(reducedRoc, add=TRUE, col="pink")
  auc(reducedRoc)
  # MLP
  reducedRoc <- roc(response = fTS_eval$cardio, fTS_eval$mlp_prob$Yes)
  plot(reducedRoc, add=TRUE, col="grey")
  auc(reducedRoc)
  
  legend("bottomright", legend=c("LR", "KNN","Decision Tree", "Random Forest", "XGBoost", "SVM", "MLP"), col=c("black", "red","green","blue","orange","pink","grey"), lwd=2)
  
  
  
  
  
  
  
  
  
  
  
  
  
  