### Train a classification model ###

### Author: Group 6
### Project 3
### ADS Spring 2018


# 1. Baseline Model: GBM 

train_gbm <- function(data_train){
  
  ###  data_train: processed features from images also contains label
  
  library(gbm)
  
  start_time_gbm = Sys.time() # Model Start Time
  
  gbm.fit = gbm(Label~., 
                data = data_train,
                n.trees = 400,
                distribution = "multinomial",
                interaction.depth = 3, 
                shrinkage = 0.2,
                n.minobsinnode = 30,
                cv.fold = 10,
                verbose=FALSE)
  end_time_gbm = Sys.time() # Model End time
  gbm_time = end_time_gbm - start_time_gbm #Total Running Time
  
  
  return(list(fit = gbm.fit, time = gbm_time))
}

# 2. SVM Model 

train_svm = function(data_train){
  
  library(e1071)
  
  control = trainControl(method = 'cv', number = 10)
  
  svmGrid = expand.grid(C = c(0.5,1,2))
  
  start_time_svm = Sys.time() # Model Start Time
  svm.fit = train(Label~., 
                  data = data_train,
                  method = "svmLinear",
                  preProc = c('center', 'scale'),
                  tuneGrid = svmGrid,
                  trControl = control)
  end_time_svm = Sys.time() # Model End time
  
  svm_time = end_time_svm - start_time_svm #Total Running Time
  
  
  return(list(fit = svm.fit, time = svm_time))
  
}

# 3. Random Forest 

train_rf = function(data_train){
  
  control = trainControl(method = 'cv', number = 10)
  
  rfGrid = expand.grid(mtry = floor(sqrt(ncol(data_train)) * 0.95) : floor(sqrt(ncol(data_train) * 1.05)))
  
  start_time_rf = Sys.time() # Model Start Time 
  rf.fit = train(Label~., 
                 data = data_train,
                 method = "rf", 
                 trControl = control,
                 ntree = 500, #number of trees to grow
                 tuneGrid = rfGrid) # Parameter Tuning
  end_time_rf = Sys.time() # Model End time
  end_time_rf - start_time_rf
  
  rf_time = end_time_rf - start_time_rf #Total Running Time
  
  return(list(fit = rf.fit, time = rf_time))
}



# 4. Logistic Regression 

train_lr.cv <- function(data_train, K=10){
  
  library(nnet)
  
  start_time_lr = Sys.time() # Model Start Time
  
  # K-folder cross-validation
  
  n <- nrow(data_train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))
  models <- list()
  cv.error.log <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- data_train[s != i,]
    test.data <- data_train[s == i,]
    
    fit.log <-  multinom(Label~., data = data_train,MaxNWts=16000)
    pred.log <-predict(fit.log, test.data, type = "class") 
    cv.error.log[i] <- mean(pred.log != test.data$Label)
    models[[i]] <- fit.log
  }
  
  end_time_lr = Sys.time() # Model End time
  end_time_lr - start_time_lr
  
  lr_time = end_time_lr - start_time_lr #Total Running Time
  
  
  return(list(fit = models[[which.min(cv.error.log)]], time = lr_time))
}






