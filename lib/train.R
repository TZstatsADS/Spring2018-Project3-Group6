#########################################################
### Train a classification model with training images ###
#########################################################

### Author: Group 6
### Project 3
### ADS Spring 2016


##### Baseline Model: GBM #####

train_gbm <- function(data_train){
  
  ###  dat_train: processed features from images also contains label
  
  library(gbm)
  
  start_time_gbm = Sys.time() # Model Start Time
  
  gbm.fit = gbm(Label~., 
                data = dat_train,
                n.trees = 400,
                distribution = "multinomial",
                interaction.depth = 3, 
                shrinkage = 0.2,
                n.minobsinnode = 30,
                verbose=FALSE)
  end_time_gbm = Sys.time() # Model End time
  gbm_time = end_time_gbm - start_time_gbm #Total Running Time
  
  
  return(list(fit = gbm.fit, time = gbm_time))
}
