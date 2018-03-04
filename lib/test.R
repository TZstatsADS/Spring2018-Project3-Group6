######################################################
### Fit the classification model with testing data ###
######################################################

### Author: Group 6
### Project 3
### ADS Spring 2016

########### Testing Function For GBM Model ###########

test_gbm <- function(fit_train, dat_test){
  library("gbm")
  pred = predict(fit_train$fit, newdata=dat_test,n.trees = 400, type="response")
  gbm_prob = predict(fit_train$fit, n.trees = 400, newdat =  dat_test, type="response")
  gbm_class = apply(gbm_prob, 1, which.max)
  return(gbm_class)
}

