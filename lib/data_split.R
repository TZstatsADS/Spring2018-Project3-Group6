###############################################################################
# This R file aims to find a balanced partition of training and testing data  #
# 'df_train' is the training data (80% or 64%)                                #
# 'df_test' is the testing data (20%)                                         #
###############################################################################


data_split = function(feature_name){
  
  ### feature name is a string contains the feature name
  
  library(caret)
  
  if("SIFT" %in% feature_name){
    df = read.csv('../data/SIFT_train.csv',header = F)
    df = df[,-1]
  }else{
    dir_df = paste0('../output/', feature_name, '.csv')
    df = read.csv(dir_df, header = F)
  }
  
  ### Load The Label Data
  label = read.csv('../data/label_train.csv', header = T)
  
  label = data.frame(label[,3])
  colnames(label) = c('Label') # Rename the Column
  
  set.seed(123)
  df_complete = cbind(label, df) # Combine the dataframe and label
  df_complete$Label = as.factor(df_complete$Label) # Make the Label as factor
  dpart = createDataPartition(df_complete$Label, p = 0.2, list = F) #Balanced Partition
  df_test = df_complete[dpart, ] # test set
  df_train = df_complete[-dpart, ] # training set
  
  return(list(df_train = df_train, df_test = df_test))
  
}