##########  Training image preparation : resize all pictures and save as RData##########################

###########################                Mengqi Chen--Mar 3
## 
## Packages and Set working directory
source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage")
library(EBImage)
setwd("/Users/mengqichen/Desktop/Pro3") 

## Import Images
temp = list.files(path="./train/images/", pattern="*.jpg", all.files = T,full.names = T)
img.all = lapply(temp, readImage) 
class(img.all) # list
img.all1 <- img.all

## Adjust to the same image size
width <- 128
height <- 128
img.all2 <- list()
for (i in 1:3000) {
  img.all2[[i]] <- resize(img.all1[[i]], width, height)
}

## Save as RData, saving time to load into
save(img.all2, file ="image.RData")


