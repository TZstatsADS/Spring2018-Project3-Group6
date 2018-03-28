
################ RGB Feature #################

library(EBImage)

mat<- list()
freq_RGB<- list()
nR<- 6
nG<- 10
nB<- 10
Rbin<- seq(0, 1, length.out=nR)
Gbin<- seq(0, 1, length.out=nG)
Bbin<- seq(0, 1, length.out=nB)
RGB_features<- matrix(nrow=3000, ncol=nR * nG * nB)
dim(RGB_features)   #600 features

files<- list()
img_dir<- "/Users/mengqichen/Desktop/Pro3/train/images/"

#  "/Users/mengqichen/Desktop/Pro3/image.RData";  /3000.jpg"

for(i in 1:3000){
  mat<- imageData(readImage(paste0(img_dir, sprintf("%04.f", i), ".jpg")))
  mat_RGB <-array(c(mat, mat, mat), dim =c(nrow(mat), ncol(mat), 3))
  freq_RGB[[i]]<- as.data.frame(table(factor(findInterval(mat_RGB[, , 1], Rbin), levels=1:nR), 
                                      factor(findInterval(mat_RGB[, , 2], Gbin), levels=1:nG), 
                                      factor(findInterval(mat_RGB[, , 3], Bbin), levels=1:nB)))
  RGB_features[i, ]<- as.numeric(freq_RGB[[i]]$Freq) / (ncol(mat) * nrow(mat))
}

RGB_features<- data.frame(RGB_features)

label_train<- read.csv("/Users/mengqichen/Desktop/Pro3/train/label_train.csv")[, 2]
names(label_train)<-"y"
RGB_features$y<- label_train
write.csv(RGB_features,file="/Users/mengqichen/Documents/Github/Spring2018-Project3-Group6/output/rgb_feature.csv", row.names = FALSE)


