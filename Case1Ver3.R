
# clear dir etc 
rm(list=ls())

#### import relevant packages ####

# for making trees: 
library(rpart)
library(party)
library(rpart.plot)
library(mboost)
library(randomForest)

# for one-hot encoding
library(onehot)

# for elastic net
library('lars') 
library('glmnet')
library('cvTools')  

# to handle missing values 
library("missForest")

# for parameter tuning: 
library(mlbench)
library(mltools)
library(caret)

library(vtreat)
#### import data ####

library(cvTools)

# change dir: 
path = "C:\\Users\\Nicolaj\\Documents\\GitHub\\Case1"
setwd(path)

# import data: 
data = read.csv("case1Data.txt",sep=",")
#dataANTON = read.csv("C:\\Users\\Nicolaj\\OneDrive - Danmarks Tekniske Universitet\\DTU mapper\\6. semester\\ComputationalDataAnalysis\\Cases\\Case1\\anton\\Case Data\\case1Data.txt",sep=",")
train.test = read.csv("split.csv",sep=",")



# the first col is y, the last 5 columns contains categorical features
# w missing values, NaN:
index.cat = (dim(data)[2]-4):(dim(data)[2])


nan.class = as.character(data$C_1[1])
data[data==nan.class] <- NA 
data[,index.cat] = lapply(data[,index.cat], factor)


train = data[train.test$train,]
test = data[train.test$test,]



#### Feature selection  ####
# made with 

# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
# impute missing values and onehot encode

y.train = train$y

# do not impute with y
set.seed(123)
train.impute = missForest(train[,-1])$ximp
dummy = dummyVars(" ~ .", data=train.impute)
train.one.hot=data.frame('y'=y.train,predict(dummy, newdata = train.impute))
train.one.hot = preProcess(train.one.hot)


# take out correlation with y and sort 
cor_y = sort(abs(cor(train.one.hot)[,1]))
plot(cor_y[-length(cor_y)])

cor_y[(length(cor_y)-7):(length(cor_y)-1)]

# there is like 6-7 important features

control <- trainControl(method="repeatedcv", number=10, repeats=3)

model <- train(y~., data=train.one.hot, method="rf", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE,order=T)
# order of importance: 
row_names = rownames(importance$importance)
ord_im = order(importance$importance,decreasing = F)

plot(ord_im)

row_names[ord_im][(length(ord_im)-10):(length(ord_im))]


# reruce feature eleminitaion

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(train.one.hot[,-1], train.one.hot[,1], sizes=c(1:length(train.one.hot)), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)   # x_42, x_73 and x_52 
# plot the results
plot(results, type=c("g", "o"))




# draw conclusion on input features

cor_y[(length(cor_y)-7):(length(cor_y)-1)]
row_names[ord_im][(length(ord_im)-10):(length(ord_im))]
head(results$variables,7)


# we should defintely use x_42, x_73 and x_52. One could include x95 as well 




#features = c('x_42','x_73','x_52','x_95')
features = c('x_42','x_73','x_52','x_51')

# data feature selected, fs
train.features = train[c('y',features)]


preProc.train <- preProcess(train.features[,-1], method = c("center", "scale"))


train.fs = cbind('y'=train[,1],predict(preProc.train,train.features[,-1]))
train.fs.x = train.fs[features]
train.fs.y = train.features['y']
n_train = dim(train.fs)[1]


test.fs = test[c('y',features)]
n_test = dim(test.fs)[1]


############################################################
########           Model selection  
############################################################


#### random forest ####

# taken from course materia
customRF <- list(type = "Regression", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree","nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree","nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

tunegrid <- expand.grid(.mtry=1:4,.ntree=c(1,5,10,15,20,50),.nodesize=c(1,5,10,20,30))
control <- trainControl(method="boot", number=10,search = "grid")

rf_train <- train(y~., data=train.fs, method=customRF, tuneGrid=tunegrid, trControl=control,importance=TRUE)
plot(rf_train)
# we see that mtry=4, ntree=20 and nodesize=5 seems reasonble 


rf.fit.train <-  randomForest(y~., data=train.fs,ntree=20,mtry=4,nodesize=5)


rmse_reg <- function(yhat=NULL, actual = NULL) {
  sqrt(mean((yhat-actual)^2))
}


RMSE_rf = rmse_reg(predict(rf.fit,train.fs[,-1]),train.fs[,1])


#### Elastic net to determine lambda and alpha ####

mse1se <- matrix(NA,101)
mseMin <- matrix(NA,101)


for (i in 0:100) {
  model = cv.glmnet(as.matrix(train.fs.x), as.matrix(train.fs.y), type.measure="mse", 
                                            alpha=i/100,family="gaussian")
  
  mse1se[i+1] = sqrt(mean((as.matrix(train.fs.y) - predict(model,model$lambda.1se, newx=as.matrix(train.fs.x)))^2))
  mseMin[i+1] = sqrt(mean((as.matrix(train.fs.y) - predict(model,model$lambda.min, newx=as.matrix(train.fs.x)))^2))
}

plot((0:100)/100,mse1se,xlab=expression(alpha))
plot((0:100)/100,mseMin,xlab=expression(alpha))

which.min(mse1se)

# it seems that alpha=1 is OK. i.e. lasso is good. We use the mse 1se hence 
lasso.fit.train.opt= cv.glmnet(as.matrix(train.fs.x), as.matrix(train.fs.y), type.measure="mse", 
                  alpha=1,family="gaussian")
lasso.fit.train = glmnet(as.matrix(train.fs.x), as.matrix(train.fs.y),alpha=1,lambda =lasso.fit.train.opt$lambda.1se )








##################################################
########   Sampling with found models 




# parameters for boostraping
n_train = dim(train.fs)[1]
n_takeout = floor(n_train*0.63)

# number of boostrap samples: 
n.boots.sample = 100
n_boots_rep = 500

# number of models: 
n_model = 2
#models = list(rf.fit,lasso.fit)

RMSE_matrix = matrix(NA,n_boots_rep,n_model)

set.seed(123)

for (i in 1:n_boots_rep) {
  takeout = sample(1:n_train,n_takeout,replace=F)
  
  bootstrap = sample((1:n_train)[takeout],n.boots.sample,replace=T)
  
  # train models
  #train.boots = train.features[bootstrap,]
  # we cannot scale because it would be too dependt
  train.boots = train.fs[bootstrap,]
  preProcValues <- preProcess(train.boots[,-1], method = c("center", "scale"))
  
  train.boots = cbind('y'=train.boots[,1],predict(preProcValues,train.boots[,-1]))
  
  
  rf.fit <-  randomForest(y~., data=train.boots,ntree=50,mtry=4,nodesize=5)
  lasso.fit <- glmnet(as.matrix(train.boots[-1]), as.matrix(train.boots[1]),alpha=1,lambda =lasso.fit.train.opt$lambda.1se)
  
  # calculate RMSE on out of bag samples 
  train.oob = cbind('y'=train.boots[-takeout,1],
                    predict(preProcValues,train.boots[-takeout,-1]))
  
  # calculate RMSE on lasso
  RMSE_lasso = rmse_reg(predict(lasso.fit,lasso.fit$lambda.1se, newx=as.matrix(train.oob[,-1])),train.oob[,1])
  RMSE_rf = rmse_reg(predict(rf.fit,train.oob[,-1]),train.oob[,1])

  RMSE_matrix[i,] = c(RMSE_lasso,RMSE_rf)
}


hist(RMSE_matrix[,1],main="Hist RMSE for Lasso reg")
hist(RMSE_matrix[,2],main="Hist RMSE for RF")



# check for normality: 



# fits normal distribution on the RMSE for RF 
require(fitdistrplus)
fit.norm.RMSE.rf = fitdist(RMSE_matrix[,2],"norm")

plot(fit.norm.RMSE.rf)

mu.rmse.train = as.numeric(fit.norm.RMSE.rf$estimate[1])
sigma.rmse.train = as.numeric(fit.norm.RMSE.rf$estimate[2])

# looks quite Gaussian. Now lets predict the RMSE interval for test set: 

mu.int.train = mu.rmse.train + qnorm(0.975) * c(-sigma.rmse.train,sigma.rmse.train)


#### Validation method on test set ####
test.fs = test[c('y',features)]

# preprocess with the data from train set
#test.fs = cbind('y'=test.features[,1],predict(preProc.train,test.features[,-1]))

# take down the fit from the wole traning set: 
set.seed(123)
rf.fit.train <-  randomForest(y~., data=train.features,ntree=20,mtry=4,nodesize=5)

# predict on the test set: 
test.rf.pred = predict(rf.fit.train,test.fs[,-1])

# calculate RMSE on test
test.rf.rmse = rmse_reg(test.rf.pred,test.fs[,1])







# try with CV 


K = 10; # K-fold cross-validation
R = 200; # number of repetitions 
folds <- cvFolds(n_train, K = K, R =R, type = "random") 

RMSE.lasso.matrix = matrix(NA,R,K)
RMSE.rf.matrix = matrix(NA,R,K)

set.seed(123)
for (r in 1:R){
  for (j in 1:K){ # K fold cross-validation
    train.cv = train.features[folds$subsets[folds$which!=j,r],]
    test.cv = train.features[folds$subsets[folds$which==j,r],]
    
    
    preProcValues <- preProcess(train.cv[,-1], method = c("center", "scale"))
    
    train.cv = cbind('y'=train.cv[,1],predict(preProcValues,train.cv[,-1]))
    
    test.cv = cbind('y'=test.cv[,1],
                    predict(preProcValues,test.cv[,-1]))
    
    
    rf.fit <-  randomForest(y~., data=train.cv,ntree=50,mtry=4,nodesize=5)
    lasso.fit <- glmnet(as.matrix(train.cv[-1]), as.matrix(train.cv[1]),alpha=1,lambda =lasso.fit.train.opt$lambda.1se)
    
    
    # calculate RMSE on lasso
    RMSE_lasso = rmse_reg(predict(lasso.fit,lasso.fit$lambda.1se, newx=as.matrix(test.cv[,-1])),test.cv[,1])
    RMSE_rf = rmse_reg(predict(rf.fit,test.cv[,-1]),test.cv[,1])
    
    RMSE.lasso.matrix[r,j] = RMSE_lasso
    RMSE.rf.matrix[r,j] = RMSE_rf
  }
  
}
  

hist(as.vector(RMSE.lasso.matrix),main="Hist Lasso")
hist(as.vector(RMSE.rf.matrix),main="Hist RF")



# fits normal distribution on the RMSE for RF 
fit.norm.RMSE.cv.lasso = fitdist(as.vector(RMSE.lasso.matrix),"norm")
fit.norm.RMSE.cv.rf = fitdist(as.vector(RMSE.rf.matrix),"norm")

plot(fit.norm.RMSE.cv.lasso)
plot(fit.norm.RMSE.cv.rf)


mu.rmse.cv.lasso = as.numeric(fit.norm.RMSE.cv.lasso$estimate[1])
sigma.rmse.cv.lasso = as.numeric(fit.norm.RMSE.cv.lasso$estimate[2])

mu.int.cv.lasso = mu.rmse.cv.lasso + qnorm(0.975) * c(-sigma.rmse.cv.lasso,sigma.rmse.cv.lasso)
mu.int.cv.lasso

mu.rmse.cv.rf = as.numeric(fit.norm.RMSE.cv.rf$estimate[1])
sigma.rmse.cv.rf = as.numeric(fit.norm.RMSE.cv.rf$estimate[2])

mu.int.cv.rf = mu.rmse.cv.rf + qnorm(0.975) * c(-sigma.rmse.cv.rf,sigma.rmse.cv.rf)
mu.int.cv.rf


# looks quite Gaussian. Now lets predict the RMSE interval for test set: 

mu.int.train = mu.rmse.train + qnorm(0.975) * c(-sigma.rmse.train,sigma.rmse.train)


#### Validation method on test set ####
test.fs = test[c('y',features)]

# preprocess with the data from train set
preProc.test <- preProcess(test.fs[,-1], method = c("center", "scale"))

test.fs = cbind('y'=test.features[,1],predict(preProc.test,test.fs[,-1]))



# take down the fit from the wole traning set: 
set.seed(123)
rf.fit.train <-  randomForest(y~., data=train.fs,ntree=50,mtry=4,nodesize=5)

# predict on the test set: 
test.rf.pred = predict(rf.fit.train,test.fs[,-1])

# calculate RMSE on test
test.rf.rmse = rmse_reg(test.rf.pred,test.fs[,1])


##### Predict xnew #####

data.all = data[c('y',features)]

preProc.all <- preProcess(data.all[,-1], method = c("center", "scale"))

data.fs = cbind('y'=data.all[,1],predict(preProc.all,data.all[,-1]))

set.seed(123)
rf.fit.train <-  randomForest(y~., data=train.fs,ntree=50,mtry=4,nodesize=5)

# import xnew

data.new = read.csv("case1Data_Xnew.txt",sep=",")

data.new.feature = data.new[c(features)]

preProc.xnew <- preProcess(data.new.feature, method = c("center", "scale"))

test.fs = predict(preProc.xnew,data.new.feature)


y.new.hat = predict(rf.fit.train,test.fs)
y.hat.new.num = as.numeric(y.new.hat)

write.csv(y.new.hat,"C:\\Users\\Nicolaj\\Documents\\GitHub\\Case1\\prediction_s184335_s210158.txt",row.names = F)
