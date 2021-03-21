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

#### import data ####

# change dir: 
path = "C:\\Users\\Nicolaj\\Documents\\GitHub\\Case1"
setwd(path)

# import data: 
data = read.csv("case1Data.txt",sep=",")
train.test = read.csv("split.csv",sep=",")

# data is investigated in seperate file 
# the first col is y, the last 5 columns contains categorical features
# w missing values, NaN:
index.cat = (dim(data)[2]-4):(dim(data)[2])


#### data processing ####

# NA as an instance i.e. just factor the last columns 
data.inst = data.frame(data)
data.inst[,index.cat] = lapply(data.inst[,index.cat], factor)

# in the remaining we will convert the data to actual NA

nan.class = as.character(data$C_1[1])
data[data==nan.class] <- NA 
data[,index.cat] = lapply(data[,index.cat], factor)

# Impute the NAs using nonparametric method:
# only do it on the x subset: 
imp =  missForest(data[,-1])$ximp
data.impute = cbind('y'=data$y,imp)


# one hot encode 
dummy = dummyVars(" ~ .", data=data.impute)
data.one.hot=data.frame(predict(dummy, newdata = data.impute))
#data.one.hot = as.data.frame(predict(encoder,data))



# devide into test and train: 

train.inst = data.inst[train.test$train,]
test.inst = data.inst[train.test$test,]

levels(test.inst[,index.cat]) = lapply(levels(train.inst[,index.cat]), levels)


train.impute = data.impute[train.test$train,]
test.impute = data.impute[train.test$test,]

train.one.hot = data.one.hot[train.test$train,]
test.one.hot = data.one.hot[train.test$test,]

#### define evaluation function #####
rmse_reg <- function(model_obj, testing = NULL, actual = NULL) {
  #Calculates rmse for a regression decision tree
  #Arguments:
  # testing - test data set
  # target  - target variable (length 1 character vector)
  yhat <- predict(model_obj, newdata = testing)
  sqrt(mean((yhat-actual)^2))
}


######### model building 
# 1. missing values as instance (mvi)
# 2. impute NAs
# 3. impute NAs and use one-hot encoding 

#### missing values as instance ####

# regression trees

# we use 10 cross validation
set.seed(123)
regTree.inst=rpart(y~., data=train.inst, method="anova",control =rpart.control(minsplit =1,minbucket=1,xval=10))
printcp(regTree.inst)
# in the following we prune the tree using 1-SE 
CVerr=regTree.inst$cptable[,"xerror"]
minCP=regTree.inst$cptable[which.min(regTree.inst$cptable[,"xerror"]),"CP"]
minSE=min(CVerr)+regTree.inst$cptable[which.min(CVerr),"xstd"]
w=which.min(CVerr[CVerr>minSE])
wv=which(CVerr>minSE)[w]
CP_SE=regTree.inst$cptable[wv,"CP"]
# prune the tress
pruned.tree.inst <- prune(regTree.inst, cp = CP_SE)
prp(pruned.tree.inst)

# on traning data 
rmse_reg(regTree.inst,train.inst[,-1],train.inst[,1])
rmse_reg(pruned.tree.inst,train.inst[,-1],train.inst[,1])

# on test data
rmse_reg(regTree.inst,test.inst[,-1],test.inst[,1])
rmse_reg(pruned.tree.inst,test.inst[,-1],test.inst[,1])


## Random forest ##

# in other script number of trees 80, nodesize=5 test mtry:
bestmtry <- tuneRF(x=train.inst[,-1], y=train.inst[,1], stepFactor=1.5, improve=1e-10, ntree=80)
print(bestmtry)
# around 80 seems good 

# trains 
rf.inst = randomForest(x = train.inst[,-1],y = train.inst[,1],ntree=80,mtry=40,nodesize=5)

# on traning data 
rmse_reg(rf.inst,train.inst[,-1],train.inst[,1])

# to fix bug of different levels in train and test
test.inst <- rbind(train.inst[1, ] , test.inst)
test.inst <- test.inst[-1,]
# on test data
rmse_reg(rf.inst,test.inst[,-1],test.inst[,1])


# lets try to train more regiously
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


control <- trainControl(method="boot", number=5,search = "grid")
seed <- 7
set.seed(seed)
tunegrid <- expand.grid(.mtry=round(seq(1,100,len=4)),.ntree=c(1,5,10,20,100,150),.nodesize=c(1,5,10,20,30))

# takes ~ 5 minutes to run.
rf_train <- train(y~., data=train.inst, method=customRF, tuneGrid=tunegrid, trControl=control,importance=TRUE)
plot(rf_train)

# mtry could be aounrd 80, ntree= 80 is OK, and nodesize= 5
# trains 
rf.inst.revist = randomForest(x = train.inst[,-1],y = train.inst[,1],ntree=80,mtry=80,nodesize=5)

# on traning data 
rmse_reg(rf.inst.revist,train.inst[,-1],train.inst[,1])

# on test data
rmse_reg(rf.inst.revist,test.inst[,-1],test.inst[,1])



# now lets try gradient boosting

gb.inst = blackboost(y ~ .,data=train.inst)


# on traning data 
rmse_reg(gb.inst,train.inst[,-1],train.inst[,1])
# on test data
rmse_reg(gb.inst,test.inst[,-1],test.inst[,1])



#### missing values imputed #### 

# regression trees

# we use 10 cross validation
set.seed(123)
regTree.impute=rpart(y~., data=train.impute, method="anova",control =rpart.control(minsplit =1,minbucket=1,xval=10))
printcp(regTree.impute)
# in the following we prune the tree using 1-SE 
CVerr=regTree.impute$cptable[,"xerror"]
minCP=regTree.impute$cptable[which.min(regTree.impute$cptable[,"xerror"]),"CP"]
minSE=min(CVerr)+regTree.impute$cptable[which.min(CVerr),"xstd"]
w=which.min(CVerr[CVerr>minSE])
wv=which(CVerr>minSE)[w]
CP_SE=regTree.impute$cptable[wv,"CP"]
# prune the tress
pruned.tree.impute <- prune(regTree.impute, cp = CP_SE)
prp(pruned.tree.impute)

# on traning data 
rmse_reg(regTree.impute,train.impute[,-1],train.impute[,1])
rmse_reg(pruned.tree.impute,train.impute[,-1],train.impute[,1])

# on test data
rmse_reg(regTree.impute,test.impute[,-1],test.impute[,1])
rmse_reg(pruned.tree.impute,test.impute[,-1],test.impute[,1])


## Random forest ##

# in other script number of trees 80, nodesize=5 test mtry:
bestmtry <- tuneRF(x=train.impute[,-1], y=train.impute[,1], stepFactor=1.5, improve=1e-10, ntree=80)
print(bestmtry)
# around 100 seems good 

# trains 
rf.impute = randomForest(x = train.impute[,-1],y = train.impute[,1],ntree=500,mtry=100,nodesize=5)

# on traning data 
rmse_reg(rf.impute,train.impute[,-1],train.impute[,1])

# to fix bug of different levels in train and test
test.impute <- rbind(train.impute[1, ] , test.impute)
test.impute <- test.impute[-1,]
# on test data
rmse_reg(rf.impute,test.impute[,-1],test.impute[,1])

# grid search: 
tunegrid <- expand.grid(.mtry=round(seq(1,100,len=4)),.ntree=c(1,5,10,20,100,150),.nodesize=c(1,5,10,20,30))

# takes ~ 5 minutes to run.
rf_train <- train(y~., data=train.impute, method=customRF, tuneGrid=tunegrid, trControl=control,importance=TRUE)
plot(rf_train)


# again it seems that mtry=80,nodesize=5, ntree=80 are optimal parameterters

# now lets try gradient boosting

gb.impute = blackboost(y ~ .,data=train.impute)


# on traning data 
rmse_reg(gb.impute,train.impute[,-1],train.impute[,1])
# on test data
rmse_reg(gb.impute,test.impute[,-1],test.impute[,1])




#### using one-hot encoding ####

# regression trees

# we use 10 cross validation
set.seed(123)
regTree.one.hot=rpart(y~., data=train.one.hot, method="anova",control =rpart.control(minsplit =1,minbucket=1,xval=10))
printcp(regTree.one.hot)
# in the following we prune the tree using 1-SE 
CVerr=regTree.one.hot$cptable[,"xerror"]
minCP=regTree.one.hot$cptable[which.min(regTree.one.hot$cptable[,"xerror"]),"CP"]
minSE=min(CVerr)+regTree.one.hot$cptable[which.min(CVerr),"xstd"]
w=which.min(CVerr[CVerr>minSE])
wv=which(CVerr>minSE)[w]
CP_SE=regTree.one.hot$cptable[wv,"CP"]
# prune the tress
pruned.tree.one.hot <- prune(regTree.one.hot, cp = CP_SE)
prp(pruned.tree.one.hot)

# on traning data 
rmse_reg(regTree.one.hot,train.one.hot[,-1],train.one.hot[,1])
rmse_reg(pruned.tree.one.hot,train.one.hot[,-1],train.one.hot[,1])

# on test data
rmse_reg(regTree.one.hot,test.one.hot[,-1],test.one.hot[,1])
rmse_reg(pruned.tree.one.hot,test.one.hot[,-1],test.one.hot[,1])


## Random forest ##

# in other script number of trees 80, nodesize=5 test mtry:
bestmtry <- tuneRF(x=train.one.hot[,-1], y=train.one.hot[,1], stepFactor=1.5, improve=1e-10, ntree=80)
print(bestmtry)
# around 100 seems good 

# trains 
rf.one.hot = randomForest(x = train.one.hot[,-1],y = train.one.hot[,1],ntree=80,mtry=100,nodesize=5)

# on traning data 
rmse_reg(rf.one.hot,train.one.hot[,-1],train.one.hot[,1])

# to fix bug of different levels in train and test
test.one.hot <- rbind(train.one.hot[1, ] , test.one.hot)
test.one.hot <- test.one.hot[-1,]
# on test data
rmse_reg(rf.one.hot,test.one.hot[,-1],test.one.hot[,1])

# grid search: 
tunegrid <- expand.grid(.mtry=round(seq(1,100,len=4)),.ntree=c(1,5,10,20,100,150),.nodesize=c(1,5,10,20,30))

# takes ~ 5 minutes to run.
rf_train <- train(y~., data=train.one.hot, method=customRF, tuneGrid=tunegrid, trControl=control,importance=TRUE)
plot(rf_train)



# now lets try gradient boosting

gb.one.hot = blackboost(y ~ .,data=train.one.hot)


## ELASTIC NET ##

x.train.hot = as.matrix(train.one.hot[,-1])
y.train.hot = as.vector(train.one.hot[,1])

x.test.hot = as.matrix(test.one.hot[,-1])
y.test.hot = as.vector(test.one.hot[,1])

mse1se <- c()
mseMin <- c()
mse1seOut <- c()

for (i in 0:100) {
  assign(paste("fit", i, sep=""), cv.glmnet(x.train.hot, y.train.hot, type.measure="mse", standardize = T, 
                                            alpha=i/100,family="gaussian"))
  
  mse1se = append(mse1se, mean((y.test.hot - predict(eval(parse(text =  paste("fit", i, sep=""))), s=eval(parse(text =  paste("fit", i, sep="")))$lambda.1se, newx=x.test.hot))^2))
  mseMin = append(mseMin, mean((y.test.hot - predict(eval(parse(text =  paste("fit", i, sep=""))), s=eval(parse(text =  paste("fit", i, sep="")))$lambda.min, newx=x.test.hot))^2))
  #mse1seOut = append(mse1seOut, mean((y_out - predict(eval(parse(text =  paste("fit", i, sep=""))), s=eval(parse(text =  paste("fit", i, sep="")))$lambda.1se, newx=x_out))^2))
}


plot((0:100)/100,mse1se,xlab=expression(alpha),main=expression(paste("MSE w. largest ",lambda," within 1 SE of ",lambda[min])))
plot((0:100)/100,mseMin,xlab=expression(alpha),main=expression(paste("MSE w. ",lambda[min])))


# on traning data 
rmse_reg(gb.one.hot,train.one.hot[,-1],train.one.hot[,1])
# on test data
rmse_reg(gb.one.hot,test.one.hot[,-1],test.one.hot[,1])

