
# for making trees: 
library(rpart)
library(party)
library(rpart.plot)
library(mboost)
library(randomForest)
library(missf)

library(onehot)

# for elastic net
library('lars') 
library('glmnet')
library('cvTools')  

# to handle missing values 
library(mice)
library("missForest")

# clear dir etc 
rm(list=ls())

# change dir: 
path = "C:\\Users\\Nicolaj\\Documents\\GitHub\\Case1"
setwd(path)

# import data: 
data = read.csv("case1Data.txt",sep=",")

#split to x and y: 

y = as.vector(data[,1])
x = as.data.frame(data[,-1])



# define into test and train
n = dim(x)[1] # number of observations
p = dim(x)[2] # number of paramter estimates

plot(y)


# it doesn't seem to look like a times series hence we can randomly split the data 
## set the seed to make your partition reproducible

# we subset 20% for testing and 80% for traning: 

smp_size = 80 
set.seed(123456789)

train_ind <- sample(100, size = smp_size)
test <- (1:100)[-train_ind]

splitting = data.frame('train'=train_ind,'test'=test)
write.csv(splitting,"C:\\Users\\Nicolaj\\Documents\\GitHub\\Case1\\split.csv",row.names = F)

ytr = y[train_ind]
xtr = x[train_ind,]
ytest = y[-train_ind]
xtest = x[-train_ind,]

# notice that y is numeric hence we want to do a regression model. 

# make a dataframe


partFram = cbind('yn'=ytr,xtr)
# save an additional dataframe for reference 
partFram_wo_factor = cbind('yn'=ytr,xtr)

# to makes sure that it knows that the last columns are categorical we use factor and treat 
# nan as an instance: 
# try
# factor(partFram[,(dim(partFram)[2]-4)])
# head(partFram[,(dim(partFram)[2]-4):(dim(partFram)[2])])
indexCat = (dim(partFram)[2]-4):(dim(partFram)[2])

partFram[,indexCat]=lapply(partFram[,indexCat], factor)


# 
fitTree=rpart(yn~., data=partFram, method="anova",control =rpart.control(minsplit =1,minbucket=1))
printcp(fitTree)

fitTree2=rpart(yn~., data=partFram_wo_factor, method="anova",control =rpart.control(minsplit =1,minbucket=1))
printcp(fitTree2)

# check the difference: 

fitTree$cptable-fitTree2$cptable

# it doesnt seem to make a difference on wether we factor or not. We can see how rpart recognizes the 
# last columns using: 

fitTree2$terms
fitTree$terms


# let us do a fitting with cross valition set to 5: 
fitTree=rpart(yn~., data=partFram, method="anova",control =rpart.control(minsplit =1,minbucket=1,xval=5))

fitTree$y-partFram$yn
ymod_1 = predict(fitTree,xtr)

plot(ytr,lty=1)
lines(ymod_1,lty=3)

summary(fitTree)

rsq.rpart(fitTree)

# R automatically find the best cp=alpha parameter hence we dont have to think about that   
# what we want to do is to find out what the number of splits should be using the 1-SE rule

# we now want to find the cp=alpha parameter that minmizeses the crossvalidation error using 1-SE rule

# below each we get at number, it would also have som col names in some of the calcualtions.
# the lowest number is the indexing number from e.g. w



## first find the number of splits that gives the smallets crovalidation error 
CVerr=fitTree$cptable[,"xerror"]
# then take that out and take out all the alpha values for this best number of splits
minCP=fitTree$cptable[which.min(fitTree$cptable[,"xerror"]),"CP"]

# take out the minSE for the number of splits that minimized error in CV and add to it the mean of the 
# CV 
minSE=min(CVerr)+fitTree$cptable[which.min(CVerr),"xstd"]

# of the mean CV values above the minSE, find the lowest
w=which.min(CVerr[CVerr>minSE])

# of the CV values above find the index in the CVerr
wv=which(CVerr>minSE)[w]

# take out the control parameter of the three with the optimal splits. 
CP_SE=fitTree$cptable[wv,"CP"]


# since the three is optimized with control parameters, we can now prune the three using the 
# control parameters. 

pruned.tree <- prune(fitTree, cp = CP_SE)
prp(pruned.tree)


# lets make a function to calculate the RMSE. Remeber that for rpart we need the entire 
# dataframe to use rpart. Therefore we have to give it the dataframe and the col name of
# the y we want to test against

rmse_reg <- function(model_obj, testing = NULL, target = NULL) {
  #Calculates rmse for a regression decision tree
  #Arguments:
  # testing - test data set
  # target  - target variable (length 1 character vector)
  yhat <- predict(model_obj, newdata = testing)
  actual <- testing[[target]]
  sqrt(mean((yhat-actual)^2))
}



# test the performce of prunes tree 
rmse_reg(pruned.tree,partFram,'yn')

# lets try the test data: 
testFrame = cbind('y'=ytest,xtest)

rmse_reg(pruned.tree,testFrame,'y')


# test the performance of the full tree: 
rmse_reg(fitTree,partFram,'yn')

# test performace on test data
rmse_reg(fitTree,testFrame,'y')
# it has greatly overfit 



# gradient bosting: 

bb = blackboost(yn ~ .,data=partFram)


rmse_reg(bb,partFram,'yn')

# greatly reduced the number RMS 

# to test on test-data

testFrame = cbind('y'=ytest,xtest)
testFrame[,indexCat]=lapply(testFrame[,indexCat], factor)

rmse_reg(bb,testFrame,'y')
# this is also greatly reduced 



#### Random Forrest ####

# define a vector of different number of trees. We let it increment by 3 simply to simply avoid 
# much computations
 
n_tree = seq(2,150,3) # from=2,to=80,by=3
mse = rep(NA,length(n_tree)); # prepare vector for all the mse values

# run a for-loop to see the MSE fall as the nunber of trees grow:
for (b in 1:length(n_tree)) {
  rf=randomForest(x = partFram[,-1],y = partFram[,1],ntree=n_tree[b],importance=T,proximity=T,mtry=80,nodesize=5)
  mse[b] = rf$mse[n_tree[b]] # Out-of-bag-estimate (take out the last as it would be the comult)
}


plot(n_tree,mse,type="o",
     xlab="Number of trees",
     ylab="MSE")



# we see that around 80, it starts it be OK. 
n_tree_opt = 80

# now we investigate the number of varibles to sample 
# in Bloomberg it is set to ntry=sqrt(number of parameters)=sqrt(100)=10
# we do the investigation:

# at most the number of parameters:
m_vec = seq(5,n,5)

mse = rep(NA,length(m_vec)); # prepare vector for all the mse values

for (m in 1:length(m_vec)) {
  rf=randomForest(x = partFram[,-1],y = partFram[,1],ntree=n_tree_opt,mtry=m_vec[m],nodesize=5)
  mse[m] = rf$mse[n_tree_opt] # Out-of-bag estimate of MSE
}

plot(m_vec,mse,type="o",
     xlab="Number of variables to sample",
     ylab="OOB Misclassification error")

# perhaps more formal find but I found the 60 seems good. 
tune = tuneRF(x = partFram[,-1],y = partFram[,1], mtryStart = 60, ntreeTry=n_tree_opt, stepFactor=2, improve=0.05,
       trace=TRUE, plot=TRUE, doBest=FALSE)


# we now try to find the best number of leaf nodes: 

nodesizevec = seq(1,30,1)

mse = rep(NA,length(nodesizevec)); # prepare vector for all the mse values

for (no in 1:length(nodesizevec)) {
  rf=randomForest(x = partFram[,-1],y = partFram[,1],ntree=n_tree_opt,mtry=60,nodesize=nodesizevec[no])
  mse[no] = rf$mse[n_tree_opt] # Out-of-bag estimate of MSE
}

plot(nodesizevec,mse,type="o",
     xlab="Minimum size of terminal nodes",
     ylab="OOB Misclassification error")


# we use 5 

n_tree_opt = 80
mtry_opt = 60 
nodesize_opt = 5

bestmtry <- tuneRF(x=train.inst[,-1], y=train.inst[,1], stepFactor=1.5, improve=1e-10, ntree=500)
print(bestmtry)

# we make the last randomforrest: 
rf=randomForest(x = partFram[,-1],y = partFram[,1],ntree=n_tree_opt,mtry=mtry_opt,nodesize=nodesize_opt)





rmse_reg(rf,partFram,'yn')

rmse_reg(rf,testFrame,'y')

# it seems that we are overfitting as there is a great discrepancy between test and train performance. 





#### Elastic net ####
# before we have simply treated NA as an instrance. Lets make it an actual NA: 
data = read.csv("case1Data.txt",sep=",")

nan_class = as.character(data$C_1[1]);
data$C_1 = replace(data$C_1, data$C_1 == nan_class, NA);
data$C_2 = replace(data$C_2, data$C_2 == nan_class, NA);
data$C_3 = replace(data$C_3, data$C_3 == nan_class, NA);
data$C_4 = replace(data$C_4, data$C_4 == nan_class, NA);
data$C_5 = replace(data$C_5, data$C_5 == nan_class, NA);

indexCat = (dim(data)[2]-4):(dim(data)[2])
# we can now factor the charactor columns
data[,indexCat]=lapply(data[,indexCat], factor)


smp_size = 80 
set.seed(123)
train_ind <- sample(n, size = smp_size)

train = data[train_ind,]
test = data[-train_ind,]

# devide into x and y
y.train = as.vector(train[,1])
x.train = as.data.frame(train[,-1])

y.test = as.vector(test[,1])
x.test = as.data.frame(test[,-1])

# we can try to impute missing values:
# here just using missForest which is a random forrest to predict it: 
missf = missForest(x.train)


# we now one-hot encode the data
encoder <- onehot(x.train)
x.train.hot <- as.matrix(predict(encoder,x.train))
x.test.hot <- as.matrix(predict(encoder,x.test))

# we have now added: 
print(paste(dim(x.hot)[2] - n, " new varibles"))


# we now do the Elastic net for different alphas remember that 
mse1se <- c()
mseMin <- c()
mse1seOut <- c()

for (i in 0:100) {
  assign(paste("fit", i, sep=""), cv.glmnet(x.train.hot, y.train, type.measure="mse", 
                                            alpha=i/100,family="gaussian"))
  
  mse1se = append(mse1se, mean((y.test - predict(eval(parse(text =  paste("fit", i, sep=""))), s=eval(parse(text =  paste("fit", i, sep="")))$lambda.1se, newx=x.test.hot))^2))
  mseMin = append(mseMin, mean((y.test - predict(eval(parse(text =  paste("fit", i, sep=""))), s=eval(parse(text =  paste("fit", i, sep="")))$lambda.min, newx=x.test.hot))^2))
  #mse1seOut = append(mse1seOut, mean((y_out - predict(eval(parse(text =  paste("fit", i, sep=""))), s=eval(parse(text =  paste("fit", i, sep="")))$lambda.1se, newx=x_out))^2))
}


plot((0:100)/100,mse1se,xlab=expression(alpha),main=expression(paste("MSE w. largest ",lambda," within 1 SE of ",lambda[min])))
plot((0:100)/100,mseMin,xlab=expression(alpha),main=expression(paste("MSE w. ",lambda[min])))

# we see that we get better performance for larger alpha. 
# remember: 
# (1-alpha)/2 sum(vp_jb_j^2) + alpha sum(vp_j)|beta|
# i.e. if alpha is great more lasso L1 regulization while small alpha would be more L2 and ridge
# we have more Lasso here the better. REMEMEBER that CV uses crossvalidation default number of folds
# is 10!!!!

# we can now take out the one with the higest lambda witin 1 se of the lambda that minimizes MSE 
rmse_reg_elas <- function(model_obj, testing = NULL, target = NULL) {
  #Calculates rmse for a regression decision tree
  #Arguments:
  # testing - test data set
  # target  - target variable (length 1 character vector)
  yhat <- predict(model_obj, newx = testing)
  actual <- testing[[target]]
  sqrt(mean((yhat-actual)^2))
}




index_min = which.min(mse1se)
alpha_opt = (index_min-1)/100

# lest investigate the best i.e. index_min-1
fit93

x.train.hot <- as.matrix(predict(encoder,x.train))
x.test.hot <- as.matrix(predict(encoder,x.test))


rmse_elas_train = sqrt(mean((predict(fit93,x.train.hot)-y.train)^2))

rmse_elas_test = sqrt(mean((predict(fit93,x.test.hot)-y.test)^2))

print(paste('train RMSE', rmse_elas_train,' test RMSE', rmse_elas_test))


