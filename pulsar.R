
#Libraries to use
library(MASS)
library(class)
library(e1071)
library(nnet)
library(rpart)
library(rpart.plot)
library(rattle)

total = 8

# This bit of code sets up the working directory and get the data in
scriptPath <- function() {
  getSrcDirectory(scriptPath);
}
wd=scriptPath()
setwd(wd)

#Reading the data into our R envrionment and setting up the data is done here
pulsar = read.csv("pulsar_stars.csv")
pulsar = pulsar[order(pulsar$target_class),]


#Drawing some plots to look at the density of the covariates
par(mfrow=c(1, 2))
ps = pulsar[which(pulsar$target_class==1),]
nps = pulsar[which(pulsar$target_class==0),]
colnames <- c("Mean IP","SD IP", "Excess Kurtosis IP", "Skew IP","Mean DM SNR","SD DM SNR", "Excess Kurtosis DM SNR", "Skew DM SNR")
for (i in 1:8) {

  d.pos = density(ps[,i])
  d.neg = density(nps[,i])
  hist(ps[,i], probability =TRUE, main=paste("Positive Density",colnames[i]))
  lines(d, col="red")
  hist(nps[,i],probability = TRUE,main=paste("Negative Density",colnames[i]))
  lines(d, col="blue")
}

#Setting the seed so we can get similar results every time
set.seed(47)

#Use about a fifth of the data for each cross validation set
rows = 1:17898
test_index1 = sample(1:17898,3580)
rest1 = rows[-test_index1]
test_index2 = sample(rest1,3580)
rest2 = rows[-test_index2]
test_index3 = sample(rows[-c(test_index1,test_index2)],3580)
rest3 = rows[-test_index3]
test_index4 = sample(rows[-c(test_index1,test_index2,test_index3)],3579)
rest4 = rows[-test_index4]
test_index5 = rows[-c(test_index1,test_index2,test_index3,test_index4)]
rest5 = rows[-test_index5]

#Descriptive Statistics about the data and different variables in the data set:
sapply(pulsar,mean)
sapply(pulsar,sd)
pos  = which(pulsar$target_class==1)
sapply(pulsar[pos,],mean)
sapply(pulsar[pos,],sd)
sapply(pulsar[-pos,],mean)
sapply(pulsar[-pos,],sd)

#Useful Functions:
ConfusionMatrix = function(p1,p2,p3,p4,p5){
  C_P = as.data.frame(matrix(nrow=17898,ncol=2))
  colnames(C_P) = c("True","Predicted")
  C_P[,1] = pulsar[,9]
  C_P[test_index1,2] = p1
  C_P[test_index2,2] = p2
  C_P[test_index3,2] = p3
  C_P[test_index4,2] = p4
  C_P[test_index5,2] = p5
  
  C.Matrix = table(C_P[,1],C_P[,2])
  if(dim(C.Matrix)[2]==1){
    print(C.Matrix)
    return(C.Matrix)
  }
  else{
    colnames(C.Matrix) <- c("Predicted=0","Predicted=1")
    rownames(C.Matrix) <- c("True=0","True=1")
    print(C.Matrix)
    
    FP = C.Matrix[2,1]
    TP = C.Matrix[2,2]
    FN = C.Matrix[1,2]
    TN = C.Matrix[1,1]
    #Calculating the False Positive:
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TN)
    Error.Rate = (FP+FN)/(FP+FN+TP+TN)
    print(paste("The False Positive Rate is ",FPR))
    print(paste("The False Negative Rate is ",FNR))
    print(paste("The Error Rate is ",Error.Rate))
    return(C.Matrix)  
    
  }
}

#Scaling and centering data:
pulsar[,1:8] = scale(pulsar[,1:8])

#in order to time how long our code takes
time.tests = rep(0,5)
names(time.tests) = c("SVM","KNN","Logistic Regression","Neural Network","Tree")
# 
# #First Method: Linear and Quadratic Discriminant Analysis
#Commented out to allow code to run faster
start.time = proc.time()
svmfit1 = svm(as.factor(target_class)~.,data=pulsar[rest1,],kernel="linear",scale=F)
svmfit2 = svm(as.factor(target_class)~.,data=pulsar[rest2,],kernel="linear",scale=F)
svmfit3 = svm(as.factor(target_class)~.,data=pulsar[rest3,],kernel="linear",scale=F)
svmfit4 = svm(as.factor(target_class)~.,data=pulsar[rest4,],kernel="linear",scale=F)
svmfit5 = svm(as.factor(target_class)~.,data=pulsar[rest5,],kernel="linear",scale=F)
svmpredict1 = predict(svmfit1,newdata = pulsar[test_index1,])
svmpredict2 = predict(svmfit2,newdata = pulsar[test_index2,])
svmpredict3 = predict(svmfit3,newdata = pulsar[test_index3,])
svmpredict4 = predict(svmfit4,newdata = pulsar[test_index4,])
svmpredict5 = predict(svmfit5,newdata = pulsar[test_index5,])
print("SVM Confusion Matrix")
ConfusionMatrix(svmpredict1,svmpredict2,svmpredict3,svmpredict4,svmpredict5)
time.tests[1] = (proc.time() - start.time)[3]
start.time = proc.time()

#Second method: K-Nearest Neighbors
#k_tuned =tune.knn(x=pulsar[,-9],y=as.factor(pulsar[,9]),k=1:10,tunecontrol=tune.control(sampling="cross",cross=5))
#Tuning gives k of 7
k=7
knn1 <- knn(train = pulsar[rest1,1:8],test = pulsar[test_index1,1:8],cl=pulsar[rest1,9],k=k)
knn2 <- knn(train = pulsar[rest2,1:8],test = pulsar[test_index2,1:8],cl=pulsar[rest2,9],k=k)
knn3 <- knn(train = pulsar[rest3,1:8],test = pulsar[test_index3,1:8],cl=pulsar[rest3,9],k=k)
knn4 <- knn(train = pulsar[rest4,1:8],test = pulsar[test_index4,1:8],cl=pulsar[rest4,9],k=k)
knn5 <- knn(train = pulsar[rest5,1:8],test = pulsar[test_index5,1:8],cl=pulsar[rest5,9],k=k)
print("KNN Confusion Matrix with 7 nearest neighbors")
ConfusionMatrix(knn1,knn2,knn3,knn4,knn5)
time.tests[2] = (proc.time() - start.time)[3]
start.time = proc.time()


#Third Method: Logistic Regression Classifiers

logitmodel1 = glm(target_class~.,data=pulsar[rest1,],family="binomial")
glm1 = round(predict.glm(logitmodel1,type="response",newdata=pulsar[test_index1,1:8]))
logitmodel2 = glm(target_class~.,data=pulsar[rest2,],family="binomial")
glm2 = round(predict.glm(logitmodel2,type="response",newdata=pulsar[test_index2,1:8]))
logitmodel3 = glm(target_class~.,data=pulsar[rest3,],family="binomial")
glm3 = round(predict.glm(logitmodel3,type="response",newdata=pulsar[test_index3,1:8]))
logitmodel4 = glm(target_class~.,data=pulsar[rest4,],family="binomial")
glm4 = round(predict.glm(logitmodel4,type="response",newdata=pulsar[test_index4,1:8]))
logitmodel5 = glm(target_class~.,data=pulsar[rest5,],family="binomial")
glm5 = round(predict.glm(logitmodel5,type="response",newdata=pulsar[test_index5,1:8]))
print("Logistic Regression Confusion Matrix")
ConfusionMatrix(glm1,glm2,glm3,glm4,glm5)

time.tests[3] = (proc.time() - start.time)[3]
start.time = proc.time()
# Fourth Method: Neural Network Classifiers
#tune.nnet(x=pulsar[,1:8],y=pulsar[,9],data=pulsar,size=1:8,trace=FALSE,tunecontrol = tune.control(sampling="cross",cross=5))
#Suggests a size of 7
sz=7
nnetfit1 = nnet(x=pulsar[,1:8],y=class.ind(pulsar$target_class),data=pulsar,size=sz,subset=rest1,softmax = T,trace=FALSE)
nnetfit2 = nnet(x=pulsar[,1:8],y=class.ind(pulsar$target_class),data=pulsar,size=sz,subset=rest2,softmax = T,trace=FALSE)
nnetfit3 = nnet(x=pulsar[,1:8],y=class.ind(pulsar$target_class),data=pulsar,size=sz,subset=rest3,softmax = T,trace=FALSE)
nnetfit4 = nnet(x=pulsar[,1:8],y=class.ind(pulsar$target_class),data=pulsar,size=sz,subset=rest4,softmax = T,trace=FALSE)
nnetfit5 = nnet(x=pulsar[,1:8],y=class.ind(pulsar$target_class),data=pulsar,size=sz,subset=rest5,softmax = T,trace=FALSE)

nfit1 = predict(nnetfit1,pulsar[test_index1,])
nfit2 = predict(nnetfit2,pulsar[test_index2,])
nfit3 = predict(nnetfit3,pulsar[test_index3,])
nfit4 = predict(nnetfit4,pulsar[test_index4,])
nfit5 = predict(nnetfit5,pulsar[test_index5,])

nfit1 = round(nfit1[,2])
nfit2 = round(nfit2[,2])
nfit3 = round(nfit3[,2])
nfit4 = round(nfit4[,2])
nfit5 = round(nfit5[,2])

print("Neural Network Confusion Matrix")
ConfusionMatrix(nfit1,nfit2,nfit3,nfit4,nfit5)

time.tests[4] = (proc.time() - start.time)[3]
start.time = proc.time()
# Fifth Method: Tree based Classifiers
#tune.rpart(target_class~.,data=pulsar,minsplit=1:10,tunecontrol = tune.control(sampling="cross",cross=5))
#tune.rpart(target_class~.,data=pulsar,cp=seq(.0001,.01,.0005),tunecontrol = tune.control(sampling="cross",cross=5))

#The tuning gives a minimum split of 2 and a cp of .0021, removing these gives the simple tree

fittree1 = rpart(target_class~.,data=pulsar,minsplit=2,cp= .0021,subset=rest1,method="class")
fittree2 = rpart(target_class~.,data=pulsar,minsplit=2,cp= .0021,subset=rest2,method="class")
fittree3 = rpart(target_class~.,data=pulsar,minsplit=2,cp= .0021,subset=rest3,method="class")
fittree4 = rpart(target_class~.,data=pulsar,minsplit=2,cp= .0021,subset=rest4,method="class")
fittree5 = rpart(target_class~.,data=pulsar,minsplit=2,cp= .0021,subset=rest5,method="class")
fancyRpartPlot(fittree5) #All the trees end up looking pretty identical

tfit1 = predict(fittree1,pulsar[test_index1,])
tfit2 = predict(fittree1,pulsar[test_index2,])
tfit3 = predict(fittree1,pulsar[test_index3,])
tfit4 = predict(fittree1,pulsar[test_index4,])
tfit5 = predict(fittree1,pulsar[test_index5,])

tfit1 = round(tfit1[,2])
tfit2 = round(tfit2[,2])
tfit3 = round(tfit3[,2])
tfit4 = round(tfit4[,2])
tfit5 = round(tfit5[,2])

time.tests[5] = (proc.time() - start.time)[3]
print("Tree Confusion Matrix")
ConfusionMatrix(tfit1,tfit2,tfit3,tfit4,tfit5)

print(time.tests)
