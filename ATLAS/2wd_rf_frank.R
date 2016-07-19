#############################
####  Preprocessing data #### 5/31/2016
#
############################## Read data and mark 999.0 as NAs
dfTrain <- read.csv('C:/wanglf2016/kaggle/ATLAS/data/training.csv', header=T,nrows = 100000)
dfTest <- read.csv('C:/wanglf2016/kaggle/ATLAS/data/test.csv', header=T,nrows = 10000)
#dfTrain[dfTrain==-999.0] <- NA
#dfTest[dfTest==-999.0] <- NA
testId = dfTest$EventId##adding interpretive variables
momenta<-function(particlename,data) {
  coln=paste(particlename,'pt',sep="")
  pt=data[,coln]
  eta=data[,paste(particlename,'eta',sep="")]
  phi=data[,paste(particlename,'phi',sep="")]
  px=pt*cos(phi)
  py=pt*sin(phi)
  tantheta2=exp(-eta)
  pz=pt*(1+tantheta2*tantheta2)/2/tantheta2
  #  pz=pt*sinh(eta)
  ptot=sqrt(px*px+py*py+pz*pz)
  px[pt<0]=-999
  py[pt<0]=-999
  pz[pt<0]=-999
  ptot[pt<0]=-999  
  df=data.frame(px,py,pz,ptot)
  sx=paste(particlename,'px',sep="")
  sy=paste(particlename,'py',sep="")
  sz=paste(particlename,'pz',sep="")
  st=paste(particlename,'ptot',sep="")
  colnames(df)=c(sx,sy,sz,st)
  return (df)
}
lep_mom=momenta('PRI_lep_',dfTrain)
jet_leading_mom =momenta('PRI_jet_leading_',dfTrain)
jet_subleading_mom = momenta('PRI_jet_subleading_',dfTrain)
tau_mom =momenta('PRI_tau_',dfTrain)
lep_mom_test = momenta('PRI_lep_',dfTest)
jet_leading_mom_test = momenta('PRI_jet_leading_',dfTest)
jet_subleading_mom_test = momenta('PRI_jet_subleading_',dfTest)
tau_mom_test = momenta('PRI_tau_',dfTest)
dfTrainNew <- data.frame(dfTrain,lep_mom, jet_leading_mom, jet_subleading_mom,tau_mom)
dfTestNew <- data.frame(dfTest,lep_mom_test, jet_leading_mom_test, jet_subleading_mom_test,tau_mom_test)##drop off phi terms
dfTrainNew <- subset(dfTrainNew, select = -c(PRI_lep_phi,PRI_jet_leading_phi,PRI_jet_subleading_phi,PRI_tau_phi))
dfTestNew <- subset(dfTestNew, select = -c(PRI_lep_phi,PRI_jet_leading_phi,PRI_jet_subleading_phi,PRI_tau_phi))#this will never be touched till we submit
dfTest <- dfTest[,-1]  # Convert PRI_jet_num to factor as instructed on the website.
dfTrainNew$PRI_jet_num <- as.factor(dfTrain$PRI_jet_num)
dfTestNew$PRI_jet_num <- as.factor(dfTest$PRI_jet_num)#assigning label columns to 1 or 0, if it ==s, it's 1.###???
dfTrainNew$outcome <- ifelse(dfTrain$Label == 's', 1, 0)##in the training dataset, split for 80 /20 test
#dfTrainNew$outcome <- dfTrain$Label

train <- sample(1:nrow(dfTrainNew), 8*nrow(dfTrainNew)/10)

df.train <- dfTrainNew[train, ]
df.test <- dfTrainNew[-train, ]
#the train and test is for the use going forward
train <-subset(df.train, select = -c(EventId, outcome,Weight))#20K record in the train
test <- subset(df.test, select = -c(EventId, outcome,Weight)) #5K record in the test#below is the real outcome of the train and test dataset.
trainOutcome <-df.train$outcome   # df.train$Label
testOutcome <- df.test$outcome
##----------------------add weight
trainWeight <- df.train$outcome


#------------------------------end of feature data clean-----------------

##test
library(caret)
#library(doSNOW)
#cl <- makeCluster(30, outfile="")
#registerDoSNOW(cl)
source('helper.R')
library(randomForest)


set.seed(0)
rf.boston = randomForest(Label ~ ., data = train, ntree=500,mtry=6,importance = TRUE)
rf.boston

####################################
####  Build random forest model ####
####################################

###### The only thing you need to change is the name of method.
###### Check all the available algorithms by typing names(getModelInfo())
###### Check avaliable tuning parameters here: http://topepo.github.io/caret/modelList.html
rfGrid <-  expand.grid(mtry = c(3,6,9))

ctrl = trainControl(method = "repeatedcv",number = 2,
                    summaryFunction = AMS_summary)
m_rf = train(x=train, y=train$Label, 
             method="rf", weights=trainWeight, 
             verbose=TRUE, trControl=ctrl, metric="AMS")
summary(m_rf)
m_rf$results
m_rf$bestTune

rfTrainPred <- predict(m_rf, newdata=train, type="prob")

library(pROC)

labels <- ifelse(train$Label=='s', 1, 0)
auc = roc(labels, rfTrainPred[,2])
plot(auc, print.thres=TRUE)

# prediction 
rfTrainPred <- predict(m_rf, newdata=test, type="raw")
summary(rfTrainPred)
sum_rf=table(rfTrainPred,test$Label)
pred_acc=sum_rf[2,2]/(sum_rf[2,1]+sum_rf[2,2])
pred_acc
