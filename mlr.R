
library(mlr)
train <- read.csv("train_u6lujuX_CVtuZ9i.csv", na.strings = c(""," ",NA))
test <- read.csv("test_Y3wMUE5_7gLdaTN.csv", na.strings = c(""," ",NA))

summarizeColumns(train)
summarizeColumns(test)

hist(train$ApplicantIncome, breaks = 300, main = "Applicant Income Chart",xlab = "ApplicantIncome")

hist(train$CoapplicantIncome, breaks = 100,main = "Coapplicant Income Chart",xlab = "CoapplicantIncome")

boxplot(train$ApplicantIncome)
train$Credit_History <- as.factor(train$Credit_History)
test$Credit_History <- as.factor(test$Credit_History)

summary(train)
summary(test)

levels(train$Dependents)[4] <- "3"
levels(test$Dependents)[4] <- "3"

imp_train <- impute(train, classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")
imp_test <- impute(test, classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")

train_imp <- imp_train$data
test_imp <- imp_test$data

summarizeColumns(train_imp)
summarizeColumns(test_imp)

listLearners("classif", check.packages = TRUE, properties = "missings")

cd <- capLargeValues(train_imp, target = "Loan_Status",cols = c("ApplicantIncome"),threshold = 40000)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("CoapplicantIncome"),threshold = 21000)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("LoanAmount"),threshold = 520)
cd_train <- cd

test_imp$Loan_Status <- sample(0:1,size = 367,replace = T)
cde <- capLargeValues(test_imp, target = "Loan_Status",cols = c("ApplicantIncome"),threshold = 33000)
cde <- capLargeValues(cde, target = "Loan_Status",cols = c("CoapplicantIncome"),threshold = 16000)
cde <- capLargeValues(cde, target = "Loan_Status",cols = c("LoanAmount"),threshold = 470)
cd_test <- cde


summary(cd_train$ApplicantIncome)

for (f in names(cd_train[, c(14:20)])) {
  if( class(cd_train[, c(14:20)] [[f]]) == "numeric"){
    levels <- unique(cd_train[, c(14:20)][[f]])
    cd_train[, c(14:20)][[f]] <- as.factor(factor(cd_train[, c(14:20)][[f]], levels = levels))
  }
}

for (f in names(cd_test[, c(13:18)])) {
  if( class(cd_test[, c(13:18)] [[f]]) == "numeric"){
    levels <- unique(cd_test[, c(13:18)][[f]])
    cd_test[, c(13:18)][[f]] <- as.factor(factor(cd_test[, c(13:18)][[f]], levels = levels))
  }
}

cd_train$Total_Income <- cd_train$ApplicantIncome + cd_train$CoapplicantIncome
cd_test$Total_Income <- cd_test$ApplicantIncome + cd_test$CoapplicantIncome

cd_train$Income_by_loan <- cd_train$Total_Income/cd_train$LoanAmount
cd_test$Income_by_loan <- cd_test$Total_Income/cd_test$LoanAmount

cd_train$Loan_Amount_Term <- as.numeric(cd_train$Loan_Amount_Term)
cd_test$Loan_Amount_Term <- as.numeric(cd_test$Loan_Amount_Term)

cd_train$Loan_amount_by_term <- cd_train$LoanAmount/cd_train$Loan_Amount_Term
cd_test$Loan_amount_by_term <- cd_test$LoanAmount/cd_test$Loan_Amount_Term


#splitting the data based on class
az <- split(names(cd_train), sapply(cd_train, function(x){ class(x)}))

#creating a data frame of numeric variables
xs <- cd_train[az$numeric]

#check correlation
cor(xs)

cd_train$Total_Income <- NULL
cd_test$Total_Income <- NULL

summarizeColumns(cd_train)
summarizeColumns(cd_test)

####################################################################################################################################################################################
#create a task
trainTask <- makeClassifTask(data = cd_train,target = "Loan_Status")
testTask <- makeClassifTask(data = cd_test, target = "Loan_Status")

trainTask <- makeClassifTask(data = cd_train,target = "Loan_Status", positive = "Y")
str(getTaskData(trainTask))

trainTask <- normalizeFeatures(trainTask,method = "standardize")
testTask <- normalizeFeatures(testTask,method = "standardize")

trainTask <- dropFeatures(task = trainTask,features = c("Loan_ID","Married.dummy"))

im_feat <- generateFilterValuesData(trainTask, method = c("information.gain","chi.squared"))
plotFilterValues(im_feat,n.show = 20)
####################################################################################################################################################################################

#load qda 
qda.learner <- makeLearner("classif.qda", predict.type = "response")
qmodel <- train(qda.learner, trainTask)
qpredict <- predict(qmodel, testTask)

submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = qpredict$data$response)
write.csv(submit, "submit1.csv",row.names = F)

#logistic regression
logistic.learner <- makeLearner("classif.logreg",predict.type = "response")

#cross validation (cv) accuracy
cv.logistic <- crossval(learner = logistic.learner,task = trainTask,iters = 3,stratify = TRUE,measures = acc,show.info = F)

fmodel <- train(logistic.learner,trainTask)
getLearnerModel(fmodel)
fpmodel <- predict(fmodel, testTask)
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = fpmodel$data$response)
write.csv(submit, "submit2.csv",row.names = F)


#Decision Tree
getParamSet("classif.rpart")

makeatree <- makeLearner("classif.rpart", predict.type = "response")
set_cv <- makeResampleDesc("CV",iters = 3L)
gs <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)
#do a grid search
gscontrol <- makeTuneControlGrid()
stune <- tuneParams(learner = makeatree, resampling = set_cv, task = trainTask, par.set = gs, control = gscontrol, measures = acc)
t.tree <- setHyperPars(makeatree, par.vals = stune$x)

t.rpart <- train(t.tree, trainTask)
getLearnerModel(t.rpart)
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = tpmodel$data$response)

#Naive Byes
bayes.learner <- makeLearner("classif.naiveBayes" , predict.type = "response" )
t.naiveBayes <- train(bayes.learner,trainTask)
fpmodel <- predict(fmodel, testTask)
submit <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = fpmodel$data$response)
write.csv(submit, "submit_bayes.csv",row.names = F)

