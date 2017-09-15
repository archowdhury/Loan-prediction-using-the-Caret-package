library(dplyr)
library(caret)


train = read.csv("C:/Users/amit.r/Desktop/Loan Prediction/Train.csv")
str(train)

# Check for missing values
sum(is.na(train))
sapply(train, function(x) sum(is.na(x)))

# Imputing missing values using KNN
preProc = preProcess(train, method=c("knnImpute","center","scale"))
train = predict(preProc, train)
sum(is.na(train))

# One hot encoding
id = train$Loan_ID
train$Loan_ID = NULL

train$Loan_Status = ifelse(train$Loan_Status=="N",0,1)

dmy = dummyVars(" ~ .",data=train, fullRank = TRUE)
train_transformed = data.frame(predict(dmy, newdata=train))


# Convert the outcome variable back to a factor
train_transformed$Loan_Status<-as.factor(train_transformed$Loan_Status)
str(train_transformed)

# Splitting the dataset into training and test
index = createDataPartition(train_processed$Loan_Status, p=0.75, list=FALSE)
trainSet = train_transformed[index,]
testSet = train_transformed[-index,]


#===================================================================#
#          Feature Selection using RFE in Caret                     #
#===================================================================#

control = rfeControl(functions=rfFuncs,
                     method="repeatedcv",
                     repeats=3,
                     verbose=FALSE)
outcome = "Loan_Status"
predictors = names(trainSet)[!names(trainSet) %in% outcome]

Loan_Profile = rfe(trainSet[,predictors], trainSet[,outcome],rfeControl = control)
Loan_Profile

# taking only the top 5 predictors of 16 as per the profiling
best_predictors = Loan_Profile$optVariables[1:5]


# Just some features of the Caret package
#----------------------------------------

# To get list of all algorithms that Caret supports
names(getModelInfo())

# To find the parameters that can be tuned for any model
modelLookup(model="rpart")



#===================================================================#
# BUILDING THE MODELS                                               #
#===================================================================#

# 1) Creating a GBM model
# -----------------------

# Setting the resampling and cross-validation parameters
fitControl = trainControl(method="repeatedcv",
                          number=5,
                          repeats=5)


# Identify what parameters can be tuned for gbm
modelLookup(model="gbm")

# Set the values required for tuning the model
grid = expand.grid(n.trees = c(10,20,50,100,500,1000), 
                   shrinkage=c(0.01, 0.05, 0.1, 0.5),
                   n.minobsinnode = c(3,5,10),
                   interaction.depth = c(1,5,10))

# Creating the model using parameters we have specified
mod_GBM = train(trainSet[,best_predictors], trainSet[,outcome], 
                method="gbm",
                trControl=fitControl,
                tuneGrid = grid)

# Let's also create another model using tuneLength.
# This automatically runs through all possible values for each parameter
mod_GBM_auto = train(trainSet[,best_predictors],trainSet[,outcome],
                     method="gbm",
                     trControl=fitControl,
                     tuneLength=10)

# Check the model variable importance
summary(mod_GBM)

# We can also use varImp (output is the same as summary)
plot(varImp(mod_GBM_auto))


# 2) Creating a RandomForest model
# --------------------------------

mod_RF = train(trainSet[,best_predictors],trainSet[,outcome],
               method="rf",
               trControl=fitControl,
               tuneLength=10,
               importance=TRUE) #have to add the importance criteria separately for RF (not done by default)

plot(varImp(mod_RF))


# 3) Creating a Neural Net model
# ------------------------------

mod_NNet = train(trainSet[,best_predictors],trainSet[,outcome],
               method="nnet",
               trControl=fitControl,
               tuneLength=10) 

mod_NNet

plot(varImp(mod_NNet))



# 4) Creating a Logistic regression model
# ---------------------------------------

mod_Log = train(trainSet[,best_predictors],trainSet[,outcome],
               method="glm",
               trControl=fitControl,
               tuneLength=10) 

summary(mod_Log)

plot(varImp(mod_Log))


#===================================================================#
# VALIDATING THE MODELS                                             #
#===================================================================#

# Note: type="raw" --> gives category classes as per the outcome variable
#       type="prob" --> gives the probability. Useful if we want to use a different cutoff than 0.5


# GBM model prediction
pred_GBM_train = predict(mod_GBM, trainSet[,predictors],type="raw")
pred_GBM_train = ifelse(pred_GBM_train >0.5,1,0)
confusionMatrix(pred_GBM_train, trainSet[,outcome])

pred_GBM_test = predict(mod_GBM, testSet[,predictors],type="raw")
pred_GBM_test = ifelse(pred_GBM_test >0.5,1,0)
confusionMatrix(pred_GBM_test, testSet[,outcome])


# Random Forest model prediction
pred_RF_train = predict(mod_RF, trainSet[,predictors],type="raw")
confusionMatrix(pred_RF_train, trainSet[,outcome])

pred_RF_test = predict(mod_RF, testSet[,predictors],type="raw")
confusionMatrix(pred_RF_test, testSet[,outcome])


# Neural Network model prediction
pred_NNet_train = predict(mod_NNet, trainSet[,predictors],type="raw")
confusionMatrix(pred_NNet_train, trainSet[,outcome])

pred_NNet_test = predict(mod_NNet, testSet[,predictors],type="raw")
confusionMatrix(pred_NNet_test, testSet[,outcome])


# Logistic Regression model prediction
pred_Log_train = predict(mod_Log, trainSet[,predictors],type="raw")
confusionMatrix(pred_Log_train, trainSet[,outcome])

pred_Log_test = predict(mod_Log, testSet[,predictors],type="raw")
confusionMatrix(pred_Log_test, testSet[,outcome])




# Summary : Of the four models the Logistic Regression gives the best results on the test set