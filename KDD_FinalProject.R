# removing all objects
rm(list=ls())
?neur
# Load dataset into R
attrition_data <- read.csv("attrition_data.csv")
attrition_drop <- attrition_data[,c(-1, -4, -5, -11, -14)]
attrition_drop[c(1, 2, 7)] <- scale(attrition_drop[c(1, 2, 6)])

# Identifying missing values
str(attrition_drop)
summary(attrition_drop)
#attrition_drop <- attrition_drop[-2935, ]

attrition_drop$SEX  <- factor(attrition_drop$SEX, order = TRUE, levels = unique(attrition_drop$SEX))
summary(attrition_drop$SEX)

attrition_drop$MARITAL_STATUS  <- factor(attrition_drop$MARITAL_STATUS, order = TRUE, levels = unique(attrition_drop$MARITAL_STATUS))
summary(attrition_drop$MARITAL_STATUS)

attrition_drop$JOB_SATISFACTION  <- factor(attrition_drop$JOB_SATISFACTION, order = TRUE, levels = unique(attrition_drop$JOB_SATISFACTION))
summary(attrition_drop$JOB_SATISFACTION)

attrition_drop$NUMBER_OF_TEAM_CHANGED  <- factor(attrition_drop$NUMBER_OF_TEAM_CHANGED, order = TRUE, levels = unique(attrition_drop$NUMBER_OF_TEAM_CHANGED))
summary(attrition_drop$NUMBER_OF_TEAM_CHANGED)

attrition_drop$REHIRE  <- factor(attrition_drop$REHIRE, order = TRUE, levels = unique(attrition_drop$REHIRE))
summary(attrition_drop$REHIRE)

attrition_drop$IS_FIRST_JOB  <- factor(attrition_drop$IS_FIRST_JOB, order = TRUE, levels = unique(attrition_drop$IS_FIRST_JOB))
summary(attrition_drop$IS_FIRST_JOB)

attrition_drop$TRAVELLED_REQUIRED  <- factor(attrition_drop$TRAVELLED_REQUIRED, order = TRUE, levels = unique(attrition_drop$TRAVELLED_REQUIRED))
summary(attrition_drop$TRAVELLED_REQUIRED)

attrition_drop$PERFORMANCE_RATING  <- factor(attrition_drop$PERFORMANCE_RATING, order = TRUE, levels = unique(attrition_drop$PERFORMANCE_RATING))
summary(attrition_drop$PERFORMANCE_RATING)

attrition_drop$DISABLED_EMP  <- factor(attrition_drop$DISABLED_EMP, order = TRUE, levels = unique(attrition_drop$DISABLED_EMP))
summary(attrition_drop$DISABLED_EMP)

attrition_drop$DISABLED_VET  <- factor(attrition_drop$DISABLED_VET, order = TRUE, levels = unique(attrition_drop$DISABLED_VET))
summary(attrition_drop$DISABLED_VET)

attrition_drop$EDUCATION_LEVEL  <- factor(attrition_drop$EDUCATION_LEVEL, order = TRUE, levels = unique(attrition_drop$EDUCATION_LEVEL))
summary(attrition_drop$EDUCATION_LEVEL)

attrition_drop$JOB_GROUP  <- factor(attrition_drop$JOB_GROUP, order = TRUE, levels = unique(attrition_drop$JOB_GROUP))
summary(attrition_drop$JOB_GROUP)

attrition_drop$PREVYR_1  <- factor(attrition_drop$PREVYR_1, order = TRUE, levels = unique(attrition_drop$PREVYR_1))
summary(attrition_drop$PREVYR_1)

attrition_drop$PREVYR_2  <- factor(attrition_drop$PREVYR_1, order = TRUE, levels = unique(attrition_drop$PREVYR_2))
summary(attrition_drop$PREVYR_2)

attrition_drop$PREVYR_3  <- factor(attrition_drop$PREVYR_3, order = TRUE, levels = unique(attrition_drop$PREVYR_3))
summary(attrition_drop$PREVYR_3)

attrition_drop$PREVYR_4  <- factor(attrition_drop$PREVYR_4, order = TRUE, levels = unique(attrition_drop$PREVYR_4))
summary(attrition_drop$PREVYR_4)

attrition_drop$PREVYR_5  <- factor(attrition_drop$PREVYR_5, order = TRUE, levels = unique(attrition_drop$PREVYR_5))
summary(attrition_drop$PREVYR_5)

attrition_final <- attrition_drop
str(attrition_final)
summary(attrition_final)

idx <- sort(sample(nrow(attrition_final),as.integer(.70*nrow(attrition_final)))) 
training_set <- attrition_final[idx,]
test_set <- attrition_final[-idx,]


########################################

# install.packages('e1071')
library(e1071)
classifier_nb <- naiveBayes(as.factor(STATUS) ~., data = training_set)

# Predicting the Test set results
y_pred <- predict(classifier_nb, newdata = test_set[-16])

# Making the Confusion Matrix
table(test_set[, 16], y_pred)

# Accuracy
NB_wrong <- sum(y_pred != test_set$STATUS)
(1 - NB_wrong / length(y_pred)) * 100

########################################

library(kknn) 

for(i in c(1, 3, 5, 6, 7)){
  classifier_knn <- kknn(formula=as.factor(STATUS)~., training_set , test_set[,-16], k=i,kernel ="rectangular"  )
  
  fit <- fitted(classifier_knn)
  
  #e.	Measure the performance of knn
  
  knn_wrong <- sum(fit != test_set$STATUS)
  acc_rate<-(1 - knn_wrong / length(fit)) * 100
  
  print('***************')
  print(i)
  print( table(test_set$STATUS,fit))
  print( acc_rate)
  print('***************') 
}

########################################

# install.packages("rpart")
# install.packages("rpart.plot")     # Enhanced tree plots
# install.packages("rattle")         # Fancy tree plot
# install.packages("RColorBrewer")   # colors needed for rattle
library(rpart)
library(rpart.plot)  			# Enhanced tree plots
library(rattle)           # Fancy tree plot
library(RColorBrewer)     # colors needed for rattle

# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = as.factor(STATUS) ~ .,
                   data = training_set)

# Graphs
rpart.plot(classifier)
prp(classifier)
# Much fancier graph
fancyRpartPlot(classifier)

# Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-16], type = 'class')

# Making the Confusion Matrix
table(test_set[, 16], y_pred)

# Accuracy
DT_wrong <- sum(y_pred != test_set$STATUS)
(1 - DT_wrong / length(y_pred)) * 100

########################################

# install.packages("randomForest")
library(randomForest)

classifier_rf <- randomForest(as.factor(STATUS) ~ ., data = training_set, ntree = 700, mtry = 6, importance = TRUE)
y_pred_rf <- predict(classifier_rf, newdata = test_set[-16], type = 'class')
table(test_set[, 16], y_pred_rf)

# Accuracy
rf_wrong <- sum(y_pred_rf != test_set$STATUS)
accuracy<-(1 - rf_wrong / length(y_pred_rf)) * 100



var <- c('ANNUAL_RATE', 'STATUS')

training_set <- training_set[var]
test_set <- test_set[var]

library(neuralnet)
# fit neural network
model_nn = neuralnet(as.factor(STATUS) ~ ., training_set, hidden = 5 , threshold = 0.01)

# plot neural network
plot(model_nn)

test_set$pred_nn = predict(model_nn, test_set[-7])
test_set$pred_nn <- ifelse(test_set$pred_nn < 0.5, '0', '1')

