dataset = read.csv("Churn_Modelling.csv")
dataset = dataset[,4:14]

dataset$Gender = as.numeric(factor(dataset$Gender, levels=c('Male', 'Female'), labels=c(0,1)))
dataset$Geography = as.numeric(factor(dataset$Geography, levels=c('France','Germany','Spain'), labels = c(1,2,3)))
#dataset$Exited = factor(dataset$Exited, levels= c(0,1), labels = c(0,1))

#Split training and test sets

#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set =subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#classifier
#create classifier
install.packages('xgboost')
library(xgboost)

classifier = xgboost(data = as.matrix(training_set[-11]), label= training_set$Exited, nrounds=10)

#use k folds cross validation to build classifier
#predict results
prob_pred= h2o.predict(classifier, newdata= as.matrix(test_set[-11]))
y_pred= (prob_pred>0.5)
y_pred = as.vector(y_pred)

#confusion matrix
cm = table(test_set[,11], y_pred)









