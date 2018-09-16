

dataset = read.csv("50_Startups.csv")
#dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN=function(x) mean(x, na.rm = TRUE)), dataset$Age)
#dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Salary)
#dataset$Country = factor(dataset$Country, levels=c('France','Spain','Germany'), labels=c(1,2,3))
#dataset$Purchased = factor(dataset$Purchased, levels=c('Yes','No'), labels=c(1,0))
dataset$State = factor(dataset$State, levels=c('California','Florida','New York'), labels=c(1,2,3))

#Split training and test sets
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set =subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

backward_elm<- function(x, sl){
  numVars= length(x)
  for(i in c(1:numVars)){
    regressor = lm(Profit ~ ., data=x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if(maxVar > sl){
      j= which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"]==maxVar)
      x=x[,-j]
      
    }
    numVars= numVars-1
    
                 
  }
  return(summary(regressor))
}


#Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3]= scale(test_set[,2:3])

#Multiple Linear Regression with Backpropagation
regressor = lm(formula= Profit ~ ., 
               data=training_set )
regressor.summary()
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)

#predict test results
y_pred= predict(regressor, newdata = test_set)

#analysis
regressor = lm(formula = Profit ~ R.D.Spend+ Administration + Marketing.Spend +State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend+ Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

y_pred= predict(regressor, newdata = test_set)


#automatic backward elimination
sl=0.05
dataset=dataset[,c(1,2,3,4,5)]

backward_elm(dataset, sl)
