dataset = read.csv("Salary_Data.csv")
#dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN=function(x) mean(x, na.rm = TRUE)), dataset$Age)
#dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Salary)
#dataset$Country = factor(dataset$Country, levels=c('France','Spain','Germany'), labels=c(1,2,3))
#dataset$Purchased = factor(dataset$Purchased, levels=c('Yes','No'), labels=c(1,0))

#Split training and test sets
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set =subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3]= scale(test_set[,2:3])

#Simple Linear Regression
regressor= lm(formula = Salary ~ YearsExperience , training_set)
y_pred= predict(regressor, newdata = test_set)

#visualize
install.packages('ggplot2', dependencies = TRUE)
library(ggplot2)
ggplot() +
  geom_point(aes(x= training_set$YearsExperience, y=training_set$Salary), 
             colour='red')+
  geom_line(aes(x= training_set$YearsExperience, y= predict(regressor, newdata = training_set),
                colour= 'blue'))+
  ggtitle("Salary vs Experience (Training Set)")+
  xlab("Years experience")+
  ylab("Salary")

ggplot()+
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary), colour='red')+
  geom_line(aes(x=training_set$YearsExperience , y=predict(regressor, newdata= training_set), colour= 'blue'))+
  ggtitle("salary vs experience (test set)")+
  xlab("exp")+
  ylab("salary")
  

