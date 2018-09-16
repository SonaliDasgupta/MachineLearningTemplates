dataset = read.csv("Position_Salaries.csv")


#Split training and test sets
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set =subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3]= scale(test_set[,2:3])

# regressor
#create regressor

#predict with regression
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))


#visualize regression
library(ggplot2)

ggplot() + geom_point(aes(x= dataset$Level, y=dataset$Salary), color='red')+ geom_line(aes(x=dataset$Level, y=predict(regressor, newdata= dataset)), color='blue')
+ ggtitle("Regression")+ xlab("Level")+ ylab("Salary")

#visualize regression smooth curve

x_grid= seq(min(dataset$Level), max(dataset$Level),0.1)
ggplot() + geom_point(aes(x= dataset$Level, y=dataset$Salary), color='red')+ geom_line(aes(x=x_grid, y=predict(regressor, newdata= data.frame(Level= x_grid))), color='blue')
+ ggtitle("Regression")+ xlab("Level")+ ylab("Salary")


