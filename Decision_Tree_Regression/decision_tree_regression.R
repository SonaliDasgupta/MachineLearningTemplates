dataset = read.csv("Position_Salaries.csv")

dataset = dataset[2:3]
#Feature Scaling

# regressor
#create regressor
library(rpart)
regressor= rpart(formula= Salary ~ ., data= dataset, control = rpart.control(minsplit=1))

#predict with regression
y_pred = predict(regressor, data.frame(Level = 6.5))


#visualize regression
library(ggplot2)



#visualize regression smooth curve

x_grid= seq(min(dataset$Level), max(dataset$Level),0.01)
ggplot() + geom_point(aes(x= dataset$Level, y=dataset$Salary), color='red')+ geom_line(aes(x=x_grid, y=predict(regressor, newdata= data.frame(Level= x_grid))), color='blue')
+ ggtitle("Decision Tree")+ xlab("Level")+ ylab("Salary")


