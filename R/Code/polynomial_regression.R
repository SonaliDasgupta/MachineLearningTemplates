dataset = read.csv("Position_Salaries.csv")
dataset= dataset[2:3]

#linear regressor
lin_reg= lm(formula = Salary ~ ., data= dataset)

#polynomial regressor
dataset$level2= dataset$Level^2
dataset$level3= dataset$Level^3
dataset$Level4= dataset$Level^4
poly_reg= lm(formula = Salary ~ ., data= dataset)

#visualize linear reg
library(ggplot2)
ggplot() + geom_point(aes(x= dataset$Level, y=dataset$Salary), color='red')+ geom_line(aes(x=dataset$Level, y=predict(lin_reg, newdata= dataset)), color='blue')
 + ggtitle("Linear Regression")+ xlab("Level")+ ylab("Salary")

#visualize polynomial reg
ggplot() + geom_point(aes(x= dataset$Level, y=dataset$Salary), color='red')+ geom_line(aes(x=dataset$Level, y=predict(poly_reg, newdata= dataset)), color='blue')
+ ggtitle("Polynomial Regression")+ xlab("Level")+ ylab("Salary")

#predict with linear regression
y_pred= predict(lin_reg, newdata=data.frame(Level = 6.5))

#predict with polynomial regression
y_pred = predict(poly_reg, newdata = data.frame(Level = 6.5, level2= 6.5^2, level3= 6.5^3, Level4= 6.5^4))
