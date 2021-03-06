plot_fn<- function(dataset, training_set){
  #install.packages('ElemStatLearn')
  library(ElemStatLearn)
  set = dataset
  X1= seq(min(set[,1])-1, max(set[,1])+1, by=0.01)
  X2= seq(min(set[,2])-1, max(set[,2])+1, by=0.01)
  grid_set = expand.grid(X1, X2)
  colnames(grid_set)= c('Age','EstimatedSalary')
  y_grid = knn(train= training_set[,-3], test= grid_set, cl=training_set[,3],k=5)
  
  plot(set[,-3], main='Classfication', xlab= 'Age', ylab='Estimated Salary',
       xlim= range(X1), ylim=range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)
  points(grid_set, pch='.', col=ifelse(y_grid==1, 'blue', 'tomato'))
  points(set, pch=21, bg= ifelse(set[,3]==1, 'green4', 'red3'))
}


dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[,3:5]

#Split training and test sets
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 3/4)
training_set =subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Feature Scaling
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2]= scale(test_set[,1:2])

#classifier
library(class)
y_pred= knn(train= training_set[,-3], test= test_set[,-3], cl= training_set[,3], k=5)

#confusion matrix
cm = table(test_set[,3], y_pred)

#plot training set results
plot_fn(training_set, training_set)

#test set visualization
plot_fn(test_set, training_set)







