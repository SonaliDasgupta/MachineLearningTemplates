plot_fn<- function(dataset){
  #install.packages('ElemStatLearn')
  library(ElemStatLearn)
  set = dataset
  X1= seq(min(set[,1])-1, max(set[,1])+1, by=0.01)
  X2= seq(min(set[,2])-1, max(set[,2])+1, by=0.01)
  grid_set = expand.grid(X1, X2)
  colnames(grid_set)= c('V1','V2')
  prob_set = predict(classifier, type='response', newdata= grid_set)
  y_grid= ifelse(prob_set>0.5, 1,0)
  plot(set[,-3], main='Logistic Regression', xlab= 'PC1', ylab='PC2',
       xlim= range(X1), ylim=range(X2))
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add=TRUE)
  points(grid_set, pch='.', col=ifelse(y_grid==1, 'blue', 'tomato'))
  points(set, pch=21, bg= ifelse(set[,3]==1, 'green4', 'red3'))
}


dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[,3:5]

#Split training and test sets
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 3/4)
training_set =subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Feature Scaling
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2]= scale(test_set[,1:2])

#KernelPCA
library(kernlab)
kpca = KernelPCA(~., data = training_set[-3], kernel = 'rbfdot', features=2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased


#classifier
classifier = glm(formula= Purchased ~ ., 
                 family= binomial,
                 data=training_set_pca)
#predict results
prob_pred= predict(classifier, type='response', newdata= test_set_pca[-3])
y_pred= ifelse(prob_pred>0.5, 1, 0)

#confusion matrix
cm = table(test_set_pca[,3], y_pred)

#plot training set results
plot_fn(training_set_pca)

#test set visualization
plot_fn(test_set_pca)













