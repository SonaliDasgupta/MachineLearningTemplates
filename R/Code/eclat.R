dataset = read.csv("Market_Basket_Optimisation.csv", header=FALSE)
install.packages('arules')
library(arules)
dataset= read.transactions("Market_Basket_Optimisation.csv",sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=10)

#training
rules= eclat(dataset, parameter = list(support= 28/7500, minlen=2))

#visualize
inspect(sort(rules, by='support')[1:10])