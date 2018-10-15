dataset_org= read.delim("Restaurant_Reviews.tsv", quote="", stringsAsFactors = FALSE)

#clean reviews
#install.packages("tm")
library(tm)
corpus = VCorpus(VectorSource(dataset_org$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)

#install.packages("SnowballC")
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#creating Bag Of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_org$Liked

#classifier
library(caTools)
set.seed(123)
split= sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split == FALSE)

library(randomForest)
training_set$Liked <- as.character(training_set$Liked)
training_set$Liked <- as.factor(training_set$Liked)
classifier= randomForest(x= training_set[-692], 
                         y = training_set$Liked,
                         ntree= 10)




#predict results
y_pred= predict(classifier, type="class", newdata= test_set[-692])


#confusion matrix
cm = table(test_set[,692], y_pred)


