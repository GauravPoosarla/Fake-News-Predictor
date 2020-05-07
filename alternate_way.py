#Importing Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the data
dataset=pd.read_csv('news.csv')
x = dataset.iloc[:, 2]
y = dataset.iloc[:, -1].values

# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,6335):
    review = re.sub("[^a-zA-Z]", " ", dataset["text"][i]) #Anything after ^: included
    review = review.lower()
    review = review.split() #Will split string into items of list.
    ps = PorterStemmer()    #Will removed all forms of word, will keep only root words.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)   #Joins the list seperated by spaces.
    corpus.append(review)
    
#Creating the bag of words model.
#A matrix containing zeroes is called sparse matrix, And sparsity is more number of zeroes.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#For Natural Language processing we generally use Naive Bayes, Decision tree, Random Forest Models.

#Using here Random Forest Classifier.
# Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = "entropy")
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

