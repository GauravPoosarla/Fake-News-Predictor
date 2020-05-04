#Importing Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

#Read the data
df=pd.read_csv('news.csv')


#Get shape and head
df.shape
df.head()


#Get the label
labels=df.label
labels.head()


#Splitting of the dataset
from sklearn.model_selection import train_test_split       
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=0)


#Initialize a TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer      #Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once.
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7) #max-df checks for the number of repetitions or the barrier.


#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier           #Online Algorithm, is designed to manage massive data.
from sklearn.linear_model import PassiveAggressiveClassifier
classifier =PassiveAggressiveClassifier()
classifier.fit(tfidf_train,y_train)


#Predict on the test set and calculate accuracy
from sklearn.metrics import accuracy_score
y_pred=classifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


#Build confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)