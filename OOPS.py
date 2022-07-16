#!/usr/bin/env python
# coding: utf-8

#Importing all the required libraries:
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import re
import nltk
import time
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

import numpy as np
import itertools

class Model:
    def __init__(self, datafile = "airline_sentiment_analysis.csv"):
        self.dataset = pd.read_csv(datafile)
        self.log_reg = LogisticRegression()
    
    def plot(self,dataset):
        sns.countplot(x='airline_sentiment',data=dataset,order=['negative','positive'])
        plt.show()
    
    def cleaning(self,dataset):
        #remove words which are starts with @ symbols
        dataset['text'] = dataset['text'].map(lambda x:re.sub('@\w*','',str(x)))
        #remove special characters except [a-zA-Z]
        dataset['text'] = dataset['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
        #remove link starts with https
        dataset['text'] = dataset['text'].map(lambda x:re.sub('http.*','',str(x)))
        # lowercasing all the words:
        dataset['text'] = dataset['text'].map(lambda x:str(x).lower())
    
    def removing_stopwords(self,dataset):
        self.corpus = []
        self.none=dataset['text'].map(lambda x:self.corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))                                     
        self.X = pd.DataFrame(data=self.corpus,columns=['comment_text'])
        self.y = dataset['airline_sentiment'].map({'negative':0,'positive':1})
    
    def wordcloud(self,X):
        self.text = X['comment_text'].astype(str).to_list()
        self.wordcloud = WordCloud().generate(' '.join(x for x in self.text))

        # Display the generated image:
        plt.imshow(self.wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
    def split(self,X,y,test_size):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size=test_size,random_state=0)
        print(self.X_train.shape,self.X_test.shape,self.y_train.shape,self.y_test.shape)
        
    def tfidf(self,X_train,X_test):
        self.vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),max_features=30000)
        #token_patten #2 for word length greater than 2>=
        self.X_train_word_feature = self.vector.fit_transform(X_train['comment_text']).toarray()
        self.X_test_word_feature = self.vector.transform(X_test['comment_text']).toarray()
        print(self.X_train_word_feature.shape,self.X_test_word_feature.shape)

        
    def log_reg_1(self,X_train_word_feature,y_train,X_test_word_feature,y_test):
        self.classifier = LogisticRegression()
        self.classifier.fit(X_train_word_feature,y_train)
        self.y_pred = self.classifier.predict(X_test_word_feature)
        self.cm = confusion_matrix(y_test,self.y_pred)
        self.acc_score = accuracy_score(y_test,self.y_pred)
        print(classification_report(y_test,self.y_pred),'\nCONFUSION MATRIX: \n',self.cm,'\nACCURACY: ',self.acc_score * 100)
    
    def svm(self,X_train_word_feature,y_train,X_test_word_feature,y_test):
        # fitting the training dataset on the Support Vector Machine(SVM) classifier
        self.SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        self.SVM.fit(X_train_word_feature,y_train)
        self.predictions_SVM = self.SVM.predict(X_test_word_feature)
        self.cm_SVM = confusion_matrix(y_test,self.predictions_SVM)
        self.acc_score_SVM = accuracy_score(y_test,self.predictions_SVM)
        print(classification_report(y_test,self.predictions_SVM),'\nCONFUSION MATRIX: \n',self.cm_SVM,'\nACCURACY: ',self.acc_score_SVM * 100)

    def ran_forest(self,X_train_word_feature,y_train,X_test_word_feature,y_test):
        # fitting the training dataset on Random Forest classifier
        self.Ran_forest = RandomForestClassifier()
        self.Ran_forest.fit(X_train_word_feature,y_train)
        self.predictions_rfc = self.Ran_forest.predict(X_test_word_feature)
        self.cm_RFC = confusion_matrix(y_test,self.predictions_rfc)
        self.acc_score_RFC = accuracy_score(y_test,self.predictions_rfc)
        print(classification_report(y_test,self.predictions_rfc),'\nCONFUSION MATRIX: \n',self.cm_RFC,'\nACCURACY: ',self.acc_score_RFC * 100)
    
    def naive_bayes(self,X_train_word_feature,y_train,X_test_word_feature,y_test):
        self.Naive = naive_bayes.MultinomialNB()
        self.Naive.fit(X_train_word_feature,y_train)
        self.predictions_NB = self.Naive.predict(X_test_word_feature)
        self.cm_NB = confusion_matrix(y_test,self.predictions_NB)
        self.acc_score_NB = accuracy_score(y_test,self.predictions_NB)
        print(classification_report(y_test,self.predictions_NB),'\nCONFUSION MATRIX: \n',self.cm_NB,'\nACCURACY: ',self.acc_score_NB * 100)
        
    def testing_unseen(self,sent,model,tfidf):
        #Load saved models
        #Get the input sentence
        sent = pd.Series(sent)
        #remove words which are starts with @ symbols
        sent = sent.map(lambda x:re.sub('@\w*','',str(x)))
        #remove special characters except [a-zA-Z]
        sent = sent.map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
        #remove link starts with https
        sent = sent.map(lambda x:re.sub('http.*','',str(x)))
        # lowercasing all the words:
        sent = sent.map(lambda x:str(x).lower())
        self.corp = []
        self.none=sent.map(lambda x:self.corp.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))                                     
        self.X_unseen = pd.DataFrame(data=self.corp,columns=['comment_text'])
        self.X_unseen = tfidf.transform(self.X_unseen['comment_text']).toarray() 
        self.y_pred_unseen = model.predict(self.X_unseen)
        if self.y_pred_unseen[0] == 1:
            return 'Positive'
        else:
            return 'Negative'
        

if __name__ == '__main__':
    model_instance = Model()
    model_instance.dataset.drop(['Unnamed: 0'],axis = 1,inplace = True)
    print("Dataset: \n",model_instance.dataset.head(5))
    print("\nLabel count:\n", model_instance.dataset['airline_sentiment'].value_counts())
    print("\nVisual Plot of Data:")
    model_instance.plot(model_instance.dataset)
    model_instance.cleaning(model_instance.dataset)
    print("\n Cleaned Text:\n",model_instance.dataset['text'].head())
    model_instance.removing_stopwords(model_instance.dataset)
    print("\n After removing stopwords:\n",model_instance.X.head())
    print("\n Wordcloud of the Dataset: \n")
    model_instance.wordcloud(model_instance.X)
    print("\nAfter Splitting data into Training and Testing:\n")
    model_instance.split(model_instance.X,model_instance.y,0.2)
    print("\nShape after applying TFIDF for feature Extraction \n")
    model_instance.tfidf(model_instance.X_train,model_instance.X_test)
    print("1. Logistic Regression")
    model_instance.log_reg_1(model_instance.X_train_word_feature,model_instance.y_train,model_instance.X_test_word_feature,model_instance.y_test)
    print("2. Naive Bayes Classifier")
    model_instance.naive_bayes(model_instance.X_train_word_feature,model_instance.y_train,model_instance.X_test_word_feature,model_instance.y_test)
    print("3. Random Forest Classifier")
    model_instance.ran_forest(model_instance.X_train_word_feature,model_instance.y_train,model_instance.X_test_word_feature,model_instance.y_test)
    print("4. Support Vector Machine")
    model_instance.svm(model_instance.X_train_word_feature,model_instance.y_train,model_instance.X_test_word_feature,model_instance.y_test)
    print(model_instance.SVM)

# # save the winning model(SVM) and TFIDF for testing on Unseen sentences
# import pickle
# pickle.dump(model_instance.vector, open('tfidf.pkl', 'wb'))
# filename = 'finalized_model_SVM.sav'
# pickle.dump(model_instance.SVM, open(filename, 'wb'))

