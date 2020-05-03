# Text Classifiation using NLP

# Importing the libraries
import numpy as np
import re
import pickle 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Importing the dataset
reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target

'''
# Storing as Pickle files
with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)
'''

# Unpickling the dataset
with open('X.pickle','rb') as f:
    X = pickle.load(f)

with open('y.pickle','rb') as f:
    y = pickle.load(f)
    
# Creating the corpus
corpus = []
for i in  range(0,len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ', review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)

"""
vectorizer = CountVectorizer(max_features=2000, min_df = 2, max_df = 0.8, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
"""

vectorizer = TfidfVectorizer(max_features=2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

text_train, text_test, sent_train, sent_test = train_test_split(X,y,test_size=0.2,random_state = 0)
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)


sent_pred = classifier.predict(text_test)

cm = confusion_matrix(sent_test,sent_pred)
accuracy = (cm[0][0] + cm[1][1])/4

'''
# Pickling the classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
'''
'''
# Pickling the vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
'''

# Unpickling the classifier and vectorizer
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)

sample = ["You are not a good person, I hope you kill yourself"]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))