# Twitter Sentiment Analysis using NLP
# Access token: 1207888491570438144-RG32orydI3YdXHKrgeSQSa118werC6
# Access token secret: z4RwAfzLpRa04zE1Pq89FOkD0OCkTOnyTSF4uyq9ioKDb
# Install tweepy - pip install tweepy

# Importing the libraries
import tweepy
import re
import pickle
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Initializing the keys
consumer_key = 'EZgzWSUwrHQj9gV7LR0K9DJrA'
consumer_secret = 'zPv6j5nFKJCJxnTmIuQuxm6lxoV7BLix1DBY3Adb4RmPd8Ifka'
access_token = '1207888491570438144-RG32orydI3YdXHKrgeSQSa118werC6'
access_secret = 'z4RwAfzLpRa04zE1Pq89FOkD0OCkTOnyTSF4uyq9ioKDb'

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_secret)
args = ['facebook']
api = tweepy.API(auth,timeout=10)


list_tweets = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search, q=query+" -filter:retweets",lang='en',result_type='recent').items(100):
        list_tweets.append(status.text)

with open('tfidfmodel.pickle','rb') as f:
    vectorizer = pickle.load(f)

with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

total_pos = 0
total_neg = 0

for tweet in list_tweets:
    tweet = re.sub(r"^http://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/9a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    sent = clf.predict(vectorizer.transform([tweet]).toarray())
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1

# Plotting the bar chart
objects = ['Positive', 'Negative']
y_pos = np.arange(len(objects))
plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Positive and Negative tweets')

plt.show()