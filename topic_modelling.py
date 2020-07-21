import tweepy
import csv
import pandas as pd
import json

import string
from tweepy import Stream
from tweepy.auth import OAuthHandler
from tweepy.streaming import StreamListener
import time
import matplotlib.pyplot as plt
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
from gensim import corpora
import regex

import pandas as pd

tweetList=[]
stops = set(stopwords.words("english"))
stemmer = nltk.stem.SnowballStemmer('english')
lemmatizer = nltk.wordnet.WordNetLemmatizer()

consumer_key  = ""
consumer_secret  = ""
access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

siaObject = SentimentIntensityAnalyzer()

# Open/Create a file to append data
csvFile = open('santander.csv', 'w', encoding='utf8')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#bancosantander",count=100,
                           lang="en",
                           since="2014-01-01").items():

    cleanString1 = regex.sub('[^A-Za-z]+', ' ', tweet.text)
    cleanString2 = cleanString1.split("http")[0]

    if (siaObject.polarity_scores(tweet._json['text'])['compound']) >= 0.05:

        csvWriter.writerow([tweet.created_at, cleanString2, tweet.user.location, '1', tweet.user.followers_count,
                            tweet.user.description, tweet.retweet_count])
    elif (siaObject.polarity_scores(tweet._json['text'])['compound']) <= -0.05:

        csvWriter.writerow([tweet.created_at, cleanString2, tweet.user.location, '-1', tweet.user.followers_count,
                            tweet.user.description, tweet.retweet_count])
    else:
        csvWriter.writerow([tweet.created_at, cleanString2, tweet.user.location, '0', tweet.user.followers_count,
                            tweet.user.description, tweet.retweet_count])

    rawTweet = tweet._json['text']
    # print(rawTweet)
    processedTweet = rawTweet.strip()
    processedTweet = processedTweet.translate(str.maketrans('', '', string.punctuation))
    tweetTokens = nltk.word_tokenize(processedTweet)

    for token in tweetTokens:
        if token in stops:
            tweetTokens.remove(token)

    for token in tweetTokens:
        oldToken = token
        tweetTokens.remove(token)
        oldToken = lemmatizer.lemmatize(oldToken)
        tweetTokens.append(oldToken)

    processedTweet = ' '.join(tweetTokens)
    # print(processedTweet)

    if len(tweetTokens) > 3:
        tweetList.append(processedTweet)
    # print(tweetList)

    texts = [[text for text in doc.split()] for doc in tweetList]
        # print(texts)
    dictionary = corpora.Dictionary(texts)
        # print("printing dictionary",dictionary.token2id)
        # print(dictionary)
    doc_term_matrix = [dictionary.doc2bow(doc.split()) for doc in tweetList]
    print(doc_term_matrix)
    ldaObject = gensim.models.ldamodel.LdaModel
    ldaModel = ldaObject(doc_term_matrix, num_topics=3, id2word=dictionary, passes=20)
    print(ldaModel.print_topics(num_topics=3, num_words=25))
    print("LDA analysis complete")

    pass


#print(tweet)