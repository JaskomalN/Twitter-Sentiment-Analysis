import pandas as pd
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation

#read train csv
tweet_df = pd.read_csv("train.csv", encoding="latin")

ps = PorterStemmer()
negation_list = ["arent","isnt","dont","doesnt","not","cant","couldnt", "werent",
                 "wont","didnt","never","nothing","nowhere","noone","none"
                "hasnt","hadnt","shouldnt","wouldnt","aint"]

def preProcessTweets(tweet):
    tweet = tweet.lower()
    tweet = re.sub('n[^A-Za-z ]t','nt', tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    tweet = re.sub('@[^\s]+', '', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = word_tokenize(tweet)
    tweet_list = [];
    negate = False
    for word in tweet:
        word = ps.stem(word)
        if word in negation_list:
            negate = True
        elif negate is True and word in list(punctuation):
            negate = False

        if negate and word not in negation_list:
            word = "not_"+word
        else:
            pass
        word = re.sub('[^A-Za-z_ ]+', '', word)
        if len(word) > 2 and word not in stopwords.words('english'):
            tweet_list.append(word)
    tweet_set = set(tweet_list)
    return " ".join(tweet_set)

tweet_df["clean_text"] = tweet_df["SentimentText"].apply(preProcessTweets)

tweet_df.to_csv("train_preprocessed.csv")
