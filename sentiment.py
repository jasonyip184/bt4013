import tweepy
import re
import pickle
import os.path
import sys
import os
import datetime
import pandas as pd
import numpy as np
from tweepy import OAuthHandler
from nltk.tokenize import TweetTokenizer
from nltk import ngrams, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter
from itertools import chain
from operator import methodcaller
from datetime import datetime

'''
Generate lexicon by scraping from twitter
'''
access_token = 'hidden'
access_token_secret = 'hidden'
consumer_key = 'hidden'
consumer_secret = 'hidden'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=1, retry_delay=5, retry_errors=set([503, 104]))
tknzr = TweetTokenizer()
dates_dict = defaultdict(list)
train_dict = defaultdict(list)
query = "realDonaldTrump" #markets realDonaldTrump

def filter_token(token):
    if (token[0] == "@") or (":" in token) or (token in [',',"'",'RT','...','…','.','-']) or token.isdigit():
        pass
    else:
        return token

c = 0
for tweet in tweepy.Cursor(api.user_timeline, screen_name=query).items():
    date = tweet.created_at.date().strftime("%Y%m%d")
    print(date, c)
    c+=1
    words = []
    text = tweet.text
    text = text.replace("Five things you need to know to start your day in Europe",'').replace("Five things you need to know at the start of Asia's trading day",'').replace("All you need to know in markets today",'').replace("Here's a rundown of your top economic news today",'')
    for token in tknzr.tokenize(text):
        if (token[0] == "@") or (":" in token) or ("," in token) or ("-" in token) or ("/" in token) or token.isdigit() or \
            (token in [',',"'",'RT','...','…','.','-','. . .','%','*','(',')','/','$','|','’','s','•',"https",'“','”','"']):
            pass
        else:
            words.append(token.lower())
    # tokens.extend([' '.join(ngram) for ngram in ngrams(tokens, 2)]) # combine single word + 2-gram in entire list
    tokens = [' '.join(ngram) for ngram in ngrams(words, 2)]
    tokens.extend([' '.join(ngram) for ngram in ngrams(words, 3)])
    dates_dict[date].extend(tokens) # add to current date's tokens

print("all tokens combined for each date")

i = 0
# for each date, count the occurrences of all tokens
for date,tokens in dates_dict.items():
    counts = Counter(tokens)
    # add the counts from today to overall token dictionary
    for k,v in counts.items():
        if (k != '_date') and (k not in train_dict.keys()):
            train_dict[k].extend([0]*i)
        train_dict[k].append(v)
    # add date to keep track
    train_dict['_date'].append(date)
    tokens_to_pad = list(set(train_dict.keys()) - set(counts.keys()))
    tokens_to_pad.remove('_date')
    for token in tokens_to_pad:
        train_dict[token].append(0)
    i+=1

d = {}
for k,v in train_dict.items():
    length = len(list(filter(lambda a: a != 0, v)))
    if (length > 2):
        d[k] = v
        print(k, length)

with open('data/trump_train_dict.pickle', 'wb') as handle:
    pickle.dump(d, handle)

'''
Train futures prices against prev day news
'''
with open('data/trump_train_dict.pickle', 'rb') as handle:
    train_dict = pickle.load(handle)
for k,v in train_dict.items():
    length = len(list(filter(lambda a: a != 0, v)))
    if (length > 10):
        print(k, length)

dates = train_dict['_date']

d = {}
for name in ['F_VX','F_GC','F_FV','F_TU','F_TY','F_US']:
    f = name + '.txt'
    future_data = pd.read_csv('tickerData/{}'.format(f))
    future_data['DATE'] = future_data['DATE'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d').date())
    future_data = future_data[['DATE','CLOSE']]
    earliest_date = datetime.strptime(dates[30],'%Y%m%d').date()
    train_df = pd.DataFrame.from_dict(train_dict, orient='index').transpose()
    train_df['_date'] = train_df['_date'].apply(lambda x: datetime.strptime(x,'%Y%m%d').date())
    train_df.set_index('_date',inplace=True)
    train_df = train_df.shift(-1) # use yesterday's news to predict today's close
    df = pd.merge(future_data,train_df,how='inner',left_on='DATE',right_on='_date').iloc[1:]
    d[name] = df

with open('data/trump_train_data.pickle', 'wb') as handle:
    pickle.dump(d, handle)
