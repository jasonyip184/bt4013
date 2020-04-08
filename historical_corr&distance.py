import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
arr = os.listdir("./tickerData")
file_names = arr[1:89] ##all 88 futures
names = []
for i in file_names:
    names.append(i[:-4])

d = {}
for i in range(len(file_names)):
    x = pd.read_csv("./tickerData/" + file_names[i])
    d[names[i]] = x
    ls = []
    for k in range(len(x.columns)):
        ls.append(x.columns[k].strip())
    ls[4] = names[i] + "_CLOSE"
    x.columns = ls


historic_distance = {}
historic_corr = {}
for i in range (len(names)):
    for k in range (i+1,len(names)):
        print(i,k)
        tup = (names[i],names[k])
        l1 = d[names[i]]
        l2 = d[names[k]]
        df = pd.merge(left=l1, right=l2, left_on='DATE', right_on='DATE')
        st1 = names[i]+"_CLOSE"
        st2 = names[k]+"_CLOSE"
        mask = (df["DATE"] <= 20191231)
        df = df.loc[mask]
        print(df.iloc[-1:,])

        dtw , _= fastdtw(df[st1],df[st2])
        corr , _ = pearsonr(df[st1],df[st2])
        x = len(df)
        historic_distance[tup] = dtw/x
        historic_corr[tup] = corr

with open('historic_distance.pickle', 'wb') as f:
    pickle.dump(historic_distance, f)
    
with open('historic_corr.pickle', 'wb') as g:
    pickle.dump(historic_corr, g)