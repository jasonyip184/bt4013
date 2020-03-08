import pandas as pd
import os

def clean():
    print("Cleaning headers in text files...")
    for f in os.listdir('tickerData'):
        if f.endswith('.txt'):
            df = pd.read_csv('tickerData/{}'.format(f))
            df.columns = [s.replace(' ','') for s in df.columns]
            df.to_csv('tickerData/{}'.format(f), index=False)
    print("Cleaning done.")