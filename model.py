import ta
import numpy as np
import pandas as pd
from indicators import ADI, ADX, BB, CCI, EMA, OBV, RSI, SMA, StochOsc, StochRSI, UltiOsc, WilliamsR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import lightgbm as lgb


def train_lgb_model(future_name):
    # Data munging
    df = pd.read_csv(f"./tickerData/{future_name}.txt")
    df['ADI'] = ADI(df['HIGH'], df['LOW'], df['CLOSE'], df['VOL'])
    df['WilliamsR'] = WilliamsR(df['HIGH'], df['LOW'], df['CLOSE'])
    df['label'] = np.where(df['CLOSE'].shift(periods=-1) - df['CLOSE'] < 0, -1, 1)
    df.loc[abs(df['CLOSE'].shift(periods=-1) - df['CLOSE']) < 30, 'label'] = 0
    # Assume stocks have pacf of 2-timestep significance
    df['LAG_1'] = df['CLOSE'].shift(periods=1)
    df['LAG_2'] = df['CLOSE'].shift(periods=2)
    df['LABEL_var'] = df['CLOSE'].shift(periods=-1)
    data = df.iloc[2:][['OPEN','HIGH','LOW','CLOSE','VOL','ADI','WilliamsR','LAG_1','LAG_2','label']]

    # Split test-train set
    train_df, test_df = train_test_split(data, test_size=0.2, shuffle=False)
    train_X = train_df[['OPEN','HIGH','LOW','CLOSE','VOL','ADI','WilliamsR','LAG_1','LAG_2']].to_numpy()
    train_Y = train_df['label'].to_numpy()
    test_X = test_df[['OPEN','HIGH','LOW','CLOSE','VOL','ADI','WilliamsR','LAG_1','LAG_2']].to_numpy()
    test_Y = test_df['label'].to_numpy()

    # Create lgb model
    lgb_clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_leaves=75, 
    learning_rate=0.05, 
    max_depth=10,
    n_estimators=100)
    lgb_clf.fit(train_X, 
                train_Y,
                eval_set=[(test_X, test_Y)],
                early_stopping_rounds=50, 
                verbose=False)
    lgb_clf.booster_.save_model(f'./model_pickle_files/{future_name}_model')
    test_accuracy = np.sum(lgb_clf.predict(test_X) == test_Y)/len(test_Y)
    print (f"Successfully trained {future_name}. Test accuracy: {test_accuracy:.3f}")
    print(f"Saved as {future_name}_model under /model_pickle_files.")


def get_lgb_prediction(model_dir, features):
    model = lgb.Booster(model_file=model_dir)
    return np.argmax(model.predict(features), axis=1)-1

if __name__ == '__main__':
    futures = ['F_ED', 'F_F', 'F_EB', 'F_ZQ', 'F_UZ', 'F_VW', 'F_SS']
    for future in futures:
        train_lgb_model(future)