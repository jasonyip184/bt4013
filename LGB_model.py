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
    # If change is too small, don't take position
    returns = abs(df['CLOSE'].shift(periods=-1) - df['CLOSE'])
    change_limit = 0.5 * returns.mean()
    df.loc[abs(df['CLOSE'].shift(periods=-1) - df['CLOSE']) < change_limit, 'label'] = 0
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
    test_accuracy = np.sum(lgb_clf.predict(test_X) == test_Y)/len(test_Y)
    if test_accuracy > 0.65:
        lgb_clf.booster_.save_model(f'./data/lgb_models/{future_name}_model')
        print (f"Successfully trained {future_name}. Test accuracy: {test_accuracy:.3f}")
        print(f"Saved as {future_name}_model under /data/lgb_models.")


def get_lgb_prediction(model_dir, features):
    model = lgb.Booster(model_file=model_dir)
    return np.argmax(model.predict(features), axis=1)-1

if __name__ == '__main__':
    futures = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']
    for future in futures:
        train_lgb_model(future)