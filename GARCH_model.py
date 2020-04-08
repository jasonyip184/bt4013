import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import datetime as dt
from arch import arch_model

def train_garch_model(future_name):
    df = pd.read_csv(f"./tickerData/{future_name}.txt", parse_dates = ['DATE'], )
    df = df.rename(columns=lambda x: x.strip())
    start_time  =df['DATE'].iloc[0]
    end_time1 = dt.datetime(2018,12,1,0,0,0,0)
    end_time2 = dt.datetime(2018,12,31,0,0,0,0)
    index = df.index[(df['DATE'] > end_time1) & (df['DATE'] < end_time2)].tolist()
    x = np.array(df['CLOSE'])
    x = x[0:index[0]+1]
    log_rtn = np.diff(np.log(x))
    AIC_mat = []
    for p in range(1, 10):
        AIC_list = []
        for q in range(1, 10):
            current_model = arch_model(log_rtn, p=p, q=q)
            current_model_fit = current_model.fit()
            AIC_list.append(current_model_fit.aic)
        AIC_mat.append(AIC_list)
    best_params = np.argwhere(AIC_mat == np.min(AIC_mat))
    p = best_params[0][0]
    q = best_params[0][1]
    #print((p,q))
    model = arch_model(log_rtn, p=int(p+1), q=int(q+1))
    model_fit = model.fit()
    params = model_fit.params
    #print(list(params))
    s = {}
    s['params'] = list(params)
    s['order'] = (int(p+1), int(q+1))

    f= open(f'./data/garch_models/{future_name}_garch_model.txt',"w")
    with open(f'./data/garch_models/{future_name}_garch_model.txt', 'w') as f:
        json.dump(s, f, ensure_ascii=False)


def find_correlation_between_vol_and_return(future_name):
    df = pd.read_csv(f"./tickerData/{future_name}.txt", parse_dates = ['DATE'], )
    df = df.rename(columns=lambda x: x.strip())
    start_time  =df['DATE'].iloc[0]
    #It is very important to address the end date
    end_time1 = dt.datetime(2018,12,1,0,0,0,0)
    end_time2 = dt.datetime(2018,12,31,0,0,0,0)
    index = df.index[(df['DATE'] > end_time1) & (df['DATE'] < end_time2)].tolist()
    x = np.array(df['CLOSE'])
    x = x[0:index[0]+1]
    log_rtn = np.diff(np.log(x))
    volList = pd.Series(log_rtn).rolling(40).std(ddof=0)
    cor = np.corrcoef(volList[40:], log_rtn[40:])
    return cor[0][1]


if __name__ == '__main__':
    futures = ['F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']
    for future in futures:
        train_garch_model(future)
        
    cor_dictionary = {}
    for future in futures:
        current_cor = find_correlation_between_vol_and_return(future)
        cor_dictionary[future] = current_cor

    with open(f'./data/garch_models/correlation.txt', 'w') as f:
         json.dump(cor_dictionary, f, ensure_ascii=False)