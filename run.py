### Quantiacs Trend Following Trading System Example
import numpy as np
from sklearn.linear_model import LinearRegression

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,
                    USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV,
                    exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    lookback = settings['lookback']
    pos = np.zeros(nMarkets)

    # to understand how this system works
    print("Using data from {} onwards to predict/take position in {}".format(DATE[0],DATE[-1]))


    if settings['model'] == 'trend_following':
        '''
        Sample Trend Following Model
        '''
        periodLonger = 200 #%[100:50:300]#
        periodShorter = 40

        # Calculate Simple Moving Average (SMA)
        smaLongerPeriod = np.nansum(CLOSE[-periodLonger:,:],axis=0) / periodLonger
        smaShorterPeriod=  np.nansum(CLOSE[-periodShorter:,:],axis=0) / periodShorter

        longEquity = smaShorterPeriod > smaLongerPeriod
        shortEquity = ~longEquity

        # Equal weight is placed on each market across the long/short positions over here
        pos[longEquity] = 1 
        pos[shortEquity] = -1

    
    elif settings['model'] == 'MLR_CLOSE':
        '''
        Multiple Linear Regression using Y=CLOSE, Xs=prev day's OHLC & Vol.
        If today's OHLC&Vol predicts tomorrow CLOSE to go up/down from today's CLOSE, go long/short
        '''
        OPEN = np.transpose(OPEN)[1:]
        HIGH = np.transpose(HIGH)[1:]
        LOW = np.transpose(LOW)[1:]
        CLOSE = np.transpose(CLOSE)[1:]
        VOL = np.transpose(VOL)[1:]
        
        for i in range(0, nMarkets-1):
            # training & prediction
            # Xs is a matrix of shape (n_samples, n_features)
            train_Xs = np.transpose([OPEN[i][:-1], HIGH[i][:-1], LOW[i][:-1], CLOSE[i][:-1], VOL[i][:-1]])
            test_Xs = [OPEN[i][-1], HIGH[i][-1], LOW[i][-1], CLOSE[i][-1], VOL[i][-1]]
            train_Y = CLOSE[i][1:]
            reg = LinearRegression()
            reg.fit(train_Xs, train_Y)
            pred_Y = reg.predict(np.array([test_Xs]))[0]
            ## Uncomment below to see rsquared
            # print("Rsquared: {}".format(round(reg.score(train_Xs, train_Y),5)))

            # taking position (pos[0] is reserved for NaN field)
            if pred_Y > CLOSE[i][-1]: # close expected to go up from today's close
                pos[i+1] = 1
            else:
                pos[i+1] = -1
        print("Today's position in the 88 futures:", pos)


    elif settings['model'] == 'ANOTHER MODEL':
        pass


    weights = pos / np.nansum(abs(pos))
    return weights, settings


def mySettings():
    settings = {}
    markets  = ['CASH', 'F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']
    budget = 10**6
    slippage = 0.05
    model = 'MLR_CLOSE' # trend_following, MLR_CLOSE
    lookback = 504
    beginInSample = '20180119'
    endInSample = None # taking the latest available

    settings = {'markets': markets, 'beginInSample': beginInSample, 'endInSample': endInSample, 'lookback': lookback,
                'budget': budget, 'slippage': slippage, 'model': model}

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)