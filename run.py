### Quantiacs Trend Following Trading System Example
import numpy as np
from sklearn.linear_model import LinearRegression


def EMA_func(closes, order):
    prev_EMA = np.nanmean(closes[:order])
    for j in range(order, len(closes)):
        alpha = 2 / (order+1)
        EMA = alpha * closes[j] + (1-alpha) * prev_EMA
        prev_EMA = EMA
    return EMA


def RSI_func(closes):
    '''
    Based on formula: https://www.investopedia.com/terms/r/rsi.asp
    Returns list of past RSIs
    '''
    CLOSE_chg = [0] + np.diff(closes)
    RSIs = []
    # very first RSI
    gains = 0
    losses = 0
    for chg in CLOSE_chg:
        if chg > 0:
            gains += chg
        else:
            losses -= chg
    prev_avg_gain = gains / 14
    prev_avg_loss = losses / 14
    RS = prev_avg_gain / prev_avg_loss
    RSI = 100 - 100 / (1 + RS)
    RSIs.append(RSI)
    
    # subsequent RSI
    for j in range(14, len(closes)):
        current_chg = closes[j] - closes[j-1]
        if current_chg > 0:
            current_gain = current_chg
            current_loss = 0
        else:
            current_gain = 0
            current_loss = -current_chg
        avg_gain = (prev_avg_gain * 13 + current_gain) / 14
        avg_loss = (prev_avg_loss * 13 + current_loss) / 14
        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)
        RSIs.append(RSI)
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss
    return RSIs


def stochastic_oscillator_func(closes, highs, lows, K_period, D_period):
    # K is fast stochastic osc
    Ks = [None] * (K_period-1)
    for j in range(K_period, (len(highs)+1)):
        highest_high = np.nanmax(highs[j-K_period:j])
        lowest_low = np.nanmin(lows[j-K_period:j])
        current_close = closes[j-1]
        K = (current_close - lowest_low) / (highest_high - lowest_low) * 100
        Ks.append(K)
    # D is slow stochastic osc, the D_period-SMA of K
    Ds = [None] * (K_period+D_period-2)
    for j in range((K_period+D_period-2), len(Ks)):
        Ds.append(np.nanmean(Ks[(j-D_period+1):(j+1)]))
    return Ks, Ds


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,
                    USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV,
                    exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    lookback = settings['lookback']
    pos = np.zeros(nMarkets)

    # to understand how this system works
    print("Using data from {} onwards to predict/take position in {}".format(DATE[0],DATE[-1]))

    OPEN = np.transpose(OPEN)[1:]
    HIGH = np.transpose(HIGH)[1:]
    LOW = np.transpose(LOW)[1:]
    CLOSE = np.transpose(CLOSE)[1:]
    VOL = np.transpose(VOL)[1:]

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


    elif settings['model'] == 'TA_multifactor':
        '''
        Based on factors from https://www.investing.com/technical/us-spx-500-futures-technical-analysis
        '''
        for i in range(0, nMarkets-1):
            latest_CLOSE = CLOSE[i][-1]

            '''
            SMA crosses
            '''
            SMA5 = np.nanmean(CLOSE[i][-5:])
            SMA10 = np.nanmean(CLOSE[i][-10:])
            SMA20 = np.nanmean(CLOSE[i][-20:])
            SMA50 = np.nanmean(CLOSE[i][-50:])
            SMA100 = np.nanmean(CLOSE[i][-100:])
            SMA200 = np.nanmean(CLOSE[i][-200:])
            SMA5_cross_buy = SMA5 > latest_CLOSE
            SMA10_cross_buy = SMA10 > latest_CLOSE
            SMA20_cross_buy = SMA20 > latest_CLOSE
            SMA50_cross_buy = SMA50 > latest_CLOSE
            SMA100_cross_buy = SMA100 > latest_CLOSE
            SMA200_cross_buy = SMA200 > latest_CLOSE

            '''
            EMA crosses
            '''
            EMA5 = EMA_func(CLOSE[i], 5)
            EMA10 = EMA_func(CLOSE[i], 10)
            EMA20 = EMA_func(CLOSE[i], 20)
            EMA50 = EMA_func(CLOSE[i], 50)
            EMA100 = EMA_func(CLOSE[i], 100)
            EMA200 = EMA_func(CLOSE[i], 200)
            EMA5_cross_buy = EMA5 > latest_CLOSE
            EMA10_cross_buy = EMA10 > latest_CLOSE
            EMA20_cross_buy = EMA20 > latest_CLOSE
            EMA50_cross_buy = EMA50 > latest_CLOSE
            EMA100_cross_buy = EMA100 > latest_CLOSE
            EMA200_cross_buy = EMA200 > latest_CLOSE

            '''
            RSI - momentum oscillator
            Signals from https://www.babypips.com/learn/forex/relative-strength-index
            '''
            RSIs = RSI_func(CLOSE[i])
            RSI_oversold_in_uptrend = RSI_overbought_in_downtrend = RSI_rising_centerline_crossover = RSI_falling_centerline_crossover = False
            # Uptrend and cross 30 to become oversold
            if (latest_CLOSE > SMA200) and (RSIs[-2] >= 30) and (RSIs[-1] < 30):
                RSI_oversold_in_uptrend = True
            # Downtrend and cross 70 to become overbought
            elif (latest_CLOSE < SMA200) and (RSIs[-2] <= 70) and (RSIs[-1] > 30):
                RSI_overbought_in_downtrend = True
            # Rising centerline crossover
            if (RSIs[-1] > 50) and (RSIs[-2] <= 50):
                RSI_rising_centerline_crossover = True
            # Falling centerline crossover
            elif (RSIs[-1] < 50) and (RSIs[-2] >= 50):
                RSI_falling_centerline_crossover = True
            
            '''
            Stochastic Oscillator
            https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full
            '''
            Ks, Ds = stochastic_oscillator_func(CLOSE[i], HIGH[i], LOW[i], 14, 3)
            stochastic_bullish_crossover = stochastic_bearish_crossover = stochastic_bullish_divergence = stochastic_bearish_divergence = False
            # K (fast) cross D (slow) from below
            if (Ks[-2] <= Ds[-2]) and (Ks[-1] > Ds[-1]):
                stochastic_bullish_crossover = True
            # K (fast) cross D (slow) from above
            elif (Ks[-2] >= Ds[-2]) and (Ks[-1] < Ds[-1]):
                stochastic_bearish_crossover = True
            # Bullish divergence
            if (LOW[i][-1] < LOW[i][-2]) and (Ks[-1] >= Ks[-2]):
                stochastic_bullish_divergence = True
            # Bearish divergence
            elif (HIGH[i][-1] > HIGH[i][-2]) and (Ks[-1] <= Ks[-2]):
                stochastic_bearish_divergence = True
            break

        # pos[longEquity] = 1 
        # pos[shortEquity] = -1

    
    elif settings['model'] == 'MLR_CLOSE':
        '''
        Multiple Linear Regression using Y=CLOSE, Xs=prev day's OHLC & Vol.
        If today's OHLC&Vol predicts tomorrow CLOSE to go up/down from today's CLOSE, go long/short
        '''
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
    budget = 1000000
    slippage = 0.05
    model = 'MLR_CLOSE' # trend_following, MLR_CLOSE, TA_multifactor
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
    print(results['stats'])
    # print(results['returns'])
    # print(results['marketEquity'])