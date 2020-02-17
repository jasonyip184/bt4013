import numpy as np
from sklearn.linear_model import LinearRegression
from indicators import ADI, ADX, BB, CCI, EMA, OBV, RSI, SMA, StochOsc, StochRSI, UltiOsc, WilliamsR


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
            ################ TREND FOLOWING ################
            '''
            '''
            Simple Moving Average (SMA) crosses, period 5,10,20,50,100,200
            '''
            # indicators
            SMA5s = SMA(CLOSE[i], 5)
            SMA10s = SMA(CLOSE[i], 10)
            SMA20s = SMA(CLOSE[i], 20)
            SMA50s = SMA(CLOSE[i], 50)
            SMA100s = SMA(CLOSE[i], 100)
            SMA200s = SMA(CLOSE[i], 200)
            # signals
            SMA5_cross_buy = SMA5_cross_sell = SMA10_cross_buy = SMA10_cross_sell = SMA20_cross_buy = SMA20_cross_sell = SMA50_cross_buy = SMA50_cross_sell = SMA100_cross_buy = SMA100_cross_sell = SMA200_cross_buy = SMA200_cross_sell = False
            if (latest_CLOSE > SMA5s[-1]) and (CLOSE[i][-2] <= SMA5s[-2]):
                SMA5_cross_buy = True
            elif (latest_CLOSE < SMA5s[-1]) and (CLOSE[i][-2] >= SMA5s[-2]):
                SMA5_cross_sell = True
            if (latest_CLOSE > SMA10s[-1]) and (CLOSE[i][-2] <= SMA10s[-2]):
                SMA10_cross_buy = True
            elif (latest_CLOSE < SMA10s[-1]) and (CLOSE[i][-2] >= SMA10s[-2]):
                SMA10_cross_sell = True
            if (latest_CLOSE > SMA20s[-1]) and (CLOSE[i][-2] <= SMA20s[-2]):
                SMA20_cross_buy = True
            elif (latest_CLOSE < SMA20s[-1]) and (CLOSE[i][-2] >= SMA20s[-2]):
                SMA20_cross_sell = True
            if (latest_CLOSE > SMA50s[-1]) and (CLOSE[i][-2] <= SMA50s[-2]):
                SMA50_cross_buy = True
            elif (latest_CLOSE < SMA50s[-1]) and (CLOSE[i][-2] >= SMA50s[-2]):
                SMA50_cross_sell = True
            if (latest_CLOSE > SMA100s[-1]) and (CLOSE[i][-2] <= SMA100s[-2]):
                SMA100_cross_buy = True
            elif (latest_CLOSE < SMA100s[-1]) and (CLOSE[i][-2] >= SMA100s[-2]):
                SMA100_cross_sell = True
            if (latest_CLOSE > SMA200s[-1]) and (CLOSE[i][-2] <= SMA200s[-2]):
                SMA200_cross_buy = True
            elif (latest_CLOSE < SMA200s[-1]) and (CLOSE[i][-2] >= SMA200s[-2]):
                SMA200_cross_sell = True

            '''
            Exponential Moving Average (EMA) crosses, period 5,10,20,50,100,200
            '''
            # indicators
            EMA5s = EMA(CLOSE[i], 5)
            EMA10s = EMA(CLOSE[i], 10)
            EMA20s = EMA(CLOSE[i], 20)
            EMA50s = EMA(CLOSE[i], 50)
            EMA100s = EMA(CLOSE[i], 100)
            EMA200s = EMA(CLOSE[i], 200)
            # signals
            EMA5_cross_buy = EMA5_cross_sell = EMA10_cross_buy = EMA10_cross_sell = EMA20_cross_buy = EMA20_cross_sell = EMA50_cross_buy = EMA50_cross_sell = EMA100_cross_buy = EMA100_cross_sell = EMA200_cross_buy = EMA200_cross_sell = False
            if (latest_CLOSE > EMA5s[-1]) and (CLOSE[i][-2] <= EMA5s[-2]):
                EMA5_cross_buy = True
            elif (latest_CLOSE < EMA5s[-1]) and (CLOSE[i][-2] >= EMA5s[-2]):
                EMA5_cross_sell = True
            if (latest_CLOSE > EMA10s[-1]) and (CLOSE[i][-2] <= EMA10s[-2]):
                EMA10_cross_buy = True
            elif (latest_CLOSE < EMA10s[-1]) and (CLOSE[i][-2] >= EMA10s[-2]):
                EMA10_cross_sell = True
            if (latest_CLOSE > EMA20s[-1]) and (CLOSE[i][-2] <= EMA20s[-2]):
                EMA20_cross_buy = True
            elif (latest_CLOSE < EMA20s[-1]) and (CLOSE[i][-2] >= EMA20s[-2]):
                EMA20_cross_sell = True
            if (latest_CLOSE > EMA50s[-1]) and (CLOSE[i][-2] <= EMA50s[-2]):
                EMA50_cross_buy = True
            elif (latest_CLOSE < EMA50s[-1]) and (CLOSE[i][-2] >= EMA50s[-2]):
                EMA50_cross_sell = True
            if (latest_CLOSE > EMA100s[-1]) and (CLOSE[i][-2] <= EMA100s[-2]):
                EMA100_cross_buy = True
            elif (latest_CLOSE < EMA100s[-1]) and (CLOSE[i][-2] >= EMA100s[-2]):
                EMA100_cross_sell = True
            if (latest_CLOSE > EMA200s[-1]) and (CLOSE[i][-2] <= EMA200s[-2]):
                EMA200_cross_buy = True
            elif (latest_CLOSE < EMA200s[-1]) and (CLOSE[i][-2] >= EMA200s[-2]):
                EMA200_cross_sell = True

            '''
            Average Directional Movement Index (ADX), period 14
            '''
            # indicators
            mDIs, pDIs, ADXs = ADX(HIGH[i], LOW[i], CLOSE[i], 14)
            # signals
            ADX_bullish_cross = ADX_bearish_cross = False
            # Bullish strong trend cross
            if (ADXs[-1] > 25) and (pDIs[-1] > mDIs[-1]) and (pDIs[-2] <= mDIs[-2]):
                ADX_bullish_cross = True
            # Bearish strong trend cross
            elif (ADXs[-1] > 25) and (pDIs[-1] < mDIs[-1]) and (pDIs[-2] >= mDIs[-2]):
                ADX_bearish_cross = True

            '''
            Moving Average Convergence Divergence (MACD) fast=12, slow=26
            '''
            # indicator
            EMA12s = EMA(CLOSE[i], 12)
            EMA26s = EMA(CLOSE[i], 26)
            MACDs = [(a_i - b_i) if b_i is not None else None for a_i, b_i in zip(EMA12s, EMA26s)]
            # signals
            MACD_bullish_zero_cross = MACD_bearish_zero_cross = False
            # Bullish zero cross which sustains for 3 days (reduce false signals)
            if (MACDs[-5] <= 0) and (MACDs[-4] > 0) and (MACDs[-3] > 0) and (MACDs[-2] > 0) and (MACDs[-1] > 0):
                MACD_bullish_zero_cross = True
            # Bearish zero cross
            elif (MACDs[-5] >= 0) and (MACDs[-4] < 0) and (MACDs[-3] < 0) and (MACDs[-2] < 0) and (MACDs[-1] < 0):
                MACD_bearish_zero_cross = True

            '''
            Commodity Channel Index (CCI), period 14
            '''
            # indicator
            CCIs = CCI(HIGH[i], LOW[i], CLOSE[i], 14)
            # signals
            CCI_emerging_bull = CCI_emerging_bear = False
            # emerging bull
            if CCIs[-1] > 100:
                CCI_emerging_bull = True
            # emerging bear
            elif CCIs[-1] < -100:
                CCI_emerging_bear = True

            '''
            ################ MOMENTUM ################
            '''
            '''
            Relative Strength Index (RSI), period 14
            '''
            # indicator
            RSIs = RSI(CLOSE[i])
            # signals
            RSI_oversold_in_uptrend = RSI_overbought_in_downtrend = RSI_rising_center_cross = RSI_falling_center_cross = False
            # Uptrend and cross 30 to become oversold (Bullish)
            if (latest_CLOSE > SMA200s[-1]) and (RSIs[-2] >= 30) and (RSIs[-1] < 30):
                RSI_oversold_in_uptrend = True
            # Downtrend and cross 70 to become overbought (Bearish)
            elif (latest_CLOSE < SMA200s[-1]) and (RSIs[-2] <= 70) and (RSIs[-1] > 30):
                RSI_overbought_in_downtrend = True
            # Bullish center cross
            if (RSIs[-1] > 50) and (RSIs[-2] <= 50):
                RSI_bullish_center_cross = True
            # Bearish center cross
            elif (RSIs[-1] < 50) and (RSIs[-2] >= 50):
                RSI_bearish_center_cross = True
            
            '''
            Stochastic Oscillator, fast 14, slow 3
            '''
            # indicators
            Ks, Ds = StochOsc(CLOSE[i], HIGH[i], LOW[i], 14, 3)
            # signals
            StochOsc_bullish_cross = StochOsc_bearish_cross = StochOsc_bullish_divergence = StochOsc_bearish_divergence = False
            # K (fast) cross D (slow) from below
            if (Ks[-2] <= Ds[-2]) and (Ks[-1] > Ds[-1]):
                StochOsc_bullish_cross = True
            # K (fast) cross D (slow) from above
            elif (Ks[-2] >= Ds[-2]) and (Ks[-1] < Ds[-1]):
                StochOsc_bearish_cross = True
            # Bullish divergence
            if (LOW[i][-1] < LOW[i][-2]) and (Ks[-1] >= Ks[-2]):
                StochOsc_bullish_divergence = True
            # Bearish divergence
            elif (HIGH[i][-1] > HIGH[i][-2]) and (Ks[-1] <= Ks[-2]):
                StochOsc_bearish_divergence = True

            '''
            Stochastic RSI 14 period
            '''
            # indicator
            StochRSIs = StochRSI(RSIs, 14)
            # signals
            StochRSI_bullish_center_cross = StochRSI_bearish_center_cross = False
            # Bullish center cross
            if (StochRSIs[-1] > 50) and (StochRSIs[-2] <= 50):
                StochRSI_bullish_center_cross = True
            # Bearish center cross
            elif (StochRSIs[-1] < 50) and (StochRSIs[-2] >= 50):
                StochRSI_bearish_center_cross = True

            '''
            Williams %R, 14 period
            '''
            # indicator
            WilliamsRs = WilliamsR(HIGH[i], LOW[i], CLOSE[i])
            # signals
            WilliamsR_bullish_center_cross = WilliamsR_bearish_center_cross = False
            # Bullish center cross & price action
            if (WilliamsRs[-2] <= -50) and (WilliamsRs[-1] > -50) and (CLOSE[i][-1] > CLOSE[i][-2]):
                WilliamsR_bullish_center_cross = True
            # Bearish center cross & price action
            elif (WilliamsRs[-2] >= -50) and (WilliamsRs[-1] < -50) and (CLOSE[i][-1] < CLOSE[i][-2]):
                WilliamsR_bearish_center_cross = True
            
            ''' 
            Ultimate Oscillator, periods 20,40,80
            '''
            # indicator
            UltiOscs = UltiOsc(HIGH[i], LOW[i], CLOSE[i], 20, 40, 80)
            # signals
            UltiOsc_bullish_center_cross = UltiOsc_bearish_center_cross = False
            # Bullish center cross
            if (UltiOscs[-1] > 50) and (UltiOscs[-2] <= 50):
                UltiOsc_bullish_center_cross = True
            # Bearish center cross
            elif (UltiOscs[-1] < 50) and (UltiOscs[-2] >= 50):
                UltiOsc_bearish_center_cross = True

            '''
            ################ VOLUME ################
            '''
            '''
            Accumulation / Distribution Index (ADI)
            '''
            # indicator
            ADIs = ADI(HIGH[i], LOW[i], CLOSE[i], VOL[i])
            # signals
            ADI_bullish_reversal = ADI_bearish_reversal = ADI_bullish_trend_confo = ADI_bearish_trend_confo = False
            # Foreshadowing bullish reversal
            if (HIGH[i][-1] < HIGH[i][-2]) and (ADIs[-1] > ADIs[-2]):
                ADI_bullish_reversal = True
            # Foreshadowing bearish reversal
            elif (HIGH[i][-1] > HIGH[i][-2]) and (ADIs[-1] < ADIs[-2]):
                ADI_bearish_reversal = True
            # bullish trend confirmation
            if (latest_CLOSE > CLOSE[i][-2]) and (ADIs[-1] > ADIs[-2]):
                ADI_bullish_trend_confo = True
            # bearish trend confirmation
            elif (latest_CLOSE < CLOSE[i][-2]) and (ADIs[-1] < ADIs[-2]):
                ADI_bearish_trend_confo = True
            
            '''
            On-Balance Volume (OBV)
            '''
            # indicator
            OBVs = OBV(CLOSE[i], VOL[i])
            # signals
            OBV_bullish_trend_confo = OBV_bearish_trend_confo = False
            if (OBVs[-1] > OBVs[-2]) and (OBVs[-2] > OBVs[-3]):
                OBV_bullish_trend_confo = True
            elif (OBVs[-1] < OBVs[-2]) and (OBVs[-2] < OBVs[-3]):
                OBV_bearish_trend_confo = True

            '''
            ################ VOLATILITY ################
            '''
            ''' 
            Bollinger Bands (BB), 20 period
            '''
            # indicator + signal
            BB_high_crosses, BB_low_crosses = BB(CLOSE[i], 20)
            BB_bullish_reversal = BB_bearish_reversal = False
            if BB_high_crosses[-1] == 1:
                BB_bearish_reversal = True
            elif BB_low_crosses[-1] == 1:
                BB_bullish_reversal = True
            print(BB_bullish_reversal, BB_bearish_reversal)
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
    model = 'TA_multifactor' # trend_following, MLR_CLOSE, TA_multifactor
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