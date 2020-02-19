import numpy as np
from sklearn.linear_model import LinearRegression
from indicators import ADI, ADX, BB, CCI, EMA, OBV, RSI, SMA, StochOsc, StochRSI, UltiOsc, WilliamsR
from economic_indicators import econ_long_short_allocation, market_factor_weights


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,
                    USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV,
                    exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    lookback = settings['lookback']
    pos = np.zeros(nMarkets)
    markets = settings['markets']
    w = settings['market_factor_weights']
    lweights, sweights = econ_long_short_allocation(markets, DATE[0], DATE[-1], w, activate=settings['dynamic_portfolio_allocation'])

    # to understand how this system works
    print("Using data from {} onwards to predict/take position in {}".format(DATE[0],DATE[-1]))

    OPEN = np.transpose(OPEN)[1:]
    HIGH = np.transpose(HIGH)[1:]
    LOW = np.transpose(LOW)[1:]
    CLOSE = np.transpose(CLOSE)[1:]
    VOL = np.transpose(VOL)[1:]

    if settings['model'] == 'MLR_CLOSE':
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

    elif settings['model'] == 'TA_multifactor':
        '''
        Based on factors from https://www.investing.com/technical/us-spx-500-futures-technical-analysis

        ################ TREND FOLOWING ################
        Simple Moving Average (SMA) crosses, period 5,10,20,50,100,200
        '''
        # # indicators
        # SMA5s = [SMA(close, 5) for close in CLOSE]
        # SMA10s = [SMA(close, 10) for close in CLOSE]
        # SMA20s = [SMA(close, 20) for close in CLOSE]
        # SMA50s = [SMA(close, 50) for close in CLOSE]
        # SMA100s = [SMA(close, 100) for close in CLOSE]
        # SMA200s = [SMA(close, 200) for close in CLOSE]

        # # signals
        # def buy_condition(close, sma):
        #     return (close[-1] > sma[-1]) and (close[-2] <= sma[-2])
        # def sell_condition(close, sma):
        #     return (close[-1] < sma[-1]) and (close[-2] >= sma[-2])
        # SMA5_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA5s)]
        # SMA5_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA5s)]
        # SMA10_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA10s)]
        # SMA10_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA10s)]
        # SMA20_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA20s)]
        # SMA20_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA20s)]
        # SMA50_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA50s)]
        # SMA50_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA50s)]
        # SMA100_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA100s)]
        # SMA100_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA100s)]
        # SMA200_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA200s)]
        # SMA200_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA200s)]

        '''
        Exponential Moving Average (EMA) crosses, period 5,10,20,50,100,200
        '''
        # # indicators
        # EMA5s = [EMA(close, 5) for close in CLOSE]
        # EMA10s = [EMA(close, 10) for close in CLOSE]
        # EMA20s = [EMA(close, 20) for close in CLOSE]
        # EMA50s = [EMA(close, 50) for close in CLOSE]
        # EMA100s = [EMA(close, 100) for close in CLOSE]
        # EMA200s = [EMA(close, 200) for close in CLOSE]

        # # signals
        # # def condition(close, ema):
        # #     return (close[-1] > ema[-1]) and (close[-2] <= ema[-2])
        # def buy_condition(close, ema):
        #     return (close[-1] > ema[-1]) and (close[-2] <= ema[-2])
        # def sell_condition(close, ema):
        #     return (close[-1] < ema[-1]) and (close[-2] >= ema[-2])
        # EMA5_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA5s)]
        # EMA5_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA5s)]
        # EMA10_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA10s)]
        # EMA10_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA10s)]
        # EMA20_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA20s)]
        # EMA20_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA20s)]
        # EMA50_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA50s)]
        # EMA50_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA50s)]
        # EMA100_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA100s)]
        # EMA100_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA100s)]
        # EMA200_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA200s)]
        # EMA200_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA200s)]

        '''
        Average Directional Movement Index (ADX), period 14
        '''
        # # indicators
        # ADXs = [ADX(high,low,close,14) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # # signals
        # # adx[0] is mDI, adx[1] is pDI, adx[2] is actual ADX
        # def bullish_condition(adx):
        #     # Bullish strong trend cross
        #     return (adx[2][-1] > 20) and (adx[1][-1] > adx[0][-1]) and (adx[1][-2] <= adx[0][-2])
        # def bearish_condition(adx):
        #     # Bearish strong trend cross
        #     return (adx[2][-1] > 20) and (adx[1][-1] < adx[0][-1]) and (adx[1][-2] >= adx[0][-2])
        # ADX_bullish_crosses = [True if bullish_condition(adx) else False for adx in ADXs]
        # ADX_bearish_crosses = [True if bearish_condition(adx) else False for adx in ADXs]

        '''
        Moving Average Convergence Divergence (MACD) fast=12, slow=26
        '''
        # # indicator
        # EMA12s = [EMA(close, 12) for close in CLOSE]
        # EMA26s = [EMA(close, 26) for close in CLOSE]
        # MACDs = [[(a-b) if b is not None else None for a, b in zip(EMA12, EMA26)] for EMA12, EMA26 in zip(EMA12s, EMA26s)]
        # # signals
        # def bullish_condition(MACD):
        #     # Bullish zero cross which sustains for 2 days (reduce false signals)
        #     return (MACD[-3] <= 0) and (MACD[-2] > 0) and (MACD[-1] > 0)
        # def bearish_condition(MACD):
        #     # Bearish zero cross
        #     return (MACD[-3] >= 0) and (MACD[-2] < 0) and (MACD[-1] < 0)
        # MACD_bullish_zero_cross = [True if bullish_condition(MACD) else False for MACD in MACDs]
        # MACD_bearish_zero_cross = [True if bearish_condition(MACD) else False for MACD in MACDs]

        '''
        Commodity Channel Index (CCI), period 14
        '''
        # # indicator
        # CCIs = [CCI(high,low,close,14) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # # signals
        # def bullish_condition(CCI):
        #     return (CCI[-1] > 100) and (CCI[-2] <= 100)
        # def bearish_condition(CCI):
        #     return (CCI[-1] < -100) and (CCI[-2] >= -100)   
        # CCI_emerging_bulls = [True if bullish_condition(CCI) else False for CCI in CCIs]
        # CCI_emerging_bears = [True if bearish_condition(CCI) else False for CCI in CCIs]

        '''
        ################ MOMENTUM ################
        Relative Strength Index (RSI), period 14
        '''
        # # indicator
        # RSIs = [RSI(close) for close in CLOSE]
        # # signals
        # def bullish_reversal(rsi, sma200, close):
        #     # Uptrend and cross 30 to become oversold (Bullish)
        #     return (close[-1] > sma200[-1]) and (rsi[-2] >= 30) and (rsi[-1] < 30)
        # def bearish_reversal(rsi, sma200, close):
        #     # Downtrend and cross 70 to become overbought (Bearish)
        #     return (close[-1] < sma200[-1]) and (rsi[-2] <= 70) and (rsi[-1] > 70)
        # def underbought_uptrend(rsi, sma200, close):
        #     # Uptrend and underbought
        #     return (close[-1] > sma200[-1]) and (rsi[-1] < 50)
        # def undersold_downtrend(rsi, sma200, close):
        #     # Downtrend and undersold
        #     return (close[-1] < sma200[-1]) and (rsi[-1] > 50)
        # RSI_bullish_reversal = [True if bullish_reversal(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        # RSI_bearish_reversal = [True if bearish_reversal(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        # RSI_underbought_uptrend = [True if underbought_uptrend(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        # RSI_undersold_downtrend = [True if undersold_downtrend(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        
        '''
        Stochastic Oscillator, fast 14, slow 3
        '''
        # # indicators
        # StochOscs = [StochOsc(close,high,low, 14, 3) for close,high,low in zip(CLOSE,HIGH,LOW)]
        # # signals
        # # stochosc[0] is Ks, stochosc[1] is Ds
        # def bullish_cross(stochosc):
        #     # K (fast) cross D (slow) from below
        #     return (stochosc[0][-2] <= stochosc[1][-2]) and (stochosc[0][-1] > stochosc[1][-1])
        # def bearish_cross(stochosc):
        #     # K (fast) cross D (slow) from above
        #     return (stochosc[0][-2] >= stochosc[1][-2]) and (stochosc[0][-1] < stochosc[1][-1])
        # StochOsc_bullish_cross = [True if bullish_cross(stochosc) else False for stochosc in StochOscs]
        # StochOsc_bearish_cross = [True if bearish_cross(stochosc) else False for stochosc in StochOscs]

        '''
        Williams %R, 14 period
        '''
        # # indicator
        # WilliamsRs = [WilliamsR(high,low,close) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # # signals
        # def bullish(wr, close, sma100):
        #     # Overbought price action
        #     return (wr[-1] > -20) and (close[-1] > sma100[-1]) and (close[-2] <= sma100[-2])
        # def bearish(wr, close, sma100):
        #     # Oversold price action
        #     return (wr[-1] < -80) and (close[-1] < sma100[-1]) and (close[-2] >= sma100[-2])
        # WilliamsR_uptrend = [True if bullish(wr,close,sma100) else False for wr,close,sma100 in zip(WilliamsRs,CLOSE,SMA100s)]
        # WilliamsR_downtrend = [True if bearish(wr,close,sma100) else False for wr,close,sma100 in zip(WilliamsRs,CLOSE,SMA100s)]
        
        ''' 
        Ultimate Oscillator, periods 20,40,80
        '''
        # # indicator
        # UltiOscs = [UltiOsc(high,low,close,20,40,80) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # # signals
        # def bullish_cross(ultiosc):
        #     # Bullish center cross
        #     return (ultiosc[-1] > 50) and (ultiosc[-2] <= 50)
        # def bearish_cross(ultiosc):
        #     # Bearish center cross
        #     return (ultiosc[-1] < 50) and (ultiosc[-2] >= 50)
        # def bullish_reversal(ultiosc):
        #     # Bullish reversal from oversold
        #     return (ultiosc[-1] < 30) and (ultiosc[-2] >= 30)
        # def bearish_reversal(ultiosc):
        #     # Bearish reversal from overbought
        #     return (ultiosc[-1] > 70) and (ultiosc[-2] >= 70)
        # UltiOsc_bullish_cross = [True if bullish_cross(ultiosc) else False for ultiosc in UltiOscs]
        # UltiOsc_bearish_cross = [True if bearish_cross(ultiosc) else False for ultiosc in UltiOscs]
        # UltiOsc_bullish_reversal = [True if bullish_reversal(ultiosc) else False for ultiosc in UltiOscs]
        # UltiOsc_bearish_reversal = [True if bearish_reversal(ultiosc) else False for ultiosc in UltiOscs]

        '''
        ################ VOLUME ################
        Accumulation / Distribution Index (ADI)
        '''
        # # indicator
        # ADIs = [ADI(high,low,close,vol) for high,low,close,vol in zip(HIGH,LOW,CLOSE,VOL)]
        # def bullish_trend(close, adi, sma200):
        #     # bullish trend confirmation
        #     return (close[-1] > sma200[-1]) and (close[-2] <= sma200[-2]) and (adi[-1] > adi[-2])
        # def bearish_trend(close, adi, sma200):
        #     # bearish trend confirmation
        #     return (close[-1] < sma200[-1]) and (close[-2] >= sma200[-2])  and (adi[-1] < adi[-2])
        # ADI_bullish_trend_confo = [True if bullish_trend(close,adi,sma200) else False for close,adi,sma200 in zip(CLOSE,ADIs,SMA200s)]
        # ADI_bearish_trend_confo = [True if bearish_trend(close,adi,sma200) else False for close,adi,sma200 in zip(CLOSE,ADIs,SMA200s)]
        
        '''
        On-Balance Volume (OBV)
        '''
        # # indicator
        # OBVs = [OBV(close,vol) for close,vol in zip(CLOSE,VOL)]
        # # signals
        # def bullish_trend(obv):
        #     return (obv[-1] > obv[-2]) and (obv[-2] > obv[-3])
        # def bearish_trend(obv):
        #     return (obv[-1] < obv[-2]) and (obv[-2] < obv[-3])
        # OBV_bullish_trend_confo = [True if bullish_trend(obv) else False for obv in OBVs]
        # OBV_bearish_trend_confo = [True if bearish_trend(obv) else False for obv in OBVs]

        '''
        ################ VOLATILITY ################
        Bollinger Bands (BB), 20 period
        '''
        # indicator + signal
        # bb[0] = BB_high_crosses, bb[1] = BB_low_crosses
        BBs = [BB(close, 20) for close in CLOSE]
        BB_bullish_reversal = [True if bb[1][-1] == 1 else False for bb in BBs]
        BB_bearish_reversal = [True if bb[0][-1] == 1 else False for bb in BBs]

        '''
        Execution
        '''
        for i in range(0, nMarkets-1):
            future_name = markets[i+1]
            # # Trend following
            # # 'sharpe': -4.9505, 'sortino': -6.6969, 'returnYearly': -0.24064, 'volaYearly': 0.048608
            # # short - 'sharpe': -0.3336, 'sortino': -0.5194, 'returnYearly': -0.0098, 'volaYearly': 0.0293
            # if (SMA5_cross_buys[i] == True) or (SMA10_cross_buys[i] == True) or (SMA20_cross_buys[i] == True) or (SMA50_cross_buys[i] == True) or (SMA100_cross_buys[i] == True) or (SMA200_cross_buys == True):
            #     pos[i+1] = 1
            # if (SMA5_cross_sells[i] == True) or (SMA10_cross_sells[i] == True) or (SMA20_cross_sells[i] == True) or (SMA50_cross_sells[i] == True) or (SMA100_cross_sells[i] == True) or (SMA200_cross_sells == True):
            #     pos[i+1] = -1

            # Mean reverting
            # 'sharpe': -3.4408, 'sortino': -5.3037, 'returnYearly': -0.17250, 'volaYearly': 0.050133
            # short - 'sharpe': 2.9931, 'sortino': 5.5113, 'returnYearly': 0.0738, 'volaYearly': 0.0247
            # if (SMA5_cross_buys[i] == True) or (SMA10_cross_buys[i] == True) or (SMA20_cross_buys[i] == True) or (SMA50_cross_buys[i] == True) or (SMA100_cross_buys[i] == True) or (SMA200_cross_buys == True):
            #     pos[i+1] = -1
            # if (SMA5_cross_sells[i] == True) or (SMA10_cross_sells[i] == True) or (SMA20_cross_sells[i] == True) or (SMA50_cross_sells[i] == True) or (SMA100_cross_sells[i] == True) or (SMA200_cross_sells == True):
            #     pos[i+1] = 1

            # # Trend following
            # # 'sharpe': -7.1276, 'sortino': -8.3732, 'returnYearly': -0.30480, 'volaYearly': 0.042763
            # # short - 'sharpe': -2.7881, 'sortino': -3.8461, 'returnYearly': -0.0875, 'volaYearly': 0.0314
            # if (EMA5_cross_buys[i] == True) or (EMA10_cross_buys[i] == True) or (EMA20_cross_buys[i] == True) or (EMA50_cross_buys[i] == True) or (EMA100_cross_buys[i] == True) or (EMA200_cross_buys == True):
            #     pos[i+1] = 1
            # if (EMA5_cross_sells[i] == True) or (EMA10_cross_sells[i] == True) or (EMA20_cross_sells[i] == True) or (EMA50_cross_sells[i] == True) or (EMA100_cross_sells[i] == True) or (EMA200_cross_sells == True):
            #     pos[i+1] = -1

            # # Mean-reverting
            # # 'sharpe': -2.2089, 'sortino': -3.8098, 'returnYearly': -0.096276, 'volaYearly': 0.043585
            # # short - 'sharpe': 4.3531, 'sortino': 8.7333, 'returnYearly': 0.0787, 'volaYearly': 0.0181
            # if (EMA5_cross_buys[i] == True) or (EMA10_cross_buys[i] == True) or (EMA20_cross_buys[i] == True) or (EMA50_cross_buys[i] == True) or (EMA100_cross_buys[i] == True) or (EMA200_cross_buys == True):
            #     pos[i+1] = -1
            # if (EMA5_cross_sells[i] == True) or (EMA10_cross_sells[i] == True) or (EMA20_cross_sells[i] == True) or (EMA50_cross_sells[i] == True) or (EMA100_cross_sells[i] == True) or (EMA200_cross_sells == True):
            #     pos[i+1] = 1

            # # 'sharpe': -0.70354, 'sortino': -1.4637, 'returnYearly': -0.05842, 'volaYearly': 0.083045
            # # short - 'sharpe': -1.12151, 'sortino': -2.0892, 'returnYearly': -0.0529, 'volaYearly': 0.0436
            # if ADX_bullish_crosses[i] == True:
            #     pos[i+1] = 1
            # elif ADX_bearish_crosses[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': 2.2280, 'sortino': 4.9203, 'returnYearly': 0.1650, 'volaYearly': 0.0741
            # # short - 'sharpe': 6.4176, 'sortino': 17.4393, 'returnYearly': 0.3672, 'volaYearly': 0.0572
            # if MACD_bullish_zero_cross[i] == True:
            #     pos[i+1] = 1
            # elif MACD_bearish_zero_cross[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': -1.7270, 'sortino': -2.7356, 'returnYearly': -0.13488, 'volaYearly': 0.078105
            # # short - 'sharpe': 0.6039, 'sortino': 1.1404, 'returnYearly': 0.0204, 'volaYearly': 0.0398
            # # avg longs per day: 5.87 , avg shorts per day: 6.391
            # if CCI_emerging_bulls[i] == True:
            #     pos[i+1] = 1
            # elif CCI_emerging_bears[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': -3.5085, 'sortino': -3.5085, 'returnYearly': -0.46177, 'volaYearly': 0.13161
            # # short - 'sharpe': 2.4844, 'sortino': 10.2102, 'returnYearly': 0.007, 'volaYearly': 0.0028
            # if RSI_bullish_reversal[i] == True:
            #     pos[i+1] = 1
            # elif RSI_bearish_reversal[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': -1.897, 'sortino': -2.608, 'returnYearly': -0.1386, 'volaYearly': 0.07309
            # # short - 'sharpe': 11.1196, 'sortino': 31.2659, 'returnYearly': 0.2095, 'volaYearly': 0.0188
            # if RSI_underbought_uptrend[i] == True:
            #     pos[i+1] = 1
            # elif RSI_undersold_downtrend[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': -5.950, 'sortino': -7.50, 'returnYearly': -0.2882, 'volaYearly': 0.0484
            # # short - 'sharpe': 2.4638, 'sortino': 6.4270, 'returnYearly': 0.0774, 'volaYearly': 0.0314
            # if StochOsc_bullish_cross[i] == True:
            #     pos[i+1] = 1
            # elif StochOsc_bearish_cross[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': 1.313, 'sortino': 2.529, 'returnYearly': 0.1504, 'volaYearly': 0.1145
            # # short - 'sharpe': 0.5885, 'sortino': 1.0835, 'returnYearly': 0.0622, 'volaYearly': 0.1057
            # # avg longs per day: 0.957 , avg shorts per day: 1.478
            # if WilliamsR_uptrend[i] == True:
            #     pos[i+1] = 1
            # elif WilliamsR_downtrend[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': -2.196, 'sortino': -3.666, 'returnYearly': -0.1454, 'volaYearly': 0.0662
            # # short - 'sharpe': 0.623, 'sortino': 1.1108, 'returnYearly': 0.0309, 'volaYearly': 0.0496
            # # avg longs per day: 2.696 , avg shorts per day: 3.435
            # if UltiOsc_bullish_cross[i] == True:
            #     pos[i+1] = 1
            # elif UltiOsc_bearish_cross[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': -0.2341, 'sortino': -0.3653, 'returnYearly': -0.01931, 'volaYearly': 0.0824
            # # short - 'sharpe': 3.7680, 'sortino': 707.2583, 'returnYearly': 0.2134, 'volaYearly': 0.0566
            # # avg longs per day: 0.217 , avg shorts per day: 0.087
            # if UltiOsc_bullish_reversal[i] == True:
            #     pos[i+1] = 1
            # elif UltiOsc_bearish_reversal[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': 5.0665, 'sortino': 11.93, 'returnYearly': 0.3819, 'volaYearly': 0.07537
            # # short - 'sharpe': 10.4469, 'sortino': 40.7201, 'returnYearly': 0.6463, 'volaYearly': 0.0619
            # # avg longs per day: 0.913 , avg shorts per day: 1.435
            # if ADI_bullish_trend_confo[i] == True:
            #     pos[i+1] = 1
            # elif ADI_bearish_trend_confo[i] == True:
            #     pos[i+1] = -1

            # # 'sharpe': 0.9950, 'sortino': 1.712, 'returnYearly': 0.05863, 'volaYearly': 0.05892
            # # short - 'sharpe': 7.2415, 'sortino': 17.849, 'returnYearly': 0.3282, 'volaYearly': 0.0453
            # # avg longs per day: 17.826 , avg shorts per day: 19.913
            # if OBV_bullish_trend_confo[i] == True:
            #     pos[i+1] = 1
            # elif OBV_bearish_trend_confo[i] == True:
            #     pos[i+1] = -1

            # # Mean-reverting
            # # 'sharpe': -6.674, 'sortino': -7.555, 'returnYearly': -0.6050, 'volaYearly': 0.09065
            # # short - 'sharpe': 7.2415, 'sortino': 17.849, 'returnYearly': 0.3282, 'volaYearly': 0.0453
            # # avg longs per day: 7.565 , avg shorts per day: 7.391
            # if BB_bullish_reversal[i] == True:
            #     pos[i+1] = 1
            # elif BB_bearish_reversal[i] == True:
            #     pos[i+1] = -1

            # Trend-following
            # 'sharpe': 7.7099, 'sortino': 22.036, 'returnYearly': 0.6505, 'volaYearly': 0.08437
            # short - 'sharpe': 8.5076, 'sortino': 27.5326, 'returnYearly': 0.5654, 'volaYearly': 0.0665
            # avg longs per day: 7.391 , avg shorts per day: 7.565
            # with portfolio allocation...
            # 'sharpe': 'sharpe': 8.536, 'sortino': 26.050, 'returnYearly': 0.6896, 'volaYearly': 0.08078
            # short - 'sharpe': 9.2273, 'sortino': 30.2474, 'returnYearly': 0.5888, 'volaYearly': 0.0635
            # avg longs per day: 7.348 , avg shorts per day: 7.391
            if BB_bullish_reversal[i] == True:
                pos[i+1] = sweights[future_name]
            elif BB_bearish_reversal[i] == True:
                pos[i+1] = lweights[future_name]
            


    elif settings['model'] == 'ANOTHER MODEL':
        pass

    # check if latest economic data suggests downturn then activate short only strats 
    # print(pos)
    if np.nansum(pos) > 0:
        pos = pos / np.nansum(abs(pos))

    settings['longs'] = settings['longs'] + sum(1 for x in pos if x > 0)
    settings['shorts'] = settings['shorts'] + sum(1 for x in pos if x < 0)
    settings['days'] = settings['days'] + 1
    return pos, settings


def mySettings():
    settings = {}
    # markets  = ['CASH','F_AD','F_BO']
    markets  = ['CASH', 'F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']
    budget = 1000000
    slippage = 0.05
    model = 'TA_multifactor' # trend_following, MLR_CLOSE, TA_multifactor
    lookback = 504 # 504
    beginInSample = '20180119' # '20180119'
    endInSample = None # None # taking the latest available
    dynamic_portfolio_allocation = True # activate=False to set even allocation for all futures and even for long/short
    if dynamic_portfolio_allocation:
        mfw = market_factor_weights(markets)

    settings = {'markets': markets, 'beginInSample': beginInSample, 'endInSample': endInSample, 'lookback': lookback,
                'budget': budget, 'slippage': slippage, 'model': model, 'longs':0, 'shorts':0, 'days':0,
                'dynamic_portfolio_allocation':dynamic_portfolio_allocation, 'market_factor_weights':mfw}
    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    print(results['stats'])
    print('avg longs per day:', round(results['settings']['longs']/results['settings']['days'],3), ', avg shorts per day:', round(results['settings']['shorts'] / results['settings']['days'],3))
    # print(results['returns'])
    # print(results['marketEquity'])