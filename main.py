import numpy as np
import pandas as pd
import pickle
import datetime
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from indicators import ADI, ADX, BB, CCI, EMA, OBV, RSI, SMA, StochOsc, StochRSI, UltiOsc, WilliamsR
from economic_indicators import econ_long_short_allocation, market_factor_weights
from utils import clean
from LGB_model import train_lgb_model, get_lgb_prediction
from pmdarima.arima import auto_arima
from scipy.stats import pearsonr
from statistics import stdev 
from utils import clean
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

ARIMA_MODELS = {}

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,
                    USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV,
                    exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    lookback = settings['lookback']
    pos = np.zeros(nMarkets)
    markets = settings['markets']
    w = settings['market_factor_weights']
    lweights, sweights = econ_long_short_allocation(markets, DATE[0], DATE[-1], w, activate=settings['dynamic_portfolio_allocation'])
    sentiment_data = settings['sentiment_data']
    covid_data = settings['covid_data']

    # to understand how this system works
    print("Using data from {} onwards to predict/take position in {}".format(DATE[0],DATE[-1]))

    OPEN = np.transpose(OPEN)[1:]
    HIGH = np.transpose(HIGH)[1:]
    LOW = np.transpose(LOW)[1:]
    CLOSE = np.transpose(CLOSE)[1:]
    VOL = np.transpose(VOL)[1:]

    if settings['model'] == 'TA':
        '''
        Based on factors from https://www.investing.com/technical/us-spx-500-futures-technical-analysis

        ################ TREND FOLOWING ################
        Simple Moving Average (SMA) crosses, period 5,10,20,50,100,200
        '''
        # indicators
        SMA5s = [SMA(close, 5) for close in CLOSE]
        SMA10s = [SMA(close, 10) for close in CLOSE]
        SMA20s = [SMA(close, 20) for close in CLOSE]
        SMA50s = [SMA(close, 50) for close in CLOSE]
        SMA100s = [SMA(close, 100) for close in CLOSE]
        SMA200s = [SMA(close, 200) for close in CLOSE]

        # signals
        def buy_condition(close, sma):
            return (close[-1] > sma[-1]) and (close[-2] <= sma[-2])
        def sell_condition(close, sma):
            return (close[-1] < sma[-1]) and (close[-2] >= sma[-2])
        SMA5_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA5s)]
        SMA5_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA5s)]
        SMA10_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA10s)]
        SMA10_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA10s)]
        SMA20_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA20s)]
        SMA20_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA20s)]
        SMA50_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA50s)]
        SMA50_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA50s)]
        SMA100_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA100s)]
        SMA100_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA100s)]
        SMA200_cross_buys = [True if buy_condition(close, sma) else False for close, sma in zip(CLOSE, SMA200s)]
        SMA200_cross_sells = [True if sell_condition(close, sma) else False for close, sma in zip(CLOSE, SMA200s)]

        '''
        Exponential Moving Average (EMA) crosses, period 5,10,20,50,100,200
        '''
        # indicators
        EMA5s = [EMA(close, 5) for close in CLOSE]
        EMA10s = [EMA(close, 10) for close in CLOSE]
        EMA20s = [EMA(close, 20) for close in CLOSE]
        EMA50s = [EMA(close, 50) for close in CLOSE]
        EMA100s = [EMA(close, 100) for close in CLOSE]
        EMA200s = [EMA(close, 200) for close in CLOSE]

        # signals
        # def condition(close, ema):
        #     return (close[-1] > ema[-1]) and (close[-2] <= ema[-2])
        def buy_condition(close, ema):
            return (close[-1] > ema[-1]) and (close[-2] <= ema[-2])
        def sell_condition(close, ema):
            return (close[-1] < ema[-1]) and (close[-2] >= ema[-2])
        EMA5_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA5s)]
        EMA5_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA5s)]
        EMA10_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA10s)]
        EMA10_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA10s)]
        EMA20_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA20s)]
        EMA20_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA20s)]
        EMA50_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA50s)]
        EMA50_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA50s)]
        EMA100_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA100s)]
        EMA100_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA100s)]
        EMA200_cross_buys = [True if buy_condition(close, ema) else False for close, ema in zip(CLOSE, EMA200s)]
        EMA200_cross_sells = [True if sell_condition(close, ema) else False for close, ema in zip(CLOSE, EMA200s)]

        '''
        Average Directional Movement Index (ADX), period 14
        '''
        # indicators
        ADXs = [ADX(high,low,close,14) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # signals
        # adx[0] is mDI, adx[1] is pDI, adx[2] is actual ADX
        def bullish_condition(adx):
            # Bullish strong trend cross
            return (adx[2][-1] > 20) and (adx[1][-1] > adx[0][-1]) and (adx[1][-2] <= adx[0][-2])
        def bearish_condition(adx):
            # Bearish strong trend cross
            return (adx[2][-1] > 20) and (adx[1][-1] < adx[0][-1]) and (adx[1][-2] >= adx[0][-2])
        ADX_bullish_crosses = [True if bullish_condition(adx) else False for adx in ADXs]
        ADX_bearish_crosses = [True if bearish_condition(adx) else False for adx in ADXs]

        '''
        Moving Average Convergence Divergence (MACD) fast=12, slow=26
        '''
        # indicator
        EMA12s = [EMA(close, 12) for close in CLOSE]
        EMA26s = [EMA(close, 26) for close in CLOSE]
        MACDs = [[(a-b) if b is not None else None for a, b in zip(EMA12, EMA26)] for EMA12, EMA26 in zip(EMA12s, EMA26s)]
        # signals
        def bullish_condition(MACD):
            # Bullish zero cross which sustains for 2 days (reduce false signals)
            return (MACD[-3] <= 0) and (MACD[-2] > 0) and (MACD[-1] > 0)
        def bearish_condition(MACD):
            # Bearish zero cross
            return (MACD[-3] >= 0) and (MACD[-2] < 0) and (MACD[-1] < 0)
        MACD_bullish_zero_cross = [True if bullish_condition(MACD) else False for MACD in MACDs]
        MACD_bearish_zero_cross = [True if bearish_condition(MACD) else False for MACD in MACDs]

        '''
        Commodity Channel Index (CCI), period 14
        '''
        # indicator
        CCIs = [CCI(high,low,close,14) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # signals
        def bullish_condition(CCI):
            return (CCI[-1] > 100) and (CCI[-2] <= 100)
        def bearish_condition(CCI):
            return (CCI[-1] < -100) and (CCI[-2] >= -100)   
        CCI_emerging_bulls = [True if bullish_condition(CCI) else False for CCI in CCIs]
        CCI_emerging_bears = [True if bearish_condition(CCI) else False for CCI in CCIs]

        '''
        ################ MOMENTUM ################
        Relative Strength Index (RSI), period 14
        '''
        # indicator
        RSIs = [RSI(close) for close in CLOSE]
        # signals
        def bullish_reversal(rsi, sma200, close):
            # Uptrend and cross 30 to become oversold (Bullish)
            return (close[-1] > sma200[-1]) and (rsi[-2] >= 30) and (rsi[-1] < 30)
        def bearish_reversal(rsi, sma200, close):
            # Downtrend and cross 70 to become overbought (Bearish)
            return (close[-1] < sma200[-1]) and (rsi[-2] <= 70) and (rsi[-1] > 70)
        def underbought_uptrend(rsi, sma200, close):
            # Uptrend and underbought
            return (close[-1] > sma200[-1]) and (rsi[-1] < 50)
        def undersold_downtrend(rsi, sma200, close):
            # Downtrend and undersold
            return (close[-1] < sma200[-1]) and (rsi[-1] > 50)
        RSI_bullish_reversal = [True if bullish_reversal(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        RSI_bearish_reversal = [True if bearish_reversal(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        RSI_underbought_uptrend = [True if underbought_uptrend(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        RSI_undersold_downtrend = [True if undersold_downtrend(rsi,sma200,close) else False for rsi,sma200,close in zip(RSIs,SMA200s,CLOSE)]
        
        '''
        Stochastic Oscillator, fast 14, slow 3
        '''
        # indicators
        StochOscs = [StochOsc(close,high,low, 14, 3) for close,high,low in zip(CLOSE,HIGH,LOW)]
        # signals
        # stochosc[0] is Ks, stochosc[1] is Ds
        def bullish_cross(stochosc):
            # K (fast) cross D (slow) from below
            return (stochosc[0][-2] <= stochosc[1][-2]) and (stochosc[0][-1] > stochosc[1][-1])
        def bearish_cross(stochosc):
            # K (fast) cross D (slow) from above
            return (stochosc[0][-2] >= stochosc[1][-2]) and (stochosc[0][-1] < stochosc[1][-1])
        StochOsc_bullish_cross = [True if bullish_cross(stochosc) else False for stochosc in StochOscs]
        StochOsc_bearish_cross = [True if bearish_cross(stochosc) else False for stochosc in StochOscs]

        '''
        Williams %R, 14 period
        '''
        # indicator
        WilliamsRs = [WilliamsR(high,low,close) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # signals
        def bullish(wr, close, sma100):
            # Overbought price action
            return (wr[-1] > -20) and (close[-1] > sma100[-1]) and (close[-2] <= sma100[-2])
        def bearish(wr, close, sma100):
            # Oversold price action
            return (wr[-1] < -80) and (close[-1] < sma100[-1]) and (close[-2] >= sma100[-2])
        WilliamsR_uptrend = [True if bullish(wr,close,sma100) else False for wr,close,sma100 in zip(WilliamsRs,CLOSE,SMA100s)]
        WilliamsR_downtrend = [True if bearish(wr,close,sma100) else False for wr,close,sma100 in zip(WilliamsRs,CLOSE,SMA100s)]
        
        ''' 
        Ultimate Oscillator, periods 20,40,80
        '''
        # indicator
        UltiOscs = [UltiOsc(high,low,close,20,40,80) for high,low,close in zip(HIGH,LOW,CLOSE)]
        # signals
        def bullish_cross(ultiosc):
            # Bullish center cross
            return (ultiosc[-1] > 50) and (ultiosc[-2] <= 50)
        def bearish_cross(ultiosc):
            # Bearish center cross
            return (ultiosc[-1] < 50) and (ultiosc[-2] >= 50)
        def bullish_reversal(ultiosc):
            # Bullish reversal from oversold
            return (ultiosc[-1] < 30) and (ultiosc[-2] >= 30)
        def bearish_reversal(ultiosc):
            # Bearish reversal from overbought
            return (ultiosc[-1] > 70) and (ultiosc[-2] >= 70)
        UltiOsc_bullish_cross = [True if bullish_cross(ultiosc) else False for ultiosc in UltiOscs]
        UltiOsc_bearish_cross = [True if bearish_cross(ultiosc) else False for ultiosc in UltiOscs]
        UltiOsc_bullish_reversal = [True if bullish_reversal(ultiosc) else False for ultiosc in UltiOscs]
        UltiOsc_bearish_reversal = [True if bearish_reversal(ultiosc) else False for ultiosc in UltiOscs]

        '''
        ################ VOLUME ################
        Accumulation / Distribution Index (ADI)
        '''
        # indicator
        ADIs = [ADI(high,low,close,vol) for high,low,close,vol in zip(HIGH,LOW,CLOSE,VOL)]
        def bullish_trend(close, adi, sma200):
            # bullish trend confirmation
            return (close[-1] > sma200[-1]) and (close[-2] <= sma200[-2]) and (adi[-1] > adi[-2])
        def bearish_trend(close, adi, sma200):
            # bearish trend confirmation
            return (close[-1] < sma200[-1]) and (close[-2] >= sma200[-2])  and (adi[-1] < adi[-2])
        ADI_bullish_trend_confo = [True if bullish_trend(close,adi,sma200) else False for close,adi,sma200 in zip(CLOSE,ADIs,SMA200s)]
        ADI_bearish_trend_confo = [True if bearish_trend(close,adi,sma200) else False for close,adi,sma200 in zip(CLOSE,ADIs,SMA200s)]
        
        '''
        On-Balance Volume (OBV)
        '''
        # indicator
        OBVs = [OBV(close,vol) for close,vol in zip(CLOSE,VOL)]
        # signals
        def bullish_trend(obv):
            return (obv[-1] > obv[-2]) and (obv[-2] > obv[-3])
        def bearish_trend(obv):
            return (obv[-1] < obv[-2]) and (obv[-2] < obv[-3])
        OBV_bullish_trend_confo = [True if bullish_trend(obv) else False for obv in OBVs]
        OBV_bearish_trend_confo = [True if bearish_trend(obv) else False for obv in OBVs]

        '''
        ################ VOLATILITY ################
        Bollinger Bands (BB), 20 period
        '''
        # indicator + signal
        BBs = [BB(close, 20) for close in CLOSE]
        BB_bullish_reversal = [True if bb[1][-1] == 1 else False for bb in BBs]
        BB_bearish_reversal = [True if bb[0][-1] == 1 else False for bb in BBs]

        '''
        Execution
        '''
        for i in range(0, nMarkets-1):
            future_name = markets[i+1]
            # Trend following
            # 'sharpe': -0.40248, 'sortino': -0.7310, short - 'sharpe': 3.47, 'sortino': 10.5
            if (SMA5_cross_buys[i] == True) or (SMA10_cross_buys[i] == True) or (SMA20_cross_buys[i] == True) or (SMA50_cross_buys[i] == True) or (SMA100_cross_buys[i] == True) or (SMA200_cross_buys == True):
                pos[i+1] = 1
            if (SMA5_cross_sells[i] == True) or (SMA10_cross_sells[i] == True) or (SMA20_cross_sells[i] == True) or (SMA50_cross_sells[i] == True) or (SMA100_cross_sells[i] == True) or (SMA200_cross_sells == True):
                pos[i+1] = -1

            # Mean reverting
            # 'sharpe': -4.143, 'sortino': -4.988, short - 'sharpe': -0.6878, 'sortino': -1.526
            if (SMA5_cross_buys[i] == True) or (SMA10_cross_buys[i] == True) or (SMA20_cross_buys[i] == True) or (SMA50_cross_buys[i] == True) or (SMA100_cross_buys[i] == True) or (SMA200_cross_buys == True):
                pos[i+1] = -1
            if (SMA5_cross_sells[i] == True) or (SMA10_cross_sells[i] == True) or (SMA20_cross_sells[i] == True) or (SMA50_cross_sells[i] == True) or (SMA100_cross_sells[i] == True) or (SMA200_cross_sells == True):
                pos[i+1] = 1

            # Trend following
            # 'sharpe': -0.5942, 'sortino': -0.9762, short - 'sharpe': 2.728, 'sortino': 7.063
            if (EMA5_cross_buys[i] == True) or (EMA10_cross_buys[i] == True) or (EMA20_cross_buys[i] == True) or (EMA50_cross_buys[i] == True) or (EMA100_cross_buys[i] == True) or (EMA200_cross_buys == True):
                pos[i+1] = 1
            if (EMA5_cross_sells[i] == True) or (EMA10_cross_sells[i] == True) or (EMA20_cross_sells[i] == True) or (EMA50_cross_sells[i] == True) or (EMA100_cross_sells[i] == True) or (EMA200_cross_sells == True):
                pos[i+1] = -1

            # Mean-reverting
            # 'sharpe': -3.806, 'sortino': -4.845, short - 'sharpe': -0.5814, 'sortino': -1.312
            if (EMA5_cross_buys[i] == True) or (EMA10_cross_buys[i] == True) or (EMA20_cross_buys[i] == True) or (EMA50_cross_buys[i] == True) or (EMA100_cross_buys[i] == True) or (EMA200_cross_buys == True):
                pos[i+1] = -1
            if (EMA5_cross_sells[i] == True) or (EMA10_cross_sells[i] == True) or (EMA20_cross_sells[i] == True) or (EMA50_cross_sells[i] == True) or (EMA100_cross_sells[i] == True) or (EMA200_cross_sells == True):
                pos[i+1] = 1

            # 'sharpe': -1.973, 'sortino': -3.275, short - 'sharpe': -3.544 'sortino': -3.988
            if ADX_bullish_crosses[i] == True:
                pos[i+1] = 1
            elif ADX_bearish_crosses[i] == True:
                pos[i+1] = -1

            # 'sharpe': -1.705, 'sortino': -2.380, short - 'sharpe': 0.9374, 'sortino': 1.5024
            if MACD_bullish_zero_cross[i] == True:
                pos[i+1] = 1
            elif MACD_bearish_zero_cross[i] == True:
                pos[i+1] = -1

            # 'sharpe': -0.0093, 'sortino': -0.0198, short - 'sharpe': -0.0094, 'sortino': -0.0198
            # avg longs per day: 5.021 , avg shorts per day: 5.872
            if CCI_emerging_bulls[i] == True:
                pos[i+1] = 1
            elif CCI_emerging_bears[i] == True:
                pos[i+1] = -1

            # 'sharpe': -4.308, 'sortino': -4.356, short - 'sharpe': 2.323, 'sortino': 9.84
            # avg longs per day: 0.319, avg shorts per day: 0.149
            if RSI_bullish_reversal[i] == True:
                pos[i+1] = 1
            elif RSI_bearish_reversal[i] == True:
                pos[i+1] = -1

            # 'sharpe': -3.174, 'sortino': -4.730, short - 'sharpe': 0.0643, 'sortino': 0.108
            # avg longs per day: 17.362, avg shorts per day: 17.404
            if StochOsc_bullish_cross[i] == True:
                pos[i+1] = 1
            elif StochOsc_bearish_cross[i] == True:
                pos[i+1] = -1

            # 'sharpe': 0.2383, 'sortino': 0.4939, short - 'sharpe': 0.9064, 'sortino': 1.599
            # avg longs per day: 0.809 , avg shorts per day: 1.553
            if WilliamsR_uptrend[i] == True:
                pos[i+1] = 1
            elif WilliamsR_downtrend[i] == True:
                pos[i+1] = -1

            # 'sharpe': -1.57, 'sortino': -2.159, short - 'sharpe': 0.8575, 'sortino': 1.248
            # avg longs per day: 2.362 , avg shorts per day: 3.0
            if UltiOsc_bullish_cross[i] == True:
                pos[i+1] = 1
            elif UltiOsc_bearish_cross[i] == True:
                pos[i+1] = -1

            # 'sharpe': 0.7773, 'sortino': 1.288, short - 'sharpe': 2.3757, 'sortino': 70.82
            # avg longs per day: 0.128 , avg shorts per day: 0.362
            if UltiOsc_bullish_reversal[i] == True:
                pos[i+1] = 1
            elif UltiOsc_bearish_reversal[i] == True:
                pos[i+1] = -1

            # 'sharpe': -0.2829, 'sortino': -0.5704, short - 'sharpe': 2.4719, 'sortino': 5.47
            # avg longs per day: 1.043 , avg shorts per day: 1.702
            if ADI_bullish_trend_confo[i] == True:
                pos[i+1] = lweights[future_name]
            elif ADI_bearish_trend_confo[i] == True:
                pos[i+1] = sweights[future_name]

            # 'sharpe': 2.3531, 'sortino': 5.9688, short - 'sharpe': 5.9294, 'sortino': 18.667
            # avg longs per day: 20.638 , avg shorts per day: 21.255
            if OBV_bullish_trend_confo[i] == True:
                pos[i+1] = 1
            elif OBV_bearish_trend_confo[i] == True:
                pos[i+1] = -1

            # Mean-reverting
            # 'sharpe': -5.577, 'sortino': -6.322, short - 'sharpe': -4.714, 'sortino': -5.181
            # avg longs per day: 8.766 , avg shorts per day: 8.511
            if BB_bullish_reversal[i] == True:
                pos[i+1] = 1
            elif BB_bearish_reversal[i] == True:
                pos[i+1] = -1

            # Trend-following
            # 'sharpe': 5.6105, 'sortino': 15.905, short - 'sharpe': 4.799, 'sortino': 12.11
            # avg longs per day: 8.511 , avg shorts per day: 8.766
            # with portfolio allocation...
            # 'sharpe': 5.6286, 'sortino': 16.472, short - 'sharpe': 5.1068, 'sortino': 13.10
            # avg longs per day: 8.511 , avg shorts per day: 8.766
            if BB_bullish_reversal[i] == True:
                pos[i+1] = sweights[future_name]
            elif BB_bearish_reversal[i] == True:
                pos[i+1] = lweights[future_name]

    elif settings['model'] == 'LIGHTGBM':
        for i in range(0, nMarkets - 1):
            future_name = markets[i + 1]
            if future_name in ["CASH", "F_ED", "F_UZ", "F_SS", "F_ZQ", "F_EB", "F_VW", "F_F"]:
                feature_ADI = ADI(HIGH[i], LOW[i], CLOSE[i], VOL[i])
                feature_WilliamsR = WilliamsR(HIGH[i], LOW[i], CLOSE[i])
                feature_BB_high_crosses, feature_BB_low_crosses = BB(CLOSE[i], 10)
                feature_CCI = CCI(LOW[i], CLOSE[i], VOL[i], 10)
                features = np.array(
                    [
                        [
                            OPEN[i][-1],
                            HIGH[i][-1],
                            LOW[i][-1],
                            CLOSE[i][-1],
                            VOL[i][-1],
                            feature_ADI[-1],
                            feature_WilliamsR[-1],
                            feature_BB_high_crosses[-1],
                            feature_BB_low_crosses[-1],
                            feature_CCI[-1],
                            CLOSE[i][-2],
                            CLOSE[i][-3],
                        ]
                    ]
                )
                model_dir = f"./data/lgb_models/{markets[i+1]}_model"
                prediction = get_lgb_prediction(model_dir, features)[0]
                pos[i + 1] = prediction

            
    elif settings['model'] == 'sentiment':
        '''
        How sentiment of tweets from Bloomberg/Trump affect VIX and Gold
        '''
        for i in range(0, nMarkets-1):
            future_name = markets[i+1]
            if future_name == 'F_VX':
                today = datetime.strptime(str(DATE[-1]),'%Y%m%d').date()
                if (today - sentiment_data['DATE'].tolist()[0]).days > 30: # at least 30 days for training
                    train = sentiment_data[sentiment_data['DATE'] < today]
                    test = sentiment_data[sentiment_data['DATE'] == today]
                    trainY = train['CLOSE']
                    del train['DATE'], train['CLOSE']
                    trainX = train
                    del test['DATE'], test['CLOSE']
                    model = RandomForestRegressor()
                    model.fit(trainX, trainY)
                    pred_CLOSE = model.predict(test)[0]
                    if pred_CLOSE > CLOSE[i][-2]:
                        pos[i+1] = 1
                    else:
                        pos[i+1] = -1

    elif settings['model'] == 'covid':
        '''
        How no. of covid cases in each country affects their overall markets
        'sharpe': 1.3048, 'sortino': 2.3477,
        avg longs per day: 1.35 , avg shorts per day: 6.6
        '''
        for i in range(0, nMarkets-1):
            country = None
            future_name = markets[i+1]
            if future_name in ['F_ES','F_MD','F_NQ','F_RU','F_XX','F_YM']:
                country = 'US'
            elif future_name in ['F_AX','F_DM','F_DZ']:
                country = 'Germany'
            elif future_name == 'F_CA':
                country = 'France'
            elif future_name == 'F_LX':
                country = 'United Kingdom'
            elif future_name == 'F_FP':
                country = 'Finland'
            elif future_name == 'F_NY':
                country = 'Japan'
            elif future_name == 'F_PQ':
                country = 'Portugal'
            elif future_name in ['F_SH','F_SX']:
                country = 'Switzerland'

            if country:
                df = covid_data[covid_data['Country/Region'] == country].T.sum(axis=1).reset_index()
                df = df.iloc[1:]
                df['index'] = df['index'].apply(lambda x: x+"20")
                df['index'] = df['index'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y').date())
                future = pd.DataFrame({'DATE':DATE,'CLOSE':CLOSE[i]})
                future['CLOSE'] = future['CLOSE'].shift(-1)
                future['DATE'] = future['DATE'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d').date())
                df = pd.merge(df,future,left_on='index',right_on='DATE')
                df = df[df[0] != 0][[0,'CLOSE']].rename(columns={0: "count"})
                if len(df) > 10:
                    print(df)
                    reg = LinearRegression().fit(np.array(df['count'].values[:-1]).reshape(-1,1), df['CLOSE'].values[:-1])
                    pred_CLOSE = reg.predict(np.array(df['count'].values[-1]).reshape(1,-1))[0]
                    if pred_CLOSE > CLOSE[i][-2]:
                        pos[i+1] = 1
                    else:
                        pos[i+1] = -1


    elif settings['model'] == 'ARIMA':
        for i in range(0, nMarkets-1):
            try:
                if markets[i+1] not in ARIMA_MODELS:
                    model = auto_arima(np.log(CLOSE[i][:-1]), trace=False, error_action='ignore', suppress_warnings=True)
                    ARIMA_MODELS[markets[i+1]] = model.fit(np.log(CLOSE[i][:-1]))
                model = ARIMA_MODELS[markets[i+1]].fit(np.log(CLOSE[i][:-1]))
                pred = model.predict(n_periods=1)[0]
                # print(markets[i+1],pred, np.log(CLOSE[i][-1]))    
                pos[i+1] = 1 if pred > np.log(CLOSE[i][-1]) else -1
            except:
                pos[i+1] = 0
        print(f"Today's position in the {len(markets)} futures: {pos}")                  


    elif settings['model'] == 'GARCH':
        # Log return of the closing data
        #Prameters
        bound1 = 1
        bound2 = 1
        cor_dir = f'./data/garch_models/correlation.txt'
        with open(cor_dir) as f:
            cor_dict = json.load(f)
        log_return = np.diff(np.log(CLOSE))
        #print(log_return)
        #log_return = log_return[~np.isnan(log_return)]
        #print(log_return[1])
        for i in range(0, nMarkets-1):
            train_Xs = log_return[i][:-1]
            #test_Xs = log_return[i][-1]
            sd = np.var(train_Xs)
            # define model
            model_dir = f'./data/garch_models/{markets[i+1]}_garch_model.txt'
            with open(model_dir) as f:
                params_dict = json.load(f)
            p = params_dict['order'][0]
            q = params_dict['order'][1]
            model = arch_model(train_Xs, p=p, q=q)
            model_fixed = model.fix(params_dict['params'])
            # forecast the test set
            forecasts = model_fixed.forecast()
            #expected = forecasts.mean.iloc[-1:]['h.1']
            var = forecasts.variance.iloc[-1:]['h.1']
            #print(type(variance))

            if(cor_dict[markets[i+1]]>0.03):
                if (float(np.sqrt(var)) > bound1*np.std(train_Xs)):
                    pos[i] = 1
                elif (float(np.sqrt(var)) < bound2*np.std(train_Xs)):
                    pos[i] = -1
                else:
                    pos[i] = 0
            elif(cor_dict[markets[i+1]]<-0.03):
                if (float(np.sqrt(var)) > bound1*np.std(train_Xs)):
                    pos[i] = -1
                elif (float(np.sqrt(var)) < bound2*np.std(train_Xs)):
                    pos[i] = 1
                else:
                    pos[i] = 0  
            else:
                pos[i] = 0             
        # With the estimated return and variance, we can apply portfolio optimization
        #print((np.array(result) * np.array(truth)).sum() / len(result))

    
    elif settings['model'] == 'fourier':
        #Parameters that filter the signal with specific range of signals
        #Note that the lower bound should be larger than 0 and upper bound smaller than 1
        filter_type = 'customed'
        my_frequency_bound = [0.05, 0.2] 
        filter_type = 'high' #Specify high/mid/low/customed for weekly signals, weekly to monthly signals, above monthly signals or customed frequency range
        if filter_type == 'high':        
            frequency_bound = [0.2, 1]
        elif filter_type == 'mid': 
            frequency_bound = [0.05, 0.2]
        elif filter_type == 'low':  
            frequency_bound = [0, 0.05]
        elif filter_type == 'customed':
            frequency_bound = my_frequency_bound

        #Transform the close data by fourier filtering, only signals within the specific range remains
        transformed_close = []
        for i in range(0, nMarkets-1):
            signal = CLOSE[i][:-1]
            fft = np.fft.rfft(signal)
            T = 1  # sampling interval
            N = len(signal)
            f = np.linspace(0, 1 / T, N)
            fft_filtered = []
            for j in range(int(N/2)):
                if f[j] > frequency_bound[0] and f[j] < frequency_bound[1]:
                    fft_filtered.append(fft[j])
                else:
                    fft_filtered.append(0)
            signal_filtered = np.fft.irfft(fft_filtered)
            transformed_close.append(list(signal_filtered))

        smaLongerPeriod = np.nansum(np.array(transformed_close)[:, -periodLonger:], axis=1) / periodLonger	
        smaShorterPeriod = np.nansum(np.array(transformed_close)[:, -periodShorter:], axis=1) / periodShorter
        longEquity = smaShorterPeriod > smaLongerPeriod
        shortEquity = ~longEquity
        pos_1 = np.zeros(nMarkets-1)
        pos_1[longEquity] = 1
        pos_1[shortEquity] = -1
        pos = np.array([0]+list(pos_1))


    elif settings['model'] == 'pearson':
        '''
        Pairwise correlation, taking position based on the greatest variation from
        average of the past 50 periods of 50 days
        '''
        #'sharpe': 0.57939, 'sortino': 0.9027189, 'returnYearly': 0.076509, 'volaYearly': 0.13205
        # with allocation
        #avg longs per day: 6.343 , avg shorts per day: 6.746
        d = {} ##Name of future : Close of all 88 futures
        names = []  ##names of all 88 future
        for i in range(0, nMarkets-1):
            n = markets[i+1]
            names.append(n)
            d[n] = (CLOSE[i])
        d_corr = settings['historic_corr']
        ## key = tuple of name of 2 futures, value = position to take for ((future1,future2),difference)
        d_position = {}
        for i in list(d_corr.keys()):
            f = i[0]
            s = i[1]
            tup = d_corr[i]
            l1 = d[f][-49:-1] ##take last 50 close
            l2 = d[s][-49:-1] ##take last 50 close
            corr , _ = pearsonr(l1,l2)
            change_f = d[f][-2] - d[f][-49]
            change_s = d[s][-2] - d[s][-49]
            diff = tup - corr
            if diff > 0.3:
                if change_f > change_s :
                    d_position[i] = ((-1,1),diff) ##assuming -1 means short while 1 means long
                else:
                    d_position[i] = ((1,-1),diff)

        for i in range (len(names)): ##find position based on greatest variation
            diff = 0
            pair = tuple()
            name = names[i]
            counter = 0
            for k in list(d_position.keys()):
                if name in k:
                    counter+=1
                    pair = k
            if counter == 1:
                if name == k[0]:
                    if d_position[k][0][0] > 0:
                        pos[i+1] = lweights[name]
                    else:
                        pos[i+1] = sweights[name]
                else:
                    if d_position[k][0][1] > 0:
                        pos[i+1] = lweights[name]
                    else:
                        pos[i+1] = sweights[name]
        
    elif settings['model'] == 'FASTDTW':
        #'sharpe': 4.8632971, 'sortino': 17.09129, 'returnYearly': 1.216714, 'volaYearly': 0.25018
        # no allocation
        # avg longs per day: 0.328 , avg shorts per day: 0.281
        d = {} ##Name of future : Close of all 88 futures
        names = []  ##names of all 88 future
        for i in range(0, nMarkets-1):
            n = markets[i+1]
            names.append(n)
            d[n] = (CLOSE[i])

        d_dist = settings['historic_dist'] ## key = tuple of name of 2 futures, value = average distance

        d_position = {}
        for i in list(d_dist.keys()):
            f = i[0]
            s = i[1]
            tup = d_dist[i]
            l1 = d[f][-49:-1] ##take last 50 close
            l2 = d[s][-49:-1] ##take last 50 close
            distance, _ = fastdtw(l1,l2) 
            distance = distance / 50
            change_f = d[f][-2] - d[f][-49]
            change_s = d[s][-2] - d[s][-49]
            diff = distance - tup
            threshold = 16*tup
            if distance > threshold:
                if change_f > change_s :
                    d_position[i] = ((-1,1),diff) ##assuming -1 means short while 1 means long
                else:
                    d_position[i] = ((1,-1),diff)


        for i in range (len(names)): ##find position based on greatest variation
            diff = 0
            name = names[i]
            for k in list(d_position.keys()):
                if name in k:
                    if d_position[k][1] > diff :
                        diff = d_position[k][1]
                        if name == k[0]:
                            if d_position[k][0][0] > 0:
                                pos[i+1] = lweights[name]
                            else:
                                pos[i+1] = sweights[name]
                        else:
                            if d_position[k][0][1] > 0:
                                pos[i+1] = lweights[name]
                            else:
                                pos[i+1] = sweights[name]

    # check if latest economic data suggests downturn then activate short only strats 
    print("Positions:", pos)
    if np.nansum(pos) > 0:
        pos = pos / np.nansum(abs(pos))

    settings['longs'] = settings['longs'] + sum(1 for x in pos if x > 0)
    settings['shorts'] = settings['shorts'] + sum(1 for x in pos if x < 0)
    settings['days'] = settings['days'] + 1
    return pos, settings


def mySettings():
    settings = {}
    markets  = ['CASH','F_AD','F_BO','F_BP','F_C','F_CC','F_CD','F_CL','F_CT','F_DX','F_EC','F_ED','F_ES','F_FC','F_FV','F_GC','F_HG','F_HO','F_JY','F_KC','F_LB','F_LC','F_LN','F_MD','F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA','F_PL','F_RB','F_RU','F_S','F_SB','F_SF','F_SI','F_SM','F_TU','F_TY','F_US','F_W','F_XX','F_YM','F_AX','F_CA','F_DT','F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU','F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM','F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP','F_RY','F_SH','F_SX','F_TR','F_EB','F_VF','F_VT','F_VW','F_GD','F_F']
    budget = 1000000
    slippage = 0.05

    model = 'LIGHTGBM' # TA, LIGHTGBM, pearson, FASTDTW, ARIMA, GARCH, fourier, sentiment, covid

    lookback = 504 # 504
    beginInSample = '20180119' # '20180119'
    endInSample = '20200331' # None # taking the latest available
    dynamic_portfolio_allocation = True # =False to set even allocation for all futures and even for long/short, set to False when downloading data
    # clean() # clean data's headers. only need to uncomment this after you have downloaded data again.
    
    if dynamic_portfolio_allocation:
        mfw = market_factor_weights(markets)
    else:
        mfw = None

    covid_data = sentiment_data = historic_corr = historic_distance = None 
    if model == 'sentiment':
        with open('data/trump_train_data.pickle', 'rb') as handle:
            sentiment_data = pickle.load(handle)
    elif model == 'covid':
        covid_data = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
        del covid_data['Province/State'], covid_data['Lat'], covid_data['Long']
    elif model == 'Pairs_trade':
        with open('data/historic_corr.pickle','rb') as f:
            historic_corr = pickle.load(f)
    elif model == 'FASTDTW':
        with open('data/historic_distance.pickle','rb') as g:
            historic_distance = pickle.load(g)

    settings = {'markets': markets, 'beginInSample': beginInSample, 'endInSample': endInSample, 'lookback': lookback,
                'budget': budget, 'slippage': slippage, 'model': model, 'longs':0, 'shorts':0, 'days':0,
                'dynamic_portfolio_allocation':dynamic_portfolio_allocation, 'market_factor_weights':mfw,
                'sentiment_data':sentiment_data,'covid_data':covid_data,'historic_corr':historic_corr,
                'historic_dist':historic_distance}

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__, )
    print(results['stats'])
    print('avg longs per day:', round(results['settings']['longs']/results['settings']['days'],3), ', avg shorts per day:', round(results['settings']['shorts'] / results['settings']['days'],3))