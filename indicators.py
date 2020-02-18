import numpy as np
import pandas as pd
import ta

def ADI(highs, lows, closes, volumes):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line
    '''
    return list(ta.volume.AccDistIndexIndicator(pd.Series(highs), pd.Series(lows), pd.Series(closes), pd.Series(volumes)).acc_dist_index())

def ADX(highs, lows, closes, period):
    '''
    Returns list of 3 lists
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    '''
    adxindicator = ta.trend.ADXIndicator(pd.Series(highs), pd.Series(lows), pd.Series(closes), period)
    ADXs = list(adxindicator.adx())
    mDIs = list(adxindicator.adx_neg())
    pDIs = list(adxindicator.adx_pos())
    return [mDIs, pDIs, ADXs]

def BB(closes, period):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    '''
    BB_high_crosses = list(ta.volatility.BollingerBands(pd.Series(closes), period).bollinger_hband_indicator())
    BB_low_crosses = list(ta.volatility.BollingerBands(pd.Series(closes), period).bollinger_lband_indicator())
    return [BB_high_crosses, BB_low_crosses]

def CCI(highs, lows, closes, period):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci
    '''
    return list(ta.trend.CCIIndicator(pd.Series(highs), pd.Series(lows), pd.Series(closes), period).cci())

def EMA(closes, order):
    '''
    Based on formula: https://www.investopedia.com/terms/e/ema.asp
    Returns list of past EMAs
    '''
    return list(ta.trend.EMAIndicator(pd.Series(closes), order).ema_indicator())

def OBV(closes, volumes):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    '''
    return list(ta.volume.OnBalanceVolumeIndicator(pd.Series(closes), pd.Series(volumes)).on_balance_volume())

def RSI(closes):
    '''
    Based on formula: https://www.babypips.com/learn/forex/relative-strength-index
    Returns list of past RSIs
    '''
    return list(ta.momentum.RSIIndicator(pd.Series(closes)).rsi())

def SMA(closes, order):
    SMAs = [None]*(order-1)
    for j in range(order, (len(closes)+1)):
        SMAs.append(np.nanmean(closes[(j-order):j]))
    return SMAs

def StochOsc(closes, highs, lows, K_period, D_period):
    '''
    Based on formula: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    Returns list of K (fast) oscillator and list of D (slow) oscillator
    '''
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
    return [Ks, Ds]

def StochRSI(RSIs, period):
    '''
    Based on formula: https://www.investopedia.com/terms/s/stochrsi.asp
    Returns list of StochRSIs
    '''
    StochRSIs = [None]*(period-1)
    for j in range(period, (len(RSIs)+1)):
        min_RSI = np.nanmin(RSIs[(j-period):j])
        max_RSI = np.nanmax(RSIs[(j-period):j])
        stoch_RSI = (RSIs[j-1] - min_RSI) / (max_RSI - min_RSI)
        StochRSIs.append(stoch_RSI)
    return StochRSIs

def UltiOsc(highs, lows, closes, period1, period2, period3):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:ultimate_oscillator
    '''
    return list(ta.momentum.UltimateOscillator(pd.Series(highs), pd.Series(lows), pd.Series(closes), period1, period2, period3).uo())
    
def WilliamsR(highs, lows, closes):
    '''
    https://tradingsim.com/blog/williams-percent-r/#Strategy_1_-_Cross_of_-50
    '''
    return list(ta.momentum.WilliamsRIndicator(pd.Series(highs), pd.Series(lows), pd.Series(closes)).wr())


