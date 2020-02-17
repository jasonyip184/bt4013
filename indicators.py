import numpy as np
import pandas as pd
import ta

def ADI(highs, lows, closes, volumes):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line
    '''
    df = pd.DataFrame({'HIGH':highs, 'LOW': lows, 'CLOSE': closes, 'VOL':volumes})
    ADIs = list(ta.volume.AccDistIndexIndicator(df['HIGH'], df['LOW'], df['CLOSE'], df['VOL']).acc_dist_index())
    return ADIs

def ADX(highs, lows, closes, period):
    '''
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    '''
    df = pd.DataFrame({'HIGH':highs, 'LOW': lows, 'CLOSE': closes})
    adxindicator = ta.trend.ADXIndicator(df['HIGH'], df['LOW'], df['CLOSE'], period)
    ADXs = list(adxindicator.adx())
    mDIs = list(adxindicator.adx_neg())
    pDIs = list(adxindicator.adx_pos())
    return mDIs, pDIs, ADXs

def BB(closes, period):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    '''
    df = pd.DataFrame({'CLOSE': closes})
    BB_high_crosses = list(ta.volatility.BollingerBands(df['CLOSE'], period).bollinger_hband_indicator())
    BB_low_crosses = list(ta.volatility.BollingerBands(df['CLOSE'], period).bollinger_lband_indicator())
    return BB_high_crosses, BB_low_crosses

def CCI(highs, lows, closes, period):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci
    '''
    df = pd.DataFrame({'HIGH':highs, 'LOW': lows, 'CLOSE': closes})
    CCIs = list(ta.trend.CCIIndicator(df['HIGH'], df['LOW'], df['CLOSE'], period).cci())
    return CCIs

def EMA(closes, order):
    '''
    Based on formula: https://www.investopedia.com/terms/e/ema.asp
    Returns list of past EMAs
    '''
    EMAs = [None]*(order-1)
    prev_EMA = np.nanmean(closes[:order])
    EMAs.append(prev_EMA)
    for j in range(order, len(closes)):
        alpha = 2 / (order+1)
        EMA = alpha * closes[j] + (1-alpha) * prev_EMA
        EMAs.append(EMA)
        prev_EMA = EMA
    return EMAs

def OBV(closes, volumes):
    '''
    https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv
    '''
    df = pd.DataFrame({'CLOSE': closes, 'VOL':volumes})
    OBVs = list(ta.volume.OnBalanceVolumeIndicator(df['CLOSE'], df['VOL']).on_balance_volume())
    return OBVs

def RSI(closes):
    '''
    Based on formula: https://www.babypips.com/learn/forex/relative-strength-index
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
    return Ks, Ds

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
    df = pd.DataFrame({'HIGH':highs, 'LOW': lows, 'CLOSE': closes})
    UltiOscs = list(ta.momentum.UltimateOscillator(df['HIGH'], df['LOW'], df['CLOSE'], period1, period2, period3).uo())
    return UltiOscs
    
def WilliamsR(highs, lows, closes):
    '''
    https://tradingsim.com/blog/williams-percent-r/#Strategy_1_-_Cross_of_-50
    '''
    df = pd.DataFrame({'HIGH':highs, 'LOW': lows, 'CLOSE': closes})
    WilliamsRs = list(ta.momentum.WilliamsRIndicator(df['HIGH'], df['LOW'], df['CLOSE']).wr())
    return WilliamsRs


