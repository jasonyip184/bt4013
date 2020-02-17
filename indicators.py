def SMA(closes, order):
    SMAs = [None]*(order-1)
    for j in range(order, len(closes)):
        SMAs.append(np.nanmean(closes[(j-order):j]))
    return SMAs


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


def RSI(closes):
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
