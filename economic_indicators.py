import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.stats.mstats import gmean


agri = ['F_BO','F_C','F_CC','F_CT','F_FC','F_KC','F_LB','F_LC','F_LN','F_NR','F_O','F_OJ','F_S','F_SB','F_SM','F_W','F_DL']
energy = ['F_CL','F_HO','F_NG','F_RB','F_BG','F_BC','F_LU','F_FL','F_HP','F_LQ']
metals = ['F_GC','F_HG','F_PA','F_PL','F_SI']
us_bonds = ['F_FV','F_TU','F_TY','F_US']
fx = ['F_AD','F_BP','F_CD','F_DX','F_EC','F_JY','F_MP','F_SF','F_LR','F_ND','F_RR','F_RF','F_RP','F_RY','F_TR']
stock_indices = ['F_ES','F_MD','F_NQ','F_RU','F_XX','F_YM','F_AX','F_CA','F_LX','F_AE','F_DM','F_DZ','F_FB','F_FM','F_FP','F_FY','F_NY','F_PQ','F_SH','F_SX']
rates = ['F_ED','F_SS','F_ZQ','F_EB','F_F']
commods = agri + energy + metals

                 
def econ_long_short_allocation(markets, start, end, w, activate):
    '''
    Dynamic long/short allocation based on economic indicators in current time frame
    Returns 2 dictionaries (1 long 1 short) of k:v as future name:weighting in portfolio
    All factors are created using zscores from the start:end period
    each zscore is then normalized to lie between 0-1, any values exceeding 2 or -2 sd will be winsorized.
    w = market_factor_weights
    '''
    lw = {}
    sw = {}

    if activate == False:
        for market in markets:
            lw[market] = 1
            sw[market] = -1
        return lw, sw

    '''
    Lower i/r -> higher commods price
    http://www.choicesmagazine.org/choices-magazine/theme-articles/will-rising-interest-rates-lead-to-intensifying-risks-for-agriculture/monetary-tightening-could-mean-lower-agricultural-commodity-prices-if-historical-relationships-hold
    Using rates futures as proxy for real i/r
    Future price is (100-rate)*currencyvalue
    The higher future price, the lower i/r, the higher commods price, the more long weightage and less short weightage
    geometric average rates_long among all different types of rates
    '''
    a = []
    for rate in rates:
        df = pd.read_csv('tickerData/{}.txt'.format(rate)).set_index(['DATE']).loc[start:end]
        z = zscore(df['CLOSE'])[-1]
        if z > 2:
            z = 2
        elif z <-2:
            z = -2
        nz = (z + 2) / 4
        a.append(nz)
    rates_long = gmean(a)
    rates_short = 1-rates_long

    '''
    Higher inflation -> lower commods demand and prices
    https://inflationdata.com/articles/2018/03/29/the-effects-of-inflation-and-interest-rates-on-commodity-prices/
    use core inflation over CPI to exclude price of most commodities to avoid "reverse causality"
    govt may respond with higher rates (cooling), therefore lower bond prices, lower rates futures price
    https://learnbonds.com/news/economic-indicators-bond-investors-should-follow/
    '''
    df = pd.read_csv('tickerData/{}.txt'.format('USA_CINF')).set_index(['DATE']).loc[start:end]
    z = zscore(df['CLOSE'])[-1]
    if z > 2:
        z = 2
    elif z <-2:
        z = -2
    nz = (z + 2) / 4
    inflation_short = nz
    inflation_long = 1-nz

    '''
    Higher nonfarm payrolls -> higher commods and stocks demand and prices
    govt may respond with higher rates, therefore lower bond prices, lower rates futures price
    https://www.dummies.com/personal-finance/investing/commodities/10-commodities-market-indicators-you-should-monitor/
    '''
    df = pd.read_csv('tickerData/{}.txt'.format('USA_NFP')).set_index(['DATE']).loc[start:end]
    z = zscore(df['CLOSE'])[-1]
    if z > 2:
        z = 2
    elif z <-2:
        z = -2
    nz = (z + 2) / 4
    nfp_long = nz
    nfp_short = 1-nz

    '''
    Higher manufacturing and industrial production -> higher energy demand and prices
    https://www.dummies.com/personal-finance/investing/commodities/10-commodities-market-indicators-you-should-monitor/
    production_long is a factor that ranges from 0-1 which shows how relatively high production is in the start:end period
    geometric avg between manufacturing production and industrial production
    '''
    a = []
    for index in ['USA_MP','USA_IP']:
        df = pd.read_csv('tickerData/{}.txt'.format(index)).set_index(['DATE']).loc[start:end]
        z = zscore(df['CLOSE'])[-1]
        if z > 2:
            z = 2
        elif z <-2:
            z = -2
        nz = (z + 2) / 4
        a.append(nz)
    production_long = gmean(a)
    production_short = 1-production_long

    '''
    Stronger USD -> commods (mainly priced in USD) become more expensive -> demand and prices fall
    https://www.thebalance.com/how-the-dollar-impacts-commodity-prices-809294
    dollar index futures as proxy for usd
    '''
    df = pd.read_csv('tickerData/{}.txt'.format('F_DX')).set_index(['DATE']).loc[start:end]
    z = zscore(df['CLOSE'])[-1]
    if z > 2:
        z = 2
    elif z <-2:
        z = -2
    nz = (z + 2) / 4
    usd_short = nz
    usd_long = 1-nz

    '''
    Higher consumer credit -> higher spending power -> higher demand for commods and stocks
    govt may respond with higher rates, therefore lower bond prices, lower rates futures price
    https://www.investopedia.com/terms/c/consumercredit.asp
    '''
    df = pd.read_csv('tickerData/{}.txt'.format('USA_CCR')).set_index(['DATE']).loc[start:end]
    z = zscore(df['CLOSE'])[-1]
    if z > 2:
        z = 2
    elif z <-2:
        z = -2
    nz = (z + 2) / 4
    consumercredit_long = nz
    consumercredit_short = 1-consumercredit_long

    '''
    Higher business optimism -> higher demand for commods and stocks
    govt may respond with higher rates, therefore lower bond prices, lower rates futures price
    https://www.dittotrade.academy/education/intermediate/fundamental-analysis/fundamental-indicators/what-is-business-confidence-index-why-is-it-important/
    '''
    df = pd.read_csv('tickerData/{}.txt'.format('USA_NFIB')).set_index(['DATE']).loc[start:end]
    z = zscore(df['CLOSE'])[-1]
    if z > 2:
        z = 2
    elif z <-2:
        z = -2
    nz = (z + 2) / 4
    bizoptimism_long = nz
    bizoptimism_short = 1-bizoptimism_long

    '''
    Commodities lead overall markets
    https://www.financialsense.com/contributors/kurt-kallaus/commodity-index-leading-economic-indicator
    goldman and bloomberg commodity index futures as proxy
    '''
    a = []
    for index in ['F_GD','F_AH']:
        df = pd.read_csv('tickerData/{}.txt'.format(index)).set_index(['DATE']).loc[start:end]
        z = zscore(df['CLOSE'])[-1]
        if z > 2:
            z = 2
        elif z <-2:
            z = -2
        nz = (z + 2) / 4
        a.append(nz)
    commods_long = gmean(a)
    commods_short = 1-commods_long

    '''
    Better housing market could lead to potentially higher rates (cooling) -> lower bond prices, lower rates futures price
    https://learnbonds.com/news/economic-indicators-bond-investors-should-follow/
    bond futures as proxy
    '''
    a = []
    for index in ['USA_NAHB','USA_PHS']:
        df = pd.read_csv('tickerData/{}.txt'.format(index)).set_index(['DATE']).loc[start:end]
        z = zscore(df['CLOSE'])[-1]
        if z > 2:
            z = 2
        elif z <-2:
            z = -2
        nz = (z + 2) / 4
        a.append(nz)
    housing_short = gmean(a)
    housing_long = 1-housing_short

    '''
    Determine the final long/short weight based on the weights of each factor
    Weights of factor is determined by correlation initially computed
    w = market_factor_weights
    '''
    for market in markets:
        if market in energy:

            long_sum = w['rates'][market]*rates_long + w['inflation'][market]*inflation_long + \
            w['nfp'][market]*nfp_long + w['usd'][market]*usd_long + w['consumercredit'][market]*consumercredit_long + \
            w['bizoptimism'][market]*bizoptimism_long
            
            lw[market] = long_sum / np.nansum(w['rates'][market]+w['inflation'][market]+w['nfp'][market]+\
            w['usd'][market]+w['consumercredit'][market]+w['bizoptimism'][market])

            short_sum = w['rates'][market]*rates_short + w['inflation'][market]*inflation_short + \
            w['nfp'][market]*nfp_short + w['usd'][market]*usd_short + w['consumercredit'][market]*consumercredit_short + \
            w['bizoptimism'][market]*bizoptimism_short

            sw[market] = -short_sum / np.nansum(w['rates'][market]+w['inflation'][market]+w['nfp'][market]+\
            w['usd'][market]+w['consumercredit'][market]+w['bizoptimism'][market])

        elif market in commods:

            long_sum = w['rates'][market]*rates_long + w['inflation'][market]*inflation_long + \
            w['nfp'][market]*nfp_long + w['usd'][market]*usd_long + w['consumercredit'][market]*consumercredit_long + \
            w['bizoptimism'][market]*bizoptimism_long + w['production'][market]*production_long
            
            lw[market] = long_sum / np.nansum(w['rates'][market]+w['inflation'][market]+w['nfp'][market]+\
            w['usd'][market]+w['consumercredit'][market]+w['bizoptimism'][market]+w['production'][market])

            short_sum = w['rates'][market]*rates_short + w['inflation'][market]*inflation_short + \
            w['nfp'][market]*nfp_short + w['usd'][market]*usd_short + w['consumercredit'][market]*consumercredit_short + \
            w['bizoptimism'][market]*bizoptimism_short + w['production'][market]*production_short

            sw[market] = -short_sum / np.nansum(w['rates'][market]+w['inflation'][market]+w['nfp'][market]+\
            w['usd'][market]+w['consumercredit'][market]+w['bizoptimism'][market]+w['production'][market])

        elif market in stock_indices:

            long_sum = w['nfp'][market]*nfp_long + w['consumercredit'][market]*consumercredit_long + \
            w['bizoptimism'][market]*bizoptimism_long + w['commods'][market]*commods_long
            
            lw[market] = long_sum / np.nansum(w['nfp'][market]+w['consumercredit'][market]+\
            w['bizoptimism'][market]+w['commods'][market])

            short_sum = w['nfp'][market]*nfp_short + w['consumercredit'][market]*consumercredit_short + \
            w['bizoptimism'][market]*bizoptimism_short + w['commods'][market]*commods_short

            sw[market] = -short_sum / np.nansum(w['nfp'][market]+w['consumercredit'][market]+\
            w['bizoptimism'][market]+w['commods'][market])

        elif (market in us_bonds) or (market in rates):

            long_sum = w['inflation'][market]*inflation_short + w['nfp'][market]*nfp_short + \
            w['consumercredit'][market]*consumercredit_short + \
            w['bizoptimism'][market]*bizoptimism_short + w['housing'][market]*housing_short

            lw[market] = long_sum / np.nansum(w['inflation'][market]+w['nfp'][market]+w['consumercredit'][market]+\
            w['bizoptimism'][market]+w['housing'][market])

            short_sum = w['inflation'][market]*inflation_long + w['nfp'][market]*nfp_long + \
            w['consumercredit'][market]*consumercredit_long + \
            w['bizoptimism'][market]*bizoptimism_long + w['housing'][market]*housing_long

            sw[market] = -short_sum / np.nansum(w['inflation'][market]+w['nfp'][market]+w['consumercredit'][market]+\
            w['bizoptimism'][market]+w['housing'][market])
            
        else:
            lw[market] = 1
            sw[market] = -1
    return lw, sw


def market_factor_weights(markets):
    '''
    Determines the factor_weight for every market and relevant factor
    Returns a dictionary with keys as factor and inner keys as market
    '''
    print("Calculating market_factor_weights...")

    d = {'rates':{},'inflation':{},'nfp':{},'production':{},'usd':{},'consumercredit':{},'bizoptimism':{},'commods':{},'housing':{}}

    for market in markets:
        if market in energy:
            a = []
            for rate in rates:
                a.append(factor_weight(rate,market))
            d['rates'][market] = gmean(a)
            d['inflation'][market] = factor_weight('USA_CINF',market)
            d['nfp'][market] = factor_weight('USA_NFP',market)
            d['usd'][market] = factor_weight('F_DX',market)
            d['consumercredit'][market] = factor_weight('USA_CCR',market)
            d['bizoptimism'][market] = factor_weight('USA_NFIB',market)

        elif market in commods:
            a = []
            for rate in rates:
                a.append(factor_weight(rate,market))
            d['rates'][market] = gmean(a)
            d['inflation'][market] = factor_weight('USA_CINF',market)
            d['nfp'][market] = factor_weight('USA_NFP',market)
            d['usd'][market] = factor_weight('F_DX',market)
            d['consumercredit'][market] = factor_weight('USA_CCR',market)
            d['bizoptimism'][market] = factor_weight('USA_NFIB',market)
            a = []
            for idx in ['USA_MP','USA_IP']:
                a.append(factor_weight(idx,market))
            d['production'][market] = gmean(a)

        elif market in stock_indices:
            d['nfp'][market] = factor_weight('USA_NFP',market)
            d['consumercredit'][market] = factor_weight('USA_CCR',market)
            d['bizoptimism'][market] = factor_weight('USA_NFIB',market)
            a = []
            for idx in ['F_GD','F_AH']:
                a.append(factor_weight(idx,market))
            d['commods'][market] = gmean(a)

        elif (market in us_bonds) or (market in rates):
            d['inflation'][market] = factor_weight('USA_CINF',market)
            d['nfp'][market] = factor_weight('USA_NFP',market)
            d['consumercredit'][market] = factor_weight('USA_CCR',market)
            d['bizoptimism'][market] = factor_weight('USA_NFIB',market)
            a = []
            for idx in ['USA_NAHB','USA_PHS']:
                a.append(factor_weight(idx,market))
            d['housing'][market] = gmean(a)

    print("Calculated market_factor_weights")
    return d


def factor_weight(f1,f2):
    '''
    Returns the weight the factor should play a part in overall long/short position
    based on the magnitude correlation since the start
    f1 should be economic indicator or future
    f2 should only be future
    '''
    df1 = pd.read_csv('tickerData/{}.txt'.format(f1)).set_index(['DATE'])['CLOSE']
    df2 = pd.read_csv('tickerData/{}.txt'.format(f2)).set_index(['DATE'])['CLOSE']
    df = pd.concat([df1,df2], join='outer', axis=1)
    df.columns = ['CLOSE_1','CLOSE_2']
    df['CLOSE_2'] = df['CLOSE_2'].fillna(method='backfill')
    df = pd.concat([df1,df], join='inner', axis=1)
    corr = np.corrcoef(df['CLOSE_1'],df['CLOSE_2'])[0,1]
    return abs(corr)