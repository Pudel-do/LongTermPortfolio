# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:49:09 2022

@author: Matthias Pudel
"""

import pandas as pd
from pandas.tseries.offsets import DateOffset
import Analysis

year_shift = 2
end = pd.Timestamp.today()
start = end - DateOffset(years=year_shift)
ticks = ['IWDA.L', 'EWG2.SG', 'LIT']

port_analysis = Analysis.PortAnalysis(start, end, ticks)
port_analysis.get_data()
port_analysis.ticker_mapping()
port_analysis.descriptive_analysis()
port_analysis.quote_plot()