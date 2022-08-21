# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:52:36 2022

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sn
import matplotlib.pyplot as plt
import os
import json
from pandas.tseries.offsets import DateOffset

year_shift = 2
end = pd.Timestamp.today()
start = end - DateOffset(years=year_shift)
ticks = ['IWDA.L', 'EWG2.SG', 'LIT']



class DescriptiveAnalysis:
    def __init__(self, start, end, ticks):
        self.start = start
        self.end = end
        self.ticks = ticks
        self.ticks_file = 'TickerMapping.json'

    def get_data(self):
        print('Start downloading data')
        raw_data = yf.download(self.ticks, self.start, self.end)
        quotes = raw_data['Adj Close']
        rets = np.log(quotes / quotes.shift(1))
        rets = rets.iloc[1:, :]
        self.quotes = quotes
        self.rets = rets
        
    def ticker_mapping(self):
        if os.path.exists(self.ticks_file):
            with open(self.ticks_file, 'r') as fp:
                tick_mapping = json.load(fp)
            keys = list(tick_mapping.keys())
            if set(keys) != set(self.ticks):
                tick_mapping = {}
                for tick in self.ticks:
                    name = yf.Ticker(tick).info['longName']
                    tick_mapping[tick] = name
            with open(self.ticks_file, 'w') as fp:
                json.dump(tick_mapping, fp)
        else:
            tick_mapping = {}
            for tick in self.ticks:
                name = yf.Ticker(tick).info['longName']
                tick_mapping[tick] = name
            with open(self.ticks_file, 'w') as fp:
                json.dump(tick_mapping, fp)
        self.tick_mapping = tick_mapping

    def descriptive_analysis(self):
        rets_corr = self.rets.corr()
        rets_plot = sn.heatmap(rets_corr, annot=True)


    def quote_plot(self):
        norm_quotes = self.quotes / self.quotes.iloc[0,:]
        
        
desc_analysis = DescriptiveAnalysis(start, end, ticks)
desc_analysis.get_data()
desc_analysis.ticker_mapping()
desc_analysis.descriptive_analysis()
desc_analysis.quote_plot()
         
