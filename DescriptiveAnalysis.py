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

year_shift = 7
end = pd.Timestamp.today()
start = end - DateOffset(years=year_shift)
analysis_ticks = ['IWDA.L', 'EWG2.SG', 'LIT', 
                  'SUES.L', 'EMIM.L'
                  ]
min_days_frac = 0.8

class DescriptiveAnalysis:
    def __init__(self, start, end, ticks):
        self.start = start
        self.end = end
        self.ticks = analysis_ticks
        self.ticks_file = 'TickerMapping.json'
        self.rets_pooled_file = 'PooledReturns.csv'

    def get_data(self):
        print('--- Start downloading data ---')
        raw_data = yf.download(self.ticks, self.start, self.end)
        quotes = raw_data['Adj Close']
        rets = np.log(quotes / quotes.shift(1))
        rets = rets.iloc[1:, :]
        self.rets = rets
        self.quotes = quotes
        
    def ticker_mapping(self):
        if os.path.exists(self.ticks_file):
            with open(self.ticks_file, 'r') as fp:
                tick_mapping = json.load(fp)
            keys = list(tick_mapping.keys())
            complement_ticks = list(set(self.ticks)-set(keys))
            if complement_ticks:
                for tick in complement_ticks:
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
        print('--- Dowloading data completed ---')
        self.tick_mapping = tick_mapping
        
    def pool_mean_data(data, ticks, file_name):
        n_total_days = len(data)
        n_days = data.count()
        days_frac = n_days / n_total_days
        days_mask = days_frac < min_days_frac
        days_mask = days_mask.where(days_mask==True)
        days_mask = days_mask.dropna()
        invalid_ticks = list(days_mask.index)
        period_min = data.index.min().strftime('%Y')
        period_max = data.index.max().strftime('%Y')
        period = f"{period_min}-{period_max}"
        if os.path.exists(file_name):
            pooled_mean = pd.read_csv(file_name, index_col=0)
            pooled_period = list(pooled_mean.index)
            pooled_ticks = list(pooled_mean.columns)
            complement_ticks = list(set(ticks)-set(pooled_ticks))
            mean = data.mean()*252
            mean = mean.round(3)
            for index, value in mean.items():
                if index in invalid_ticks:
                    mean.loc[index] = f'({value})'
            mean = pd.DataFrame(mean)
            mean.columns = [period]
            mean = mean.transpose()
            if period not in pooled_period:
                pooled_mean = pd.concat([pooled_mean, mean])
            elif complement_ticks:
                mean = mean.loc[:, complement_ticks]
                pooled_mean = pd.concat([pooled_mean, mean], axis=1)
            else:
                mean_adj = pooled_mean.loc[period]
                adj_mask = pd.isnull(mean_adj)
                mean_adj_idx = list(mean_adj[adj_mask].index)
                for value in mean_adj_idx:
                    try:
                        fill_value = float(mean.loc[:, value])
                        pooled_mean.loc[period, value] = fill_value
                    except:
                        break
            pooled_mean.to_csv(file_name)
        else:            
            pooled_mean = data.mean()*252
            pooled_mean = pooled_mean.round(3)
            for index, value in pooled_mean.items():
                if index in invalid_ticks:
                    pooled_mean.loc[index] = f'({value})'

            pooled_mean = pd.DataFrame(pooled_mean)
            pooled_mean.columns = [period]
            pooled_mean = pooled_mean.transpose()
            pooled_mean.to_csv(file_name)
            
        return pooled_mean

    def return_analysis(self):
        rets_corr = self.rets.corr()
        rets_plot = sn.heatmap(rets_corr, annot=True)
        test = DescriptiveAnalysis.pool_mean_data(self.rets, 
                                                  self.ticks,
                                                  self.rets_pooled_file)
        
        print('Test')


    def quote_plot(self):
        norm_quotes = self.quotes / self.quotes.iloc[0,:]
        
        
desc_analysis = DescriptiveAnalysis(start, end, analysis_ticks)
desc_analysis.get_data()
desc_analysis.ticker_mapping()
desc_analysis.return_analysis()
desc_analysis.quote_plot()
         
