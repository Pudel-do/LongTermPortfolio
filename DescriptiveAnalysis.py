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
from tabulate import tabulate
from Config import Config

import warnings
warnings.filterwarnings("ignore")

parameter_config = Config("Parameter.json")

year_shift = 5
end = pd.Timestamp.today()
start = end - DateOffset(years=year_shift)
analysis_ticks = ['IWDA.L', 'EWG2.SG', 'LIT']


class DescriptiveAnalysis:
    def __init__(self, start, end, ticks, parameter_config):
        self.start = start
        self.end = end
        self.ticks = analysis_ticks
        self.ticks_file = 'TickerMapping.json'
        self.rets_pooled_file = 'PooledReturns.csv'
        self.period = f"{start.strftime('%Y-%m')} - {end.strftime('%Y-%m')}"
        self.plot_settings = parameter_config.config['Plot_Settings']
        self.parameters = parameter_config.config['Parameters']
        self.flags = parameter_config.config['Flags']

    def get_data(self):
        print('\n')
        print(f'--- Start downloading data---')
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
        
    def pool_mean_data(self):
        n_total_days = len(self.rets)
        n_days = self.rets.count()
        days_frac = n_days / n_total_days
        days_mask = days_frac < self.parameters['min_share_obs']
        days_mask = days_mask.where(days_mask==True)
        days_mask = days_mask.dropna()
        invalid_ticks = list(days_mask.index)
        period_min = self.rets.index.min().strftime('%Y')
        period_max = self.rets.index.max().strftime('%Y')
        period = f"{period_min}-{period_max}"
        
        pooled_mean_flag = False
        if os.path.exists(self.rets_pooled_file):
            pooled_mean = pd.read_csv(self.rets_pooled_file, index_col=0)
            pooled_period = list(pooled_mean.index)
            pooled_ticks = list(pooled_mean.columns)
            complement_ticks = list(set(self.ticks)-set(pooled_ticks))
            mean = self.rets.mean()*252
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
                    
            pooled_mean.to_csv(self.rets_pooled_file)
            
        else:            
            pooled_mean = self.rets.mean()*252
            pooled_mean = pooled_mean.round(3)
            for index, value in pooled_mean.items():
                if index in invalid_ticks:
                    pooled_mean.loc[index] = f'({value})'

            pooled_mean = pd.DataFrame(pooled_mean)
            pooled_mean.columns = [period]
            pooled_mean = pooled_mean.transpose()
            pooled_mean.to_csv(self.rets_pooled_file)
        
        if pooled_mean.shape[0] > 1:
            pooled_mean['Sort_Col'] = pooled_mean.index.values
            pooled_mean['Sort_Col'] = pooled_mean['Sort_Col'].str[:4]
            pooled_mean['Sort_Col'] = pooled_mean['Sort_Col'].astype(int)
            pooled_mean.sort_values(by='Sort_Col', ascending=False, inplace=True)
            pooled_mean.drop('Sort_Col', axis=1, inplace=True)
            pooled_mean = pooled_mean.transpose()
            pooled_mean_flag = True
        else:
            pooled_mean_flag = False
           
        self.pooled_mean = pooled_mean
        self.pooled_mean_flag = pooled_mean_flag

    def return_analysis(self):
        self.rets_corr = self.rets.corr()
        
        rets_stat_full = self.rets.describe()
        rets_stat_full = rets_stat_full.transpose()
        stat_cols = ['count', 'mean', 'std']
        adj_cols = {'count': 'Count', 'mean': 'Mean', 'std': 'Volatility'}
        rets_stat = rets_stat_full[stat_cols]
        rets_stat['mean'] = rets_stat['mean']*252
        rets_stat['std'] = rets_stat['std']*np.sqrt(252)
        rets_stat.sort_values(by='mean', ascending=False, inplace=True)
        rets_stat.rename(adj_cols, axis=1, inplace=True)
        self.rets_stat = rets_stat
        
    def quote_analysis(self):
        norm_quotes = self.quotes.copy()
        for col, values in norm_quotes.items():
            values = values.dropna()
            initial_value = values.iloc[0]
            norm_quotes[col] = norm_quotes[col] / initial_value
        norm_quotes = norm_quotes*1000
        
    def visualization(self):
        df_tick_mapping = pd.DataFrame(self.tick_mapping.values())
        df_tick_mapping.index = self.tick_mapping.keys()
        df_tick_mapping.columns = ['Name']
        print('\n')
        print(tabulate(
                df_tick_mapping, 
                headers = self.plot_settings['headers'],
                tablefmt = self.plot_settings['tablefmt']
                    ))
        
        print('\n')
        print(
            tabulate(
                self.rets_stat, 
                headers = self.plot_settings['headers'],
                tablefmt = self.plot_settings['tablefmt'],
                floatfmt = self.plot_settings['floatfmt']
                    ))
        
        if self.pooled_mean_flag:
            print('\n')
            print(
                tabulate(
                    self.pooled_mean, 
                    headers = self.plot_settings['headers'],
                    tablefmt = self.plot_settings['tablefmt'],
                    floatfmt = self.plot_settings['floatfmt']
                        ))
        
        sn.heatmap(self.rets_corr, annot=True)
        hist_rets = self.rets.rename(self.tick_mapping, axis=1)
        hist_rets.hist(bins=self.plot_settings['bins'], figsize=tuple(self.plot_settings['figsize']))

          
desc_analysis = DescriptiveAnalysis(
    start, end, analysis_ticks, 
    parameter_config
    )

print('\n')
print(40*'=')
print(f'Analysis Period: {desc_analysis.period}')
print(40*'=')

desc_analysis.get_data()
desc_analysis.ticker_mapping()
desc_analysis.pool_mean_data()
desc_analysis.return_analysis()
desc_analysis.quote_analysis()
desc_analysis.visualization()
