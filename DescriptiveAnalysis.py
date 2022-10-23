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
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate

class DescriptiveAnalysis:
    def __init__(self, start, end, ticks, parameter_config):
        self.start = start
        self.start_year = start.year
        self.end = end
        self.end_year = end.year
        self.ticks = ticks
        self.period = f"{start.strftime('%Y-%m')} - {end.strftime('%Y-%m')}"
        self.plot_settings = parameter_config.config['Plot_Settings']
        self.parameters = parameter_config.config['Parameters']
        self.flags = parameter_config.config['Flags']
        self.main_path = os.getcwd()
        self.plot_path = 'Plots_Analysis'
        self.data_path = 'Data_Analysis'
    def remove_data(self):
        if self.flags['remove_analysis_files']:
            if os.path.exists(os.path.join(self.main_path, self.data_path)):
                directory = os.path.join(self.main_path, self.data_path)
                filelist = [file for file in os.listdir(directory)]
                for file in filelist:
                    os.remove(os.path.join(directory, file))

        if self.flags['remove_analysis_plots']:
            if os.path.exists(os.path.join(self.main_path, self.plot_path)):
                directory = os.path.join(self.main_path, self.plot_path)
                filelist = [file for file in os.listdir(directory)]
                for file in filelist:
                    os.remove(os.path.join(directory, file))
    def get_data(self):
        print('\n')
        quotes_index = pd.date_range(self.start, self.end, freq='B')
        quotes_index = quotes_index.strftime('%Y-%m-%d')
        quotes = pd.DataFrame(index=quotes_index)
        for tick in self.ticks:
            data = yf.download(tick, start=self.start, end=self.end, progress=False)
            data = pd.DataFrame(data['Adj Close'])
            data.index = data.index.strftime('%Y-%m-%d')
            data.columns = [tick]
            quotes = quotes.join(data)
        quotes.index = pd.to_datetime(quotes.index)
        rets = np.log(quotes / quotes.shift(1))
        rets = rets.iloc[1:, :]
        self.rets = rets
        self.quotes = quotes
        
    def ticker_mapping(self):
        file_name = 'TickerMapping.json'
        if os.path.exists(file_name):
            with open(file_name, 'r') as fp:
                tick_mapping = json.load(fp)
            keys = list(tick_mapping.keys())
            complement_ticks = list(set(self.ticks)-set(keys))
            if complement_ticks:
                for tick in complement_ticks:
                    name = yf.Ticker(tick).info['longName']
                    tick_mapping[tick] = name
                    with open(file_name, 'w') as fp:
                        json.dump(tick_mapping, fp)
        else:
            tick_mapping = {}
            for tick in self.ticks:
                name = yf.Ticker(tick).info['longName']
                tick_mapping[tick] = name
            with open(file_name, 'w') as fp:
                json.dump(tick_mapping, fp)
        self.ticks_file = file_name
        self.tick_mapping = tick_mapping
        
    def pool_mean_data(self):
        rets_pooled_file = 'PooledReturns.csv'
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
        if os.path.exists(os.path.join(self.data_path, rets_pooled_file)):
            pooled_mean = pd.read_csv(os.path.join(self.data_path, rets_pooled_file), index_col=0)
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
                    
            pooled_mean.to_csv(os.path.join(self.data_path, rets_pooled_file))
            
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)

            pooled_mean = self.rets.mean()*252
            pooled_mean = pooled_mean.round(3)
            for index, value in pooled_mean.items():
                if index in invalid_ticks:
                    pooled_mean.loc[index] = f'({value})'

            pooled_mean = pd.DataFrame(pooled_mean)
            pooled_mean.columns = [period]
            pooled_mean = pooled_mean.transpose()
            pooled_mean.to_csv(os.path.join(self.data_path, rets_pooled_file))
        
        index_flag = pooled_mean.shape[0] > 1
        if index_flag:
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
        self.rets_pooled_file = rets_pooled_file
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
        self.norm_quotes = norm_quotes
        
        moving_avg_dict = {}
        for col, values in self.quotes.items():
            values = values.dropna()
            ma_values = pd.DataFrame(values)
            for ma_day in self.parameters['moving_avg_days']:
                ma_values[f'MA{ma_day}'] = values.rolling(ma_day).mean()
            moving_avg_dict[col] = ma_values
        self.moving_avg_dict = moving_avg_dict
    def print_table(self, df):
        try:
            df.rename(self.tick_mapping, inplace=True)
        except:
            pass
        print('\n')
        print(tabulate(df,
                headers=self.plot_settings['headers'],
                tablefmt=self.plot_settings['tablefmt'],
                floatfmt=self.plot_settings['floatfmt']
                    ))

    def visualization(self):
        print(40 * '=')
        print(f'Analysis Period: {self.period}')
        print(40 * '=')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        sn.heatmap(self.rets_corr, annot=True)
        file_name = f'Correlation_{self.start_year}_{self.end_year}'
        plt.savefig(os.path.join(self.plot_path, file_name))

        hist_rets = self.rets.rename(self.tick_mapping, axis=1)
        hist_rets.hist(bins=self.plot_settings['bins'], figsize=tuple(self.plot_settings['figsize']))
        file_name = f'Histogram_{self.start_year}_{self.end_year}'
        plt.savefig(os.path.join(self.plot_path, file_name))

        for key, df in self.moving_avg_dict.items():
            file_name = self.tick_mapping[key]
            if len(self.parameters['moving_avg_days'])>1:
                plt.figure(figsize=tuple(self.plot_settings['figsize']))
                plt.plot(df.iloc[:, 0], lw=self.plot_settings['main_line'], label='Quote')
                plt.plot(df.iloc[:, 1], lw=self.plot_settings['sub_line'], label=df.columns[1])
                plt.plot(df.iloc[:, 2], lw=self.plot_settings['sub_line'], label=df.columns[2])
                plt.legend(loc=0)
                plt.grid(True)
                plt.xlabel('Date', fontsize=self.plot_settings['label_size'])
                plt.ylabel('Quote', fontsize=self.plot_settings['label_size'])
                plt.title(self.tick_mapping.get(key), fontsize=self.plot_settings['title_size'])
                plt.savefig(os.path.join(self.plot_path, file_name))
            else:
                plt.figure(figsize=tuple(self.plot_settings['figsize']))
                plt.plot(df.iloc[:, 0], lw=self.plot_settings['main_line'], label='Quote')
                plt.plot(df.iloc[:, 1], lw=self.plot_settings['sub_line'], label=df.columns[1])
                plt.legend(loc=0)
                plt.grid(True)
                plt.xlabel('Date', fontsize=self.plot_settings['label_size'])
                plt.ylabel('Quote', fontsize=self.plot_settings['label_size'])
                plt.title(self.tick_mapping.get(key), fontsize=self.plot_settings['title_size'])
                plt.savefig(os.path.join(self.plot_path, file_name))

        file_name = f'NormalizedQuotes{self.start_year}_{self.end_year}'
        plt.figure(figsize=tuple(self.plot_settings['figsize']))
        for name, values in self.norm_quotes.iteritems():
            values.dropna(inplace=True)
            plt.plot(values, lw=self.plot_settings['main_line'], label=name)
        plt.legend(loc=0)
        plt.grid(True)
        plt.xlabel('Date', fontsize=self.plot_settings['label_size'])
        plt.ylabel('Normed Quote', fontsize=self.plot_settings['label_size'])
        plt.title(f'Normed quotes for period {self.period}', fontsize=self.plot_settings['title_size'])
        plt.savefig(os.path.join(self.plot_path, file_name))
        
        df_tick_mapping = pd.DataFrame(self.tick_mapping.values())
        df_tick_mapping.index = self.tick_mapping.keys()
        df_tick_mapping.columns = ['Name']
        self.print_table(df_tick_mapping)
        self.print_table(self.rets_stat)
        if self.pooled_mean_flag:
            self.print_table(self.pooled_mean)

