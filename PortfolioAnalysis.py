import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
import scipy.optimize as sco
import json
from tabulate import tabulate

class PortfolioAnalysis:
    def __init__(self, start, end, ticks, custom_weights, saving_rate, parameter_config):
        self.start = start
        self.start_year = start.year
        self.end = end
        self.end_year = end.year
        self.ticks = ticks
        self.custom_weights = custom_weights
        self.saving_rate = saving_rate
        self.period = f"{start.strftime('%Y-%m')} - {end.strftime('%Y-%m')}"
        self.plot_settings = parameter_config.config['Plot_Settings']
        self.parameters = parameter_config.config['Parameters']
        self.flags = parameter_config.config['Flags']
        self.main_path = os.getcwd()
        self.plot_path = 'Plots_Portfolio'

    def get_data(self, filename):
        with open(filename, 'r') as fp:
            data = json.load(fp)
        return data

    def plot_dataframe(self, df, title, filename):
        plt.figure(figsize=tuple(self.plot_settings['figsize']))
        for col, values in df.iteritems():
            plt.plot(values, lw=self.plot_settings['main_line'], label=col)
        plt.legend(loc=0)
        plt.grid(True)
        plt.xlabel('Date', fontsize=self.plot_settings['label_size'])
        plt.ylabel('Quote', fontsize=self.plot_settings['label_size'])
        plt.title(title, fontsize=self.plot_settings['title_size'])
        plt.savefig(os.path.join(self.plot_path, filename))

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

    def get_returns(self):
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
        benchmark_quotes = yf.download(self.parameters['benchmark'], start=self.start, end=self.end, progress=False)
        benchmark_quotes = pd.DataFrame(benchmark_quotes['Adj Close'])
        benchmark_quotes.index = benchmark_quotes.index.strftime('%Y-%m-%d')
        benchmark_quotes.columns = [self.parameters['benchmark']]

        quotes.index = pd.to_datetime(quotes.index)
        benchmark_quotes.index = pd.to_datetime(benchmark_quotes.index)
        rets = np.log(quotes / quotes.shift(1))
        rets = rets.iloc[1:, :]
        benchmark_rets = np.log(benchmark_quotes / benchmark_quotes.shift(1))
        benchmark_rets = benchmark_rets.iloc[1:, :]
        self.rets = rets
        self.benchmark_rets = benchmark_rets

    def port_ret(self, weights):
        return np.sum(self.rets * weights, axis=1).mean() * 252

    def port_vol(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.rets.cov() * 252, weights)))

    def min_func_sharpe(self, weights):
        return -self.port_ret(weights) / self.port_vol(weights)

    def get_opt_weights(self):
        noa = len(self.ticks)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
        bnds = tuple((0,1) for x in range(noa))
        eweights = np.array(noa * [1./noa,])

        max_sharpe_ratio = sco.minimize(self.min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
        min_var = sco.minimize(self.port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
        max_sharp_weights = max_sharpe_ratio['x']
        min_var_weights = min_var['x']
        self.max_sharp_weights = max_sharp_weights
        self.min_var_weights = min_var_weights
        self.eweights = eweights

    def get_aggregated_returns(self):
        weights = {'MinimumVariance': self.min_var_weights,
                   'MaximumSharpeRatio': self.max_sharp_weights,
                   'CustomWeights': self.custom_weights,
                   'EqualWeights': self.eweights}

        total_rets = pd.DataFrame()
        for key, value in weights.items():
            port_rets = (self.rets * value).sum(axis=1)
            port_rets = pd.DataFrame(port_rets, columns=[key])
            total_rets = pd.concat([total_rets, port_rets], axis=1)

        total_rets = total_rets.join(self.benchmark_rets)
        self.total_rets = total_rets
        self.weights = weights

    def return_analysis(self):
        annualized_rets = self.total_rets.mean()*252
        annualized_vol = np.sqrt(self.total_rets.std())
        sharpe_ratio = (annualized_rets - self.parameters['risk_free_rate']) / annualized_vol
        annualized_rets = annualized_rets.to_frame('Return')
        annualized_vol = annualized_vol.to_frame('Volatility')
        sharpe_ratio = sharpe_ratio.to_frame('Sharpe Ratio')
        rets_statistic = pd.concat([annualized_rets, annualized_vol, sharpe_ratio], axis=1)
        self.rets_statistic = rets_statistic

    def portfolio_construction(self, use_custom_weights):
        tick_mapping = self.get_data('TickerMapping.json')
        if use_custom_weights:
            cols = ['CustomWeights', self.parameters['benchmark']]
            port_rets = self.total_rets[cols]
        else:
            cols = [self.parameters['port_type'], self.parameters['benchmark']]
            port_rets = self.total_rets[cols]

        port_quotes = port_rets.cumsum().apply(np.exp)
        port_quotes = port_quotes * 1000
        port_quotes.rename(tick_mapping, axis=1, inplace=True)
        self.port_quotes = port_quotes

    def visualization(self, use_custom_weights):
        print(40 * '=')
        print(f'Portfolio Period: {self.period}')
        print(40 * '=')
        self.tick_mapping = self.get_data('TickerMapping.json')
        if not os.path.exists(os.path.join(self.main_path, self.plot_path)):
            os.makedirs(os.path.join(self.main_path, self.plot_path))

        title = f'Portfolio performance'
        filename = 'PortfolioPerformance.png'
        plot_data = self.port_quotes.rename(self.tick_mapping, axis=1)
        self.plot_dataframe(plot_data, title, filename)

        plot_data_hist = self.total_rets.rename(self.tick_mapping, axis=1)
        title = f'Histogram for period {self.period}'
        plot_data_hist.hist(bins=self.plot_settings['bins'], figsize=tuple(self.plot_settings['figsize']))
        file_name = f'Histogram_{self.start_year}_{self.end_year}'
        plt.savefig(os.path.join(self.plot_path, file_name))

        self.rets_statistic.sort_values(by='Sharpe Ratio', inplace=True, ascending=False)
        self.print_table(self.rets_statistic)

        total_port_costs = pd.DataFrame(index=self.ticks)
        if use_custom_weights:
            port_weights = self.weights['CustomWeights']
            port_weights = np.array(port_weights)
            port_weights = np.transpose(port_weights)
        else:
            port_weights = self.weights[self.parameters['port_type']]
            port_weights = np.transpose(port_weights)
        port_costs = np.dot(port_weights, self.saving_rate)
        total_port_costs['Weights'] = port_weights
        total_port_costs['Costs'] = port_costs
        self.print_table(total_port_costs)
