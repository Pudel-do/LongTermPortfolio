import pandas as pd
import numpy as np
import os
import yfinance as yf
import scipy.optimize as sco

class PortfolioAnalysis:
    def __init__(self, start, end, ticks, custom_weights, saving_rate, parameter_config):
        self.start = start
        self.end = end
        self.ticks = ticks
        self.custom_weights = custom_weights
        self.saving_rate = saving_rate
        self.period = f"{start.strftime('%Y-%m')} - {end.strftime('%Y-%m')}"
        self.plot_settings = parameter_config.config['Plot_Settings']
        self.parameters = parameter_config.config['Parameters']
        self.flags = parameter_config.config['Flags']
        self.main_path = os.getcwd()
        self.plot_path = 'Plots_Portfolio'

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
        return np.sum(self.rets.mean() * weights * 252)

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

    def port_construction(self):
        weights = {'MinimumVariance': self.min_var_weights,
                   'MaximumSharpeRatio': self.max_sharp_weights,
                   'CustomWeights': self.custom_weights,
                   'EqualWeights': self.eweights}

        total_portfolio_rets = pd.DataFrame()
        for key, value in weights.items():
            port_rets = self.rets.dropna() * value
            port_rets = port_rets.sum(axis=1)
            port_rets = pd.DataFrame(port_rets, columns=[key])
            total_portfolio_rets = pd.concat([total_portfolio_rets, port_rets], axis=1)
            print('Test')


