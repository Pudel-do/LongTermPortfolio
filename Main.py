# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:49:09 2022

@author: Matthias Pudel
"""

import pandas as pd
from pandas.tseries.offsets import DateOffset
from DescriptiveAnalysis import DescriptiveAnalysis
from PortfolioAnalysis import PortfolioAnalysis
from Config import Config
import pickle
parameter_config = Config("Parameter.json")

"""Settings for descreptive Analysis"""
analysis_year_shift = 3
end = pd.Timestamp.today()
start = end - DateOffset(years=analysis_year_shift)
analysis_ticks = ['IWDA.L', 'LIT']

"""Settings for Portfolio Analysis"""
port_end = pd.Timestamp('2022-10-27')
port_start = pd.Timestamp('2020-01-01')
port_ticks = ['IWDA.L', 'LIT']
use_custom_weights = False
custom_weights = [0.2, 0.7]
saving_rate = 300


if parameter_config.config['Flags']['use_desc_analysis']:
    desc_analysis = DescriptiveAnalysis(start, end, analysis_ticks, parameter_config)
    desc_analysis.remove_data()
    desc_analysis.get_data()
    desc_analysis.ticker_mapping()
    desc_analysis.pool_mean_data()
    desc_analysis.return_analysis()
    desc_analysis.quote_analysis()
    desc_analysis.visualization()
    with open('DescriptiveAnalysis.pkl', 'wb') as outp:
        pickle.dump(desc_analysis, outp, pickle.HIGHEST_PROTOCOL)
else:
    with open('DescriptiveAnalysis.pkl', 'rb') as inp:
        desc_analysis = pickle.load(inp)

port_analysis = PortfolioAnalysis(port_start, port_end, port_ticks, custom_weights, saving_rate, parameter_config)
port_analysis.get_returns()
port_analysis.get_opt_weights()
port_analysis.get_aggregated_returns()
port_analysis.return_analysis()
port_analysis.portfolio_construction(use_custom_weights)
port_analysis.visualization(use_custom_weights)
