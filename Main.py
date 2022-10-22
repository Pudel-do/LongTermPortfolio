# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:49:09 2022

@author: Matthias Pudel
"""

import pandas as pd
from pandas.tseries.offsets import DateOffset
from DescriptiveAnalysis import DescriptiveAnalysis
from Config import Config

parameter_config = Config("Parameter.json")

analysis_year_shift = 3
end = pd.Timestamp.today()
start = end - DateOffset(years=analysis_year_shift)
analysis_ticks = ['IWDA.L', 'LIT']

desc_analysis = DescriptiveAnalysis(start, end, analysis_ticks, parameter_config)
print('\n')
print(40*'=')
print(f'Analysis Period: {desc_analysis.period}')
print(40*'=')
desc_analysis.remove_data()
desc_analysis.get_data()
desc_analysis.ticker_mapping()
desc_analysis.pool_mean_data()
desc_analysis.return_analysis()
desc_analysis.quote_analysis()
desc_analysis.visualization()



