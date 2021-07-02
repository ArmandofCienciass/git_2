# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:14:07 2021

@author: Armando
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import importlib
from scipy.stats import skew, kurtosis, chi2, linregress

import file_classes
importlib.reload(file_classes) 
import file_functions
importlib.reload(file_functions)

benchmark = '^STOXX50E'

security = '^GDAXI'

capm =  file_classes.capm_manager(benchmark,security)
capm.load_timeseries()
capm.plot_timeseries()
capm.compute()
capm.plot_linear_regression()
print(capm)

inputs = file_classes.capm_inputs
inputs.benchmark = 'SPY'
inputs.security = 'BBVA.MC'

nb_decimals = 4


# synchronise

t =file_functions.load_synchronise_timeseries(ric_x = benchmark,ric_y = security)

x = t['return_x'].values
y = t['return_y'].values
slope, intercept, r_value, p_value, std_err = linregress(x,y)


plt.figure(figsize = (12,5))
plt.title('time series of')
plt.xlabel('Time')
plt.ylabel('Prices')
ax = plt.gca()
ax1 = t.plot(kind = 'line', x = 'date', y = 'price_x', ax = ax, grid = True,\
            color = 'blue', label = benchmark)
ax2 = t.plot(kind = 'line', x = 'date', y = 'price_y', ax = ax, grid = True,\
            color = 'red',secondary_y = True, label = security)
    
ax1.legend(loc =2)
ax2.legend(loc =1)
plt.show()

slope = np.round(slope, nb_decimals)
nb_decimals = 4
intercept = np.round(intercept,nb_decimals)
p_value = np.round(p_value,nb_decimals)
is_null_hypothesis = p_value > 0.05
r_value = np.round(r_value,nb_decimals)
r_squared = np.round(r_value**2,nb_decimals)
predictor_linreg = intercept + slope*x

str_title = 'Scatterplot of returns' + '\n'\
    + 'Linear regression / security' + security\
    + 'alpha (intercept)' + benchmark + '\n'\
    + '/ beta (slope)' + str(slope) + '\n'\
    + 'p_value' + str(p_value)\
    + '/ null hypothesis' + str(is_null_hypothesis) + '\n'\
    + 'r_value (correl)' + str(r_value)\
    + '/ r_squared' + str(r_squared)
    
plt.figure()
plt.title(str_title)
plt.scatter(x,y)
plt.plot(x,predictor_linreg, color = 'green')
plt.ylabel(security)
plt.xlabel(benchmark)
plt.grid()
plt.show()
        


