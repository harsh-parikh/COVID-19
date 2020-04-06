#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:34:35 2020

@author: Harsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

df_total = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_death = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recovered =  pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
#preprocessing
df_total = df_total.drop(columns=['Province/State','Lat', 'Long'])
df_death = df_death.drop(columns=['Province/State','Lat', 'Long'])
df_recovered = df_recovered.drop(columns=['Province/State','Lat', 'Long'])

df_total = df_total.groupby(['Country/Region']).sum()
df_death = df_death.groupby(['Country/Region']).sum()
df_recovered = df_recovered.groupby(['Country/Region']).sum()

assert (df_death.index == df_recovered.index).all()

df_active = df_total - ( df_death + df_recovered )

total = df_total.to_numpy()
daily = total[:,1:] - total[:,:-1] 

df_daily = pd.DataFrame(daily,index=df_total.index,columns=df_total.columns[1:])

t = np.arange(len(df_daily.columns))
fig = plt.figure()
for country in ['India','US','Italy','United Kingdom','France','Korea, South']:
    daily_t = df_daily.loc[country]
    active_t_1 = df_total.loc[country][:-1]
    #plt.plot(active_t_1, daily_t)
    sns.regplot(x=active_t_1, y=daily_t, scatter_kws={"s": 8},order=2,truncate=True)

# plt.xlim((0,10000))  
# plt.ylim((0,2000))
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Total Cases (t-1)')
plt.ylabel('New Cases (t)')
plt.legend(['India','US','Italy','United Kingdom','France','South Korea'])
