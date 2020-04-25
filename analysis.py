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

sns.set()

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
for country in ['India']:
    daily_t = df_daily.loc[country].rolling(7).mean()
    active_t_1 = df_total.loc[country][:-1].rolling(7).mean()
    #plt.plot(active_t_1, daily_t)
    #, scatter_kws={"s": 8},order=2,truncate=True)
    plt.xlim(left=100,right=10**5)
    sns.regplot(x=active_t_1, y=daily_t, line_kws={'alpha':0.8},scatter=True,order=5,truncate=False)
    sns.lineplot(x=active_t_1, y=daily_t)
# plt.xlim((0,10000))  
# plt.ylim((0,2000))
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Total Cases (t-1)')
plt.ylabel('New Cases (t)')
plt.xscale('log')
plt.yscale('log')

plt.legend(['India'])

fig = plt.figure()
for country in ['India']:
    daily_t_a = df_daily.loc[country,'4/3/20':]
    active_t_1_a = df_total.loc[country,'4/2/20':][:-1]
    #plt.plot(active_t_1, daily_t)
    # sns.lineplot(x=active_t_1_a, y=daily_t_a)#, scatter_kws={"s": 8},order=4,truncate=False)
    
    daily_t = df_daily.loc[country]
    active_t_1 = df_total.loc[country]
    #plt.plot(active_t_1, daily_t)
    sns.lineplot(x=active_t_1, y=daily_t)#, scatter_kws={"s": 8},order=2,truncate=False)
    
    daily_t = df_daily.loc[country,'3/16/20':'4/2/20']
    active_t_1 = df_total.loc[country,'3/15/20':'4/1/20']
    
    sns.regplot(x=active_t_1_a, y=daily_t_a, scatter=False,line_kws={'alpha':0.5}, order=2,truncate=True,ci=67)
    sns.regplot(x=active_t_1, y=daily_t,scatter=False,line_kws={'alpha':0.6}, order=2,truncate=False,ci=67)
    
    plt.axvline(x=df_total.loc[country,'3/22/20'],alpha=0.2,c='black')
    plt.axvline(x=df_total.loc[country,'3/24/20'],alpha=0.2,c='black')
    plt.axvline(x=df_total.loc[country,'4/2/20'],alpha=0.2,c='black')
    
    plt.text(x=df_total.loc[country,'3/22/20']-130,y=6,s='Janta Curfew', fontsize=8.7)
    plt.text(x=df_total.loc[country,'3/24/20']+25,y=12,s='Lockdown declared', fontsize=8.7)
    plt.text(x=df_total.loc[country,'4/2/20']+20,y=18,s='10 days since the lockdown', fontsize=8.7)
    # plt.text(x=df_total.loc[country,'4/23/20']-130,y=16,s='1 month since the lockdown', fontsize=8.7)
    
# plt.xlim((0,10000))  
# plt.ylim((0,2000))
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(left=100)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Cases (t-1)')
plt.ylabel('New Cases (t)')

from datetime import datetime

# plt.legend(['India (10 days after Lockdown)','India (before lockdown to 10 days after lockdown)'])

df_total_india = pd.DataFrame(df_total.transpose()['India'])
df_total_india['Date'] = list(map(lambda x: datetime.strptime(str(x),'%m/%d/%y') ,df_total_india.index))
df_total_india.set_index('Date',inplace=True)


df_oxfam = pd.read_csv('OxCGRT_Download_200420_151611_Full.csv')
df_oxfam_trimmed = df_oxfam[['CountryName', 'CountryCode', 'Date', 'S1_School closing', 'S2_Workplace closing', 'S3_Cancel public events','S4_Close public transport','S5_Public information campaigns', 'S6_Restrictions on internal movement','S7_International travel controls', 'S8_Fiscal measures','S9_Monetary measures','S10_Emergency investment in health care','S11_Investment in Vaccines','S12_Testing framework','S13_Contact tracing', 'ConfirmedCases','ConfirmedDeaths', 'StringencyIndex', 'StringencyIndexForDisplay']]
df_oxfam_trimmed['Date'] = list(map(lambda x: datetime.strptime(str(x),'%Y%m%d'), df_oxfam_trimmed['Date']  ))
df_oxfam_trimmed_india = df_oxfam_trimmed.loc[df_oxfam_trimmed['CountryName']=='India']
df_oxfam_trimmed_india.set_index('Date',inplace=True)

df_india = df_total_india.join(df_oxfam_trimmed_india,how='inner')
# df_india.set_index('Date',inplace=True)
df_india['Date'] = df_india.index
df_india.rename(columns={'India':'Total Cases'},inplace=True)

# ax = df_india.plot(x="Date", y="Total Cases", legend=False)
# plt.ylabel('Confirmed Cases')
# plt.yscale('log')
# ax2 = ax.twinx()
# df_india.plot(x="Date", y="StringencyIndex", ax=ax2, legend=False, color="r")
# plt.ylabel('Stringency of non-pharmacheutical intervention')
# ax.figure.legend(['Confirmed Cases','Stringency of non-pharmacheutical intervention'],loc='upper center',ncol=2)


# l1 = plt.plot(df_india.index,df_india['Total Cases'])
# # plt.legend(['Confirmed Cases'])
# plt.yscale('log')
# plt.xticks(rotation=65)
# plt.ylabel('Confirmed Cases')
# ax2 = plt.twinx()
# sns.lineplot(data=df_india['StringencyIndex'],ax=ax2,c='orange')
# plt.ylabel('Stringency of non-pharmacheutical intervention')
# # plt.legend(['Stringency of non-pharmacheutical intervention'])

# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.api import VAR

series = df_india[['Total Cases','StringencyIndexForDisplay']].rolling(7).mean()
series['New Cases'] = series['Total Cases'].diff()
series['Growth Rate of New Cases'] = series['New Cases'].diff()


# # model = VAR(series[['Total Cases','New Cases']].loc[datetime.strptime('20/2/2','%y/%m/%d'):],exog=series[['StringencyIndexForDisplay']].loc[datetime.strptime('20/2/2','%y/%m/%d'):])
# # model_a = ARIMA(endog=np.log(df_india['Total Cases']).loc[datetime.strptime('20/4/2','%y/%m/%d'):], order=(7,1,0))
# model = VAR(series[['Growth Rate of New Cases','StringencyIndexForDisplay']].loc[datetime.strptime('20/2/2','%y/%m/%d'):])
# model_fit = model.fit(maxlags=14)

# model = ARIMA(series['Growth Rate of New Cases'].loc[datetime.strptime('20/3/2','%y/%m/%d'):],exog=series['StringencyIndexForDisplay'].loc[datetime.strptime('20/3/2','%y/%m/%d'):],order=(10,1,0))
# model_fit_arima = model.fit(disp=0)
# model_fit_arima.plot_predict(3,75,exog=np.array([100,100,100,100,100,100,100,100,100,100,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,30,30,30,30,30,30,30,30,30,30]))
# model_fit_arima.plot_predict(3,75,exog=np.array([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]))


# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# residuals.plot(kind='kde')

# from statsmodels.tsa.statespace.varmax import VARMAX
# model = VARMAX(series[['New Cases','StringencyIndexForDisplay']].loc[datetime.strptime('20/3/2','%y/%m/%d'):],exog=series['Total Cases'].loc[datetime.strptime('20/3/2','%y/%m/%d'):])
# model_fit = model.fit(maxlags=14)

def seir_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for i in t[1:]:
        next_S = S[-1] - (rho[int(i)]*beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (rho[int(i)]*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T

# Define parameters
def loss(param):
    t_max = len(series['StringencyIndexForDisplay'].to_numpy())-1
    dt = 1
    t = np.arange(33, t_max)
    N = 1.3e+9
    I0 = 3*series.iloc[0]['Total Cases']/N
    R0 = 0
    S0 = 1 - I0
    E0 = 5*I0
    init_vals = S0, E0, I0, R0
    alpha = param[0]#0.1
    beta = param[1]#1.75
    gamma = param[2]#0.5
    cons = param[3]#2
    rho = 1 - (series['StringencyIndexForDisplay'].to_numpy()/(cons*100))
    params = alpha, beta, gamma, rho
    # Run simulation
    results = seir_model_with_soc_dist(init_vals, params, t)
    truth = df_active.loc['India'].to_numpy()[34:]
    predicted = (results[:,2])*N
    error = np.linalg.norm(np.log(truth+1)-np.log(predicted+1),ord=2)
    return error

import scipy.optimize as opt

res = opt.minimize( loss, x0 = np.array([0.2608 , 2.2513 , 1.3735 , 5]), method='COBYLA')
param = np.array([0.3808 , 2.313 , 1.2735 , 5])
future_1 = [100 for i in range(120)]
future_2 = [100 for i in range(12)] + [75 for i in range(108)]
future_3 = [100 for i in range(12)] + [50 for i in range(108)]
future_4 = [100 for i in range(12)] + [0 for i in range(108)]
stringency_1 = series['StringencyIndexForDisplay'].to_list() + future_1
stringency_2 = series['StringencyIndexForDisplay'].to_list() + future_2
stringency_3 = series['StringencyIndexForDisplay'].to_list() + future_3
stringency_4 = series['StringencyIndexForDisplay'].to_list() + future_4


t_max = len(stringency_1)-1
dt = 1
t = np.arange(33, t_max)
N = 1.3e+8
I0 = series.iloc[33]['Total Cases']/N
R0 = 0
S0 = 1 - I0
E0 = 3*I0
init_vals = S0, E0, I0, R0
alpha = param[0]#0.1
beta = param[1]#1.75
gamma = param[2]#0.5
cons = param[3]#2
rho_1 = 1 - (stringency_1/(cons*100))
rho_2 = 1 - (stringency_2/(cons*100))
rho_3 = 1 - (stringency_3/(cons*100))
rho_4 = 1 - (stringency_4/(cons*100))

params_1 = alpha, beta, gamma, rho_1
params_2 = alpha, beta, gamma, rho_2
params_3 = alpha, beta, gamma, rho_3
params_4 = alpha, beta, gamma, rho_4
# Run simulation
results_1 = seir_model_with_soc_dist(init_vals, params_1, t)
results_2 = seir_model_with_soc_dist(init_vals, params_2, t)
results_3 = seir_model_with_soc_dist(init_vals, params_3, t)
results_4 = seir_model_with_soc_dist(init_vals, params_4, t)

truth = df_total.loc['India'].to_numpy()
predicted_1 = np.cumsum(results_1[:,2])
predicted_2 = np.cumsum(results_2[:,2])
predicted_3 = np.cumsum(results_3[:,2])
predicted_4 = np.cumsum(results_4[:,2])

plt.plot(truth)
plt.plot(t,predicted_1*N)
plt.plot(t,predicted_2*N)
plt.plot(t,predicted_3*N)
plt.plot(t,predicted_4*N)
plt.yscale('log')
