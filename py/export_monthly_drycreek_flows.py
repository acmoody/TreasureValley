# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:56:24 2018

@author: amoody
"""

import os, re, fnmatch, sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

datapath = r'D:/TreasureValley/data/flows'
datapath = r'D:/TreasureValley/data/met'
#%%
def qparser( file ):
    df = pd.read_csv( file, 
               parse_dates = True,
               index_col = 0,
               header = 19,
               na_values = [-6999,-6934])
    return df 

#%%
files = [f for f in os.listdir(datapath) if re.match(r'Lower+.*\.csv', f)]


df = qparser(os.path.join(datapath, files[-1]))

for i,year in enumerate(files):
    if i == 0:
        df = qparser(os.path.join(datapath, files[-1]))
    else:
        df = pd.concat([ df, 
                        qparser(os.path.join(datapath,year))])
    

#%%
# 1 cfs = 28.317 L/s
df = df.div(28.317)
dfMonth=df.resample('M').mean()
dfMonth.columns = ['discharge_mean (cfs)']

fout = os.path.join(datapath,'DryCreekLowerGage_monthly_1999_2016.csv')
dfMonth.to_csv(fout,
               float_format='%3.2f',
               na_rep = '-999',
               index_label='timestamp',
               date_format='%Y-%m')
#%%
plt.style.use('bmh')

plt.plot(dfMonth)
plt.plot(df)
fig,ax = plt.subplots(2,1,sharex=True)


dfMonth.plot()
df.plot()

#%%
dfday = pd.concat([ df['Precipitation-mm'].resample('D').sum(),
                   df['AirTemperature-C'].resample('D').mean(),
                   df['AirTemperature-C'].resample('D').min(),
                   df['AirTemperature-C'].resample('D').max(),
                   df['RelativeHumidity-%'].resample('D').mean(),
                   df['RelativeHumidity-%'].resample('D').min(),
                   df['RelativeHumidity-%'].resample('D').max(),
                   df['NetRadiation-Watts/m2'].resample('D').mean(),
                   df['SolarRadiation-Watts/m2'].resample('D').sum()], axis=1 )
                   
dfday.columns = ['P','TA_avg','TA_min','TA_max','RH_avg','RH_min','RH_max',
                 'NETRAD','GLOBRAD']
sys.path.append(r'D:\code\PyETo')
from pyeto import fao

dfday['svp']=dfday.apply(lambda x: fao.mean_svp(x.loc[:,'TA_min'],x.loc[:,'TA_max']))

fao.avp_from_rhmean()




