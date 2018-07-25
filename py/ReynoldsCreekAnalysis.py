# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:14:04 2018

@author: amoody
"""
import pandas as pd 
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt

root = r'D:/ET/eddyflux/'
sites = ['Rls','Rws']

# a date parser for older AF files (2007-2008)
# jday, hr, mm, yr
def dparse1( doy, hr, mm, yr ):       
    # Get serial/ordinal date
    sDate = dt.datetime( int(yr), 1, 1 ).toordinal() + int(doy) - 1
    ts = dt.datetime.fromordinal(sDate) + dt.timedelta(hours = int(hr), minutes = int(mm) )
    return ts
        
    
# Open shrubland: RC Wyoming Big Sagebrush
Rws = pd.read_csv(os.path.join( root, 'US-Rws_daily_aflx.csv'),
                  parse_dates=True,
                  header = 5,
                  index_col = 0,
                  na_values = -9999)

RwsMet = pd.read_csv( os.path.join( root, 'met','098c.wea.30min.csv' ),
                     parse_dates={ 'Date':[ 0, 1, 2, 3 ] }, 
                     date_parser=dparse1,
                     index_col = 'Date',
                     usecols = ['jday','hour','minute','year','ppta'])

Rws['P'] = RwsMet.resample('D').sum()

RwsMnth = Rws.loc[:,('PET_mm_dayint','ET_mm_dayint','P')].dropna().resample('M').sum()
RwsMnth['SWC_avg'] = Rws['SWC_avg'].resample('M').mean()
RwsYr = RwsMnth.resample('A').sum()
#%%
plt.style.use('bmh')

#%% Monthly ET and P
fig1,axs1 = plt.subplots(2,1)

Rws.loc[:,('PET_mm_dayint','ET_mm_dayint','P')].resample('M').sum().plot( ax = axs1[0])
axs1[0].set_xlabel('')
axs1[0].set_ylabel('mm/month')
axs1[0].legend(ncol=3)

# ET/P ratio by month 
#RwsMnth['ET_mm_dayint'].div(RwsMnth.P).plot(kind='bar',ax = axs1[1])

# Qrtly, year ends in November. Label is end of quarter
RwsSeas = Rws.loc[:,('PET_mm_dayint','ET_mm_dayint','P')].resample('BQS-NOV',closed='right',label='right').sum()
RwsSeas.loc[:,('PET_mm_dayint','ET_mm_dayint')].div(RwsSeas['P'],axis=0).plot(kind='bar',ax = axs1[1])
plt.ylabel('ET/P (mm/mm)')
axs1[1].set_xlabel('')
ax = axs1[1]
ax.set_xticklabels(RwsSeas.index.strftime('%b %y'))
plt.tight_layout()

#%% Cumulative Monthly data by year
fig2,axs2 = plt.subplots( 2,1, sharex = True)
RwsMnth.groupby(RwsMnth.index.year).cumsum().plot(ax=axs2[0])
axs2[0].set_ylabel('Cumulative ET and P')
# et/p
RwsAccum=RwsMnth.groupby(RwsMnth.index.year).cumsum()
RwsAccum['ET_mm_dayint'].div(RwsAccum['P']).plot( ax = axs2[1])
plt.xlabel('')
plt.ylabel('ET/P cumulative')
plt.tight_layout()

# Monthly mean, min, max of ET
# Rws.ET_mm_dayint.resample('M',how=['mean','min','max']).plot()

# Cumulative hourly data by year
# Rws.loc[:,('ET_mm_dayint','P')].groupby(Rws.index.year).cumsum().plot()

#%% 

fig3,axs3 = plt.subplots(2,1)

Rws.loc[:,('PET_mm_dayint','P')].resample('M').sum().plot( ax = axs3[0])
axs3[0].set_xlabel('')
axs3[0].set_ylabel('mm/month')

# ET/P ratio by month 
#RwsMnth['ET_mm_dayint'].div(RwsMnth.P).plot(kind='bar',ax = axs1[1])

# Qrtly, year ends in November. Label is end of quarter
RwsSeas = Rws.loc[:,('PET_mm_dayint','P')].resample('BQS-NOV',closed='right',label='right').sum()
RwsSeas['PET_mm_dayint'].div(RwsSeas['P']).plot(kind='bar',ax = axs3[1])
plt.ylabel('PET/P (mm/mm)')
axs3[1].set_xlabel('')
axs3[1].set_xticklabels(RwsSeas.index.strftime('%b %y'))
plt.tight_layout()


#%% SWC vs ET
plt.style.use('seaborn-ticks')
f1,ax1 = plt.subplots()

ax1.plot(Rws['ET_mm_dayint'].resample('M').sum(),':ko',label='ET')
#ax1.set_xticklabels(RwsMnth.index.strftime('%b %y'))
ax2 = ax1.twinx()
plt.legend()
ax2.plot(Rws['SWC_avg'].resample('M').mean(),label='SWC')

ax1.set_ylabel('ET (mm)')
ax2.set_ylabel('SWC (%)')
plt.legend()

#%%
plt.style.use('seaborn-ticks')
f1,ax1 = plt.subplots()
RwsAccum=RwsMnth.groupby(RwsMnth.index.year).cumsum()
RwsAccum['ET_mm_dayint'].div(RwsAccum['P'])

ax2 = ax1.twinx()

ax1.plot(RwsMnth['ET_mm_dayint'].div(RwsMnth['P']),':ko',label='ET/P')
ax2.plot(Rws['SWC_avg'].resample('M').mean(),label='SWC')
ax1.set_ylabel('ET/P (mm)')
ax2.set_ylabel('SWC (%)')
plt.legend()

#%%
plt.style.use('seaborn-ticks')
f1,ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(RwsMnth['P']- RwsMnth['ET_mm_dayint'] ,':ko')
ax1.axhline()
ax2.plot(Rws['SWC_avg'].resample('M').mean(),label='SWC')
ax1.set_ylabel('P - ET (mm)')

plt.legend()

#%%
RwsMnth['ETPratio']=RwsMnth['ET_mm_dayint'].div(RwsMnth['P'])
RwsMnth['month']=RwsMnth.index.month
RwsMnth['delSWC']=RwsMnth['SWC_avg'].diff()
g = sns.lmplot(x='ETPratio',y='delSWC',data=RwsMnth)
