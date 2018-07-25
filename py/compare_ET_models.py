# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:45:30 2018

@author: amoody
"""
import os
import sys
import fnmatch
import re
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
root = r'D:\TreasureValley'
datapath = os.path.join(root , r'data')

#---------
# CDL Data
# -------
index = pd.date_range(dt.datetime(1985,1,1),
                      dt.datetime(2015,12,31),
                      freq='MS')

d = pd.DataFrame(index=index,columns=['CDL'])
for root,dirs,fnames in os.walk( datapath ):
        for fname in fnmatch.filter(fnames,'ET_stats_????_??.csv'):
            # Extract year from ETI raster
            year = int( re.search('(\d{4})',fname).group(1))
            month = int( re.search('\d{4}_(\d{2})',fname).group(1))
            if (year != 1992) & (year!=2001) & (year!=2006):
                ts = dt.datetime(year,month,1)
                # String of crop raster map
                df = pd.read_csv(os.path.join(datapath,'out',fname),
                                 usecols=['label','mean' ,'coeff_var'])
              
                d.loc[ ts ] = df['mean'].div(100).round(2).mean()

#%%
# -----------
# ETI station applied to whole county cropmix
# --------------------
plt.style.use('bmh')
df2 = pd.read_csv(os.path.join(datapath,'out','ETtrad_county.csv'),
                 parse_dates = True,
                 index_col=[0]) 
df2.rename(columns={'MEAN':'CROPMIX' , 'MIN':'CROPMIX_MIN','MAX':'CROPMIX_MAX'},inplace=True)

data=df2.iloc[:,0:5].groupby(df2.index.month).mean()
data.plot()
countymin = df2.iloc[:,0:5].groupby(df2.index.month).min().min(axis=1)
countymax = df2.iloc[:,0:5].groupby(df2.index.month).max().max(axis=1)
plt.fill_between(data.index.values,
       countymin.values,
       countymax.values,
       facecolor=[0.8, 0.8, 0.8])
plt.xlabel('Month',size=12)
plt.ylabel('Average Montly ET [mm]',size=12)
#plt.savefig(os.path.join(root,'figures','MeanMonthlyCounty.png'))

#%%
data=df2.iloc[:,0:5].groupby(df2.index.month).min()
data.plot()
myerr = df2.iloc[:,0:5].groupby(df2.index.month).std()

plt.xlabel('Month',size=12)
plt.fill_between(data.index.values,
       countymin.values,
       countymax.values,
       facecolor=[0.8, 0.8, 0.8])
plt.ylabel('Average Montly ET [mm]',size=12)
#plt.savefig(os.path.join(root,'figures','MinMonthlyCounty.png'))
#%%

df2.drop(['ADA','CANYON','PAYETTE','ELMORE','GEM', 'WASHINGTON'],axis=1,inplace=True)         
              
df3=df2.merge(d,right_index=True,left_index=True)

del d
del df2
del df
#%%
# -------------
# Compare 2011 NLCD converted 2 ETI with CDL converted 2 ETI
# ------------------------
d2 = pd.DataFrame(index=index,columns=['NLCD'])
for root,dirs,fnames in os.walk( datapath ):
        for fname in fnmatch.filter(fnames,'ET_stats_*'):
            if re.search('.*NLCD',fname):
                print(fname)
                
                 # Extract year from ETI raster
                year = int( re.search('(\d{4})',fname).group(1))
                month = int( re.search('\d{4}_(\d{2})',fname).group(1))
                ts = dt.datetime(year,month,1)
                
                # String of crop raster map
                data = pd.read_csv(os.path.join(datapath,'out',fname),
                                 usecols=['label','mean' ,'coeff_var'])
              
                d2.loc[ ts ] = data['mean'].mean() / 100      

df = df3.merge(d2,right_index=True,left_index=True)
df.fillna(-999,inplace=True)
df[df==-999]=np.nan
#%%
#print(plt.style.available)
plt.style.use('bmh')
df.filter(items=['CROPMIX','CDL','NLCD']).plot()
plt.grid(linestyle=':')      
plt.legend(labels= ('Cropmix','CDL','NLCD'),ncol=3) 

#%% Scale CROPMIX by 0.53
plt.style.use('bmh')
df.loc[:,('CROPMIX','CDL','NLCD')].mul([0.53 ,1, 1]).plot()
plt.grid(linestyle=':')      
plt.legend(labels= ('Cropmix','CDL','NLCD'),ncol=3) 

#%%
df.loc[:,('CROPMIX','CDL')].mul([0.53,1]).groupby(df.index.month).mean().plot()
#%%
df.resample('A').mean().loc[:,{'CROPMIX','CDL','NLCD'}].plot()
plt.ylabel(r'Mean Annual ET [mm]')
# Compare means of 'winter' months vs all others
df.groupby(df.index.month <= 3).mean()
#%% DIEL PLOTS

data.iloc[:,[0,3,4]].plot()
plt.fill_between(data.index.values,
       data.iloc[:,1].values,
       data.iloc[:,2].values,
       facecolor=[0.8, 0.8, 0.8])
plt.xlabel('Month',size=12)
plt.ylabel('Average Montly ET [mm]',size=12)
plt.savefig(os.path.join(root,'figures','MeanMontlyAdjust.png'))

#%%
fig, axes = plt.subplots(4,4, figsize = (12.5,12),  sharex=True, sharey=True )

#%%
g=sns.jointplot("CROPMIX", "CDL", data=df3, kind="reg",
                  xlim=(0, 240), ylim=(0, 240), color=[.1,.4,.5], size=7)
import scipy.stats as st
data.apply(lambda x: st.kendalltau(x.index.year,x['CROPMIX']))

def dielplot( df, plotid = None, fill = None, xlabel='Month',ylabel=None ):
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    
    if plotid is None:
        data = df.iloc[:,[plotid]].groupby(df.index.month).mean()
    else:
        data = df.iloc[:,[plotid]].groupby(df.index.month).mean()
        
    data.plot()    
    
    return 