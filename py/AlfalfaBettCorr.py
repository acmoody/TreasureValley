# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:39:33 2018

@author: amoody
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#------------------------
# County Meta
# -----------------------
df = pd.read_csv('D:\TreasureValley\config\CountyAcres.csv',
                    sep=',',
                    parse_dates = [4],
                    index_col=[4])
meta={}
for county in pd.unique(df['county']):
    idx=df['county'] == county
    meta[county]=df[idx].drop(['acres','cat','county'],axis=1).to_dict('series')['eti_stn']
    
# -------------
# IMPORT CROP MIX FLAT FILE FOR ADA,GEM,PAYETTE, & CANYON 
# ------------

fname = r'D:\TreasureValley\data\landuse\NASS_1985_2015.xlsx'
d = pd.read_excel( fname ,sheetname='CropMixAcres' , 
                  parse_dates={ 'Date':[ 0 ] },
                  index_col = 'Date')

# PUT CROP MIXES IN DICTIONARY OF COUNTIES
# Map station list to county list
cropmix = {}
for county in meta.keys():
    idx = d['County'] == county
    cropmix[ county ] = d[idx].drop('County',axis=1)
    
    
g = sns.lmplot(x='SUGARBEETS - ACRES HARVESTED',y='Alfalfa',size=5,data=d,col='County',truncate=True)

# Log transformed values
dlog = d.drop('County',axis=1).apply(np.log)

dlog[dlog.apply(np.abs) == np.inf] = np.nan
dlog['County']=d['County']
h = sns.lmplot(x='SUGARBEETS - ACRES HARVESTED',y='Alfalfa',col='County',data=dlog,truncate=True,robust=True,n_boot=100)
df2 = dlog.loc[:,('County','Alfalfa','SUGARBEETS - ACRES HARVESTED')]
df2.dropna(inplace=True)
results = df2.groupby('County').apply(lambda x: st.linregress(x['SUGARBEETS - ACRES HARVESTED'],x['Alfalfa']))

df2.groupby('County').apply(lambda x: sns.regplot(x='SUGARBEETS - ACRES HARVESTED',y='Alfalfa',data=x,robust=True,fit_reg=True))

# Ordinary least squares
df
df3 = df2[df2['County'] == 'CANYON']
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df3['SUGARBEETS - ACRES HARVESTED'].values,df3['Alfalfa'].values)

i = sns.regplot(x='SUGARBEETS - ACRES HARVESTED',y='Alfalfa',data=df3,robust=True,fit_reg=True)
# Compare Census years 
years=np.array([1996,1997,2002,2007,2012])
census=d[d.index.year.isin(years)]
i = sns.lmplot(x='SUGARBEETS - ACRES HARVESTED',y='Alfalfa',col='County',data=census,truncate=True,n_boot=50)


# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

networks=['ADA', 'CANYON', 'ELMORE', 'GEM', 'PAYETTE']
network_pal = sns.husl_palette(5, s=.45)
network_lut = dict(zip(map(str,networks), network_pal))
network_colors = pd.Series(networks).map(network_lut)

sns.clustermap(d.corr(),center=0,cmap='vlag',
               linewidths=.75, figsize=(13, 13))
#%%
data.apply(lambda x: st.kendalltau(x.index.year,x['CROPMIX']))

#%% THEIL-SEN
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import time
estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),]

colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold'}
lw = 2
data = dlog[dlog['County']=='CANYON'].filter(items=('Alfalfa','SUGARBEETS - ACRES HARVESTED')).dropna()
data = d[d['County']=='PAYETTE'].filter(items=('Alfalfa','SUGARBEETS - ACRES HARVESTED')).dropna()
#data = d.loc[:,('Alfalfa','SUGARBEETS - ACRES HARVESTED')].groupby(d.index.year).sum().dropna()
#data = data[~data.index.isin(years)]
X = data['SUGARBEETS - ACRES HARVESTED'].dropna().values[:,np.newaxis]
y = data['Alfalfa'].values
fig = plt.figure()
#data['year']=data.index
plt.scatter(X, y, color='indigo', marker='x', s=40)
#fig = sns.FacetGrid(data = data, hue='year',palette='GnBu_d')
#fig.map(plt.scatter,'SUGARBEETS - ACRES HARVESTED','Alfalfa')
line_x = np.array([1,5e2])
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    plt.plot(line_x, y_pred, color=colors[name], linewidth=lw,
             label='%s (slope = %.2f)' % (name,  estimator.coef_[0]))

plt.axis('tight')
plt.legend(loc='upper left')
plt.title("PAYETTE" )
plt.ylabel('Alfalfa (acres)')
plt.xlabel('Sugar beets (acres)')

#%% Timeseries of Alfalfa v Sugarbeet
fig, ax = plt.subplots(2, sharex=True )

ax[0] = plt.plot(datadata['Alfalfa']
ax[1] = plt.plot(data[~data.index.isin(years)])
    
