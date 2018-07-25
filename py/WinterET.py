# -*- coding: utf-8 -*-
"""

Working script to calculate ET in the Treasure Valley using the traditional 
method of ET depth * Area along with crop mix values calculated by aggregation
of NASS crop acreage values

This script will calculate ET this way for 1985 to 2015, though values are 
only intended for non-irrigation months (Nov-Feb) and for years without the
NASS Crop Data Layer (1985-2004,2006)

ETIdaho data has been converted to mm/month by parseMonthlyETI.py
Acreage data is in CountyAcres.csv


Created on Tue Jan  2 15:16:02 2018

@author: amoody
"""
import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt


#------------
# ETI CLASSES
# -----------
fname = r'D:\TreasureValley\data\cats\ETIclass.csv'
ETIclass = pd.read_csv(fname,header=0,delimiter=',',index_col='ETI_code')

#------------------------
# County Acreage
# -----------------------
df = pd.read_csv('D:\TreasureValley\config\CountyAcres.csv',
                    sep=',',
                    parse_dates = [4],
                    index_col=[4])
#%% Create dictionary of series. Each county has a series of timestamps and
# ETIdaho stations to use following the timestamp
meta={}
for county in pd.unique(df['county']):
    idx=df['county'] == county
    meta[county]=df[idx].drop(['acres','cat','county'],axis=1).to_dict('series')['eti_stn']
    
#%%
# -------------
# IMPORT CROP MIX FLAT FILE FOR ADA,GEM,PAYETTE, & CANYON 
# ------------
# Excel File
#fname = r'D:\TreasureValley\data\landuse\NASS_1985_2015.xlsx'
#d = pd.read_excel( fname ,sheetname='CropMixFlat' , 
#                  parse_dates={ 'Date':[ 0 ] },
#                  index_col = 'Date')
# Pandas pivot table export
fname = r'D:\TreasureValley\data\landuse\NASS_cropmix.csv'
d = pd.read_csv( fname ,
                parse_dates={ 'Date':[ 0 ] },
                index_col = 'Date')
#%%
# PUT CROP MIXES IN DICTIONARY OF COUNTIES
# Map station list to county list
cropmix = {}
for county in meta.keys():
    idx = d['County'] == county
    cropmix[ county ] = d[idx].drop('County',axis=1)
    
#%%-------------
# IMPORT ETIDAHO DATA
# ------------
etipath = r'D:\TreasureValley\data\ETIdaho'
stations = ['BOISE WSFO AIRPORT','CALDWELL','DEER FLAT DAM','KUNA',
                'MOUNTAIN HOME','PARMA EXP STN','PAYETTE','NAMPA','EMMETT 2 E',
                'WEISER']
files = [os.path.join(etipath,'{}_{}'.format( station ,'monthly.csv')) for station in df['eti_stn'].unique() ]

# Loop through ETI files (station files)
dET = {}
for i,station in enumerate(df['eti_stn'].unique()):
    ET = pd.read_csv(files[i] ,
                       skiprows=[0,1,2,3,4,5],
                       parse_dates = { 'TIMESTAMP' :[ 0 ] },
                       index_col = 'TIMESTAMP',
                       na_values = -999) 
  
    # Index of crop codes in ETIclass that match the ETIdaho column headers
    idx = list(set(ET.columns.astype(int)).intersection(ETIclass.index))
    # Rename dET columns
    ET.columns = ETIclass['ETI_class'][idx]
    dET[station] = ET
    
#%%%-----------
yearlist = np.arange(1985,2016)
monthlist = np.arange(1,13)

columns = meta.keys()
index = pd.date_range(dt.datetime(1985,1,1),
                      dt.datetime(2015,12,31),
                      freq='MS')

outDF = pd.DataFrame(index=index, columns=columns)

for year in yearlist:
    for month in monthlist:
        timestamp = dt.datetime(year,month,1)
        for county in meta.keys():
             # Kind of weird logic here
             # 1. Get a boolean of dates less than the timestamp
             # 2. Get last index of 'True' results, this should (hopefully)
             #    be the ETIdaho station to use
             idx = [meta[county].index <= timestamp][0].nonzero()[-1][-1]
             stn = meta[county][idx]
             
             ET = dET[stn].iloc[ dET[stn].index == timestamp].T
             mix = cropmix[ county ].iloc[ cropmix[ county ].index.year == timestamp.year].T
             # Rename the column (datetime) so that pd.DataFrame.mul works
             mix.rename(columns={dt.datetime(year,1,1):timestamp},inplace=True)
             
             # Multiply the right crops together to get the mean ET depth! 
             # Output sum to something
             outDF.loc[ timestamp , county] = ET.mul(mix).sum().values[0]

#%%
outDF['MEAN']=outDF.iloc[:,0:4].mean(axis=1)
outDF['MIN']=outDF.iloc[:,0:4].min(axis=1)
outDF['MAX']=outDF.iloc[:,0:4].max(axis=1)             

outDF.to_csv(r'D:\TreasureValley\data\out\ETtrad_county.csv',
             float_format = '%0.2f')             
#%% 
             
def pltfingerprints( df, clrmap = 'Spectral' ):
    
    # ------
    # Plot fingerprint plots of monthly ET
    # -----
    # Rehape each site into an array with R=years, C = months
    maxval= np.ceil(df.max().max())
    minval = np.floor(df.min().min())
    sites = df.columns 
    fig, axes = plt.subplots(2,3, figsize = (12,7),  sharex=True, sharey=True )
    yearticks = pd.date_range(df.index[0],df.index[-1],freq='A-JAN').strftime('%Y')
    monthticks = pd.date_range(dt.datetime(1970,1,1),periods=12,freq='M').strftime('%b')
    
    for ax, site in zip(axes.flat,sites):
        im = ax.pcolor( df[site].values.astype(float).reshape(31,12) , 
                       vmin=minval, 
                       vmax=maxval,
                       cmap=clrmap)
        ax.set_title( df[site].name )
        ax.set_yticklabels(yearticks[[ 0,  5, 10, 15, 20, 25, 30]])
        ax.set_xticks(np.arange(0,14,2))
        ax.set_xticklabels(monthticks[np.arange(0,11,2)])
    
    fig.text(0.5, 0.01,'Month', ha='center' , size = 12)
    fig.text(0.01, 0.5, 'Year', va='center', size = 12,rotation='vertical')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_title('mm month$^{-1}$')



#%%
plt.close()
pltfingerprints( outDF )

# Plot minus Alfalfa 
ref = dET['BOISE WSFO AIRPORT']['Alfalfa Hay - peak (no cutting effects )']
pltfingerprints(outDF.sub(ref,axis=0))
