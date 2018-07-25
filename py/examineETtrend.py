"""
    parseMonthlyETI.py
    ---------------
    
    Extract ETact for landcover types from ETidaho <site>ETC_monthly.dat
    
    OUTPUT: CSV of ET timeseries for ETI crop classes. One CSV for each station processed

    @author: amoody
"""
import os, sys, re
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
#  ET Idaho data
ETIpath = r"D:/TreasureValley/data/ETIdaho/"
sys.path.append( ETIpath )

#%%
files = [f for f in os.listdir(ETIpath) if re.match(r'[0-9]+.*\.dat', f)]
d = {}
sitenames = {}
crops = {}

index = pd.date_range(dt.datetime(1900,1,1),
                      dt.datetime(2015,12,31),
                      freq='MS')

df3 = pd.DataFrame(index=index)

for site in files:   
    # Extract Site name  and crops
    with open(ETIpath + site) as infile:
        flines = infile.readlines()
        infile.close()
        namestr = re.findall(r'Results for (.*?) .Computed',flines[0])[0]
        # Capitalize
        namestr = namestr.upper()
        # Remove idiosyncracies from sites
        namestr = namestr.replace(' 1 W','')
        namestr = namestr.replace('Station','STN')
        namestr = namestr.replace('Agrimet','')
        # Remove punctuation and white spaces
        #namestr = namestr.replace(' ','_')
        namestr = namestr.replace('.','')
        sitenames[site] = namestr
    
        ## Crops ------
        # Original method that messes up crop codes
        # 2 Regex
        a= re.findall(r'(\d.*)\d*',flines[8])  
        b = re.split(r'\d{1,2}', ''.join(a))
        b = [x.strip(' ') for x in b]
          # Conversion to string adds an empty '' row, remove      
        cropnames = filter(None,b)
        cropnames = b
        crops[site]= cropnames
        # get crop codes
        cropcodes =  re.findall(r'(\d{1,2})' ,flines[8].strip(' '))           
    
    # Get ET data    
    df = pd.read_csv(ETIpath + site,
                    sep='\s+',
                    skipinitialspace=True,
                    skiprows = np.arange(0,9),
                    na_values = [-999],
                    parse_dates = { 'Date': [0, 1]},
                    infer_datetime_format = True,
                    index_col = 'Date')
    #mask = (df.index >= '1985-01-01') & (df.index <= '2015-12-31')
    # or df.loc['2000-1-1':'2015-1-1']
    #df = df.loc[mask]
    # There are some weird values the columns. Check them to see if the separator
    # was applied correctly. Fix for now.
    df = df.apply(pd.to_numeric, errors= 'coerce')
    # Extract V.Dys and ETact columns for all crop types
    #df2 = df[ df.columns[ df.columns.str.contains('ETact|V.Dys') ] ]
    # This works,too
    #df.filter(regex = 'ETact'))
    
    # Convert mm/day to mm/month
    ET = df.filter(regex = 'ETr')
    days = df.filter(regex = 'V.Dys')
    df2 = days.values * ET
    df3[namestr] = df2.rename(columns={'ETr':namestr})
    # Label with crop codes
    #df2.columns = cropcodes
    
    # Place station in dictionary
    d[namestr] = df2
#%%
    
df3=df3.loc['1985-1-1':'2015-12-1']
# Fingerprints
fig, axes = plt.subplots(4,4, figsize = (12.5,12),  sharex=True, sharey=True )

dfSiteAvg = df3.mean(axis=1)
for ax, site in zip(axes.flat,df3.columns):  
    data =  df3[site] - dfSiteAvg
    im = ax.pcolor( data.reshape(66,12) , vmin=0., vmax=50,cmap='RdYlGn_r')
    ax.set_title( df3[site].name )
    
fig.text(0.5, 0.01,'Month', ha='center' , size = 12)
fig.text(0.01, 0.5, 'Year', va='center', size = 12,rotation='vertical')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_title('mm month$^{-1}$')

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n