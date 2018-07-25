# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:46:55 2017

@author: amoody
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
# Path to ETI ET depths 
ETIpath = r"D:\TreasureValley\data\ETIdaho"
# Sites and corresponding CATS in GRASS
sites = ['BOISE WSFO AIRPORT', 'CALDWELL', 'DEER FLAT DAM',
'EMMETT 2 E','KUNA','MOUNTAIN HOME','PARMA EXP STN','PAYETTE','NAMPA']
sitecodes = [ 12, 18,26, 32, 56 , 70, 74, 75, 109 ]
   
# Get full file path
files = [os.path.join(ETIpath,'{}_{}'.format(site,'monthly.csv')) for site in sites ]

d = {}
for i,f in enumerate(files):
    d[sites[i]]=pd.read_csv( f, skiprows=( 0,1,2,3,4,5), header=0,
                    na_values='-999',parse_dates=True,index_col=0)
f = {}
f[sites[0]] = d[sites[0]]
for i in range(len(sites)-1):
    f[sites[i+1]] = pd.merge(d[sites[i+1]], f[sites[i]], left_index=True, right_index=True, how='outer')
#%%
# Extract measurement start/end dates and stats 
    
    
from datetime import datetime, date, timedelta
from matplotlib.pyplot import cm 


q = {}
m = {}

for site in sites:
    q[site] = d[site].first_valid_index()
    m[site] = d[site].last_valid_index()

sitestart = pd.DataFrame(data=q, index=[0])
sitefinish = pd.DataFrame(data=m, index=[0])
sitestart = sitestart.transpose()
sitestart['start_date'] = sitestart[0]
sitestart = sitestart.drop([0],axis=1)
sitefinish = sitefinish.transpose()
sitefinish['fin_date'] = sitefinish[0]
sitefinish = sitefinish.drop([0],axis=1)
start_fin = pd.merge(sitefinish,sitestart,
                          left_index=True, right_index=True, how='outer' )

#%%
# ------------------------------------
# Get missing timestamps from the dataframes. Make an array of site index,
# with two columns defining each valid period, if there is only one, you only
# have one column

dict = {} 
data = pd.DataFrame()       
for site in sites:
    df = d[site]
    #missing = np.array([1])
    validrows = pd.notnull( df ).all(1).nonzero()[0] 
    timestamps = df.index
    missing =[ timestamps[0]]
    for rank in range(0, len(validrows)-1):
        if validrows[rank+1] - validrows[rank]>2:
            missing = np.concatenate((missing,[timestamps[validrows[rank]] ,timestamps[validrows[rank+1]] ]) )
        #elif validrows[rank+1] - validrows[rank]==2:
        #    missing.append(validrows[rank]+1)
        
    last_idx = np.where( df.index == df.last_valid_index() )[0][0]
    missing = np.concatenate( (missing, [timestamps[last_idx]] ) )      
    dict[site] =  missing


DataRanges = pd.concat( [pd.DataFrame( np.transpose(array),columns=[k]) for k,array in dict.items()],axis=1).T
DataRanges.index.name = 'station'
DataRanges.columns = ['datestart_01','dateend_01','datestart_02','dateend_02']
DataRanges.to_csv(r'D:\TreasureValley\WinterET\data\ETIstation_dates.csv')
#[ValidIdx.loc[ key ].replace( ) for key,df in dict.items()
#%%
x2 = start_fin['fin_date'].astype(datetime).values
x1 = start_fin['start_date'].astype(datetime).values
y = np.arange(1,10)
names = start_fin.index

labs, tickloc, col = [], [], []

# create color iterator for multi-color lines in gantt chart
color=iter(cm.Dark2(np.linspace(0,1,len(y))))

plt.figure(figsize=[8,10])
fig, ax = plt.subplots()

# generate a line and line properties for each station
for i in range(len(y)):
    c=next(color)
    
    plt.hlines(i+1, x1[i], x2[i], label=y[i], color=c, linewidth=3)
    labs.append(names[i].title()+" ("+str(y[i])+")")
    tickloc.append(i+1)
    col.append(c)
plt.ylim(0,len(y)+1)
plt.yticks(tickloc, labs)
# create custom x labels
plt.xticks(np.arange(datetime(np.min(x1).year,1,1),np.max(x2)+timedelta(days=365.25),timedelta(days=365.25*3)),rotation=45)
plt.xlim(datetime(np.min(x1).year,1,1),np.max(x2)+timedelta(days=365.25))
plt.xlabel('Date')
plt.ylabel('ETIdaho station name')
plt.grid()
plt.title('ETIdaho Treasure Valley Measurement Duration')
plt.tight_layout()

#plt.savefig( './' + 'gantt.png');

#%% Describe means of sites
s = {}
for site in sites:
    s[site] = d[site].mean()
    s[site].index=s[site].index.map(int)
    
meanET=pd.DataFrame( data=s )

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))

# Plot the orbital period with horizontal boxes
# Data have to be in a different format for this to work
#sns.boxplot( x=1, y="sitename", data=a,  palette="vlag")

#%% Boxplots
f, ax = plt.subplots(figsize=(7, 6))
plt.boxplot(meanET.dropna().values,labels=meanET.columns.str[0:4],patch_artist=True)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.ylabel('mean ET for all ETI crops [ft/month]')
plt.savefig('./'+'meanET_box.png')    


#%% Timeseries of mean monthly ET

f, ax = plt.subplots(figsize=(7, 6))
for i,site in enumerate(d.keys()):
    lab=str(site)[0:4]
    plt.plot(d[site].mean(axis=1),label=lab)

plt.legend(loc='upper right',ncol=9)




