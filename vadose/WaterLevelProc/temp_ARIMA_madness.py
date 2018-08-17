# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:03:21 2018

@author: amoody
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
VROOT = r'D:/TreasureValley/vadose'


#---------------------------
# Import nullset of TV wells, Data and site info
# ---------------------------
f = os.path.join(VROOT,'data/groundwater/null/WellInfo_null.csv')
#Read Wellsite
df = pd.read_pickle(os.path.join(VROOT,'data/groundwater/null/WellLevels_null.pkl'))
df[['StatusName','MethodName','AgencyName']] = df[['StatusName','MethodName','AgencyName']].astype(str)
arrays = [df.filter(regex=r'WellNum').values.squeeze(),df.filter(regex='Date').values.squeeze()]
idx = pd.MultiIndex.from_arrays(arrays,names=('WellNumber','TIMESTAMP'))
df.index = idx
del arrays, idx
df = df.pop('WaterLevelBelowLSD')
df = df.unstack( level = 0)
a = df.notnull().sum().sort_values()
wells = a[a >= 200].index
df = df.resample('MS').mean()
#df = df[df.notnull().any(axis=1)]
# Find long time series

## Transform data
f = lambda x: (x - x.mean()) / x.std()
data = df[wells].apply(f,axis=0)
# Diagnoal Correlation Matrix
data_ma = data.groupby(data.index.month).mean().dropna(axis=1)
data_ma_corr = data_ma.corr(method='spearman',min_periods=7)  
## Mask for upper triangle
mask = np.zeros_like(data_ma_corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
## Mask non- significant values

pvalmat = np.zeros((data_ma_corr.shape))
for i,well1 in enumerate(data_ma):
    for j,well2 in enumerate(data_ma):
        corrtest = pearsonr(data_ma[well1],data_ma[well2])
        pvalmat[i,j] = corrtest[1]
        
mask = mask + pvalmat > 0.05
## Sort by variance and plot
wells_by_var = data_ma.var().sort_values().index
f,ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(10,240,as_cmap=True,center='dark')
sns.heatmap(data_ma_corr[wells_by_var], mask=mask, cmap =cmap,vmax = 1, vmin = -1, center = 0, square=True)

# Dynamic Time Warping
def skclustsearch( data, n_neighbors,n_clusters ):
    
    from sklearn.metrics import calinski_harabaz_score, silhouette_score
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    #d=dict(list(zip(range(0,n_clust),plt.cm.tab10(range(0,n_clust)))))
    #d=dict(zip(range(0,6),['k','c','m','y','r','g']))
    f,ax = plt.subplots(2,5,figsize=(16,4.5),tight_layout=True)
    ax = ax.ravel()
    count = 0
    
    for neighbor in n_neighbors:
        knn_graph = kneighbors_graph(data.T,neighbor,include_self=False)
        for n_clust in n_clusters: 
            model = AgglomerativeClustering(linkage = 'ward',
                                            connectivity= knn_graph,
                                            n_clusters= n_clust)
            model.fit(data.T)
            # Compute metric scores
            ss = silhouette_score(data.T,model.labels_) #-1 bad cluster, 1 highly dense clustering, 0 overlapping
            chs = calinski_harabaz_score(data.T,model.labels_)
            #plt.rc('axes',prop_cycle=cycler('color',[d[l] for l in model.labels_]))
            temp=pd.DataFrame(data=data.values,
                              index=range(1,13),
                              columns= pd.MultiIndex.from_arrays(
                                      [data.columns, model.labels_],
                                      names=['WellNumber','cat']))
            catlist = temp.columns.tolist()
            temp = temp.unstack().reset_index().rename(columns={0:'dtw','level_2':'month'})
            
            sns.tsplot(ax=ax[count],
                       legend=False,
                       data=temp,
                       time='month',
                       condition='cat',
                       value='dtw',
                       unit='WellNumber',
                       ci=[68,96])
            ax[count].set_xlabel('')
            ax[count].set_ylabel('')
            ax[count].set_title('nclust = {} neighbor = {}'.format(n_clust,neighbor),fontsize=9)
            del temp
            count += 1
    # Reset plot parameters
    #plt.rc(plt.rcParamsDefault)
    return 

def skclust(data,n_neighbors,n_clusters):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
   
    
    knn_graph = kneighbors_graph(data.T,n_neighbors,include_self=False)
    model = AgglomerativeClustering(linkage = 'ward',
                                            connectivity= knn_graph,
                                            n_clusters= n_clusters)
    model.fit(data.T)
    temp=pd.DataFrame(data=data.values,
                      index=range(1,13),
                      columns= pd.MultiIndex.from_arrays(
                              [data.columns, model.labels_],
                              names=['WellNumber','cat']))
    catlist = temp.columns.tolist()
    temp = temp.unstack().reset_index().rename(columns={0:'dtw','level_2':'month'})
    temp=temp.sort_values(by=['cat','WellNumber'])
    
    f,ax = plt.subplots(1,1,figsize=(4.5,4.5),tight_layout=True)  
    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]        
    sns.tsplot(ax=ax,
               legend=True,
               data=temp,
               time='month',
               condition='cat',
               value='dtw',
               unit='WellNumber',
               ci=[68,96],
               color=sns.xkcd_palette(colors))
    ax.set_xlabel('Month')
    ax.set_ylabel('Normalized DTW')
    ax.set_title('nclust = {} neighbor = {}'.format(n_clusters,n_neighbors),fontsize=9)
    return {'model':model,'catlist':catlist}

skclustsearch(data_ma,[5,20],[2,3,4,5,6])
d = skclust(data_ma,10,5)

# There doesn't seem to be a difference between the number of neighbors for the
# kneighbors array. variations around the mean and the normalized to unit variance
# plots parse out similar groups
cats = d['catlist']
catsdf = pd.DataFrame(index= [x[0] for x in cats],
                      data=[x[1] for x in cats],
                      columns=['group_5'])
data_ma.columns = pd.MultiIndex.from_tuples(cats,names=['WellNumber','group5'])
#data_ma.index = pd.MultiIndex.from_arrays(
#        [pd.period_range(start='2015-1-1',end='2015-12-31',freq='M').strftime('%b'),
#         ['W','W','Sp','Sp','Sp','Su','Su','Su','F','F','F','W']],names=['month','season'])
    
data_ma.T.groupby(data_ma.columns.get_level_values(1)).boxplot()
# Box plot of Months across groups? So 12 subplots (3x4) with 5 boxes each
data_ma.T.boxplot(by='group5')
# ---------------------------------------------------
# GEOPANDAS
# Use geopandas to append groups to data shapefile
# ---------------------------------------------------
import geopandas
from shapely.geometry import Point

f = r'D:/TreasureValley/vadose/data/groundwater/null/WellInfo_null.csv'
df_meta = pd.read_csv(f,header=0,infer_datetime_format=True,parse_dates=True,index_col='WellNumber')
temp=df_meta.merge(catsdf,left_index=True,right_index=True)
temp['geometry'] =temp.apply(lambda x: Point((float(x.XIDTM), float(x.YIDTM))),axis=1)


proj4str = '+proj=tmerc +lat_0=42 +lon_0=-114 +k=0.9996 +x_0=2500000 +y_0=1200000 +datum=nad83 +ellps=GRS80 +units=m +no_defs'
temp = geopandas.GeoDataFrame(temp,geometry='geometry',crs = proj4str)
f = os.path.join(VROOT,'data','groundwater','null','null_grouped.shp')
temp.to_file(f,driver='ESRI Shapefile')

#---------------
# Spectrum stuff
# ----------------
# Get a few long time series from each group and plot a spectrogram or CWT or
# multitaper
from matplotlib import gridspec
gs = gridspec.GridSpec(2,5)

fig, ax = plt.subplots(2,5,sharey = True)
ax = ax.ravel()


for cat in range(0,5):
    # Get wells in group
    w = data_ma.T.groupby(level=1).get_group(cat).index.get_level_values(0)
    # Extract longest timeseries from daily dataframe
    dtw = df[w].resample('W').mean()
    wellmax = dtw.notnull().sum().idxmax()
    ts=dtw[wellmax].dropna().resample('W').mean()
    ts=ts.interpolate(method='time')
    ax1=plt.subplot(gs[0,cat])
    ax1.plot(ts)
    ax1.set_title('Group {} - {}'.format(cat,wellmax),fontsize=9)
    dt = 1/52
    ax2=plt.subplot(gs[1,cat])
    ax2.psd(ts,2**8,1/dt)
    ax2.set_xlabel(r'Frequency, $\frac{cycles}{yr}$')
    # Write dataframe for export
    if cat == 0:
        temp = ts
    else:
        temp = pd.concat([temp,ts],axis=1)


for g in range(0,5):
    x1 = dtw.iloc[:,g].resample('M').mean()
    x1=x1.transform(lambda x: (x - x.mean())/x.std())
    x2 = spei['SPEI_monthly_3']
    dfout = pd.concat([x1,x2],axis=1,join='inner').dropna()
    dfout.plot()
    f='D:/TreasureValley/vadose/data/groundwater/wavelets/group{}.csv'.format(g)
#dfout.to_csv(f)
# Decision Tree Classifier
# Make a dataset that describes dtw by month
# i.e. data.groupby(data.index.month).describe() and use these as classifiers.
# Better yet, maybe by season? 

#
#nwells = data.shape[-1]
#nsplots = np.arange(0,5) 
#nplots = np.arange(0,nwells//5)
#mod = nwells%(nplots[-1]+1)
#gridspecs = list(itertools.product(nplots,nsplots)) + list(itertools.product([nplots[-1]+1],np.arange(0,mod)))
#
#for w in data:
#    fig,ax = plt.subplots(1,1,figsize=(8,2),tight_layout=True)
#    data[w].plot(ax=ax)
#    ax.set_title(w,fontsize=10)     
#    
#
#
#import statsmodels.api as sm
#import itertools
#import warnings
#
#p = d = q = range(0, 3)
#
#pdq = list(itertools.product(p, d, q))
##seasonal_pdq = [(x[0], x[1], x[2], 365) for x in list(itertools.product(p, d, q))]
#p = d = q = range(0,2)
#s = [12]
#seasonal_pdq = list(itertools.product(p, d, q,s))
#
#data = df[wells[0]].dropna()
## Remove data more with more than two months of 
#data = data[np.append((np.diff(data.index.to_julian_date()) <= 2 * 31),True)]
#data = data.resample('MS').mean().interpolate(method='pchip')
#
#warnings.filterwarnings("ignore")
#l = len(list(itertools.product(pdq,seasonal_pdq)))
#pdqs = list(itertools.product(pdq,seasonal_pdq))
##
#wells_min_aic = {}
#well = wells[-1]
#data = df[well].dropna()
## Remove data more with more than two months of 
#data = data[np.append((np.diff(data.index.to_julian_date()) <= 2 * 31),True)]
#data = data.resample('MS').mean().interpolate(method='bicubic')
#
## Start progress Par
#AIC_list = pd.DataFrame({}, columns=['param','param_seasonal','AIC'])
#for i,params in enumerate(pdqs):
#        try:
#            mod = sm.tsa.statespace.SARIMAX(data,
#                                            order=pdqs[i][0],
#                                            seasonal_order=pdqs[i][1],
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#            
#            results = mod.fit()
#            
#            #print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
#            temp = pd.DataFrame([[ pdqs[i][0] ,  pdqs[i][1] , results.aic ]], columns=['param','param_seasonal','AIC'])
#            AIC_list = AIC_list.append( temp, ignore_index=True)  
#            del temp           
#        except:
#            continue
#
#m = np.amin(AIC_list['AIC'].values) # Find minimum value in AIC
#l = AIC_list['AIC'].tolist().index(m) # Find index number for lowest AIC
#Min_AIC_list = AIC_list.iloc[l,:]
#wells_min_aic[well] = Min_AIC_list
##
#
#Min_AIC_list = wells_min_aic[well]
#mod = sm.tsa.statespace.SARIMAX(data,
#                                order= Min_AIC_list['param'],
#                                seasonal_order=Min_AIC_list['param_seasonal'],
#                                enforce_stationarity=False,
#                                enforce_intertibility=False)
#results=mod.fit()
#
#results.summary()
#results.plot_diagnostics()
