# -*- coding: utf-8 -*-
"""
Exploratory data analysis for Treasure Valley NPDES data from various
sources.

19 July 2018
"""
# 1 MGD = 3.07 AF
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
plt.style.use('bmh')
# Change workding directory
os.chdir(r'D:/TreasureValley/waterbudget')

# --------------------------------------
# 1. DMR data from Karen Burgess (EPA)
# --------------------------------------
fid = r'./NPDES/DMR_Data_Request-DNR_6-21-18.xlsx'
df = pd.read_excel(fid, sheet_name='Report 1', header=0, 
              parse_dates = True, index_col=[15])
# ------------------------------------------------------------------------
# Extract only Effluent Gross rows from 'Limits.Monitoring Location Desc'
# Remove upstream monitoring data for Caldwell
# Remove extreme values 
df = df[ df['Limits.Monitoring Location Desc' ] == 'Effluent Gross' ]
df = df[ df['Statistical Base Short Desc'].str.contains('MO AVG')]
df['DMR Value'][df['DMR Value'] > 1000] = np.nan
# Describe 
dfgrouper = df.groupby(['Permit Name','NPDES ID','Perm Feature ID'])
SiteDictKeys = dfgrouper.groups.keys()
SiteList = [x for x in SiteDictKeys]
SiteDict = dict.fromkeys(SiteDictKeys)

dfstats = dfgrouper['DMR Value'].describe()


for site in SiteList:
    s = dfgrouper['DMR Value'].get_group(site)
    # Convert to AF per month
    s = s * s.index.days_in_month * 3.07
    s.name = site
    SiteDict[site] = s  
    
# convert to data frame
temp=pd.concat(SiteDict.values(),keys=SiteDict.keys(),names=['Permit Name','NPDES ID','Perm Feature ID'])
temp = pd.DataFrame(temp)
a = temp.unstack(-1,fill_value=np.nan)
temp = pd.DataFrame(data = a.values.T,columns = a.index, index=pd.date_range(start='2000-1-1',end='2018-12-31',freq='M'))
temp.dropna(how='all', inplace=True)
df_DMR = temp
del temp

# Make some plots
sorted_cols = df_DMR.groupby(axis=1,level=0).sum().sum().sort_values(ascending=False).index
fig1 = df_DMR[sorted_cols].groupby(axis=1,level=0,sort=False).sum().plot(kind='area',cmap='tab10',alpha=.85,linewidth=1,figsize=(12,6))
ax = fig1.axes
ax.set_xlim(360,581)
ax.set_ylabel('Effluent (AF)')
plt.tight_layout()

# ----------------------------------------------------
# 2. Smaller effluent sources not provided by EPA yet
# ----------------------------------------------------

# Load into dataframe          
for i,fpath in enumerate(glob(r'.\\NPDES\\*\\out*')):
    permit = fpath.split('\\')[3]
    if i == 0:
        data = pd.read_csv(fpath,header=0,
                           parse_dates = True,
                           infer_datetime_format = True)
        data['permit_name'] = permit
    else:
        temp = pd.read_csv(fpath, header= 0,
                           parse_dates = True,
                           infer_datetime_format = True)
        temp['permit_name'] = permit
        data = pd.concat([data,temp])
            
data = data[ data['monitoring_location_desc' ] == 'Effluent Gross' ]
data.index = data['monitoring_period_end_date'].astype(np.datetime64)
#df = df[ df['Statistical Base Short Desc'].str.contains('MO AVG')]
data['dmr_value_nmbr'][data['dmr_value_nmbr'] > 1000] = np.nan
grouper = data.groupby(['permit_name','npdes_id','statistical_base_short_desc'])
#grouper['dmr_value_nmbr'].size()
# Check stat codes for permits
  
for i,name in enumerate(grouper['dmr_value_nmbr']):
    if i == 0:
        temp = pd.DataFrame(data=name[1])
        temp.columns = pd.MultiIndex.from_tuples([name[0]],
                                                 names=['permit_name','npdes_id','stat'])
    else:
        temp2 = pd.DataFrame(data=name[1])
        temp2.columns = pd.MultiIndex.from_tuples([name[0]],
                                                 names=['permit_name','npdes_id','stat'])
        temp=temp.merge(temp2,how='outer',right_index=True,left_index=True)
            
    
dfNPDES = temp * 3.07
sorted_cols = dfNPDES.groupby(axis=1,level=0).min().sum().sort_values(ascending=False).index
dfNPDES = dfNPDES.loc[:,sorted_cols]
# Get boolean of smaller municipalities
small_idx=~dfNPDES.columns.get_level_values(0).str.contains('boise|meridian|cald',case=False)
del temp

# Plot
fig2 = dfNPDES.loc[:,small_idx].groupby(axis=1,level=0,sort=False).min().plot(kind='area',cmap='tab20c_r',figsize=(12,6))
ax = fig2.axes
ax.set_xlim(521,580)
ax.set_ylabel('Effluent (AF)')
plt.tight_layout()
plt.legend(ncol=3)

#-----------------------------------------
# 3. Combine dataframes, dropping common NPDES IDs
# ----------------------------------------
dropIDs = dfNPDES.columns.levels[1].isin(df_DMR.columns.levels[1])
dfNPDES.columns.levels[1][~dropIDs]
temp = dfNPDES.drop(level=1,labels=dfNPDES.columns.levels[1][dropIDs],axis=1)
ids = temp.T.index.get_level_values(1).unique()
ids=ids[ids != 'ID0021199']
temp=temp.groupby(axis=1,level=0,sort=False).min()# Extract minimum level to get one DMR value
# Make a multiindex so that join is happy
temp.columns = pd.MultiIndex.from_arrays([temp.columns,ids, np.array(['001'] * 8)])
a=df_DMR.join(temp) # Join
sorted_cols=a.groupby(axis=1,level=0).min().sum().sort_values(ascending=False).index
a = a[sorted_cols]
fig3 = a.groupby(axis=1,level=0,sort=False).sum().plot(kind='area',cmap='tab20b',figsize=(12,6))
ax = fig3.axes
ax.set_xlim(360,581)
ax.set_ylabel('Effluent (AF)')
plt.tight_layout()
plt.legend(fontsize=9,ncol=4)

#--------------
# Mapping
# ---------------
#SiteCoords = {}
#for site in SiteList:
#    dms2dd = lambda x: x[0] + x[1]/60 + x[2]/3600
#    latstr = dfgrouper['Latitude in DMS'].get_group(site)[0]
#    lat=dms2dd([int(x) for x in re.split('\'|\s|\'|\"',latstr) if x.isdigit()])
#    lonstr = dfgrouper['Longitude in DMS'].get_group(site)[0]
#    long = -dms2dd([int(x) for x in re.split('\'|\s|\'|\"',lonstr) if x.isdigit()])
#    SiteCoords[site] = pd.Series([lat, long])
#    
#temp = pd.concat(SiteCoords.values(),keys=SiteCoords.keys(),axis=1).T
#temp.columns = ['lat','long']
#temp['geometry'] = temp.apply(lambda x: Point((float(x.lat), float(x.long))),axis=1)
#temp = geopandas.GeoDataFrame(temp,geometry='geometry',crs = 'epsg:4326')
#temp
#-------------------------------------------
# 4. Boise Geothermal
# Flow in GPM
# GPM 
# 1 gallon = 0.1337 ft3
# ------------------------------------------
f = glob('.\\NPDES\\*\\Inj*')[0]

# concatenate a few separate files. 2009 - present
d = {}
for i,f in enumerate(glob('.\\NPDES\\*\\Inj*')):
    df = pd.read_excel( f, sheet_name='Data',
                       header=19, usecols=[0,4], index_col=0,
                       parse_dates=True, infer_datetime_format=True)
    d[i] = df
    
df = pd.concat(d.values())
df.sort_index(inplace=True)   
df = df.squeeze()
df[df < 1] = 0

d2={}
for i,f in enumerate(glob('.\\NPDES\\*\\DMS\\*xls')):
    # The pandas parser doesnt like these old excel files. Go a little lower level
    
    try:
        xls = pd.ExcelFile(f)
    except:
        print('Failed to read {}, corrupt'.format(f))
    
    else:
        data = xls.parse(sheet_name='Sheet1',header=None,na_values=['Null','#N/A','-'])
        xls.close()
        data = data.dropna(how='all').reset_index(drop=True)
        # Find header by searching for DateTime
        hline = (data == 'DateTime').any(axis=1).nonzero()[0][0]
        # Find column by looking for Flow
        col = (data.iloc[hline,:] == 'RTU39_FLOW').nonzero()[0][0]
        s = pd.Series(data = data.iloc[hline+1:,col].values, 
                       index = pd.to_datetime(data.iloc[hline+1:,0], errors='coerce'),
                       dtype=float)
        
        label = np.array(f.split('\\')[-1].split('-')[0:3]).astype(int)
        label =pd.datetime(label[2],label[0],label[1]).date()
        d2[label] = s
   
df2 = pd.concat(d.values())
df2.sort_index(inplace=True)
df2.name = 'RTU39_FLOW'
    
df3=pd.concat([df,df2]).sort_index()
df3 = df3[~df3.index.duplicated(keep='first')]
df3.resample('D').sum().apply(lambda x: x * 60 * .1337 / 43560).plot(c='k',linewidth=1)
plt.ylabel('Outfall (AF)')
plt.tight_layout()
del d
#-------------------------------
# 5. ACHD Stormflow Phase 1
# --------------------------------
import xlrd
getsheets = False
cfs2cfd = lambda x: np.nansum(x)*60*60
f = r'D:/TreasureValley/waterbudget/NPDES/ACHD/2011 Phase I Flow.xlsx'

if getsheets:
    with xlrd.open_workbook(f) as wb:
        sheets = wb.sheet_names()
else:
    sheets = ['Walnut', 'Franklin', 'Production', 'Lucky', 
              'Walnut Alt', 'Koppels']
    
for i,sheet in enumerate(sheets):
    print('Parsing {}'.format(sheet))
    df = pd.read_excel(f, sheet_name=sheet,header=2,
                       parse_dates=[[0,1]],index_col=0,infer_datetime_format=True)
    # Remove bad data
     # These conditionals describe values we wish to keep
    idx = \
    ( df['Vel (fps)'] > df['Velocity Cutoff (cfs)']) | \
    ( df['Velocity Cutoff (cfs)'].isnull() )              & \
    ( df['Vel (fps)'] >= 0.)                             & \
    ( df['Flow (cfs)'] >= 0.0 )
    df = df.where(idx, np.nan)
    #df['Flow (cfs)'] = df['Flow (cfs)'].where(df['Flow (cfs)'] > 0.00,0)
    #df = df.where(df['Vel (fps)'] >= 0.00,np.nan)
    
 
    # Check frequency. If in minutes, resample to hour
    freq_inf = df.index.to_series().diff().min()
    print('    Detected a frequency of {}'.format(freq_inf) )
    if freq_inf.seconds == 60:
        print('    Resampling to hourly data with hourly means')
        df = df.resample('H').mean()        
        
    # CFS -> CFH -> AFD
    if i ==0:
        s=df['Flow (cfs)'].resample('D').apply(cfs2cfd )
        dfAF = pd.DataFrame(s)
        dfAF.columns = [sheet]
    else:
        temp = df['Flow (cfs)'].resample('D').apply(cfs2cfd)
        temp.name = sheet
        dfAF = dfAF.join(temp,how='outer')
        
fig4,ax4 = plt.subplots(1,1)
dfAF.plot(linewidth=1,ax =ax4)
ax4.set_ylabel('Flow (AF)')
fig4.tight_layout()

dfM=dfAF.resample('M').sum()
fig5, ax5 = plt.subplots(1,1)
dfM.plot(kind='bar',ax=ax5,stacked=True,sort_columns=True)
ax5.set_xticklabels(dfM.index.strftime('%Y-%b'))
ax5.set_xlabel('')
fig5.tight_layout()

dfAF = dfAF.resample('D').asfreq()
df2csv_head(dfAF,fprefix='Stormwater_ACHD',units='ft^3')
## Walnut is the largest watershed and responds to storm events not seen in 
## the Phase I report (ACHD_NPDES_ANNUAL_REPORT_2011.pdf). Looking at historical
## climate data shows some precip during these days        
#vols = dfAF.loc['2010-10-24'].values
#print((vols * ureg.acre_feet).to(ureg.cu_ft))
        
#-------------------------------
# 6. ACHD Stormflow Phase 1
# --------------------------------
## Timestamps are wonky
# Convert mean hourly cfs to AF per day
cfs2cfd = lambda x: np.nansum(x)*60*60

for i,f in enumerate(glob('.\\NPDES\\ACHD\\*Flow.csv')):
    
    site =  f.split('\\')[-1].replace('.csv','')
    print('Parsing {} from {}'.format(site,f))
    
    
    df = pd.read_csv(f,header=0,parse_dates=[[1,2]]) 
    df.set_index(keys='Date_Time',drop=True,inplace=True)
    if site == 'Whitewater Flow':
        fig6 = PlotFs(df,site)
    # Some comments exist in the cutoff column
    df['Velocity_Cutoff_950(fps)']=pd.to_numeric(df['Velocity_Cutoff_950(fps)'],errors='coerce')
    
    # These conditionals describe values we wish to keep
    idx = \
    ( df['Velocity(fps)'] > df['Velocity_Cutoff_950(fps)']) | \
    ( df['Velocity_Cutoff_950(fps)'].isnull() )              & \
    ( df['Velocity(fps)'] >= 0.)                             & \
    ( df['Flow(cfs)'] >= 0.0 )
    
    df = df.where(idx, np.nan)
    # Upsample to minutes to get all timesteps at the same level
    # Mean CFS per minute
    df = df['Flow(cfs)'].resample('1T').asfreq()
    df = df.resample('H').mean() # Hourly mean to get on the same fotting
    if i == 0:
        s=df.resample('D').apply(cfs2cfd)
        dfAF2 = pd.DataFrame(s)
        dfAF2.columns = [site]
    else:
        temp = df.resample('D').apply(cfs2cfd)
        temp.name = site
        dfAF2 = dfAF2.join(temp,how='outer') 
        # Check frequency. If in minutes, resample to hour
        df.index.to_series().diff().dropna().apply(lambda x:pd.Timedelta.total_seconds(x))
        #freq_inf = df.index.to_series().diff().max()
        #print('    Detected a frequency of {}'.format(freq_inf) )

dfAF2=dfAF2.resample('D').asfreq()
dfAF2 = dfAF2['2013-10-1':].dropna(how='all',axis=1)
# Select some dates for comparing with the NPDES Phase I B&C report(pg.37,Table 5)
dfAF2summary = pd.concat([dfAF2.loc['2016-12-4'],dfAF2.loc['2017-2-16'],dfAF2.loc['2017-3-24'],dfAF2.loc['2017-4-24']],axis=1)    

# PLOT IT
fig7,ax7 = plt.subplots(1,1,figsize=(11,4.5))    
dfAF2['2013-10-1':].div(1000).plot(ax=ax7,cmap='tab20_r',linewidth=1)
ax7t = ax7.twinx()
dfAF2['2013-10-1':].div(43560).plot(alpha=0,ax=ax7t,legend=False)
ax7t.grid()
ax7t.set_ylabel(r'$AF$')
ax7.set_ylabel(r'Thousands of $ft^3$')
fig7.tight_layout()

# WRITE FILE!
df2csv_head(dfAF2['2013-10-1':].dropna(how='all',axis=1),fprefix='Stormwater_ACHD',units=r'ft^3')

def PlotFs( df , title = None ):
    # Gant Plot of timestamps
    delt = df.index.to_series().diff().dropna().apply(lambda x:pd.Timedelta.total_seconds(x)/60)
    delt[delt > 120] = np.nan
    delt=delt.dropna()
    y = delt.unique()
    y.sort()
    
    color = plt.cm.cubehelix(np.linspace(0,1,30))
    colordict = dict(zip(y,color))
    # Start iterating through values. If they are the same, skip, if it is different,
    # plot a line
    # Initiate first timestamp
    fig,ax=plt.subplots(1,1)
    x1 = delt.index[0]
    for i in range(len(delt)):
        if delt[i] != delt[x1]:
            c = colordict[delt[i-1]]
            x2 = delt.index[i-1]  
            plt.hlines(delt[x1], x1, x2, color = c, linewidth=5)
            x1 = delt.index[i]        
    
    ax.set_ylabel('Sampling Frequency',fontsize=10)
    
    if title:
        ax.set_title(title,fontsize=10)
    plt.yticks([1,15,30,60],['1 min','15 min','30 min','60 min']) 
    ax.set_ylim(0,70)    
    plt.tight_layout()
            
    return fig

def df2csv_head(df,fprefix='data',units=None):
    ''' Write dataframe with some metadata to cwd '''
   
    outpath = '{}_{}_{}.csv'.format( fprefix, \
               df.first_valid_index().strftime('%Y%m%d'),\
               df.last_valid_index().strftime('%Y%m%d') )
    if df.index.freq:
        outpath = outpath.replace(fprefix,'{}_{}'.format(fprefix,df.index.freqstr))
    outpath = outpath.replace(' ','_')
    
    meta = pd.Series([
            ('{}'.format(fprefix)),
            ('date generated: {0}'.format(pd.datetime.now().strftime('%d %b %Y @ %H%M'))),
            ('by: {}'.format(os.environ.get("USERNAME"))),
            ('units: {}'.format(units)),
            ])
    
    with open(outpath , 'w') as fout:
        meta.to_csv(fout, index=False)
        df.to_csv(fout,float_format='%4.3f')
    