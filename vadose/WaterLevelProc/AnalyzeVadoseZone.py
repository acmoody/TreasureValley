# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(r'D:/TreasureValley/vadose/WaterLevelProc')
from WaterlevelTools import filterData, getWLalt, normscore

VROOT = r'D:/TreasureValley/vadose'

#---------------------------
# Import nullset of TV wells, Data and site info
# ---------------------------
f = os.path.join(VROOT, 'data/groundwater/null/WellInfo_null.csv')
#Read Wellsite
WellsNull = pd.read_csv(f,
                        header=0,
                        infer_datetime_format=True, parse_dates=True,
                        index_col='WellNumber')
if os.path.isfile(os.path.join(VROOT,'data/groundwater/null/WellLevels_null.pkl')):
    DataNull = pd.read_pickle(os.path.join(VROOT,'data/null/WellLevels_null.pkl'))
    DataNull[['StatusName','MethodName','AgencyName']] = DataNull[['StatusName', 'MethodName', 'AgencyName']].astype(str)
    DataNull.index = DataNull.pop('MeasurementDate')
    #DataNull = DataNull.unstack( level = 0)
    #df = df.resample('D').mean()
    #df = df[df.notnull().any(axis=1)]
else:
    print('Pickle file does not exist. Create nullset pickle file to speed up reading')
    
# ---------------------------------------------------------------------------
# Define a dictionary of 'scenarios' or conditions for selecting wells
# ------------------------------------------------------------------------

scenarios = [{'description': 'Median1986_2018',
              'minrec'     : 5, 
              'date_start' : pd.datetime(1980,1,1),
              'date_end'   : pd.datetime(2018,1,1),
              'data'       : (),
              'maxdepth'   : 200},
            
            {'description' : 'WinterWY2016',
             'minrec'      : 1,
             'date_start'  : pd.datetime(2015,11,1),
             'date_end'    : pd.datetime(2016,4,1),
             'data'        : (),
             'maxdepth'    : 200 } ,
            
            {'description' : 'null',
             'minrec'      : 1,
             'date_start'  : pd.datetime(1910,1,1),
             'date_end'    : pd.datetime(2018,12,1),
             'data'        : (),
             'maxdepth'    : 1000}
            ]
# Populate well data
wldata = [filterData(s,DataNull, WellsNull) for i,s in enumerate(scenarios)]
wldata = [getWLalt( df ) for df in wldata]
wldata = [normscore(df,['DTWmed','ALTdtw'])[0] for df in wldata]
for i, df in enumerate(wldata):
    scenarios[i]['data'] = df
#Attempt to populate with dictionary/list comprehension
#data = [ {key: filterData(s,DataNull) for key,val in s.items() if key == 'data' } for s in scenarios]


#-------------------------------------
# Export some files
import pickle

# Make directories for data 
[os.mkdir(os.path.join(VROOT,'data/groundwater',s['description'])) 
for s in scenarios 
if not os.path.isdir(os.path.join(VROOT,'data/groundwater',s['description']))]

for s in scenarios:
    fdir = os.path.join(VROOT,r'data/groundwater',s['description'])
    with open( fdir + '\\' + s['description'] + '.pkl','wb') as handle:
        pickle.dump(s ,handle)

geoeas=False
shapef = False
if geoeas:
    dfout = wldata[0].filter(regex='IDTM|DTW|dtw')
    toGEOEAS(dfout.replace(np.nan,-999), r'D:/TreasureValley/vadose/data/groundwater/WL_all2.dat','Water level, 1980-present')
    
elif shapef:
# ---------- Export to shapefile
    from shapely.geometry import Point
    import geopandas
    # Name and directory creation
    SiteInfo = scenarios[0]['data']
    desc = scenarios[0]['description']
    f = 'TV_watertable_{}.shp'.format(desc)
    fdir = os.path.join(VROOT,'data',desc)
    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    # Geospatial
    SiteInfo['geometry'] = SiteInfo.apply(lambda x: Point((float(x.XIDTM), float(x.YIDTM), float(x.ALTdtw))),axis=1)
    proj4str = '+proj=tmerc +lat_0=42 +lon_0=-114 +k=0.9996 +x_0=2500000 +y_0=1200000 +datum=nad83 +ellps=GRS80 +units=m +no_defs'
    SiteInfoGeo = geopandas.GeoDataFrame(SiteInfo,geometry='geometry',crs = proj4str)
    SiteInfoGeo.loc[:,~SiteInfoGeo.columns.str.contains('Date')].to_file(
            os.path.join(fdir,f),driver='ESRI Shapefile')


#------Various Data Description queries
# 753 Wells in study area
# Wells without depth or opening data
WellsNull.filter(regex='TotalDepth|Opening').isnull().all(axis=1).sum()
# Wells deeper than 200 ft
((WellsNull['TotalDepth'] > 200) | (WellsNull['OpeningMin'] > 200)).sum()