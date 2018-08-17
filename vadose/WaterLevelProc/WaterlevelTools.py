# -*- coding: utf-8 -*-
"""
WaterlevelTools.py

@author: amoody
"""
import pandas as pd

def filterData( d, df , metadf):
    ''' Filter dataframe df with scenario dictionary d'''
    
    # Applies to all wells
    # Filter depths
    keepwells = metadf.index[((metadf['TotalDepth'] < 200) | (metadf['OpeningMin'] < 200) & (metadf['OpeningMax'] < 350)).nonzero()]
    df = df[df['WellNumber'].isin(keepwells)]
    # Remove levels influenced by pumping
    df = df[~df.StatusName.str.contains('pump|flow',case=False)]
    
    
    
    # Filter date
    df = df[(df.index >= d['date_start']) & (df.index <= d['date_end'])]
    # Force negative values to 0
    df.loc[df['WaterLevelBelowLSD'] < 0 ,'WaterLevelBelowLSD'] = 0
    
    # Remove Flowing
    # Pivot and resample. Wells are now columns
    df = pivotData(df)
    df=df.resample('D').mean()
    df=df.dropna(how='all')
    
    # Minimum records
    dropwells = df.columns[(df.notnull().sum() < d['minrec']).nonzero()]
    df.drop(columns=dropwells,inplace=True)
    
    # Merge median water level with metadata and return
    return metadf.merge(pd.DataFrame(df.median(),columns=['DTWmed']),left_index=True, right_index=True)

def pivotData( df ):
    ''' Pivot DataNull after filtering out based on scenario config'''
    return df.pivot(columns='WellNumber',values='WaterLevelBelowLSD')

def getWLalt( df ):
    df['ALTdtw'] = df['Altitude'] - df['DTWmed']
    return df

def normscore( df, varnames ):
    from scipy.stats import probplot
    from scipy.interpolate import interp1d
    
    for v in varnames:
        nscores, dataranked = probplot(df[v])[0]
        # forward interpolation
        f = interp1d(dataranked,nscores,bounds_error=False)
        #finv
        finv = interp1d(nscores,dataranked,bounds_error=False)
        nscore = df.loc[:,v].transform(f)
        df[v + '_nscore'] = nscore
    return df,finv


def toGEOEAS(df, f, name):
    with open(f,'w') as fout:
        fout.write(name + '\n')
        fout.write(str(len(df.columns)) +'\n')
        for c in df.columns:
            fout.write(c+'\n')
        df.to_csv(fout,index=False,header=False,sep=' ',float_format='%.2f')
    return 0

def getMeridianWL( f = None, fmeta=None, fdata=None, writemeta=False, writedata = False ):
    """ Read City of Meridian Wells in the format provided by Jim
    Bartolino """
    # Read in data
    if not f:
        f = (r'D:/TreasureValley/vadose/data/'+
            '2-29-2016 LONG-TERM MONITORED WATER-LEVELS in the '+
             'City of Meridian Monitoring Wells.xls') 
        print( 'Reading City of Meridian wells from default file {}'.format(f))
        
    df = pd.read_excel(f,header=1,parse_dates=True)
    # Dictionary metadata
    d={
     'MW-10/10B': {'WellNumber': '03N 01E 06DB', 'cols': (0,6),  'MP':2.6},
     'MW-14': {'WellNumber': '03N 01E 19BA', 'cols': (7,8)    ,  'MP':0  },
     'MW-16-B': {'WellNumber': '03N 01E 08ACA', 'cols': (9,18),  'MP':2.6},
     'MW-17': {'WellNumber': '03N 01E 20CB', 'cols': (19,21)  ,  'MP':1.2},
     'MW-18': {'WellNumber': '04N 01E 32CC', 'cols': (22,24)  ,  'MP':1.5},
     'MW-19': {'WellNumber': '03N 01W 03A', 'cols': (25,26)   ,  'MP':0  },
     'MW-20': {'WellNumber': '04N 01W 36DDC', 'cols': (27,31) ,  'MP':2  },
     'MW-21': {'WellNumber': '03N 01E 18BB', 'cols': (32,33)  ,  'MP':0  },
     'MW-24': {'WellNumber': '03N 01W 02AAA', 'cols': (34,38) ,  'MP':1.96},
     'MW-25': {'WellNumber': '03N 01E 29ABD', 'cols': (39,43) ,  'MP':1.5}, 
     'MW-26AB': {'WellNumber': '04N 01E 32BA ', 'cols': (44,48), 'MP':1.05},
     'MW-27': {'WellNumber': '03N 01W 10DD', 'cols': (49,53)  ,  'MP':1.2},
     'MW-28': {'WellNumber': '03N 01E 32ACD', 'cols': (54,60),   'MP':1.9},
     }
    
    # Empty dictionary to hold plucked data. Each key is a well
    d2 = {}   
    # Wrangle dataframes into a series
    for key in d.keys():      
        # Use range in dictionary to pick columns from df
        df2 = df.iloc[:,range(d[key]['cols'][0], d[key]['cols'][1]+1)]
        df2 = df2.dropna(how='all').copy()
        # Filter spces and weird text
        df2.columns=df2.columns.str.replace(r'\n.*|\..*|\s','')
        df2['Date/Time']=pd.to_datetime(df2.filter(regex='Date').squeeze())
        df2.index = df2['Date/Time']
        df2.drop(labels='Date/Time',axis=1,inplace=True)
        # Eds convention on meridian wells is opposite the typical convention
        df2 = df2.mul(-1)
        # Subtract off MP
        df2 = df2 - d[key]['MP']
        #d[key]['data']=df2
        # Put cut dataframe into dictionary
        well= d[key]['WellNumber']
        d2[well] = df2.resample('M').mean()
    
    # Form long series ALA WellSite with dataframes in dictionary
    # Option 2, discovered later concat and MELT from dictionary!
    dfconcat=pd.concat(d2.values(),axis=1,keys=d2.keys())   
    dfconcat['date']=dfconcat.index
    # Melt  
    dfmelt = dfconcat.melt(id_vars='date',var_name=['Well','Zone'])
    dfmelt['WellNumber'] =  dfmelt['Well'] + '_' + dfmelt['Zone']
    dfmelt.dropna(inplace=True)
    
    # Output
    dfout = dfmelt.loc[:,['WellNumber','date','value']]
    dfout.columns = ['WellNumber','MeasurementDate','WaterLevelBelowLSD']
    dfconcat=dfconcat.drop(labels=['date'],axis=1)
    
    # Write data to pickle (binary) or whatnot
    if writedata:
    #Append to existing treasure valley well data file. Use excel writer to 
    # write appended data. This seems to take a pretty long time, so a csv might be better
        fappend = 'D:/TreasureValley/vadose/data/null/WellLevels_null.xlsx'
        data = pd.read_excel(fappend,sheet_name='Water Levels')
        data = data[~data['WellNumber'].str.contains('baro',case=False)]
        pd.concat([data,dfout],sort=False).to_csv(fappend.replace('.xlsx','.csv'),index=False)
        pd.concat([data,dfout],sort=False).to_pickle(fappend.replace('.xlsx','.pkl'))
 
 
    
    # ---------------------------------------------------
    # Append the Site Info to the output from well site
    # Import site info output by wellsite 
    f = r'D:/TreasureValley/vadose/data/CityMeridianWellLocs_25Jun18.csv'
    dmeta = pd.read_csv(f,header=0,index_col=0).to_dict(orient='index') 
    
    # Merge dictionaries based on well number, making new entry for each zone
    meta_list = []
    for w in dfconcat.columns.get_level_values(0).unique():
        wn = [ key for key in d.keys() if d[key]['WellNumber'] == w][0]
        for z in dfconcat.loc[:,w]:
            if len(dfconcat.loc[:,w].columns) > 1: 
                a = {'WellNumber':'{}_{}'.format(w,z)}
                meta_list.append( {**dmeta[wn],**a})
            else:
                a = {'WellNumber':'{}_{}'.format(w,z)}
                meta_list.append( {**dmeta[wn],**a})
    #Write meta to file  
    if writemeta:
        if not fmeta:
            fmeta = r'D:/TreasureValley/vadose/data/null/WellInfo_null.xlsx'
            dest = pd.read_excel(fmeta,header=0)
            src = pd.DataFrame(meta_list)
            src.rename(columns={'DEM_Alt_ft':'Altitude'},inplace=True)
            # Append only like columns
            src=src.loc[:,['WellNumber', 'Altitude','TotalDepth', 'XIDTM', 'YIDTM', 'CountyName', 'TwpRge']]
            # wellscreen data
            screens = pd.read_csv(r'D:/TreasureValley/vadose/data/MeridianWells.csv',header=0,usecols=[0,1,2])
            # Merge screen data
            src = src.merge(screens,how='outer',on='WellNumber')
            pd.concat([dest,src],sort=False).to_csv(fmeta.replace('.xlsx','.csv'),index=False)
            
    return 0