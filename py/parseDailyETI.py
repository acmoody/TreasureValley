# -*- coding: utf-8 -*-
"""
    parseDailyETI.py
    ---------------
    
    Extract ETr and Precip for calculating SPEI
    
    OUTPUT: CSV of ETr and Precip timeseries for ETI crop classes. 

    @author: amoody
"""
import os, sys, re
import pandas as pd
import datetime as dt
import numpy as np
#  ET Idaho data
# DESKTOP
#
# IDWR FileShares
ETIpath = r'N:/ETIdaho_2017/finished_daily'
sys.path.append( ETIpath )
ETIoutpath = r"D:/TreasureValley/data/ETIdaho/"
#%%
files = [f for f in os.listdir(ETIpath) if re.match(r'[0-9]+.*ca.dat', f)]

TVstn= ['100448',
 '109638',
 '101022',
 '101380',
 '102444',
 '102942',
 '103760',
 '105038',
 '106174',
 '106844',
 '106891',
 '107648']

keepfiles = [[ s for s in files if sub in s] for sub in TVstn]


d = {}
sitenames = {}

#df3=pd.DataFrame()
for site in keepfiles:   
    site=site[0]
    # Extract Site name  and crops
    with open(os.path.join(ETIpath,site)) as infile:
        flines = infile.readlines()
        infile.close()
        namestr = re.findall(r'Results for (.*?) .Computed',flines[0])[0]
        # Capitalize
        namestr = namestr.upper()
        # Remove idiosyncracies from sites
        namestr = namestr.replace(' 1 W','')
        namestr = namestr.replace('STATION','STN')
        namestr = namestr.replace('Agrimet','')
        namestr = namestr.replace('WEISER','WEISER 2 SE')
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
        
    
    # Get ET data    
    df = pd.read_csv(os.path.join(ETIpath,site),
                    sep='\s+',
                    skipinitialspace=True,
                    skiprows = np.arange(0,4),
                    na_values = [-999],
                    parse_dates = { 'Date': [0, 2, 3]},
                    infer_datetime_format = True,
                    index_col = 'Date')
    
    # There are some weird values the columns. Check them to see if the separator
    # was applied correctly. Fix for now.
    df = df.apply(pd.to_numeric, errors= 'coerce')
  
    
    # Convert mm/day to mm/month
    df= df.filter(items=['PMETr','Pr.mm'])
        
    # Place station in dictionary
    d[site] = df
    
#%%   
for site in d.keys():
    meta_data = pd.Series([('site: {0}'.format( sitenames[ site ] ) ),
        ('date generated: {0}'.format(str(dt.datetime.now()))),
        ('script: parseDailyETI.py'),
        ('units: mm')] )   
    outf = ETIoutpath + str.join('',sitenames[ site ]) + '_PET_daily.csv'
    outf = outf.replace(' ','_')
    with open(outf, 'w') as fout:
        fout.write('---file metadata---\n')  
        meta_data.to_csv(fout, index=False)
        d[site].to_csv(fout, na_rep='-999')
    