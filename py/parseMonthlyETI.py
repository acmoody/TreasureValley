# -*- coding: utf-8 -*-
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
#  ET Idaho data
# DESKTOP
#
# IDWR FileShares
ETIpath = r'N:/ETIdaho_2017/Statistics'
sys.path.append( ETIpath )
ETIoutpath = r"D:/TreasureValley/data/ETIdaho/"
#%%
files = [f for f in os.listdir(ETIpath) if re.match(r'[0-9]+.*_monthly\.dat', f)]

TVstn= ['000008ETc_monthly.dat',
 '000009ETc_monthly.dat',
 '100448ETc_monthly.dat',
 '109638ETc_monthly.dat',
 '101017ETc_monthly.dat',
 '101022ETc_monthly.dat',
 '101380ETc_monthly.dat',
 '102444ETc_monthly.dat',
 '102942ETc_monthly.dat',
 '103760ETc_monthly.dat',
 '105038ETc_monthly.dat',
 '106174ETc_monthly.dat',
 '106844ETc_monthly.dat',
 '106891ETc_monthly.dat',
 '107648ETc_monthly.dat']

keepfiles = list( set(TVstn).intersection(files) )

d = {}
sitenames = {}
crops = {}
#df3=pd.DataFrame()
for site in keepfiles:   
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
        cropnames = filter(None,b)
        cropnames = b
        crops[site]= cropnames
        # get crop codes
        cropcodes =  re.findall(r'(\d{1,2})' ,flines[8].strip(' '))           
    
    # Get ET data    
    df = pd.read_csv(os.path.join(ETIpath,site),
                    sep='\s+',
                    skipinitialspace=True,
                    skiprows = np.arange(0,9),
                    na_values = [-999],
                    parse_dates = { 'Date': [0, 1]},
                    infer_datetime_format = True,
                    index_col = 'Date')
    mask = (df.index >= '1985-01-01') & (df.index <= '2015-12-31')
    # or df.loc['2000-1-1':'2015-1-1']
    df = df.loc[mask]
    # There are some weird values the columns. Check them to see if the separator
    # was applied correctly. Fix for now.
    df = df.apply(pd.to_numeric, errors= 'coerce')
    # Extract V.Dys and ETact columns for all crop types
    #df2 = df[ df.columns[ df.columns.str.contains('ETact|V.Dys') ] ]
    # This works,too
    #df.filter(regex = 'ETact'))
    
    # Convert mm/day to mm/month
    ET = df.filter(regex = 'ETact')
    days = df.filter(regex = 'V.Dys')
    df2 = days.values * ET
    #df3[namestr] = df2
    # Label with crop codes
    df2.columns = cropcodes
    
    
    # Place station in dictionary
    d[site] = df2

#%% Write dataframes to excel sheet
   
for site in d.keys():
    meta_data = pd.Series([('site: {0}'.format( sitenames[ site ] ) ),
        ('date generated: {0}'.format(str(dt.datetime.now()))),
        ('script: parseMonthlyETI.py'),
        ('units: mm/month')] )   
    crop_data = pd.DataFrame(columns=crops[site])
    #crop_data = crop_data.transpose()
    with open(ETIoutpath + str.join('',sitenames[ site ]) +
            '_monthly.csv', 'w') as fout:
        fout.write('---file metadata---\n')  
        meta_data.to_csv(fout, index=False)
        crop_data.to_csv(fout, index=False)
       # fout.write(crop_data + '\n')
        # Write in crop names somehow...
        d[site].to_csv(fout, na_rep='-999')
    


