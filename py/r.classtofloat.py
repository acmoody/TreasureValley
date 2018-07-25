"""
Created on Thu Jan 11 12:56:22 2018

@author: amoody
"""


"""
First attempt to apply the ETIdaho crop-specific ET depths to the reclassified CDL raster (CDL 
classified to ETI equivalent codes)

"""

import os, re, fnmatch, sys
import pandas as pd
import numpy as np
import datetime as dt
#import tempfile
from grass.script import parser, run_command, parse_command
import grass.script as gscript
from grass.exceptions import CalledModuleError

def cleanup():
    pass

def getETImeta():

    #------------------------
    # County Acreage in Model
    # -----------------------
    df = pd.read_csv('D:\TreasureValley\WinterET\data\CountyAcres.csv',
                        sep=',',
                        index_col = 1)
    return(df)
    
def raster2df( raster ):
    
    """
    Written with tricks from the notebook
    http://nbviewer.jupyter.org/github/zarch/workshop-pygrass/blob/master/03_Raster.ipynb
    
    """
    from __future__ import (nested_scopes, generators, division, absolute_import,
                        with_statement, print_function, unicode_literals)
    from grass.pygrass.raster import RasterRow
 
    
    # Use PYGRASS API to stdout to a np/pandas array
    # null = -2147483648
    crops = RasterRow( raster )
    crops.open('r')
    df = pd.read_csv( np.array(crops))   
    df.replace(-2147483648,np.nan, inplace=True)
    
    
    return df

def df2raster( df , newrastname ):
    """
    Writes a pandas dataframe to a GRASS raster
    
    """
    new = newrastname
    new = RasterRow('newET')
    new.open('w',overwrite=True)
    for row in df.iterrows():
        new.put_row( row )
    
    new.close()


def main():
    # Path to ETI crop rasters
    rastpath= r"C:\Users\amoody\Documents\grassdata\idaho_idtm83\IDlandcover\cell"
    # Path to ETI ET depths 
    ETIpath = r"D:\TreasureValley\WinterET\TV_ETIdaho_Stations"
    # Sites and corresponding CATS in GRASS
    
    # ----------------
    # Site Metadata: County, county area, ETI station
    meta = getETImeta()
    # FIXME This assumes all station have data available at all time
    stations = meta['station'].values
   
    # Get full file path
    files = [os.path.join(ETIpath,'{}_{}'.format(site,'monthly.csv')) for site in stations ]

    # Open files into a dictionary of structured arrays
    d= {} 
    
    for i,site in enumerate(stations):
        df = pd.read_csv(files[i],
                         skiprows=[0,1,2,3,4,5],
                         parse_dates=True,
                         index_col='Date',
                         na_values=-999)
        
        d[site] = df
    
            
    # ----------
    # GRASS STUFF
    # -----------
    # Set region
    run_command("g.region",
                res=30,
                vector="TV_Bound_2010@TV")
    # Mask with TV boundary
    run_command("r.mask",
                overwrite = True,
                vector = "TV_Bound_2010@TV")
    
    # TODO
    # TESTING PANDAS RECLASS
    countyrast = pd.DataFrame(np.random.randint(1,4,size=(5,5)))
    # DUMMY RASTER
    croprast = pd.DataFrame(np.random.randint(1,57,size=(5,5)))
    # ETI data
    df2.columns=df2.columns.astype(int)
    ETIdf = df2
    # index the transposed ETIdaho data from month whatever
    #  df2.iloc[ date, allcrops]
    # DUMMY OUTPUT RASTER OF ET DEPTHS
    a=croprast.replace(ETIdf.iloc[0,:].to_dict())
    
    # Masks 
    croprast.mask(countyrast == 1)
    
    
    
    # Loop ETI  years
 
    for root,dirs,fnames in os.walk(rastpath):
        for fname in fnmatch.filter(fnames,'ETI_Idaho*'):
            ETIrast = fname  # String of raster map
            # Extract year from ETI raster
            year = int( re.search('(\d{4})',ETIrast).group(1)  )
            if year >= 2009 and year <= 2012:         
                # If the year is 2009 - 2012, loop
                # loop months and make a raster for each month
                for month in range(1,13):
                    # Output raster
                    ETrast = 'ETdepths_%s_%02d' %( year, month)
                    
                    
                    


    return 0
        
if __name__ == "__main__":
    main()
  