# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:56:53 2018

@author: amoody
"""

import pandas as pd
import numpy as np
import datetime as dt

from grass.script import run_command
from grass.exceptions import CalledModuleError

def cleanup():
    pass

def getETImeta():
    """ County Acreage in Model """

    df = pd.read_csv('D:\TreasureValley\config\CountyAcres.csv',
                        sep=',',
                        usecols=['county','cat'],index_col= 'county')
    df.drop_duplicates(inplace=True)
    return df
    
def raster2df( raster ):
    """ Written with tricks from the notebook
    http://nbviewer.jupyter.org/github/zarch/workshop-pygrass/blob/master/03_Raster.ipynb
    """
    
    #from __future__ import (nested_scopes, generators, division, absolute_import,
    #                    with_statement, print_function, unicode_literals)
    from grass.pygrass.raster import RasterRow
 
    # Use PYGRASS API to stdout to a np/pandas array
    # null = -2147483648
    crops = RasterRow( raster )
    crops.open('r')
    df = pd.DataFrame( np.array(crops))   
    #df.replace(-2147483648,np.nan, inplace=True) 
    return df

def df2raster( df , newrastname ,mtype='CELL' ):
    """Writes a pandas dataframe to a GRASS raster"""

    from grass.pygrass.raster import numpy2raster   
    numpy2raster( df.values , mtype ,newrastname, overwrite=True)

    return 0

def getETIstationname(cat):
    """
    Return the name of the weather station given the category from the
    ETIdaho stations layer
    """
    
    df = pd.read_csv('D:\TreasureValley\config\ETI_attr.csv',
                header=0,
                usecols = ['cat','STNSNAME'], 
                index_col = 'cat' )

    return df.STNSNAME[df.index == cat].values[0]



def applyETdepths( ETs, dfc, ts, outrast):
    """
    Calculate a monthly raster of ET depth in mm/month with an 
    classified crop raster and output it to a new raster
    INPUT: Croprast, string, name of CDL or some other classified crop layer
           Outrast , string, name of new rater to make
           ETdata, dictionary, all sites
           timestamp, python datetime object
    """
    
    import time
    # Initiate dataframe for output raster
    [m ,n] = dfc.shape
    temp = pd.DataFrame( np.empty( ( m , n ) ) )
        
    # Loop through counties and make a mask with the numpy array
   
    print('----------{}---------------'.format(ts.strftime('%Y-%m')))
    
    a = time.clock()
    temp = dfc.replace(ETs.to_dict()) 
    # write new rasters. convert back to floating point
    df2raster( temp, outrast, mtype = 'CELL' )
    b=time.clock()
    return b-a


def main():
    
    run_command("g.region",region='treasurevalley_extended_30m')
    
    # Load ET timeseries
    ET = pd.read_csv(r'D:\TreasureValley\data\out\ETtrad_county.csv',
                header=0,
                parse_dates=True,
                index_col=[0],
                usecols = np.arange(0,6))
    # Dictionary of raster values for counties
    meta = {'ADA': 1, 'CANYON': 14, 'ELMORE': 20, 'GEM': 23, 'PAYETTE': 38 , 'WASHINGTON':44}
    # Rename counties to county codes
    ET.rename(columns=meta,inplace=True)
    # Load the model boundary raster
    TVcounty = raster2df('TVcounties_extint@IDlandcover')
    
    for row in ET.iterrows():
        # Extract series
        ts = row[0]
        sET = row[1]
        year = ts.year
        month = ts.month
        
        if year > 1984:
            ETrastout = 'ETdepthsCnty_%d_%02d' %( year, month)
            time = applyETdepths( sET.mul(100).astype(int), 
                                 TVcounty, 
                                 ts ,  
                                 ETrastout )
            run_command("r.support",
                        map = ETrastout,
                        title ='ET depths based on cropmix and ETIdaho 2017',
                        units = 'mm*100',
                        history= 'Produced by WinterETcorpmixrasters.py on {}'.format(dt.datetime.now().strftime('%Y-%m-%d %H:%M') ) )
            print('Trad. ET raster created for %d %02d in %2.1f seconds' %( year, month,time))  
            
    return 0
        
if __name__ == "__main__":
    main()
  