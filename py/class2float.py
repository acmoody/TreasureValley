
"""
Created on Thu Jan 11 12:56:22 2018

@author: amoody
"""


import os, re, fnmatch, sys
import pandas as pd
import numpy as np
import datetime as dt
#import tempfile
from grass.script import parser, run_command, parse_command

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
    return df
    
def raster2df( raster ):
    
    """
    Written with tricks from the notebook
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
    """
    Writes a pandas dataframe to a GRASS raster
    
    """
    from grass.pygrass.raster import numpy2raster
    
    numpy2raster( df.values , mtype ,newrastname, overwrite=True)

    return 0


def makeVoronoiRaster( timestamp ):
     
    
    # are available during the month
    ts = pd.read_csv('D:\TreasureValley\WinterET\data\ETIstation_dates.csv',
                        index_col = 0,
                        parse_dates=[1,2,3,4],
                        infer_datetime_format=True)
    # Open previous ignore str
    sqlfile = open(r'D:\TreasureValley\procconfig\sqlstr.txt','r')
    previous_ignore_str = sqlfile.readline()
    sqlfile.close()
    
    # MAKE SQL Query String
    # Add a boolean column signifying if the station is present during this
    # month
    a = pd.Series( 
            ( ( ts['datestart_01'] <= timestamp ) & ( ts['dateend_01'] >= timestamp ) ) 
            | ((ts['datestart_02'] <= timestamp) & (ts['dateend_02'] >= timestamp   ) )  
            )
    print(a[a])
    ignorestr = []
    ignorestr = ''.join(" AND STNSNAME != '%s'" %(x) for x in a.index if ~a[x])

    doVor=True
    if ignorestr == previous_ignore_str:
        print(' --- Previous Voronoi Polygons are equivalent --- ')
        doVor = False 
        wheresuffix = previous_ignore_str
    elif ignorestr:
        wheresuffix = ignorestr
        print(' --- Supposedly making a new Voronoi ---')
    elif not ignorestr:    
        wheresuffix = ''
      
    wherestr= ("NOT(LAT_DD < 44.567 AND LONG_DD < -114.83 AND"+
    " STNSNAME != 'Parma' AND STNSNAME != 'REYNOLDS'"+
    " AND STNSNAME != 'BOISE 7 N' AND STNSNAME != 'ARROWROCK DAM'"+
    " AND STNSNAME != 'ANDERSON DAM 1 SW' {})" ).format(wheresuffix)
    
    # Write the string to a file for referencing later
    sqlfile = open(r'D:\TreasureValley\procconfig\sqlstr.txt','w')
    sqlfile.write(ignorestr)
    sqlfile.close()
    
    
    if doVor: 
        print('hi from inside the loop')
        # Expand region to allow for more voronoi areas
        run_command('g.region',n=1496318,s=1326789,e=2469728,w=2231411)
        # Copy point vector of all ETI stations
        run_command('g.copy', vector = '{},{}'.format('ETIdahoStations','tempETI'),
                    overwrite=True)
        # Delete unwanted points
        run_command('v.edit', 
                    map = 'tempETI@IDlandcover', 
                    tool ='delete',
                    where = wherestr,
                    quiet=True)
             
        # Make new temporary voronoi vector
        run_command('v.voronoi',
                    overwrite = True,
                    input = 'tempETI',
                    output = 'tempVOR',
                    quiet=True)
        
        # Delete the old voronoi raster mask. This has caused issues in the past
    #    run_command('g.remove',
    #                type ='raster',
    #                name='tempVORRAST')
        
        # Make raster mask 
        run_command('v.to.rast', 
                    input='tempVOR@IDlandcover',
                    output='tempVORRAST@IDlandcover',
                    use='cat',
                    label_column='STNSNAME',
                    overwrite=True,
                    quiet=True)
    return 0

def getETIstationname(cat):
    """
    Return the name of the weather station given the category from the
    ETIdaho stations layer
    """
    
    df = pd.read_csv('D:\TreasureValley\WinterET\data\ETItable\ETItable.csv',
                header=0,
                usecols = ['cat','STNSNAME'], 
                index_col = 'cat' )

    return df.STNSNAME[df.index == cat].values[0]



def applyETdepths( croprast , outrast, ETdata , timestamp, desertdf):
    """
    Calculate a monthly raster of ET depth in mm/month with an 
    classified crop raster and output it to a new raster
    INPUT: Croprast, string, name of CDL or some other classified crop layer
           Outrast , string, name of new rater to make
           ETdata, dictionary, all sites
           timestamp, python datetime object
    """
    
    import difflib
    import time
    
    # Raster with values of cat from voronoi shapefile
    stationmask =  raster2df('tempVORRAST')
    # Classified crop raster
    croprast = raster2df(croprast)
    # Get unique station codes in raster
    # Can't shake off the nans!!
    stationmask.replace(np.nan,0,inplace=True)
    stncodes =  np.unique(stationmask[stationmask > 0])
    
    # Initiate dataframe for output raster
    [m ,n] = croprast.shape
    temp = pd.DataFrame( np.empty( ( m , n ) ) )
    # Loop through stations and make a mask with the numpy array
    a = time.clock()
    stncodes=stncodes[~np.isnan(stncodes)]
    
    print('----------{}---------------'.format(timestamp.strftime('%Y-%m')))
    for i,cat in enumerate(stncodes):
        # Deal with mismatch between flat file station names and shapefile
        # station names          
        name = difflib.get_close_matches( 
                getETIstationname(cat).upper(),
                ETdata.keys(), 
                cutoff=0.5)[0]  
        print('    Applying ET depths for {} (cat {})'.format(name,cat))
 
        # Get dataframe for this station
        ET = ETdata[name]
        # Convert columns of crop codes to integers. This cuts it in about half! 
        ET.columns=ET.columns.astype(int)
        # Mask according to the ETI station
        # Make new raster to hold ET values as we step through masks
        try:
            temp[stationmask == cat] = croprast.replace(ET.loc[ timestamp ].to_dict())
        except:
            print('Station with cat = {} is giving you trouble'.format(cat))

        # Mean of ETI grasses 47,48,49
        fixdesertgrass = ET.loc[ timestamp ][ [47, 48, 49] ].mean() * 100
        temp[stationmask==cat].where( (desertdf==1) ) == fixdesertgrass
        
    run_command("g.region",
                res=30,
                vector="TV_Bound_2010@TV")    
    # write new rasters. convert back to floating point
    df2raster( temp, outrast, mtype = 'CELL' )
    b=time.clock()
    return b-a

def main():   
    #------------
    # ETIdaho
    #-------------
    # Path to ETI crop rasters
    rastpath= r"C:\Users\amoody\Documents\grassdata\idaho_idtm83\IDlandcover\cell"
    # Path to ETI ET depths 
    ETIpath = r"D:\TreasureValley\WinterET\TV_ETIdaho_Stations"
    # Site Metadata: County, county area, ETI station
    meta = getETImeta()
    stations = ['BOISE WSFO AIRPORT','CALDWELL','DEER FLAT DAM','KUNA',
                'MOUNTAIN HOME','PARMA EXP STN','PAYETTE','NAMPA','EMMETT 2 E']
    # Get full file path for ETI Data
    files = [os.path.join(ETIpath,'{}_{}'.format(site,'monthly.csv')) for site in stations ]
    # Open files into a dictionary of structured arrays
    d= {}     
    for i,site in enumerate(stations):
        df = pd.read_csv(files[i],
                         skiprows=[0,1,2,3,4,5],
                         parse_dates=True,
                         index_col='Date')
        # Create integer data to make this faster
        d[site] = df.multiply(100).astype(int)
               
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
    # CALCULATE ET WITH CDL
    # Loop ETI  years
   
    for root,dirs,fnames in os.walk(rastpath):
        for fname in fnmatch.filter(fnames,'ETI_Idaho_2013'):
            ETIrast = fname  # String of crop raster map
            # Extract year from ETI raster
            year = int( re.search('(\d{4})',ETIrast).group(1)  )
            if year > 2000:           
                # Make a desert mask for this year
                # Based on where CDL signifies highly managed grassland (176) but
                # the GAP raster signifies range grasses
                run_command('r.mapcalc',
                            overwrite=True,
                            expression = 'desert = if((idveg >= 3000 && idveg < 4000 && CDL_Idaho_{} == 176), 1, null())'.format(year))
                desertdf = raster2df( 'desert' )
                # For each month generate Voronoi raster
                    # loop months and make a raster for each month
                for month in range(1,13):
                    timestamp = dt.datetime(year,month,1 )
                        
                    # Generate this months Voronoi raster
                    makeVoronoiRaster( timestamp )
    
                    # Output raster
                    ETrastout = 'ETdepths_%d_%02d' %( year, month)
                    # DO THE CALCS HERE!
                    # Timer start               
                    time = applyETdepths( ETIrast, ETrastout, d , timestamp , desertdf )
                    print('Trad. ET raster created for %d %02d in %2.1f seconds' %( year, month,time))             
                    # Write Report
                    run_command('r.univar',
                                overwrite=True,
                                flags = 't',
                                map = ETrastout, 
                                zones = 'tempVORRAST', 
                                separator = 'comma',
                                output=r'D:\TreasureValley\out\ET_stats_%d_%02d.csv'%( year, month))
 
    return 0
        
if __name__ == "__main__":
    main()
  