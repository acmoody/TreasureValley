# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:20:05 2018

@author: amoody
"""
import sys
import os
import numpy as np
import rasterio
import fiona
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio import Affine, transform
import math

# Rasterize vector with
def vtorast( vect, cell_size=None, bounds=None, select_attr=None):
    """ Convert a vector to a raster at a given pixel size. """
    # Open streams shapefile 
    with fiona.open( vect ) as src:
        records = [r for r in src]
        geoms = [r['geometry'] for r in records ]
        attr = [r['properties'] for r in records ]
        geoms = [(g, attr[i]['cat']) for i, g in enumerate(geoms) if g]
        
        # Rasterize streams shapefile at new cell size without rotation
        if bounds:
            trans = Affine(cell_size, 0, bounds[0],
                           0, -cell_size, bounds[3])
            nrow, ncol = rowcol(trans,bounds[2],bounds[1])
        else:
            trans = Affine(cell_size, 0, src.bounds[0],
                           0, -cell_size, src.bounds[3])
            
        # Get appropriate number of rows and columns with rowcol
        # Max row and column is at bottom right (east,south)
            nrow, ncol = rowcol(trans,src.bounds[2],src.bounds[1])
        raster_array = rasterize(geoms, out_shape = (nrow,ncol), transform=trans, all_touched=False)
    return {'raster':raster_array,'profile':src.profile,'affine':trans}


def resample( rast, res, bounds=None, method=None ):
    """
    RAST: Path to original raster
    res: Cell size of new raster in original raster units
    method: Rasterio resampling class methods, i.e. Resampling.average 
    """   

    with rasterio.open(rast) as src:
        # Read raster to array and get transform
        arr = src.read(1)
        trans = src.affine 
        crs = src.crs
        profile = src.profile
        
        # Make new array to hold resampled dataset  
        ktrans = res / trans[0]
        # If there are new bounds to which to clip, clip!
        if bounds:
            newtrans = transform.from_origin(bounds[0],bounds[3], res ,res)
            nrow, ncol = rowcol(newtrans,bounds[2],bounds[1])
        else:
            newtrans = Affine(trans.a * ktrans, trans.b, trans.c, 
                              trans.d, trans.e * ktrans, trans.f)           
            ncol = np.ceil((src.bounds.right - src.bounds.left)/res).astype(int)
            nrow = np.ceil((src.bounds.top - src.bounds.bottom)/res).astype(int) 
                   
        newarr = np.empty(shape=(nrow,ncol))
        reproject(arr, newarr,
                  src_transform = trans,
                  dst_transform = newtrans,
                  src_crs = crs,
                  dst_crs = crs,
                  resampling = method)
        newarr[np.isnan(newarr)] = -999
    
        
    return {'raster':newarr,'profile':profile,'affine':newtrans}


def writeraster( array,out,crs,trans,dtype=None):
    dtype=np.float64
    with rasterio.open( out, 'w', driver='GTiff',
                       height=array.shape[0], width=array.shape[1],
                       count=1, dtype=dtype,
                       crs=crs,transform = trans ) as new_dataset:
        new_dataset.write(array,1)
     
    return 0    

def rowcol(transform, xs, ys, op=math.floor, precision=6):
    import math
    import collections
    """
    Returns the rows and cols of the pixels containing (x, y) given a
    coordinate reference system.

    Use an epsilon, magnitude determined by the precision parameter
    and sign determined by the op function:
        positive for floor, negative for ceil.

    Parameters
    ----------
    transform : Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
    xs : list or float
        x values in coordinate reference system
    ys : list or float
        y values in coordinate reference system
    op : function
        Function to convert fractional pixels to whole numbers (floor, ceiling,
        round)
    precision : int
        Decimal places of precision in indexing, as in `round()`.

    Returns
    -------
    rows : list of ints
        list of row indices
    cols : list of ints
        list of column indices
    """

    single_x = False
    single_y = False
    if not isinstance(xs, collections.Iterable):
        xs = [xs]
        single_x = True
    if not isinstance(ys, collections.Iterable):
        ys = [ys]
        single_y = True

    eps = 10.0 ** -precision * (1.0 - 2.0 * op(0.1))
    invtransform = ~transform

    rows = []
    cols = []
    for x, y in zip(xs, ys):
        fcol, frow = invtransform * (x + eps, y - eps)
        cols.append(op(fcol))
        rows.append(op(frow))

    if single_x:
        cols = cols[0]
    if single_y:
        rows = rows[0]

    return rows, cols