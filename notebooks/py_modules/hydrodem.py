# -*- coding: utf-8 -*-
"""
Created on Wed May  9 08:48:29 2018

@author: amoody
"""
import sys
import os
import numpy as np
# GIS 
import fiona
from rasterio import Affine, transform
from rasterio.warp import Resampling
import matplotlib.pyplot as plt
import flopy
from flopy.utils.reference import SpatialReference
root = r'D:\TreasureValley\notebooks'
sys.path.append(root)
from gridtools import resample,vtorast,rowcol

#%% -------------------------
#  BUILD MODFLOW DEM
#  1. Make a streams raster
#  2. Resample 30m DEM to 1609.344( 1 mile ) DEM with a cell average
#  3. Resample 30m with minimum
#  4. Where the stream raster exists, use minimum cell values
    
# Set up grid parameters
dx = dy = 1609.344
xul = 2250706.91 
yul = 1437102

## Inputs
rast = os.path.join(root,'data','gis','dem.tif')
vect = os.path.join(root,'data','gis','streams_simple.shp')
seds = os.path.join(root,'data','gis','sediments.tif')
basalts = rast = os.path.join(root,'data','gis','basalts.tif')
## 1. Downsample to coarser cell size
DEMavg= resample( rast, dx, method=Resampling.average)
DEMmin = resample( rast, dx, method=Resampling.min)
npDEM = np.copy(DEMavg['raster'])

## 2. Rasterize streams
arr_stream = vtorast( vect, cell_size = dx)['raster']

### 2i. expand streams to have same dimensions as DEM
arr_stream_expand = np.full_like(npDEM,0)
arr_stream_expand[0:arr_stream.shape[0],0:arr_stream.shape[1]] = arr_stream
arr_stream = np.copy(arr_stream_expand)

## 3. Set elevation to minimum cell elevation where stream cells exist 
npDEM[arr_stream> 0] = DEMmin['raster'][arr_stream> 0]

### 3.ii Read in basalt and sediment depth rasters
region = transform.array_bounds(*npDEM.shape,DEMavg['affine'])
dseds = resample(seds, dx, bounds=region, method=Resampling.average)  
dbasalt = resample(basalts, dx, bounds=region, method=Resampling.average)  
DEMbottom2 = npDEM - dseds['raster'] - dbasalt['raster']
DEMbottom1 = npDEM - dseds['raster']

#%%
### 3.i. Plot up these DEMs
# mask for plotting river elevations only
mask = np.zeros(npDEM.shape,dtype=bool) 
mask[ arr_stream == 0 ] = True 
cmin = -500
cmax = np.ceil(npDEM.max()/100)*100
fig,ax = plt.subplots(2,3,figsize=(15,10))
ax=ax.ravel()
im = ax[0].imshow(DEMavg['raster'],cmap='terrain',vmin=cmin,vmax=cmax)
ax[1].imshow(DEMmin['raster'],cmap='terrain',vmin=cmin,vmax=cmax)
ax[2].imshow(np.ma.masked_array(DEMmin['raster'],mask),cmap='terrain',vmin=cmin,vmax=cmax)
ax[3].imshow(dbasalt['raster'],cmap='terrain',vmin=0,vmax=1000)
ax[4].imshow(dseds['raster'],cmap='terrain',vmin=0,vmax=1000)
ax[5].imshow(DEMbottom2,cmap='terrain',vmin=cmin,vmax=cmax)
ax[0].set_title('Average')
ax[1].set_title('Minimum')
ax[2].set_title('Stream Elevation')
# Colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6]) #l, b, w, h
fig.colorbar(im,cax=cbar_ax)

#%%
## 4. Export to model top
# Spatial Reference Module
xll, yll = 2247000.00, 1330950.00 # origin of the model [m] (lower left corner)
dxdy = 1609.344 # grid spacing (in model units) 
delc = np.ones(nrow, dtype=float) * dxdy
delr = np.ones(ncol, dtype=float) * dxdy
nrow , ncol = npDEM.shape
rot = -2 # rotation (positive ccw)
# Specify coordinate system with custom Proj4 string for IDTM83
model_proj4 = '+proj=tmerc +lat_0=42 +lon_0=-114 +k=0.9996 +x_0=2500000 +y_0=1200000 +ellps=GRS80 +units=m +no_defs'
sr = SpatialReference(delr=delr, delc=delc, xll=xll, yll=yll, rotation=rot, proj4_str = model_proj4)

# Modflow discretization
# row and column spacings
# (note that delc is column spacings along a row; delr the row spacings along a column)
nlay = 1
delr = dxdy
delc = dxdy
# Top of model is the DEM we just created
ztop = npDEM
botm = np.stack( (DEMbottom1,DEMbottom2))
botm = DEMbottom2
# Initialize model objext
ml = flopy.modflow.Modflow(modelname = 'TV', exe_name = 'mf2005', model_ws = root )
dis = flopy.modflow.ModflowDis(ml, nlay ,nrow, ncol , delr=delr, delc=delc ,top=ztop, botm=botm)

# BAS Package
# Use model boundary to set ibound
model_bound = os.path.join(root,'gis','TV_Bound_2010.shp') 
rbound = vtorast2( model_bound, dx)
bas = flopy.modflow.ModflowBas(ml, ibound=ibound)

# Write inputs to NAM file
ml.write_input()

# Conductivity
np.arange(1E-2,1E-7


