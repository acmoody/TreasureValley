# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:33:45 2018

@author: amoody
"""
import os
import pickle
import numpy as np
import pandas as pd

from KrigingTools import drift
from WaterlevelTools import normscore
from pykrige.uk import UniversalKriging
from pykrige.core import _adjust_for_anisotropy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import rasterio
from shapely.geometry import LinearRing
import fiona


root = r'D:\TreasureValley\vadose\data\groundwater'
# data
datasets = ['Median1986_2018','WinterWY2016']

# Load data (deserialize)
fdir = datasets[0]
with open( os.path.join(root, fdir, fdir + '.pkl' ) ,'rb') as handle:
    dataset = pickle.load(handle)

data = dataset['data'].filter(regex=r'IDTM|DTW|dtw').values
df = dataset['data'].filter(regex=r'IDTM|DTW|dtw')
# -----------
# Grid and geospatial

proj4str = '+proj=tmerc +lat_0=42 +lon_0=-114 +k=0.9996 +x_0=2500000 +y_0=1200000 +ellps=GRS80 +units=m +no_defs'
gridx = np.arange(2250000.0,2353000.0,500)
gridy = np.arange(1334000.0,1439000.0,500)
xll, yll = 2250000.0, 1334000.0
xur, yur = 2353000,1439000
dxy = 500.0
ncells = 210. 
#ncells = ( xur - xll)/dxy
gridx = np.arange(xll, xll + (ncells * dxy), dxy)
gridy = np.arange(yll, yll + (ncells * dxy), dxy)
#%%
fbound=r'D:\TreasureValley\notebooks\data\gis\TV_bound_2010.shp'
with fiona.open(fbound,'r') as shp:
    geoms = [feature['geometry'] for feature in shp]
#%%-----------------
# Get drift function for dtw scores
var = 4 # DTW nscores
# Krige to get similar transforms as pykrige
UK = UniversalKriging(data[:, 0], data[:, 1], data[:, var], 
                     variogram_model='exponential',weight=True,nlags=20,lagdist =1000,
                     anisotropy_angle = 10 , anisotropy_scaling=1.6)

UKgrid = UniversalKriging(gridx,gridy,gridy)

XADJ,YADJ = \
    _adjust_for_anisotropy(np.vstack((data[:, 0],data[:, 1])).T,
                          [UK.XCENTER, UK.YCENTER],
                          [UK.anisotropy_scaling],
                          [UK.anisotropy_angle]).T
gridx_adj, gridy_adj = \
    _adjust_for_anisotropy(np.vstack((gridx,gridy)).T,
                           [UKgrid.XCENTER,UKgrid.YCENTER],
                           [UK.anisotropy_scaling],
                           [UK.anisotropy_angle]).T
# Calculate drift data on adjusted coordinates
data_adj = np.vstack((XADJ,YADJ,data[:,var])).T
f = drift( data_adj, gridx_adj, gridy_adj, plot=True)

#%%-------------------------------------
# DTW NScores
var=4
UKdtw = UniversalKriging(data[:, 0], data[:, 1], data[:, var], 
                     variogram_model='exponential',weight=True,nlags=20,lagdist =1000,
                     drift_terms=['functional'],functional_drift= [f],
                     enable_plotting=True,verbose=True,
                     anisotropy_angle = 10 , anisotropy_scaling=1.6)

# Creates the kriged grid and the variance grid---
z_dtw, ss_dtw = UKdtw.execute('grid', gridx, gridy)
z_dtw[np.sqrt(ss_dtw)> 2] = np.ma.masked
# Unmask array

# Invert the normal scores---
_,finv = normscore(df,['DTWmed'])
finv = np.vectorize(finv,otypes=[np.float])
z_dtw = finv(z_dtw)


# Plot ---
xv,yv = np.meshgrid(gridx,gridy)
ring= LinearRing(geoms[0]['coordinates'][0])

fig2, ax2 = plt.subplots(1,1,figsize=(10,8))
c1 = ax2.contourf(xv,yv,z_dtw,np.arange(0,225,25),vmin=0,vmax=200,cmap='gist_earth')
plt.colorbar(c1)
ax2.plot(*ring.xy,c='r')
c2 = ax2.contour(xv,yv,np.sqrt(ss_dtw),np.arange(0,3,.5),colors='k')
plt.clabel(c2,fmt = '%1.1f',inline_spacing=2)
plt.title('DTW')
plt.tight_layout()
#ax2.scatter(data[:,0],data[:,1])

#%%---------------------
# ALTITUDES
var = 5 
f = drift(data[:,[0,1,var]] ,gridx, gridy,degree=3,plot=True)      
UKalt = UniversalKriging(data[:, 0], data[:, 1], data[:,5], 
                     variogram_model='gaussian',weight=True,nlags=12,lagdist=1000,
                     drift_terms=['functional'],functional_drift = [f],enable_plotting=True)
                     
 
z_alt, ss_alt = UKalt.execute('grid', gridx, gridy)
z_alt[np.sqrt(ss_alt) > 2] = np.ma.masked
# Invert the normal scores---
_,finv = normscore(df,['ALTdtw'])
finv = np.vectorize(finv,otypes=[np.float])
z_alt = finv(z_alt)

xv,yv = np.meshgrid(gridx,gridy)
fig3,ax3 = plt.subplots(1,1,figsize=(10,8))
c1 = ax3.contourf(xv,yv,z_alt,cmap='gist_earth')
plt.colorbar(c1)
ax3.plot(*ring.xy,c='r')
c2 = ax3.contour(xv,yv,np.sqrt(ss_alt),np.arange(0,3,.5),colors='k')
plt.clabel(c2, fmt = '%1.1f',inline_spacing=2)
ax3.set_title('WL altitude')
plt.tight_layout()
#ax3.imshow(np.flipud(z_alt),vmin=vmin, vmax=vmax, extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()))
#ax3.scatter(data[:,0],data[:,1], c=data[:,5])

#%% ------------
# DEM
with rasterio.open(r'D:\TreasureValley\vadose\gis\dem.tif','r') as src:
    dem = src.read(1)
    crs = src.crs
    bounds_src = src.bounds
    affine_src = src.affine
    
# Define AFffine for kriged surfaces

aff = rasterio.Affine(500, 0 , gridx.min(), 0, 500, gridy.min())
krige_aff = rasterio.transform.from_bounds(gridx.min(),gridy.min(),gridx.max(),gridy.max(),206,210)
krige_aff = rasterio.transform.from_origin(gridx.min() - 500/2,gridy.max()+500/2,500,500)
krige_bounds = rasterio.transform.array_bounds(210,206,krige_aff)
#%%
f = os.path.join(r'D:\\TreasureValley\\vadose\\data',datasets[0],'dtw.tif')
dtw_rast = rasterio.open(f, 'w',driver='GTiff',
                         height=z_dtw.shape[0], width = z_dtw.shape[1],
                         count = 1, dtype=rasterio.float64,
                         crs = crs, transform = krige_aff)
dtw_rast.write(np.flipud(z_dtw),1)
dtw_rast.close()
#%% Upsample to 30m for raster math
f = os.path.join(r'D:\\TreasureValley\\vadose\\data',datasets[0],'alt.tif')
alt_rast = rasterio.open(f, 'w',driver='GTiff',
                         height=z_dtw.shape[0], width = z_dtw.shape[1],
                         count = 1, dtype=rasterio.float64,
                         crs = crs, transform = krige_aff)
alt_rast.write(np.flipud(z_alt),1)
alt_rast.close()
z_dtw_us = np.empty(shape=(round(z_dtw.shape[0]/.06),round(z_dtw.shape[1]/.06)))
rasterio.warp.reproject(z_dtw,z_dtw_us,
                          src_transform = krige_aff,
                          dst_transform = affine_src,
                          src_crs = crs,
                          dst_crs = crs,
                          resampling=Resampling.bilinear)
#%%
z_alt_us = np.empty(shape=(round(z_dtw.shape[0]/.06),round(z_dtw.shape[1]/.06)))
rasterio.warp.reproject(z_alt,z_alt_us,
                          src_transform = krige_aff,
                          dst_transform = affine_src,
                          src_crs = crs,
                          dst_crs = crs,
                          resampling=Resampling.bilinear)

#%% Raster math! Average the elevations, one from subtracting depth to water 
# from the DEM and the other from the altitude
# TODO: Check to see if everything is aligned. DEM appears to be upside down...
z1=(dem - (z_dtw_us * .3048))[np.newaxis,...]
z2 = (z_alt_us*.3048)[np.newaxis,...]
zavg = np.nanmean(np.vstack([z1,z2]),axis=2)
#%%
fig,ax = plt.subplots(1,3)
ax[0].contourf(z_dtw_us)
ax[1].contourf(z_alt_us)
ax[2].contourf(dem)
#%%-----------
# Export rasters

#from rasterio.warp import reproject, Resampling
#with rasterio.open(os.path.join(outpath,'gis','dem.tif')) as r:\n",
#  dem = r.read(1)\n",
#  r.read_mask\n",
# newdem = np.empty(shape=rbound.shape)\n",
#  aff = r.affine\n",
#    "    newaff = Affine(aff.a / (30./1000), aff.b, aff.c, aff.d, aff.e / (30./1000), aff.f)\n",
#    "    reproject(dem, newdem,\n",
#    "              src_transform = aff,\n",
#    "              dst_transform = newaff,\n",
#    "              src_crs = r.crs,\n",
#    "              dst_crs = r.crs,\n",
#    "              resampling = Resampling.bilinear)\n",
#    "    newdem[np.isnan(newdem)] = -999"
