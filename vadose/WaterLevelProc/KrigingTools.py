# -*- coding: utf-8 -*-
"""
KrigingTools.py

Helper function for KrigeVadoseZone.py workbook

@author: amoody
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 2D Kriging parameter optimization
def SelectModel(data):
    from pykrige.compat import GridSearchCV
    from pykrige.rk import Krige
    # Set up parameters
    param_dict = {"method": ["universal"],
              "variogram_model": [ "exponential", "spherical"],
               "nlags": [10, 15],
               "lagdist": [ 1000],
               "weight": [True],
               "n_closest_points" : [0],
               "anisotropy_scaling":[1,1.5,2],
               "anisotropy_angle":[10,15]
              }
    estimator = GridSearchCV(Krige(), param_dict, verbose=True, error_score=np.nan)
    # Run gridsearch
    estimator.fit(X=data[:,0:2], y= data[:,2])
    
    if hasattr(estimator, 'best_score_'):
        print('best_score R² = {:.3f}'.format(estimator.best_score_))
        print('best_params = ', estimator.best_params_)    
    print('\nCV results::')
    if hasattr(estimator, 'cv_results_'):
        for key in ['mean_test_score', 'mean_train_score',
                    'param_method', 'param_variogram_model']:
            print(' - {} : {}'.format(key, estimator.cv_results_[key]))
    return

def drift(data, gridx, gridy, 
          degree = 3, plot = False, threed=True, anisotropy_ang = None):
    # Get drift function
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from affine import Affine
    
    # Make grid and setup data     
    X = data[:,0:2]
    z = data[:,2]
    xv,yv = np.meshgrid(gridx,gridy)
    
    # Set up coordinates for prediction
    coord1, coord2 = xv.ravel(), yv.ravel()
    coord=np.vstack((coord1,coord2)).transpose()
    
    if anisotropy_ang:
        print('Method not ready yet')
        pass
        print('Rotating data and grid {} degrees CCW'.format(anisotropy_ang))
        aff = Affine.rotation(anisotropy_ang)
        # Rotate data
        X_rot = [np.round(aff * a,2) for a in X]
        X = np.asarray(X_rot)
        # Rotate grid coords
        grid = [np.round(aff * a,2) for a in coord]
        coord = np.asarray(grid)
        
  
    # Do we need to rotate for the functional drift in UK?
           
    # Set up interpolation model
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, z)
      
    if plot:
        z_plot = model.predict(coord).reshape(np.shape(xv))
        resid = model.predict(X) - z
        extent=(gridx.min(),gridx.max(), gridy.min(), gridy.max())
        fig = plt.figure()
        if threed:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xv,yv,z_plot,cmap='gist_earth')
            ax.scatter(data[:,0],data[:,1],data[:,2],c=resid)
        else:
            ax = fig.add_subplot(111)
            ax.contourf(xv,yv,z_plot,cmap='gist_earth')
            ax.scatter(data[:,0],data[:,1],data[:,2],c=resid)
    # Try using model in drift terms as drift function
    return  lambda x,y:model.predict(np.vstack((x,y)).T)