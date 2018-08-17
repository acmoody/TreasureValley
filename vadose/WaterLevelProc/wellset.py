# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:00:59 2018

@author: amoody
"""
from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

class wellset( object ):
    '''
    The idea: Subset the null set of wells pulled from well site to prep
    data for analyses like trends, interpolation
    
    Attributes
    ----------
    data: pd.DataFrame, Timeseries with each column being a well
    config: Dict, Dictionary of metadata from wellsite with wells as keys
    '''
    root = r'D:/TreasureValley/vadose/data'
    def __init__(self, data = None, name = None, nullset = None):
        
        if not nullset:
            f = r'D:/TreasureValley/vadose/config/Null/WellInfo_null.xlsx'
            print('Reading nullset metadata from {}'.format(f))
            self.config = pd.read_excel(f,index_col='WellNumber')
            
        if not data:
            f =r'D:/TreasureValley/vadose/config/Null/WellLevels_null.xlsx'
            print('Reading nullset data from {}'.format(f)) 
            self.data = pd.read_excel(f,index_col=0,parse_dates=True)
        else:
            self.data = pd.DataFrame(data.copy())
            
        if not name:
            self.name = 'Wells_' + pd.datetime.now().strftime('%Y%m%d_%H%M')
            
      
       # Save start and enddate of the timeserie
        self._start_date = self.data.index[0]
        self._end_date = self.data.index[-1]
     
        
            

    def __str__(self):
        return self.data.__repr__()
    
    def __repr__(self):
        
        message = '{} wells ranging from ' + \
            self._start_date.strftime("%H:%M:%S %d/%m/%Y") + \
            ' until ' + self._end_date.strftime("%H:%M:%S %d/%m/%Y")
        wellcount = len(self.data)
        message.format(wellcount)
        
        return message
