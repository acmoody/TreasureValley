# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:08:18 2018

@author: amoody
"""
from progress.bar import Bar
import numpy as np

bar = Bar('Processing', max=20)
for i in range(20):
    # Do some work
    a = 1
    bar.next()
bar.finish()
# CLUSTERS FOR ARIMA STUFF
import numpy as np
import scipy.ndimage.measurements as mnts

A = np.array([
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
])

# labeled is a version of A with labeled clusters:
#
# [[1 0 0 0]
#  [0 2 2 0]
#  [0 2 0 0]
#  [0 2 0 0]]
#
# clusters holds the number of different clusters: 2
labeled, clusters = mnts.label(A)

# sizes is an array of cluster sizes: [0, 1, 4]
sizes = mnts.sum(A, labeled, index=range(clusters + 1))

# mnts.sum always outputs a float array, so we'll convert sizes to int
sizes = sizes.astype(int)

# get an array with the same shape as labeled and the 
# appropriate values from sizes by indexing one array 
# with the other. See the `numpy` indexing docs for details
labeledBySize = sizes[labeled]

print(labeledBySize)