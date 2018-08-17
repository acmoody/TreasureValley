# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:46:53 2018

@author: amoody
"""

import pywt
from wavelets import WaveletAnalysis
wa = WaveletAnalysis((s-s.mean()).values, dt=1/365)
# wavelet power spectrum
power = wa.wavelet_power
# scales 
scales = wa.scales
# associated time vector
t = wa.time
# reconstruction of the original data
rx = wa.reconstruction()

fig, ax = plt.subplots()
T, S = np.meshgrid(t, scales)
c1=ax.contourf(T, S, power, 100,cmap='jet')
plt.colorbar(c1)
ax.set_yscale('log')


chisquare(power,axis=1)

#%%
plt.style.use('bmh')
data=DataNull.loc[DataNull.WellNumber == '01N 05E 17BCA1','WaterLevelBelowLSD']
#%%
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
s = [ 90, 180, 365]
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
seasonal_pdq = list(itertools.product(p, d, q, s))
warnings.filterwarnings("ignore") # specify to ignore warning messages
#%%
minAIC = 1e9
minparams = ''
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data.resample('MS').mean(),
                                            order=param,
                                            trend='c',
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=True,
                                            enforce_invertibility=False)

            results = mod.fit()
            if results.aic < minAIC:
                minAIC = results.aic
                minparams = 'ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,results.aic)
                
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            print('uhhh')
            continue
#%%     
mod = sm.tsa.statespace.SARIMAX(data.resample('MS').mean(),
                                trend='c',
                                order= (0,1,1),
                                seasonal_order=(0,1,1,12),
                                enforce_stationarity=True,
                                enforce_intertibility=False)
results=mod.fit()

results.summary()
results.plot_diagnostics(figsize=(15,12))
