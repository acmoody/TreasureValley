# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:00:36 2018

@author: amoody

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fname = r'D:\TreasureValley\data\landuse\NASS_1985_2015.xlsx'
d = pd.read_excel(fname,sheetname='NASS_with_potatoes')
d.drop(['CV (%)', 'Unnamed: 8', 'Unnamed: 9', 'Domain','Domain Category'],axis=1,inplace=True)

#------------------------
# Group and map to ETIdaho
# --------------------------

def mapCrops():
    mix2ETI={'BARLEY - ACRES HARVESTED':'Spring Grain—Irrigated (wheat, barley, oats, triticale)',
       'OATS - ACRES HARVESTED':'Spring Grain—Irrigated (wheat, barley, oats, triticale)',
       'WHEAT - ACRES HARVESTED':'Spring Grain—Irrigated (wheat, barley, oats, triticale)',
       'WHEAT, SPRING, (EXCL DURUM) - ACRES HARVESTED':'Spring Grain—Irrigated (wheat, barley, oats, triticale)',      
       'BEANS, DRY EDIBLE - ACRES HARVESTED':'Snap and Dry Beans - fresh',      
       'CORN, GRAIN - ACRES HARVESTED':'Field Corn having moderate lengthed season',
       'CORN, SILAGE - ACRES HARVESTED':'Silage Corn (same as field corn, but with truncated season)',        
       'GRASSES & LEGUMES TOTALS, SEED - ACRES HARVESTED':'Alfalfa Hay - peak (no cutting effects )',	
       'HAY - ACRES HARVESTED	HAY & HAYLAGE - ACRES HARVESTED':'Alfalfa Hay - peak (no cutting effects )',
       'HAY, ALFALFA - ACRES HARVESTED':'Alfalfa Hay - peak (no cutting effects )',
       'HAYLAGE - ACRES HARVESTED':'Alfalfa Hay - peak (no cutting effects )',
       'HAYLAGE, ALFALFA - ACRES HARVESTED':'Alfalfa Hay - peak (no cutting effects )',
       'LEGUMES, ALFALFA, SEED - ACRES HARVESTED':'Alfalfa Hay - peak (no cutting effects )',                         
       'MINT, OIL - ACRES HARVESTED':'Mint',        
       'PEAS, DRY EDIBLE - ACRES HARVESTED':'Garden Peas--fresh',
       'SUGARBEETS - ACRES HARVESTED':'Sugar beets',  
       'WHEAT, WINTER - ACRES HARVESTED':'Winter Grain—Irrigated (wheat, barley)'	, 
       'ONIONS, DRY - ACRES HARVESTED':'Onions',
       'POTATOES - ACRES HARVESTED':'Potatoes--processing (early harvest)',
       'Range grass':'Range Grasses',
       'Shrub/Scrub':'Sage brush'}
    return mix2ETI


# PARE DOWN TO ACRES HARVESTED, Range grass, and Shrub/scrub
# d= d[ d['Data Item'].str.contains('ACRES HARVESTED|Range grass|Shrub')]
d= d[ d['Data Item'].str.contains('ACRES HARVESTED')]

# Keep only Ada, Canyon, Elmore, Gem, Payette
d = d[ d['County'].isin(['ADA','ELMORE', 'GEM','PAYETTE','CANYON']) ]
# Remove strings in Value column (D) (Z). Convert value column to float
d.loc[d['Value'].str.contains('D|Z').notnull() , 'Value' ] = np.nan
d['Value']=d['Value'].astype(float)
# Drop NAN
d.dropna(inplace=True)

# Apply NASS 2 ETI map
d = d[d['Data Item'].map( mapCrops() ).notnull()]

d.loc[:,'Data Item'] = d['Data Item'].map(mapCrops())

# Examine crops less than 50 acres
smallacresdf = d.where(d['Value'] < 50 ).dropna()

# Get an idea of the commodity crop partitions
cropdf = d.loc[:,('Commodity','Data Item','Value')].pivot_table(
        index=['Commodity','Data Item'], 
        values='Value',
        aggfunc=np.nanmean)

table = pd.pivot_table( d,
                       values = 'Value',
                       columns = ['Data Item'],
                       index = ['Year', 'County'],
                       aggfunc=np.sum)

# Split Alfalfa into 50% beef, 50% dairy
alf = pd.DataFrame(columns = {'Alfalfa Hay - frequent cuttings - dairy style',
                              'Alfalfa Hay - less frequent cuttings - beef cattle style'},
                    index = table.index,
                    data = 0.5 * np.ones( ( 155, 2 ) ) )
alf = alf.mul(table['Alfalfa Hay - peak (no cutting effects )'].values,axis=0)
table=table.merge(alf,left_index=True,right_index=True)
table.drop('Alfalfa Hay - peak (no cutting effects )',axis=1,inplace=True)


# Divide by total. How???
cropmixtable=table.apply(lambda x: x/np.nansum(x),axis=1)

# Unstack. Keep it in mind
#table.unstack(level=0)

# Model Wide cropmix 
cropmixTV = table.groupby(table.index.get_level_values(0)).sum()
cropmixTV = cropmixTV.apply(lambda x: x/np.nansum(x),axis=1)


cropmixTV.plot(kind='bar',stacked=True,colormap='tab20')					
plt.legend(ncol=3,loc='lower right', bbox_to_anchor=(0.9,-0.21))