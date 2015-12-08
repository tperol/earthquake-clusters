# ------------------------------------------
# LOAD CLASSIC LIBRARIES
# ------------------------------------------

import numpy as np
import scipy as sp
import json
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from pyquery import PyQuery as pq
import requests
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import random
import json
import time
import csv

# Default plotting
from matplotlib import rcParams
# Load package for linear model
import statsmodels.formula.api as sm
import itertools as it


# ------------------------------------------
# LOAD DATAFRAMES
# ------------------------------------------
 
# Load the earthquakes datafram 
eq_df = pd.DataFrame.from_csv('./tempdata/earthquakes_catalog.csv',sep = '|')
# filter to keep magnitude >= 3
eq_df  = eq_df[eq_df.prefmag >= 3.0]
# filter after 2010
eq_df  = eq_df[eq_df.year_float >= 2010]
# for ease add column year
eq_df['year'] = map(lambda x: int(x), eq_df['year_float'])

# Load the wells dataframe.  
welldf = pd.DataFrame.from_csv('./tempdata/wells_data.csv',sep = ',')



# Make ranges
xregions1 = np.arange(33.5, 37., 0.5)
xregions2 = np.arange(34., 37.5, 0.5) 
xregions = zip(xregions1, xregions2)
yregions1 = np.arange(-103.,-94. , 0.5) 
yregions2 = np.arange(-102.5 ,-93.5, 0.5)
yregions = zip(yregions1, yregions2)

# Create a dictionary with keys = (slice in long, slice in latitude)
# value = number of the grid cell
regions = it.product(xregions,yregions)
locdict = dict(zip(regions, range(len(xregions)*len(yregions))))

print 'total number of region', len(locdict.keys())


reg_df = pd.DataFrame(index = range(len(locdict.keys()))\
			, columns = ['quakes', 'wells', 'volume'])
reg_df.index.name = 'region'

def mask_region(df, region):
	mask_region = (df['latitude'] < region[0][1]) \
	        & (df['latitude'] >= region[0][0]) \
	        & (df['longitude'] < region[1][1]) \
	        & (df['longitude'] >= region[1][0])
	return mask_region

for region in locdict.keys():
	# just do one print to see what is hapenning
	if locdict[region] == 23:
		print 'region number', locdict[region]
		print 'number of quakes in this region'\
			,eq_df[mask_region(eq_df,region)].count().values[0]
		print 'number of wells in this region'\
			,welldf[mask_region(welldf,region)].count().values[0]	
		print 'total volume injected in this region'\
			,welldf[mask_region(welldf,region)].volume.sum()			
	reg_df.loc[(locdict[region]),'quakes']=eq_df[mask_region(eq_df,region)].count().values[0]
	reg_df.loc[(locdict[region]),'wells']= welldf[mask_region(welldf,region)].count().values[0]
	reg_df.loc[(locdict[region]),'volume'] = welldf[mask_region(welldf,region)].volume.sum()


olsQuakes = sm.ols(formula = 'quakes ~ wells + volume ', data = reg_df).fit()
# print olsQuakes.summary()


