
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


eq_df = pd.DataFrame.from_csv('./tempdata/earthquakes_catalog.csv',sep = '|')

# Separate data into areas
# Observe extreme values
print min(eq_df.longitude), max(eq_df.longitude)
print min(welldf.X), max(welldf.X)
print min(eq_df.latitude), max(eq_df.latitude)
print min(welldf.Y), max(welldf.Y)

# Clean up data by picking correct coordinates and delete missing data
wellvol = pd.read_excel('tempdata/wells.xlsx')
wellvol = wellvol[['X', 'Y', 'Volume']]
wellvol = wellvol[np.isfinite(wellvol['X'])]
wellvol = wellvol[np.isfinite(wellvol['Y'])]
wellvol = wellvol[np.isfinite(wellvol['Volume'])]
welldf = wellvol[(wellvol.X >= -102.918) & (wellvol.X <= -94.466)]
welldf = welldf[(welldf.Y >= 33.811) & (welldf.Y <= 36.99869)]

# Make ranges
xregions1 = np.arange(33.5, 37., .5); xregions2 = np.arange(34., 37.5, .5); xregions = zip(xregions1, xregions2)
yregions1 = np.arange(-103.,-94. , .5); yregions2 = np.arange(-102.5 ,-93.5, .5); yregions = zip(yregions1, yregions2)
regions = []
# Return a tuple. First tuple is latitude, second is longitude.
for x in xregions:
    for y in yregions:
        regions.append((x,y))
len(regions)

locdict = dict(zip(regions, range(len(regions))))

def find_region(rowquake):
    for region in regions:
        if (rowquake['latitude'] < region[0][1]) \
        & (rowquake['latitude'] >= region[0][0]) \
        & (rowquake['longitude'] < region[1][1]) \
        & (rowquake['longitude'] >= region[1][0]):
                reg = locdict[region]
                break
    return 1, 0, rowquake['year'], reg, 0
    
quakes = eq_df.apply(find_region, axis=1)
quakedf = {}; quakedf['quakes'] = []; quakedf['wells'] = []; quakedf['year'] = []; quakedf['region'] = []
quakedf['Volume'] = []
for quake, well, year, region, press in quakes:
    quakedf['quakes'].append(quake); quakedf['wells'].append(well); quakedf['year'].append(year)
    quakedf['region'].append(region); quakedf['Volume'].append(press)
quakedf =  pd.DataFrame(quakedf)

# Same function as above but for wells
def find_region(rowquake):
    for region in regions:
        if (rowquake['Y'] < region[0][1]) \
        & (rowquake['Y'] >= region[0][0]) \
        & (rowquake['X'] < region[1][1]) \
        & (rowquake['X'] >= region[1][0]):
                reg = locdict[region]
                break
    return 0, 1, 2009, reg, rowquake['Volume'] # WE ASSUME THAT THE WELLS STARTED IN 2009. WE NEED MORE DATA TO PROVE THIS

wells = welldf.apply(find_region, axis = 1)
welldf = {}; welldf['quakes'] = []; welldf['wells'] = []; welldf['year'] = []; welldf['region'] = []
welldf['Volume'] = []
for quake, well, year, region, press in wells:
    welldf['quakes'].append(quake); welldf['wells'].append(well); welldf['year'].append(year)
    welldf['region'].append(region); welldf['Volume'].append(press)
welldf =  pd.DataFrame(welldf)


datadf = pd.concat([quakedf, welldf]) # Some maxpressure's are empty. Fix this
datadf['year'] = 1*(datadf['year'] >= 2009)
datadf_reg = datadf.groupby(['region', 'year']).sum().reset_index()
#type(datadf_reg.max_pressure)


print datadf_reg.head()

olsQuakes = sm.ols(formula = 'quakes ~ wells + year + Volume + wells*Volume', data = datadf_reg).fit()
print olsQuakes.summary()
