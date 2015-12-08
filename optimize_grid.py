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

# Load the wells dataframe.  
welldf = pd.DataFrame.from_csv('./tempdata/wells_data.csv',sep = ',')

print welldf.head()

# Make ranges
xregions1 = np.arange(33.5, 37., .5)
xregions2 = np.arange(34., 37.5, .5) 
xregions = zip(xregions1, xregions2)
yregions1 = np.arange(-103.,-94. , .5) 
yregions2 = np.arange(-102.5 ,-93.5, .5)
yregions = zip(yregions1, yregions2)

# Create a dictionary with keys = (slice in long, slice in latitude)
# value = number of the grid cell
regions = it.product(xregions,yregions)
locdict = dict(zip(regions, range(len(xregions)*len(yregions))))



