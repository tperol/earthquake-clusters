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
import itertools as it


# Default plotting
from matplotlib import rcParams

# Load package for linear model
import statsmodels.formula.api as sm
from sklearn import linear_model


def do_regression(eq_df, welldf, interval):

	# ------------------------------------------
	# PARTITION THE MAP INTO CELLS = CREATE THE GRID
	# ------------------------------------------

	# Make ranges
	xregions1 = np.arange(33.5, 37., interval)
	xregions2 = np.arange(34., 37.5, interval) 
	xregions = zip(xregions1, xregions2)
	yregions1 = np.arange(-103.,-94. , interval) 
	yregions2 = np.arange(-102.5 ,-93.5, interval)
	yregions = zip(yregions1, yregions2)

	# Create a dictionary with keys = (slice in long, slice in latitude)
	# value = number of the grid cell
	regions = it.product(xregions,yregions)
	locdict = dict(zip(regions, range(len(xregions)*len(yregions))))

	print 'total number of region', len(locdict.keys())


	def mask_region(df, region):
		mask_region = (df['latitude'] < region[0][1]) \
		        & (df['latitude'] >= region[0][0]) \
		        & (df['longitude'] < region[1][1]) \
		        & (df['longitude'] >= region[1][0])
		return mask_region

	# Filter by time
	eq_df_prior = eq_df[eq_df.year < 2010]
	welldf_prior = welldf[welldf.year < 2010]
	eq_df_post = eq_df[eq_df.year >= 2010]
	welldf_post = welldf[welldf.year >= 2010]


	quakes_prior = []
	wells_prior = []
	volume_prior = []
	quakes_post = []
	wells_post = []
	volume_post = []
	### Start grid size loop here
	for region in locdict.keys():
		# just do one print to see what is hapenning
		if locdict[region] == 23:
			print 'region number', locdict[region]
			print 'number of quakes_prior in this region'\
				,eq_df[mask_region(eq_df,region)].count().values[0]
			print 'number of wells in this region'\
				,welldf[mask_region(welldf,region)].count().values[0]	
			print 'total volume injected in this region'\
				,welldf[mask_region(welldf,region)].volume.sum()

		# generate dataframe for regression with data < 2010

		# add the number of quakes per region			
		quakes_prior.append(int(eq_df_prior[mask_region(eq_df_prior,region)].count().values[0]))
		# add the number of wells per region
		wells_prior.append(int(welldf_prior[mask_region(welldf_prior,region)].count().values[0]))
		# add the total volume injected per region
		volume_prior.append(int(welldf_prior[mask_region(welldf_prior,region)].volume.sum()))

		# generate dataframe for regression with data >= 2010

		# add the number of quakes per region			
		quakes_post.append(eq_df_post[mask_region(eq_df_post,region)].count().values[0])
		# add the number of wells per region
		wells_post.append(welldf_post[mask_region(welldf_post,region)].count().values[0])
		# add the total volume injected per region
		volume_post.append(welldf_post[mask_region(welldf_post,region)].volume.sum())

	# Make two dataframes for <2010 and >=2010
	reg_df_prior = {'quakes':quakes_prior, 'wells':wells_prior, 'volume':volume_prior}
	reg_df_prior = pd.DataFrame(reg_df_prior)
	reg_df_post = {'quakes':quakes_post, 'wells':wells_post, 'volume':volume_post}
	reg_df_post = pd.DataFrame(reg_df_post)


	# ------------------------------------------
	# DOING THE REGRESSION
	# ------------------------------------------
	# Using sm.ols
	olsQuakes_prior = sm.ols(formula = 'quakes ~ wells+volume', data = reg_df_prior).fit()
	print olsQuakes_prior.summary()

	return


if __name__ == '__main__':
	

	# ------------------------------------------
	# LOAD DATAFRAMES
	# ------------------------------------------
	# Load the earthquakes datafram 
	eq_df = pd.DataFrame.from_csv('./tempdata/earthquakes_catalog.csv',sep = '|')
	# filter to keep magnitude >= 3
	eq_df  = eq_df[eq_df.prefmag >= 3.0]
	# for ease add column year
	eq_df['year'] = map(lambda x: int(x), eq_df['year_float'])


	# Load the wells dataframe.  
	welldf = pd.DataFrame.from_csv('tempdata/wells_data.csv')


	do_regression(eq_df, welldf, interval = 0.5)










