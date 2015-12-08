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
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


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



	X_prior = []
	X_post = []
	Y_prior = []
	Y_post = [] 
	### Start grid size loop here
	for region in locdict.keys():

		# generate dataframe for regression with data < 2010

		# add the number of quakes per region		
		Y_prior.append(int(eq_df_prior[mask_region(eq_df_prior,region)].count().values[0]))
		# add the number of wells per region
		# add the total volume injected per region
		# add them with into X_prior as [nb wells, volume]
		X_prior.append([int(welldf_prior[mask_region(welldf_prior,region)].count().values[0])
			, int(welldf_prior[mask_region(welldf_prior,region)].volume.sum())])

		# generate dataframe for regression with data >= 2010

		# add the number of quakes per region		
		Y_post.append(eq_df_post[mask_region(eq_df_post,region)].count().values[0])	
		# add the number of wells per region
		# add the total volume injected per region
		# add them with into X_post as [nb wells, volume]
		X_post.append([welldf_post[mask_region(welldf_post,region)].count().values[0]
			, welldf_post[mask_region(welldf_post,region)].volume.sum()])

	X_prior = np.array(X_prior)
	X_post = np.array(X_post)

	# ------------------------------------------
	# DOING THE REGRESSION
	# ------------------------------------------
	# Using scikit learn

	# Split in train - test 
	X_train, X_test, y_train, y_test = train_test_split(X_prior, Y_prior, test_size=0.33, random_state=42)

	clf = linear_model.LinearRegression()
	clf.fit(X_train, y_train)
	print clf.score(X_test, y_test)


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










