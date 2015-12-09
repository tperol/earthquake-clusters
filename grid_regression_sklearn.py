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
import random
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )

# library for multithreading
import threading



# Default plotting
from matplotlib import rcParams

# Load package for linear model
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing


def do_regression(eq_df, welldf, intervals, lock ,cv = 5, standardization = None):

	global best_grid_prior
	global best_grid_post

	for interval in intervals:

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
			Y_prior.append(eq_df_prior[mask_region(eq_df_prior,region)].count().values[0])
			# add the number of wells per region
			# add the total volume injected per region
			# add them with into X_prior as [nb wells, volume]
			X_prior.append([welldf_prior[mask_region(welldf_prior,region)].count().values[0]
				, welldf_prior[mask_region(welldf_prior,region)].volume.sum()])

			# generate dataframe for regression with data >= 2010

			# add the number of quakes per region		
			Y_post.append(eq_df_post[mask_region(eq_df_post,region)].count().values[0])	
			# add the number of wells per region
			# add the total volume injected per region
			# add them with into X_post as [nb wells, volume]
			X_post.append([welldf_post[mask_region(welldf_post,region)].count().values[0]
				, welldf_post[mask_region(welldf_post,region)].volume.sum()])

		X_prior = np.array(X_prior,dtype=np.float64)
		X_post = np.array(X_post,dtype=np.float64)
		Y_post = np.array(Y_post, dtype=np.float64).reshape(-1,1)
		Y_prior = np.array(Y_prior, dtype = np.float64).reshape(-1,1)

		# ------------------------------------------
		# DOING THE REGRESSION
		# ------------------------------------------

		reg_for = ['prior', 'post']
		for reg in reg_for:
			if reg == 'prior':
				X = X_prior
				Y = Y_prior
			elif reg == 'post':
				X = X_post
				Y = Y_post


			# --------------------
			# SPLIT INTO TRAIN AND TEST
			# --------------------

			# Split in train - test 
			X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


			# --------------------
			# STANDARDIZATION OF THE DATA -- SCALING
			# --------------------

			if standardization == 'scaler':

				scaler = preprocessing.StandardScaler().fit(X_train)
				X_train = scaler.fit_transform(X_train)
				X_test = scaler.transform(X_test)
				y_train = scaler.fit_transform(y_train)
				y_test = scaler.transform(y_test)

			elif standardization == 'MinMaxScaler':
				min_max_scaler = preprocessing.MinMaxScaler()
				X_train = min_max_scaler.fit_transform(X_train)
				X_test = min_max_scaler.transform(X_test)
				y_train = min_max_scaler.fit_transform(y_train)
				y_test = min_max_scaler.transform(y_test)
			else:
				pass


			# --------------------
			# OPTIMIZE CLASSIFIER WITH RIDGE REGRESSION
			# AND ORDINARY LEAST SQUARE REGRESSION
			# no need for Lasso because only 2 features
			# --------------------


			# # Using Ordinary Least Square Regression
			# clf = linear_model.LinearRegression()
			# clf.fit(X_train, y_train)
			# logging.debug('For {} cells the score is {}'.format(len(locdict.keys()),clf.score(X_test, y_test)))


			# # Using Ridge Regression and cross-validation
			# # doing the selection manually
			# # uncomment this part to check it matches the next paragraph
			# clf = linear_model.Ridge()
			# parameters = {'alpha': [0.1, 0.5]}
			# gs = GridSearchCV(clf, param_grid=parameters, cv=5)
			# gs.fit(X_train, y_train)

			# best = gs.best_estimator_
			# best.fit(X_train, y_train)
			# logging.debug('For {} cells the score of manual Ridge is {}'.format(len(locdict.keys()),best.score(X_test, y_test)))



			# Using Ridge Regression with built-in cross validation
			# of the alpha parameters
			# note that alpha = 0 corresponds to the Ordinary Least Square Regression
			clf = linear_model.RidgeCV(alphas=[0.0, 0.1, 1, 10.0, 100.0, 1e3,1e4 ,1e5], cv = cv)
			clf.fit(X_train, y_train)

			logging.debug('{}: For {} cells the score of RidgeCV is {} with alpha = {}'\
				.format(reg,len(locdict.keys()),clf.score(X_test, y_test),clf.alpha_))

			with lock:
				if reg == 'prior': 
					best_grid_prior.append([clf,clf.score(X_test, y_test),interval])
				elif reg == 'post':
					best_grid_post.append([clf,clf.score(X_test, y_test),interval])



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

	# define the intervals
	intervals = [0.05, 0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
	# intervals = [0.8,0.9, 1.0, 1.5]

	# define the number of threads
	num_threads = 4

    # split randomely the letters in batch for the various threads
	all_batch = []
	size_batch = len(intervals) / num_threads
	for i in range(num_threads-1):
		batch_per_threads = random.sample(intervals,size_batch)
		all_batch.append(batch_per_threads)
		# new set
		intervals  = list(set(intervals) - set(batch_per_threads))
	# now get the rest
	all_batch.append(intervals)
	print('look at all_batch {}'.format(all_batch))

	best_grid_prior = []
	best_grid_post = []

	# Vary the standardization and find optimum
	for standardization in ['None','scaler','MinMaxScaler'] :
		# parallelize the loop of interval
		threads = []
		lock = threading.Lock()
		for thread_id in range(num_threads):
			interval = all_batch[thread_id]
			t = threading.Thread(target = do_regression, \
				args = (eq_df, welldf, interval, lock ,5,standardization))
			threads.append(t)

		print 'Starting multithreading'
		map(lambda t:t.start(), threads)
		map(lambda t: t.join(), threads)


	best_score_prior = [[c[1],c[2]] for c in best_grid_prior]
	best_score_prior = np.array(best_score_prior)
	best_index_prior = np.where(best_score_prior[:,0] == max(best_score_prior[:,0]))[0][0]
	print('Best classifier <2010 is for alpha ={}'.format(best_grid_prior[best_index_prior][0].alpha_))
	print('Coefs <2010 are ={}'.format(best_grid_prior[best_index_prior][0].coef_))
	print('R^2 <2010 = {}'.format(best_grid_prior[best_index_prior][1]))
	print('Best interval <2010 is {}'.format(best_grid_prior[best_index_prior][2]))


	best_score_post = [[c[1],c[2]] for c in best_grid_post]
	best_score_post = np.array(best_score_post)
	best_index_post = np.where(best_score_post[:,0] == max(best_score_post[:,0]))[0][0]
	print('Best classifier >= 2010 is for alpha ={}'.format(best_grid_post[best_index_post][0].alpha_))
	print('Coefs >= 2010 are ={}'.format(best_grid_post[best_index_post][0].coef_))
	print('R^2 >= 2010 = {}'.format(best_grid_post[best_index_post][1]))
	print('Best interval >= 2010 is {}'.format(best_grid_post[best_index_post][2]))










