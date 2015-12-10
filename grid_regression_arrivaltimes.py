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

# for time split
import datetime



# Default plotting
from matplotlib import rcParams

# Load package for linear model
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

# import function from regression
from grid_regression_sklearn import mask_region,partition_state



def do_grid_interarrival_regression(eq_df, welldf, intervals, lock ,cv = 5, standardization = None):

	global best_grid_prior
	global best_grid_post

	for interval in intervals:
		# Get dictionary for the partitioned state
		locdict = partition_state(interval)

		# Filter by time
		welldf_prior = welldf[welldf.year < 2010]
		welldf_prior.reset_index(inplace=True)
		welldf_post = welldf[welldf.year >= 2010]
		welldf_post.reset_index(inplace=True)



		X_prior = []
		X_post = []
		Y_prior = []
		Y_post = [] 

		for region in locdict.keys():

		# DO THIS FOR PRIOR

		# get mask for the cluster_id
		mask = mask_region(eq_df_prior,region)
		Y_prior_append = get_hours_between(  eq_df_prior[ mask] )   
		for y in Y_prior_append:
			Y_prior.append(y)
		# add the number of wells per region
		# add the total volume injected per region
		# add them with into X_prior as [nb wells, volume]
		X_prior.append([welldf_prior[mask_region(welldf_prior,region)].count().values[0]
			, welldf_prior[mask_region(welldf_prior,region)].volume.sum()])

		for i in range(len(Y_prior_append)):			
				X_prior.append(X_prior_append)	


		# DO THIS FOR POST

		# get mask for the cluster_id
		mask = mask_region(eq_df_post,region)
		Y_post_append = get_hours_between( eq_df_post[ mask] )
		for y in Y_post_append:
			Y_post.append(y)
		# add the number of wells per region
		# add the total volume injected per region
		# add them with into X_post as [nb wells, volume]
		X_post.append([welldf_post[mask_region(welldf_post,region)].count().values[0]
			, welldf_post[mask_region(welldf_post,region)].volume.sum()])

		for i in range(len(Y_post_append)):
			X_post.append(X_post_append)

		X_prior = np.array(X_prior,dtype=np.float64)
		X_post = np.array(X_post,dtype=np.float64)		
		Y_post = np.array(Y_post, dtype=np.float64).reshape(-1,1)
		Y_prior = np.array(Y_prior, dtype = np.float64).reshape(-1,1)

		print '------------------------------------------------'
		print '------------------------------------------------'
		print('Length of Y_prior is {}'.format( len(Y_prior  )))
		print('Length of X_prior is {}'.format( len(X_prior  )))
		print '------------------------------------------------'
		print '------------------------------------------------'
		print('Length of Y_post is {}'.format( len(Y_post  )))
		print('Length of X_post is {}'.format( len(X_post  )))
		print '------------------------------------------------'
		print '------------------------------------------------'
		# ------------------------------------------
		# DOING THE REGRESSION
		# ------------------------------------------
		# logging.debug(' For {} cells, Total number of quakes: prior {}, post {}'\
		# 		.format(len(locdict.keys()),sum(X_prior[:,0]), sum(X_post[:,0]) ))		

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
			# --------------------

			# Using Ridge Regression with built-in cross validation
			# of the alpha parameters
			# note that alpha = 0 corresponds to the Ordinary Least Square Regression
			clf = linear_model.RidgeCV(alphas=[0.0, 0.1, 1, 10.0, 100.0, 1e3,1e4 ,1e5], cv =cv)
			# clf = linear_model.LinearRegression()
			clf.fit(X_train, y_train)
			print 'score', clf.score(X_test, y_test)

			# logging.debug('{}: For {} cells the score of RidgeCV is {} with alpha = {}'\
			# 	.format(reg,len(locdict.keys()),clf.score(X_test, y_test),clf.alpha_))

			with lock:
				if reg == 'prior': 
					best_grid_prior.append([clf,clf.score(X_test, y_test), eps])
				elif reg == 'post':
					best_grid_post.append([clf,clf.score(X_test, y_test),eps])		


	return

def get_hours_between(df):
    dates=[]
    origintimes = df.origintime.values
    for date in origintimes:
        year, month, day = date.split('-')
        day, hour = day.split(' ')
        hour, minute, second = hour.split(':')
        if len(second.split('.'))==2:
            second, microsecond = second.split('.')
        elif len(second.split('.'))==1:
            microsecond=0
        dates.append(datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), 
                                       int(second), int(microsecond)))
    dates=sorted(dates)
    intertimes =[]
    for i in range(1,len(dates)):
        delta = dates[i] - dates[i-1]
        delta = delta.total_seconds()/3600
        intertimes .append(delta)
    return intertimes 


def do_regression(eq_df, welldf, intervals ,lock,cv = 5, standardization = None):

	global best_grid_prior
	global best_grid_post


	for interval in intervals:

		# Get dictionary for the partitioned state
		locdict = partition_state(interval)


		# Filter by time
		eq_df_prior = eq_df[eq_df.year < 2010]
		welldf_prior = welldf[welldf.year < 2010]
		eq_df_post = eq_df[eq_df.year >= 2010]
		welldf_post = welldf[welldf.year >= 2010]



		X_prior = []
		X_post = []
		Y_prior = []
		Y_post = [] 
		# Start grid size loop here
		for region in locdict.keys():

			# generate dataframe for regression with data < 2010

			# add the intert		
			Y_prior_append = get_hours_between(eq_df_prior[mask_region(eq_df_prior,region)])
			# print('Number od quakes  is {}'.format(eq_df_prior[mask_region(eq_df_prior,region)].count()))
			# print('Y_prior_append is {}'.format(Y_prior_append))
			for y in Y_prior_append:
				Y_prior.append(y)


			# add the number of wells per region
			# add the total volume injected per region
			# add them with into X_prior as [nb wells, volume]

			X_prior_append = [welldf_prior[mask_region(welldf_prior,region)].count().values[0]
				, welldf_prior[mask_region(welldf_prior,region)].volume.sum()]

			for i in range(len(Y_prior_append)):			
				X_prior.append(X_prior_append)	


			# generate dataframe for regression with data >= 2010

			# add the number of quakes per region
			Y_post_append = get_hours_between(eq_df_post[mask_region(eq_df_post,region)])
			for y in Y_post_append:
				Y_post.append(y)	
			# add the number of wells per region
			# add the total volume injected per region
			# add them with into X_post as [nb wells, volume]
			X_post_append = [welldf_post[mask_region(welldf_post,region)].count().values[0]
				, welldf_post[mask_region(welldf_post,region)].volume.sum()]

			for i in range(len(Y_post_append)):
				X_post.append(X_post_append)

		print '------------------------------------------------'
		print '------------------------------------------------'
		print('Length of Y_prior is {}'.format( len(Y_prior  )))
		print('Length of X_prior is {}'.format( len(X_prior  )))
		print '------------------------------------------------'
		print '------------------------------------------------'
		print('Length of Y_post is {}'.format( len(Y_post  )))
		print('Length of X_post is {}'.format( len(X_post  )))
		print '------------------------------------------------'
		print '------------------------------------------------'

		if len(X_prior) == 0:
			logging.debug('interval is {}, number of regions {}'.format(interval,len(locdict.keys())))
			break

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
			# gs = GridSearchCV(clf, param_grid=parameters, cv=5)clf.score(X_test, y_test)
			# gs.fit(X_train, y_train)

			# best = gs.best_estimator_
			# best.fit(X_train, y_train)
			# logging.debug('For {} cells the score of manual Ridge is {}'.format(len(locdict.keys()),best.score(X_test, y_test)))



			# Using Ridge Regression with built-in cross validation
			# of the alpha parameters
			# note that alpha = 0 corresponds to the Ordinary Least Square Regression
			clf = linear_model.RidgeCV(alphas=[0.0, 0.1, 1, 10.0, 100.0, 1e3,1e4 ,1e5], cv = cv)
			clf.fit(X_train, y_train)

			print('{}: For {} cells the score of RidgeCV is {} with alpha = {}'\
					.format(reg,len(locdict.keys()),clf.score(X_test, y_test),clf.alpha_))
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
	# intervals = [0.05, 0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
	intervals = [0.8,0.9, 1.0, 1.5]
	# intervals = [1.5]

	# define the number of threads
	num_threads = 2

 #    # split randomely the letters in batch for the various threads
	# all_batch = []
	# size_batch = len(intervals) / num_threads
	# for i in range(num_threads-1):
	# 	batch_per_threads = random.sample(intervals,size_batch)
	# 	all_batch.append(batch_per_threads)
	# 	# new set
	# 	intervals  = list(set(intervals) - set(batch_per_threads))
	# # now get the rest
	# all_batch.append(intervals)
	# print('look at all_batch {}'.format(all_batch))

	best_grid_prior = []
	best_grid_post = []

	# # Vary the standardization and find optimum
	# for standardization in ['None','scaler','MinMaxScaler'] :
	# 	# parallelize the loop of interval
	# 	threads = []
	# 	lock = threading.Lock()
	# 	for thread_id in range(num_threads):
	# 		interval = all_batch[thread_id]
	# 		t = threading.Thread(target = do_regression, \
	# 			args = (eq_df, welldf, interval, lock ,5,standardization))
	# 		threads.append(t)

	# 	print 'Starting multithreading'
	# 	map(lambda t:t.start(), threads)
	# 	map(lambda t: t.join(), threads)

	lock = threading.Lock()
	do_regression(eq_df, welldf, intervals, lock)


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










