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
from geopy.distance import great_circle


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
    return intertimes * 60


def get_furthest_distance(eq_df, mask, centroid):
	furthest_point = None
	furthest_dist = None

	for point0, point1 in zip(eq_df[mask].latitude,eq_df[mask].longitude):
		point = (point0, point1)
		dist = great_circle( centroid, point ).km
		if (furthest_dist is None) or (dist > furthest_dist):
			furthest_dist = dist
			furthest_point = point
	return furthest_dist 

def cluster_centroid(eq_df, mask):
	n = eq_df[ mask].shape[0]
	sum_lon = eq_df[ mask].longitude.sum()
	sum_lat = eq_df[mask].latitude.sum()
	return (sum_lat/n, sum_lon/n )

def get_cluster_nwells_volume(welldf, centroid, radius):
	
	n_wells = 0
	volume = 0

	for (i, coords) in enumerate(zip(welldf.latitude,welldf.longitude)):
		if great_circle((coords[0],coords[1]),centroid ) < radius:
			n_wells += 1
			volume += welldf.loc[i, 'volume']

	return [n_wells, volume]


def mask_cluster(df, period, eps,  cluster_id): 

	# reconstruct the column name
	col_name = 'cluster_' + period + '_eps_' + str(eps)

	mask_cluster = df[ col_name  ] == cluster_id

	return mask_cluster

def mask_region(df, region):
	mask_region = (df['latitude'] < region[0][1]) \
	        & (df['latitude'] >= region[0][0]) \
	        & (df['longitude'] < region[1][1]) \
	        & (df['longitude'] >= region[1][0])
	return mask_region

def partition_state(interval):

	# ------------------------------------------
	# PARTITION THE MAP INTO CELLS = CREATE THE GRID
	# ------------------------------------------

	# Make ranges 
	# Since all earthquakes are in Oklahoma we partition roughly
	# using the upper bound of the state limit
	xregions1 = np.arange(33.5, 37.0, interval)
	xregions2 = np.arange(33.5 + interval, 37.0 + interval, interval) 
	xregions = zip(xregions1, xregions2)
	yregions1 = np.arange(-103.0,-94. , interval) 
	yregions2 = np.arange(-103.0 + interval ,-94.0 + interval, interval)
	yregions = zip(yregions1, yregions2)

	# Create a dictionary with keys = (slice in long, slice in latitude)
	# value = number of the grid cell
	regions = it.product(xregions,yregions)
	locdict = dict(zip(regions, range(len(xregions)*len(yregions))))

	return locdict

def do_grid_regression(eq_df, welldf, intervals, lock ,cv = 5, standardization = None):

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

			clf, X_test, y_test  = do_regression(X,Y,reg,locdict,lock, cv ,standardization)	

			logging.debug('{}: For {} cells the score of RidgeCV is {} with alpha = {}'\
				.format(reg,len(locdict.keys()),clf.score(X_test, y_test),clf.alpha_))

			with lock:
				if reg == 'prior': 
					best_grid_prior.append([clf,clf.score(X_test, y_test),interval])
				elif reg == 'post':
					best_grid_post.append([clf,clf.score(X_test, y_test),interval])



	return

def do_cluster_regression(eq_df, welldf, eps_s, lock ,cv = 5, standardization = None):

	global best_grid_prior
	global best_grid_post

	# Filter by time
	welldf_prior = welldf[welldf.year < 2010]
	welldf_prior.reset_index(inplace=True)
	welldf_post = welldf[welldf.year >= 2010]
	welldf_post.reset_index(inplace=True)

	for eps in eps_s:

		X_prior = []
		X_post = []
		Y_prior = []
		Y_post = [] 
		total_prior = []
		total_post = []

		logging.debug('eps {} from batch {}, standardization method: {}'\
		.format(eps, eps_s,standardization))


		# DO THIS FOR PRIOR

		# find the list of clusters
		col_name = 'cluster_' + 'prior' + '_eps_' + str(eps)
		clusters = list(set(eq_df[col_name].values) - set([-10]))
		# this is for the clusters that are not noise
		for cluster_id in clusters:
			# get mask for the cluster_id
			mask = mask_cluster(eq_df, 'prior', eps,  cluster_id)
			Y_prior_append = get_hours_between(  eq_df[ mask] )   
			for y in Y_prior_append:
				Y_prior.append(y)

			# find the centroid of the cluster
			centroid = cluster_centroid(eq_df, mask)
			# find the largest radius = largest distance between centroid and points
			# in the cluster
			radius = get_furthest_distance(eq_df, mask, centroid)
			# find the numbe of wells and volume within this radius
			X_prior_append=get_cluster_nwells_volume(welldf_prior, centroid, radius)
			total_prior.append(X_prior_append)
			for i in range(len(Y_prior_append)):			
					X_prior.append(X_prior_append)	

		# add the interarrival for the events classified as noise
		cluster_id = -1
		# ------
		mask = mask_cluster(eq_df, 'prior', eps,  cluster_id)
		Y_prior_append = get_hours_between(  eq_df[ mask] )   
		for y in Y_prior_append:
			Y_prior.append(y)

		# find the volume
		total_prior = np.array(total_prior)
		X_prior_append=[welldf_prior.count().values[0] - sum(total_prior[:,0]) , welldf_prior.volume.sum() - sum(total_prior[:,1]) ]
		for i in range(len(Y_prior_append)):			
				X_prior.append(X_prior_append)	

		#------

		# DO THIS FOR POST

		# find the list of clusters
		col_name = 'cluster_' + 'post' + '_eps_' + str(eps)
		clusters = list(set(eq_df[col_name].values) - set([-10]))
		for cluster_id in clusters:
			# get mask for the cluster_id
			mask = mask_cluster(eq_df, 'post', eps,  cluster_id)
			Y_post_append = get_hours_between( eq_df[ mask] )
			for y in Y_post_append:
				Y_post.append(y)

			# find the centroid of the cluster
			centroid = cluster_centroid(eq_df, mask)
			# find the largest radius = largest distance between centroid and points
			# in the cluster
			radius = get_furthest_distance(eq_df, mask, centroid)
			# find the numbe of wells and volume within this radius
			X_post_append=get_cluster_nwells_volume(welldf_post, centroid, radius) 
			total_post.append(X_post_append)
			for i in range(len(Y_post_append)):
				X_post.append(X_post_append)

		# add the interarrival for the events classified as noise
		cluster_id = -1
		# ------
		mask = mask_cluster(eq_df, 'post', eps,  cluster_id)
		Y_post_append = get_hours_between(  eq_df[ mask] )   
		for y in Y_post_append:
			Y_post.append(y)

		# find the volume
		total_post = np.array(total_post)
		X_post_append=[welldf_post.count().values[0] - sum(total_post[:,0]) , welldf_post.volume.sum() - sum(total_post[:,1]) ]
		for i in range(len(Y_post_append)):			
				X_post.append(X_post_append)	

		#------

		X_prior = np.array(X_prior,dtype=np.float64)
		X_post = np.array(X_post,dtype=np.float64)		

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

			clf.fit(X_train, y_train)
			logging.debug('For eps = {}, score : {}'.format(eps,clf.score(X_test, y_test)))

			with lock:
				if reg == 'prior': 
					best_grid_prior.append([clf,clf.score(X_test, y_test), eps])
				elif reg == 'post':
					best_grid_post.append([clf,clf.score(X_test, y_test),eps])		


	return

def do_grid_interarrival_regression(eq_df, welldf, intervals, lock ,cv = 5, standardization = None):

	global best_grid_prior
	global best_grid_post

	for interval in intervals:
		# Get dictionary for the partitioned state
		locdict = partition_state(interval)

		# Filter by time
		eq_df_prior = eq_df[eq_df.year < 2010]
		welldf_prior = welldf[welldf.year < 2010]
		welldf_prior.reset_index(inplace=True)
		welldf_post = welldf[welldf.year >= 2010]
		welldf_post.reset_index(inplace=True)
		eq_df_post = eq_df[eq_df.year >= 2010]


		X_prior = []
		X_post = []
		Y_prior = []
		Y_post = [] 

		for region in locdict.keys():

			# DO THIS FOR PRIOR

			# get mask for the cluster_id
			mask = mask_region(eq_df_prior,region)
			Y_prior_append = get_hours_between(  eq_df_prior[ mask] ) 
			if len(Y_prior_append) != 0:
				Y_prior_append = [1.0/y for y in Y_prior_append]
			else:
				Y_prior_append = [0.0]  
			for y in Y_prior_append:
				Y_prior.append(y)
			# add the number of wells per region
			# add the total volume injected per region
			# add them with into X_prior as [nb wells, volume]
			X_prior_append = [welldf_prior[mask_region(welldf_prior,region)].count().values[0]
				, welldf_prior[mask_region(welldf_prior,region)].volume.sum()]

			for i in range(len(Y_prior_append)):			
					X_prior.append(X_prior_append)	


			# DO THIS FOR POST

			# get mask for the cluster_id
			mask = mask_region(eq_df_post,region)
			Y_post_append = get_hours_between( eq_df_post[ mask] )
			# logging.debug('Y_post {}'.format(Y_post))
			if len(Y_post_append) != 0:
				Y_post_append = [1.0/y for y in Y_post_append if y!=0]
			else:
				Y_post_append = [0.0] 			
			for y in Y_post_append:
				Y_post.append(y)
			# add the number of wells per region
			# add the total volume injected per region
			# add them with into X_post as [nb wells, volume]
			X_post_append = [welldf_post[mask_region(welldf_post,region)].count().values[0]
				, welldf_post[mask_region(welldf_post,region)].volume.sum()]

			for i in range(len(Y_post_append)):
				X_post.append(X_post_append)

		X_prior = np.array(X_prior,dtype=np.float64)
		X_post = np.array(X_post,dtype=np.float64)		
		Y_post = np.array(Y_post, dtype=np.float64).reshape(-1,1)
		Y_prior = np.array(Y_prior, dtype = np.float64).reshape(-1,1)


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

			clf, X_test, y_test = do_regression(X,Y,reg,locdict,lock, cv ,standardization)


			logging.debug('{}: For {} cells the score of RidgeCV is {} with alpha = {}'\
				.format(reg,len(locdict.keys()),clf.score(X_test, y_test),clf.alpha_))

			with lock:
				if reg == 'prior': 
					best_grid_prior.append([clf,clf.score(X_test, y_test), interval])
				elif reg == 'post':
					best_grid_post.append([clf,clf.score(X_test, y_test),interval])		


	return


def do_regression(X,Y,reg,locdict,lock,cv, standardization):

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


	return clf, X_test, y_test

def split_in_batch(intervals):

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

	return all_batch

def print_best_score(type):

	best_score_prior = [[c[1],c[2]] for c in best_grid_prior]
	best_score_prior = np.array(best_score_prior)
	best_index_prior = np.where(best_score_prior[:,0] == max(best_score_prior[:,0]))[0][0]
	print('Best classifier <2010 is for alpha ={}'.format(best_grid_prior[best_index_prior][0].alpha_))
	print('Coefs <2010 are ={}'.format(best_grid_prior[best_index_prior][0].coef_))
	print('R^2 <2010 = {}'.format(best_grid_prior[best_index_prior][1]))
	if type == 'grid':
		print('Best interval <2010 is {}'.format(best_grid_prior[best_index_prior][2]))
	elif type == 'cluster':
		print('Best eps <2010 is {}'.format(best_grid_prior[best_index_prior][2]))



	best_score_post = [[c[1],c[2]] for c in best_grid_post]
	best_score_post = np.array(best_score_post)
	best_index_post = np.where(best_score_post[:,0] == max(best_score_post[:,0]))[0][0]
	print('Best classifier >= 2010 is for alpha ={}'.format(best_grid_post[best_index_post][0].alpha_))
	print('Coefs >= 2010 are ={}'.format(best_grid_post[best_index_post][0].coef_))
	print('R^2 >= 2010 = {}'.format(best_grid_post[best_index_post][1]))
	if type == 'grid':
		print('Best interval >= 2010 is {}'.format(best_grid_post[best_index_post][2]))
	elif type == 'cluster':
		print('Best eps >= 2010 is {}'.format(best_grid_post[best_index_post][2]))
 
	return

if __name__ == '__main__':
	

	# ------------------------------------------
	# LOAD DATAFRAMES
	# ------------------------------------------
	# Load the earthquakes datafram 
	eq_df = pd.DataFrame.from_csv('./tempdata/earthquakes_catalog_treated.csv',sep = '|')
	# filter to keep magnitude >= 3
	eq_df  = eq_df[eq_df.prefmag >= 3.0]
	# for ease add column year
	eq_df['year'] = map(lambda x: int(x), eq_df['year_float'])


	# Load the wells dataframe.  
	welldf = pd.DataFrame.from_csv('tempdata/wells_data.csv')


	best_grid_prior = []
	best_grid_post = []

	# define the number of threads
	num_threads = 4

	# ------------------------------------------
	# GRID REGRESSION
	# ------------------------------------------	

	# # define the intervals
	# intervals = [0.05, 0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
	# # intervals = [0.8,0.9, 1.0, 1.5]


 #    # split randomely the letters in batch for the various threads
	# all_batch = split_in_batch(intervals)

	# # Vary the standardization and find optimum
	# for standardization in ['None','scaler','MinMaxScaler'] :
	# 	# parallelize the loop of interval
	# 	threads = []
	# 	lock = threading.Lock()
	# 	for thread_id in range(num_threads):
	# 		interval = all_batch[thread_id]
	# 		t = threading.Thread(target = do_grid_regression, \
	# 			args = (eq_df, welldf, interval, lock ,5,standardization))
	# 		threads.append(t)

	# 	print 'Starting multithreading'
	# 	map(lambda t:t.start(), threads)
	# 	map(lambda t: t.join(), threads)

	# print_best_score('grid')


	# ------------------------------------------
	# CLUSTER INTERARRIVAL REGRESSION
	# ------------------------------------------ 



	eps_batch = range(5,30)
	# eps_batch = [5,7,9,11]

    # split randomely the eps in batch for the various threads
	all_batch = split_in_batch(eps_batch)

	# Vary the standardization and find optimum
	for standardization in ['None','scaler','MinMaxScaler'] :
		# parallelize the loop of interval
		threads = []
		lock = threading.Lock()
		for thread_id in range(num_threads):
			eps_s = all_batch[thread_id]
			t = threading.Thread(target = do_cluster_regression, \
				args = (eq_df, welldf, eps_s, lock ,5, standardization))
			threads.append(t)

		print 'Starting multithreading'
		map(lambda t:t.start(), threads)
		map(lambda t: t.join(), threads)

	print_best_score('cluster')


	# ------------------------------------------
	# GRID 1/INTERARRIVAL REGRESSION
	# ------------------------------------------ 

	# # define the intervals
	# # intervals = [0.05, 0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
	# intervals = [0.8,0.9, 1.0, 1.5]


 #    # split randomely the letters in batch for the various threads
	# all_batch = split_in_batch(intervals)

	# # Vary the standardization and find optimum
	# for standardization in ['None','scaler','MinMaxScaler'] :
	# 	# parallelize the loop of interval
	# 	threads = []
	# 	lock = threading.Lock()
	# 	for thread_id in range(num_threads):
	# 		interval = all_batch[thread_id]
	# 		t = threading.Thread(target = do_grid_interarrival_regression, \
	# 			args = (eq_df, welldf, interval, lock ,5,standardization))
	# 		threads.append(t)

	# 	print 'Starting multithreading'
	# 	map(lambda t:t.start(), threads)
	# 	map(lambda t: t.join(), threads)

	# print_best_score('grid')

