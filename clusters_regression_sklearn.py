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


from grid_regression_sklearn import mask_cluster, \
		cluster_centroid, get_furthest_distance, \
		get_cluster_nwells_volume, do_cluster_regression


from geopy.distance import great_circle

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


	# ------------------------------------------
	# COMMENT OUT TO DO PARALLEL
	# ------------------------------------------

	# # define the number of threads
	# num_threads = 4

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

	# best_grid_prior = []
	# best_grid_post = []

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

	# ------------------------------------------
	# COMMENT OUT TO DO SERIAL
	# ------------------------------------------


	X_prior = []
	X_post = []
	Y_prior = []
	Y_post = [] 

	# reconstruct the column name
	period= 'prior'
	eps = 15

	lock = threading.Lock()

	best_grid_prior = []
	best_grid_post = []

	global best_grid_prior
	global best_grid_post

	do_cluster_regression(eq_df, welldf, eps, lock)


	# ------------------------------------------
	# RESULTS SUMMARY
	# ------------------------------------------

	# best_score_prior = [[c[1],c[2]] for c in best_grid_prior]
	# best_score_prior = np.array(best_score_prior)
	# best_index_prior = np.where(best_score_prior[:,0] == max(best_score_prior[:,0]))[0][0]
	# print('Best classifier <2010 is for alpha ={}'.format(best_grid_prior[best_index_prior][0].alpha_))
	# print('Coefs <2010 are ={}'.format(best_grid_prior[best_index_prior][0].coef_))
	# print('R^2 <2010 = {}'.format(best_grid_prior[best_index_prior][1]))
	# print('Best interval <2010 is {}'.format(best_grid_prior[best_index_prior][2]))


	# best_score_post = [[c[1],c[2]] for c in best_grid_post]
	# best_score_post = np.array(best_score_post)
	# best_index_post = np.where(best_score_post[:,0] == max(best_score_post[:,0]))[0][0]
	# print('Best classifier >= 2010 is for alpha ={}'.format(best_grid_post[best_index_post][0].alpha_))
	# print('Coefs >= 2010 are ={}'.format(best_grid_post[best_index_post][0].coef_))
	# print('R^2 >= 2010 = {}'.format(best_grid_post[best_index_post][1]))
	# print('Best interval >= 2010 is {}'.format(best_grid_post[best_index_post][2]))










