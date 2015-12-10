# test debug get_hours_between

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
from grid_regression_sklearn import mask_region,partition_state, mask_cluster


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

	# implement serial code for the clusters here
	eps = 15
	cluster_id = 0


	eq_df_prior = eq_df[eq_df.year < 2010]
	welldf_prior = welldf[welldf.year < 2010]
	eq_df_post = eq_df[eq_df.year >= 2010]
	welldf_post = welldf[welldf.year >= 2010]

	X_prior = []
	X_post = []
	Y_prior = []
	Y_post = [] 

	print eq_df_prior[  eq_df_prior['cluster_M_eps_' + str(eps)] == cluster_id  ].head()
	# print eq_df_prior[mask_cluster(eq_df_prior,eps, cluster_id)].head()

	# Y_prior.append(eq_df_prior[mask_cluster(eq_df_prior,eps, cluster_id)].count().values[0])

	# X_prior.append([welldf_prior[mask_cluster(welldf_prior,eps, cluster_id)].count().values[0]
	# 	, welldf_prior[mask_cluster(welldf_prior,eps, cluster_id)].volume.sum()])



	# best_grid_prior = []
	# best_grid_post = []



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


