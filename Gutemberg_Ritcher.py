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



# modified from 
# http://eqrm.googlecode.com/svn-history/r143/trunk/preprocessing/recurrence_from_catalog.py

def calc_recurrence(subset, interval = 0.1):


	print 'Minimum magnitude:', subset.prefmag.min()
	print 'Total number of earthquakes:', subset.shape[0]
	num_years = int(subset.year_float.max()) - int(subset.year_float.min())
	annual_num_eq = subset.shape[0]/num_years
	print 'Annual number of earthquakes greater than Mw', subset.prefmag.min(),':', \
	annual_num_eq
	print 'Maximum catalog magnitude:', subset.prefmag.max()

	# Magnitude bins
	bins = np.arange(subset.prefmag.min(), subset.prefmag.max(), interval)
	# Magnitude bins for plotting - we will re-arrange bins later
	plot_bins = np.arange(subset.prefmag.min(), subset.prefmag.max(), interval)


	###########################################################################
	# Generate distribution
	###########################################################################
	# Generate histogram


	hist = np.histogram(subset.prefmag, bins=bins)

	# Reverse array order
	hist = hist[0][::-1]
	bins = bins[::-1]

	# Calculate cumulative sum
	cum_hist = hist.cumsum()
	# Ensure bins have the same length has the cumulative histogram.
	# Remove the upper bound for the highest interval.
	bins = bins[1:]

	# Get annual rate
	cum_annual_rate = cum_hist/num_years

	new_cum_annual_rate = []
	for i in cum_annual_rate:
	    new_cum_annual_rate.append(i+1e-20)

	# Take logarithm
	log_cum_sum = np.log10(new_cum_annual_rate)

	###########################################################################
	# Plot the results
	###########################################################################

	# Plotting
	print 'size bins', len(bins)
	print 'size new_cum_annual_rate', len(new_cum_annual_rate)
	fig = plt.scatter(bins, new_cum_annual_rate, label = 'Catalogue')
	ax = plt.gca()
	# ax.plot(plot_bins, log_ls_fit, c = 'r', label = 'Least Squares')
	# ax.plot(plot_bins, ls_bounded, c = 'r', linestyle ='--', label = 'Least Squares Bounded')
	# ax.plot(plot_bins, log_mle_fit, c = 'g', label = 'Maximum Likelihood')
	# ax.plot(plot_bins, mle_bounded, c = 'g', linestyle ='--', label = 'Maximum Likelihood Bounded')
	# ax.plot(plot_bins, log_fit_data, c = 'b', label = 'b = 1')

	ax.set_yscale('log')
	ax.legend()
	ax.set_ylim([min(new_cum_annual_rate) * 0.1, max(new_cum_annual_rate) * 10.])
	ax.set_xlim([subset.prefmag.min() - 0.5, subset.prefmag.max() + 0.5])
	ax.set_ylabel('Annual probability')
	ax.set_xlabel('Magnitude')
	plt.show()
	return







def my_plotting_function(subset):
	eq_count, base = np.histogram(subset['prefmag'], bins = subset.shape[0]/20)
	plt.figure(2,figsize = (9,7))
	plt.plot(base[-1:0:-1], np.log(np.cumsum(eq_count)), lw=3,c='r', label='Earthquakes' )
	plt.ylabel('Log of occurences')
	plt.xlabel('Magnitude')
	plt.legend(loc =1);
	plt.show()
	return

if __name__=="__main__":
	# ------------------------------------------
	# LOAD DATAFRAMES 
	# ------------------------------------------
	 
	# Load the earthquakes datafram 
	eq_df = pd.DataFrame.from_csv('./tempdata/earthquakes_catalog.csv',sep = '|')


	# ------------------------------------------
	# PLOT G-R FOR <= 2010
	# ------------------------------------------
	# filter after 2010
	subset  = eq_df[eq_df.year_float <= 2010]
	my_plotting_function(subset)









