import pandas as pd
import pickle
results_dic = pickle.load( open( "./tempdata/results_dic.p", "rb" ) )


'''
{'regression_grid_interarrival':
{'prior': {'interval': 0.8, 'r2': 0.19350546319008832, 'coefs': array([[ -1.71288504e-06,   1.19736194e-11]])},
'post': {'interval': 0.3, 'r2': 0.0011716005451117839, 'coefs': array([[ -1.63254401e+03,  -5.34621836e-05]])}},
'regression_cluster_interarrival':
{'prior': {'eps': 26, 'r2': 0.22250060743783628, 'coefs': array([ 1.82393251, -1.7680387 ])},
'post': {'eps': 26, 'r2': 0.22576308680614376, 'coefs': array([ 1.35792326, -1.30444587])}}, 
'grid_regression':
 {'prior': {'interval': 2.0, 'r2': 0.74391360527313899, 'coefs': array([[-1.91436847,  2.79175104]])},
 'post': {'interval': 1.5, 'r2': 0.43457634500713527, 'coefs': array([[ -3.29028433e-01,   2.39351110e-06]])}}}
'''

print results_dic['grid_regression']['prior']

results = {}
results['grid_regression_prior'] = results_dic['grid_regression']['prior']
results['grid_regression_post'] = results_dic['grid_regression']['post']

results['regression_cluster_interarrival_prior'] = results_dic['regression_cluster_interarrival']['prior']
results['regression_cluster_interarrival_post'] = results_dic['regression_cluster_interarrival']['post']

results['regression_grid_interarrival_prior'] = results_dic['regression_grid_interarrival']['prior']
results['regression_grid_interarrival_post'] = results_dic['regression_grid_interarrival']['post']

results_df = pd.DataFrame.from_dict(results, orient = 'index')

results_df.to_csv('./tempdata/results_df.csv', sep = ',')