# Run the experiments to benchmark computational time and memory usage. Run the script via
# /usr/bin/time -v python rev_timeMEM.py
import sys
sys.path.insert(1, '/home/sccaspurity/program/ASTER')
import epiaster as aster
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import gc

import os
res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])

data_name = 'subset_0.5'
print('=======', data_name)
adata = sc.read("/home/sccaspurity/data/scCASpurity/downsample/Droplet/%s.h5ad"%data_name)
k_search = pd.read_csv('/home/sccaspurity/data/scCASpurity/downsample/Droplet/%s_search.csv'%data_name, header=None).iloc[0,:].values
true_k = k_search[0]
search_list = list(k_search[1:])
gc.collect()

estimated_k = aster.estimate_k(adata, search_list)
est_error = estimated_k - true_k
est_deviation = est_error / true_k

res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                            'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
print(res_table)