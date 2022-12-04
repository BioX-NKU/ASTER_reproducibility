# Run the experiments on high-noise scCAS datasets.
import sys
sys.path.insert(1, '/home/sccaspurity/program/ASTER')
import epiaster as aster
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import gc

from os import listdir
all_data_name = ['.'.join(f.split('.')[:-1]) for f in listdir('/home/sccaspurity/data/scCASpurity/downsample/ALL_blood/') if 'h5ad' in f and 'dropout_' in f]
all_data_name = np.sort(all_data_name).tolist()
print(len(all_data_name))
print(all_data_name)

res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])

for data_name in all_data_name:
    print('=======', data_name)
    adata = sc.read('/home/sccaspurity/data/scCASpurity/downsample/ALL_blood/%s.h5ad'%data_name)
    k_search = pd.read_csv('/home/sccaspurity/data/scCASpurity/downsample/ALL_blood/%s_search.csv'%data_name, header=None).iloc[0,:].values
    gc.collect()
    true_k = k_search[0]
    search_list = list(k_search[1:])
    
    estimated_k = aster.estimate_k(adata, search_list)
    est_error = estimated_k - true_k
    est_deviation = est_error / true_k

    res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                                'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
    print(res_table)