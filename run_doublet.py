# Run the experiments on the scCAS datasets with doublets. 
import sys
sys.path.insert(1, '/home/sccaspurity/program/ASTER')
import epiaster as aster
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import gc

from os import listdir
all_data_name = [f.split('.')[0] for f in listdir('/data/cabins/chenshengquan/scCAS221012/GSE162690/') if 'h5ad' in f]
print(all_data_name)
print(len(all_data_name))

import os
res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])

for data_name in all_data_name:
    print('=======', data_name)
    adata = sc.read('/data/cabins/chenshengquan/scCAS221012/GSE162690/%s.h5ad'%data_name)
    gc.collect()
    adata = adata[adata.obs.cell_type!='AMB']
    k_search = pd.read_csv('/data/cabins/chenshengquan/scCAS221012/GSE162690/%s_search.csv'%data_name, header=None).iloc[0,:].values
    true_k = k_search[0]
    search_list = list(k_search[1:])
    estimated_k = aster.estimate_k(adata, search_list)
    est_error = estimated_k - true_k
    est_deviation = est_error / true_k

    res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                                'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
    print(res_table)