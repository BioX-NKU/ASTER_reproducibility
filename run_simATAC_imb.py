# Run the experiments on the imbalanced scCAS datasets.
import scanpy as sc
import episcanpy.api as epi
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import hdf5storage
import scipy.io as scio
import sklearn
import pickle
import scipy
from sklearn.metrics import silhouette_score
from IPython.display import display
from os import listdir

import warnings
warnings.filterwarnings("ignore")
import gc
gc.collect()
import os


data_name = 'donor_BM0828'
adata = sc.read('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_50.h5ad'%(data_name,'CLP'))
adata.obs['cell_type'] = 'CLP'
adata_concat = adata.copy()

for celltype in ['HSC', 'MPP', 'LMPP', 'CMP', 'GMP', 'MEP']:
    adata = sc.read('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_500.h5ad'%(data_name,celltype))
    adata.obs['cell_type'] = celltype
    adata_concat = sc.concat([adata_concat, adata])

adata_concat.write('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_imbalance1.h5ad'%(data_name,data_name))


res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])
adata = sc.read("/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_imbalance1.h5ad"%(data_name,data_name))
k_search = pd.read_csv('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_search.csv'%data_name, header=None).iloc[0,:].values
true_k = k_search[0]
search_list = list(k_search[1:])
gc.collect()

import sys
sys.path.insert(1, '/home/sccaspurity/program/ASTER')
import epiaster as aster

estimated_k = aster.estimate_k(adata, search_list)
est_error = estimated_k - true_k
est_deviation = est_error / true_k

res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                            'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
print(res_table)






data_name = 'donor_BM0828'
adata = sc.read('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_9700.h5ad'%(data_name,'CLP'))
adata.obs['cell_type'] = 'CLP'
adata_concat = adata.copy()

for celltype in ['HSC', 'MPP', 'LMPP', 'CMP', 'GMP', 'MEP']:
    adata = sc.read('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_50.h5ad'%(data_name,celltype))
    adata.obs['cell_type'] = celltype
    adata_concat = sc.concat([adata_concat, adata])

adata_concat.write('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_imbalance2.h5ad'%(data_name,data_name))


res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])
adata = sc.read("/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_imbalance2.h5ad"%(data_name,data_name))
k_search = pd.read_csv('/home/sccaspurity/data/scCASpurity/simATAC/output/%s/%s_search.csv'%data_name, header=None).iloc[0,:].values
true_k = k_search[0]
search_list = list(k_search[1:])
gc.collect()

import sys
sys.path.insert(1, '/home/sccaspurity/program/ASTER')
import epiaster as aster

estimated_k = aster.estimate_k(adata, search_list)
est_error = estimated_k - true_k
est_deviation = est_error / true_k

res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                            'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
print(res_table)

