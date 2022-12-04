# Run the experiments on the scCAS datasets with multiple batches. We demonstrate here how to integrate ASTER with Harmony to correct batch effects.
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
import scipy.cluster.hierarchy as shc
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


import sys
sys.path.insert(1, '/home/sccaspurity/program/ASTER')
import epiaster as aster


data_name = 'Muto-2021-ATAC'
res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])

adata = sc.read('/data/cabins/chenshengquan/scglue/%s.h5ad'%data_name)
k_search = pd.read_csv('/data/cabins/chenshengquan/scglue/%s_search.csv'%data_name, header=None).iloc[0,:].values
gc.collect()
true_k = k_search[0]
search_list = list(k_search[1:])

estimated_k = aster.estimate_k(adata, search_list)
est_error = estimated_k - true_k
est_deviation = est_error / true_k

res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                            'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
print(res_table)





import random
import numpy as np
import scanpy as sc
import scanpy.external as sce

def setup_seed(seed):
    """
    Set random seed.
    
    Parameters
    ----------
    seed
        Number to be set as random seed for reproducibility.
        
    """
    np.random.seed(seed)
    random.seed(seed)

def getNClusters(adata, n_cluster, range_min=0, range_max=3, max_steps=20, method='louvain', key_added=None):
    """
    Tune the resolution parameter in Louvain or Leiden clustering to make the number of clusters and the specified number as close as possible.
    
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    n_cluster
        Specified number of clusters.
    range_min
        Minimum clustering resolution for the binary search. By default, `range_min=0`.
    range_max
        Maximum clustering resolution for the binary search. By default, `range_max=3`.
    max_steps
        Maximum number of steps for the binary search. By default, `max_steps=20`.
    method
        Method (`louvain` or `leiden`) used for cell clustering. By default, `method='louvain'`.
    key_added
        The obs variable name of clustering results. By default, `key_added` is the same as the method name used.

    Returns
    -------
    adata
        AnnData object with clustering assignments in `adata.obs`:

        - `adata.obs['louvain']` - Louvain clustering assignments if `method='louvain'` and `key_added=None`.
        - `adata.obs['leiden']` - Leiden clustering assignments if `method='leiden'` and `key_added=None`.
    """
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    pre_min_cluster = 0
    pre_max_cluster = None
    min_update = False
    max_update = False
    while this_step < max_steps:
        this_step += 1
        if this_step==1: 
            this_resolution = this_min + ((this_max-this_min)/2)
        else:
            if max_update and not min_update:
                this_resolution = this_min + (this_max-this_min) * (n_cluster-pre_min_cluster)/(pre_max_cluster-pre_min_cluster)
            if min_update and not max_update:
                if pre_max_cluster is not None:
                    this_resolution = this_min + (this_max-this_min) * (n_cluster-pre_min_cluster)/(pre_max_cluster-pre_min_cluster)
                else:
                    this_resolution = this_min + ((this_max-this_min)/2)
                    
        if (method == 'louvain') and (key_added==None):
            sc.tl.louvain(adata, resolution=this_resolution)
        elif (method == 'louvain') and (type(key_added)==str):
            sc.tl.louvain(adata, resolution=this_resolution, key_added=key_added)
        elif (method == 'leiden') and (key_added==None):
            sc.tl.leiden(adata,resolution=this_resolution)
        elif (method == 'leiden') and (type(key_added)==str):
            sc.tl.leiden(adata,resolution=this_resolution, key_added=key_added)
        else:
            print('Error settings of method and key_added.')
        
        if key_added==None:
            this_clusters = adata.obs[method].nunique()
        else:
            this_clusters = adata.obs[key_added].nunique()
                
        if this_clusters > n_cluster:
            this_max = this_resolution
            pre_max_cluster = this_clusters
            min_update = False
            max_update = True
        elif this_clusters < n_cluster:
            this_min = this_resolution
            pre_min_cluster = this_clusters
            min_update = True
            max_update = False
        elif this_clusters == n_cluster:
            break

import scipy
import numpy as np
import scanpy as sc
sc.settings.verbosity = 0

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfTransformer


def ssd_knee_est(adata, search_list, seed=None):
    """
    Estimate the number of cell types by sum of squared distances.
    
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    search_list
        List of optional numbers of cell types for the estimation.
    seed
        Random seed for reproducibility. By default, `seed=None`.
        
    Returns
    -------
    sk_est
        Estimated number of cell types by sum of squared distances.

    """ 
    
    print('Estimating by sum of squared distances...')
    if seed is not None: setup_seed(seed)

    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(adata.X)
    model.idf_ -= 1
    tf_idf = scipy.sparse.csr_matrix(model.transform(adata.X))
    adata.X = tf_idf.copy()
        
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack', use_highly_variable=False)
    sce.pp.harmony_integrate(adata, 'batch')
    
    distances = []
    for k in search_list:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(adata.obsm['X_pca_harmony'])
        distances.append(kmeanModel.inertia_)
    kl = KneeLocator(search_list, distances, S=1.0, curve="convex", direction="decreasing")
    sk_est = kl.knee
    
    return sk_est


def davies_bouldin_est(adata, search_list, seed=None):
    """
    Estimate the number of cell types by Davies-Bouldin score.
    
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    search_list
        List of optional numbers of cell types for the estimation.
    seed
        Random seed for reproducibility. By default, `seed=None`.
        
    Returns
    -------
    db_est
        Estimated number of cell types by Davies-Bouldin score.

    """ 
    
    print('Estimating by Davies-Bouldin score...')
    if seed is not None: setup_seed(seed)

    count_mat = adata.X.T.copy()
    nfreqs = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    tfidf_mat = np.multiply(nfreqs, np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1])))
    adata.X = scipy.sparse.csr_matrix(tfidf_mat).T
        
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack', use_highly_variable=False)
    sce.pp.harmony_integrate(adata, 'batch')
    
    scores = []
    for k in search_list:
        kmeans = KMeans(n_clusters=k)
        model = kmeans.fit_predict(adata.obsm['X_pca_harmony'])
        score = davies_bouldin_score(adata.obsm['X_pca_harmony'], model)
        scores.append(score)
    db_est = search_list[np.argmin(scores)]
    
    return db_est


def silhouette_est(adata, search_list, seed=None):
    """
    Estimate the number of cell types by silhouette coefficient.
    
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    search_list
        List of optional numbers of cell types for the estimation.
    seed
        Random seed for reproducibility. By default, `seed=None`.
        
    Returns
    -------
    sil_est
        Estimated number of cell types by silhouette coefficient.

    """ 
    
    print('Estimating by silhouette coefficient...')
    if seed is not None: setup_seed(seed)

    count_mat = adata.X.T.copy()
    nfreqs = 1.0 * count_mat / np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    tfidf_mat = np.multiply(nfreqs, np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1])))
    adata.X = scipy.sparse.csr_matrix(tfidf_mat).T
        
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack', use_highly_variable=False)
    sce.pp.harmony_integrate(adata, 'batch')
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15, n_pcs=50, method='umap', metric='euclidean')
    
    sil_louvain = []
    sil_leiden  = []
    for k in search_list:
        getNClusters(adata, n_cluster=k, method='louvain');
        sil_louvain.append(silhouette_score(adata.obsm['X_pca_harmony'], adata.obs['louvain'], metric='correlation'))
        getNClusters(adata, n_cluster=k, method='leiden');
        sil_leiden.append(silhouette_score(adata.obsm['X_pca_harmony'], adata.obs['leiden'], metric='correlation'))  
    sil_est = search_list[np.argmax(np.array(sil_louvain) + np.array(sil_leiden))]
    
    return sil_est

import gc
import numpy as np
import episcanpy.api as epi

def estimate_k(adata, search_list, binary=True, fpeak=0.01, seed=2022):
    """
    Estimate the number of cell types in scCAS data by ASTER.
    
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    search_list
        List of optional numbers of cell types for the estimation.
    binary
        Whether to convert the count matrix into a binary matrix. By default, `binary=True`.
    fpeak
        Select peaks/regions that have at least one read count in at least `fpeak` of the cells in the count matrix. By default, `fpeak=0.01`.
    seed
        Random seed for reproducibility. By default, `seed=None`.
        
    Returns
    -------
    estimated_k
        Estimated number of cell types.

    """
    print('Raw dataset shape: ', adata.shape)
    if binary: epi.pp.binarize(adata)
    epi.pp.filter_features(adata, min_cells=np.ceil(fpeak*adata.shape[0]))
    if np.sum(np.sum(adata.X, axis=1)==0) > 0: 
        print("There are empty cells after filtering features. These cells will be removed. Alternatively, use a smaller fpeak to avoid empty cells.")
        epi.pp.filter_cells(adata, min_features=1)
    print('Dataset shape after preprocessing: ', adata.shape)
    
    adata_sk = adata.copy()
    sk_est = ssd_knee_est(adata_sk, search_list, seed=seed)
    del adata_sk
    gc.collect()
    
    adata_db = adata.copy()
    db_est = davies_bouldin_est(adata_db, search_list, seed=seed)
    del adata_db
    gc.collect()
    
    adata_sil = adata.copy()
    sil_est = silhouette_est(adata_sil, search_list, seed=seed)
    del adata_sil
    gc.collect()
    
    est_sum = 0; cnt = 0
    if sk_est  is not None: est_sum += sk_est;  cnt += 1
    if db_est  is not None: est_sum += db_est;  cnt += 1
    if sil_est is not None: est_sum += sil_est; cnt += 1
    estimated_k = int(np.ceil(1.0*est_sum/cnt))
    
    return estimated_k, sk_est, db_est, sil_est


data_name = 'Muto-2021-ATAC'
res_table = pd.DataFrame(columns=['data_name', 'true_k', 'estimated_k', 'est_error', 'est_deviation'])

adata = sc.read('/data/cabins/chenshengquan/scglue/%s.h5ad'%data_name)
k_search = pd.read_csv('/data/cabins/chenshengquan/scglue/%s_search.csv'%data_name, header=None).iloc[0,:].values
gc.collect()
true_k = k_search[0]
search_list = list(k_search[1:])

estimated_k, sk_est, db_est, sil_est = estimate_k(adata, search_list)
est_error = estimated_k - true_k
est_deviation = est_error / true_k

res_table = res_table.append({'data_name':data_name, 'true_k':true_k, 
                            'estimated_k':estimated_k, 'est_error':est_error, 'est_deviation':est_deviation}, ignore_index=True)
print(res_table)
