# Run the model ablation experiments.
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

def variant_res(res_table_pd, method_list):
    res_test = pd.DataFrame(columns=['data_name', 'true_num', 'pred_num', 'pred_err', 'deviation'])
    for data in res_table_pd['data_name']:
        cnt = 0
        res_all = 0
        for method in method_list:
            res_num = res_table_pd.loc[res_table_pd['data_name']==data][method].values[0]
            if not np.isnan(res_num):
                res_all += res_num
                cnt += 1
        true_num = res_table_pd.loc[res_table_pd['data_name']==data]['true_num'].values[0]
        if cnt==0: 
            continue
        else:
            pred_num = np.ceil(1.0*res_all/cnt)
	        pred_err = pred_num - true_num
	        deviation= (pred_num - true_num)/true_num
	        res_test = res_test.append({'data_name':data, 
	                                    'true_num':true_num, 'pred_num':pred_num, 
	                                    'pred_err':pred_err, 'deviation':deviation}, ignore_index=True)
    return res_test


reproduce_single = pd.read_csv('./res/variants/reproduce_single.csv')
ASTER = variant_res(reproduce_single, ['sk_est', 'db_est', 'sil_est'])
ASTER_WSS_DB = variant_res(reproduce_single, ['sk_est', 'db_est'])
ASTER_DB_SC = variant_res(reproduce_single, ['db_est', 'sil_est'])
ASTER_WSS_SC = variant_res(reproduce_single, ['sk_est', 'sil_est'])
ASTER_WSS = variant_res(reproduce_single, ['sk_est'])
ASTER_DB = variant_res(reproduce_single, ['db_est'])
ASTER_SC = variant_res(reproduce_single, ['sil_est'])

# ASTER, ASTER_WSS_DB, ASTER_DB_SC, ASTER_WSS_SC, ASTER_WSS, ASTER_DB, ASTER_SC
res_table = pd.DataFrame(columns=['data_name', 'true_k', 'method', 'metric', 'value'], dtype=float)
gc.collect()

tmp_res = ASTER
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

tmp_res = ASTER_WSS_DB
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_WSS_DB']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_WSS_DB']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

tmp_res = ASTER_DB_SC
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_DB_SC']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_DB_SC']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

tmp_res = ASTER_WSS_SC
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_WSS_SC']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_WSS_SC']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

tmp_res = ASTER_WSS
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_WSS']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_WSS']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

tmp_res = ASTER_DB
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_DB']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_DB']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

tmp_res = ASTER_SC
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_SC']*tmp_res.shape[0], 'metric':'pred_err', 'value':tmp_res.pred_err}), ignore_index=True)
res_table = res_table.append(pd.DataFrame({'data_name':tmp_res.data_name.values, 'true_k':tmp_res.pred_num.values, 
                              'method':['ASTER_SC']*tmp_res.shape[0], 'metric':'deviation', 'value':tmp_res.deviation}), ignore_index=True)

print(res_table)
    

import matplotlib as mpl
import seaborn as sns
import matplotlib.ticker as ticker
mpl.rcParams['pdf.fonttype'] = 42
plt.style.use('default')

res_table.value = np.abs(res_table.value)
ax = sns.boxplot(data=res_table[res_table.metric=='deviation'], x="metric", y="value", hue="method", 
                 boxprops={'alpha': 0.5})
sns.stripplot(data=res_table[res_table.metric=='deviation'], x = "metric", y = "value", hue="method",
              dodge=True, ax=ax)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig("./res/variants_box_plot.pdf", format="pdf", bbox_inches="tight")
