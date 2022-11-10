#!/usr/bin/env python
# coding: utf-8

# # MSCI Multiome Torch Quickstart Submission
# This notebook creates submissions from the models trained in [this notebook](https://www.kaggle.com/fabiencrom/msci-multiome-torch-quickstart-w-sparse-tensors).
# 
# We only predict the Multiome data and then merge in the CITEseq results from [this notebook](https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart/notebook) by AmbrosM, which has the highest public score at the time I am publishing.
# 
# So far we do not get better results than the one obtained by the much simpler PCA+Ridge Regression method (that you can find in [this notebook](https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart) as initially proposed by AmbrosM or in [this notebook](https://www.kaggle.com/code/fabiencrom/msci-multiome-quickstart-w-sparse-matrices) for a version using sparse matrices for better results). But I expect it can be made to perform better after improving the architecture/hyperparameters.
# 

# In[1]:


import os
import copy
import gc
import math
import itertools
import pickle
import glob
import joblib
import json
import random
import re
import operator

import collections
from collections import defaultdict
from operator import itemgetter, attrgetter

from tqdm.notebook import tqdm

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import plotly.express as px

import scipy

import sklearn
import sklearn.cluster
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.preprocessing

import copy


# In[2]:


def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation.
    """
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


# # Utility functions for loading and batching the sparse data in device memory

# In[3]:


# Strangely, current torch implementation of csr tensor do not accept to be moved to the gpu. 
# So we make our own equivalent class
TorchCSR = collections.namedtuple("TrochCSR", "data indices indptr shape")

def load_csr_data_to_gpu(train_inputs):
    """Move a scipy csr sparse matrix to the gpu as a TorchCSR object
    This try to manage memory efficiently by creating the tensors and moving them to the gpu one by one
    """
    th_data = torch.from_numpy(train_inputs.data).to(device)
    th_indices = torch.from_numpy(train_inputs.indices).to(device)
    th_indptr = torch.from_numpy(train_inputs.indptr).to(device)
    th_shape = train_inputs.shape
    return TorchCSR(th_data, th_indices, th_indptr, th_shape)

def make_coo_batch(torch_csr, indx):
    """Make a coo torch tensor from a TorchCSR object by taking the rows indicated by the indx tensor
    """
    th_data, th_indices, th_indptr, th_shape = torch_csr
    start_pts = th_indptr[indx]
    end_pts = th_indptr[indx+1]
    coo_data = torch.cat([th_data[start_pts[i]: end_pts[i]] for i in range(len(start_pts))], dim=0)
    coo_col = torch.cat([th_indices[start_pts[i]: end_pts[i]] for i in range(len(start_pts))], dim=0)
    coo_row = torch.repeat_interleave(torch.arange(indx.shape[0], device=device), th_indptr[indx+1] - th_indptr[indx])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [indx.shape[0], th_shape[1]])
    return coo_batch


def make_coo_batch_slice(torch_csr, start, end):
    """Make a coo torch tensor from a TorchCSR object by taking the rows within the (start, end) slice
    """
    th_data, th_indices, th_indptr, th_shape = torch_csr
    if end > th_shape[0]:
        end = th_shape[0]
    start_pts = th_indptr[start]
    end_pts = th_indptr[end]
    coo_data = th_data[start_pts: end_pts]
    coo_col = th_indices[start_pts: end_pts]
    coo_row = torch.repeat_interleave(torch.arange(end-start, device=device), th_indptr[start+1:end+1] - th_indptr[start:end])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [end-start, th_shape[1]])
    return coo_batch


# # GPU memory DataLoader

# In[4]:


class DataLoaderCOO:
    """Torch compatible DataLoader. Works with in-device TorchCSR tensors.
    Args:
         - train_inputs, train_targets: TorchCSR tensors
         - train_idx: tensor containing the indices of the rows of train_inputs and train_targets that should be used
         - batch_size, shuffle, drop_last: as in torch.utils.data.DataLoader
    """
    def __init__(self, train_inputs, train_targets, train_idx=None, 
                 *,
                batch_size=512, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        
        self.train_idx = train_idx
        
        self.nb_examples = len(self.train_idx) if self.train_idx is not None else train_inputs.shape[0]
        
        self.nb_batches = self.nb_examples//batch_size
        if not drop_last and not self.nb_examples%batch_size==0:
            self.nb_batches +=1
        
    def __iter__(self):
        if self.shuffle:
            shuffled_idx = torch.randperm(self.nb_examples, device=device)
            if self.train_idx is not None:
                idx_array = self.train_idx[shuffled_idx]
            else:
                idx_array = shuffled_idx
        else:
            if self.train_idx is not None:
                idx_array = self.train_idx
            else:
                idx_array = None
            
        for i in range(self.nb_batches):
            slc = slice(i*self.batch_size, (i+1)*self.batch_size)
            if idx_array is None:
                inp_batch = make_coo_batch_slice(self.train_inputs, i*self.batch_size, (i+1)*self.batch_size)
                if self.train_targets is None:
                    tgt_batch = None
                else:
                    tgt_batch = make_coo_batch_slice(self.train_targets, i*self.batch_size, (i+1)*self.batch_size)
            else:
                idx_batch = idx_array[slc]
                inp_batch = make_coo_batch(self.train_inputs, idx_batch)
                if self.train_targets is None:
                    tgt_batch = None
                else:
                    tgt_batch = make_coo_batch(self.train_targets, idx_batch)
            yield inp_batch, tgt_batch
            
            
    def __len__(self):
        return self.nb_batches


# # Simple Model: MLP

# In[5]:


class MLP(nn.Module):
    def __init__(self, layer_size_lst, add_final_activation=False):
        super().__init__()
        
        assert len(layer_size_lst) > 2
        
        layer_lst = []
        for i in range(len(layer_size_lst)-1):
            sz1 = layer_size_lst[i]
            sz2 = layer_size_lst[i+1]
            layer_lst += [nn.Linear(sz1, sz2)]
            if i != len(layer_size_lst)-2 or add_final_activation:
                 layer_lst += [nn.ReLU()]
        self.mlp = nn.Sequential(*layer_lst)
        
    def forward(self, x):
        return self.mlp(x)
    
def build_model():
    model = MLP([INPUT_SIZE] + config["layers"] + [OUTPUT_SIZE])
    if config["head"] == "softplus":
        model = nn.Sequential(model, nn.Softplus())
    else:
        assert config["head"] is None
    return model


# # test_fn function

# In[6]:


def test_fn_ensemble(model_list, dl_test):

    res = torch.empty(
        (dl_test.nb_examples, OUTPUT_SIZE), 
        device=device, dtype=torch.float32)
    
#     all_preds = []
    for model in model_list:
        model.eval()
        
    cur = 0
    for inpt, tgt in tqdm(dl_test):
        mb_size = inpt.shape[0]

        with torch.no_grad():
            pred_list = []
            for model in model_list:
                pred = model(inpt)
                pred_list.append(pred)
            pred = sum(pred_list)/len(pred_list)
            
#         print(res.shape, cur, cur+pred.shape[0], res[cur:cur+pred.shape[0]].shape, pred.shape)
        res[cur:cur+pred.shape[0]] = pred
        cur += pred.shape[0]
            
    return {"preds":res}


# # Loading Data

# In[7]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"machine has {torch.cuda.device_count()} cuda devices")
    print(f"model of first cuda device is {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")


# In[8]:


INPUT_SIZE = 228942 
OUTPUT_SIZE = 23418


# In[9]:


max_inputs = np.load("../input/msci-multiome-torch-quickstart-w-sparse-tensors/max_inputs.npz")["max_inputs"]
max_inputs = torch.from_numpy(max_inputs)[0].to(device)


# In[10]:


get_ipython().run_cell_magic('time', '', 'test_inputs = scipy.sparse.load_npz(\n    "../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_values.sparse.npz")')


# In[11]:


get_ipython().run_cell_magic('time', '', 'test_inputs = load_csr_data_to_gpu(test_inputs)\ngc.collect()')


# In[12]:


test_inputs.data[...] /= max_inputs[test_inputs.indices.long()]


# In[13]:


torch.max(test_inputs.data)


# # Load trained models

# In[14]:


model_list = []
for fn in tqdm(glob.glob("../input/msci-multiome-torch-quickstart-w-sparse-tensors/*_best_params.pth")):
    prefix = fn[:-len("_best_params.pth")]
    config_fn = prefix + "_config.pkl"
    
    config = pickle.load(open(config_fn, "rb"))
    
    model = build_model() 
    model.to(device)
    
    params = torch.load(fn)
    model.load_state_dict(params)
    
    model_list.append(model)


# # Generate Multiome predictions

# In[15]:


dl_test = DataLoaderCOO(test_inputs, None, train_idx=None,
                batch_size=512, shuffle=False, drop_last=False)


# In[16]:


test_pred = test_fn_ensemble(model_list, dl_test)["preds"]


# In[17]:


del model_list
del dl_test
del test_inputs
gc.collect()


# In[18]:


test_pred.shape


# # Creating the final submission

# In[19]:


get_ipython().run_cell_magic('time', '', '# Read the table of rows and columns required for submission\neval_ids = pd.read_parquet("../input/multimodal-single-cell-as-sparse-matrix/evaluation.parquet")\n\n# Convert the string columns to more efficient categorical types\n#eval_ids.cell_id = eval_ids.cell_id.apply(lambda s: int(s, base=16))\n\neval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())\neval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())')


# In[20]:


# Prepare an empty series which will be filled with predictions
submission = pd.Series(name='target',
                       index=pd.MultiIndex.from_frame(eval_ids), 
                       dtype=np.float32)
submission


# In[21]:


get_ipython().run_cell_magic('time', '', 'y_columns = np.load("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_idxcol.npz",\n                   allow_pickle=True)["columns"]\n\ntest_index = np.load("../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz",\n                    allow_pickle=True)["index"]')


# In[22]:


cell_dict = dict((k,v) for v,k in enumerate(test_index)) 
assert len(cell_dict)  == len(test_index)

gene_dict = dict((k,v) for v,k in enumerate(y_columns))
assert len(gene_dict) == len(y_columns)

eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))

valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)


# In[23]:


valid_multi_rows = valid_multi_rows.to_numpy()


# In[24]:


eval_ids_gene_num[valid_multi_rows].to_numpy()


# In[25]:


submission.iloc[valid_multi_rows] = test_pred[eval_ids_cell_num[valid_multi_rows].to_numpy(),
eval_ids_gene_num[valid_multi_rows].to_numpy()].cpu().numpy()

del eval_ids_cell_num, eval_ids_gene_num, valid_multi_rows, eval_ids, test_index, y_columns
gc.collect()


# In[26]:


submission


# In[27]:


submission.reset_index(drop=True, inplace=True)
submission.index.name = 'row_id'


# # Merging in the CITEseq submission
# 
# We take the CITEseq results from [this notebook](https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart/notebook) by AmbrosM.

# In[28]:


cite_submission = pd.read_csv("../input/msci-citeseq-keras-quickstart/submission.csv")
cite_submission = cite_submission.set_index("row_id")
cite_submission = cite_submission["target"]


# In[29]:


submission[submission.isnull()] = cite_submission[submission.isnull()]


# In[30]:


submission


# In[31]:


submission.isnull().any()


# In[32]:


submission.to_csv("submission.csv")


# In[33]:


get_ipython().system('head submission.csv')


# In[ ]:





# In[ ]:




