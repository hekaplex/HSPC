#!/usr/bin/env python
# coding: utf-8

# # Multimodal Single-Cell Integration: Creating a Sparse Matrix Dataset

# This notebook goal is to generate a more efficient version of the dataset of the competition "Open Problems: Multimodal Single-Cell Integration", using scipy's sparse matrices.
# 
# Using sparse matrices will lead to:
# - smaller file sizes
# - faster loading
# - much smaller memory footprint (so that you can actually keep all data in memory)
# 
# (Especially for the multiome data, which is very big and very sparse; not so much for the citeseq data which is smaller and only ~75% sparse).
# 
# The downside is that we cannot use the nice pandas DataFrame anymore. Instead, each "*xxx.h5*" file is converted into two files:
# - One "*xxx_values.sparse*" file that can be loaded with `scipy.sparse.load_npz` and contains all the values of the corresponding dataframe (i.e. the result of `df.values` in a sparse format)
# - One "*xxx_idxcol.npz*" file that can be loaded with `np.load` and contains the values of the index and the columns of the corresponding dataframe (i.e the results of `df.index` and `df.columns`)
# 
# For convenience, the csv files are also converted into a more efficient parquet version.
# 
# The generated dataset is available [here](https://www.kaggle.com/datasets/fabiencrom/multimodal-single-cell-as-sparse-matrix).

# In[1]:


get_ipython().system('conda install pytables -y')


# In[2]:


import pandas as pd
import numpy as np
import scipy.sparse


# # Conversion Functions

# In[3]:


def convert_to_parquet(filename, out_filename):
    df = pd.read_csv(filename)
    df.to_parquet(out_filename + ".parquet")


# In[4]:


import scipy
def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None
    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start+chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize: 
            del df_chunk
            break
        del df_chunk
        start += chunksize
        
    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list
    
    all_indices = np.hstack(chunks_index_list)
    
    scipy.sparse.save_npz(out_filename+"_values.sparse", all_data_sparse)
    np.savez(out_filename+"_idxcol.npz", index=all_indices, columns =columns_name)
    
    


# # H5 -> Sparse Conversion

# In[5]:


convert_h5_to_sparse_csr("../input/open-problems-multimodal/train_multi_targets.h5", "train_multi_targets")


# In[6]:


convert_h5_to_sparse_csr("../input/open-problems-multimodal/train_multi_inputs.h5", "train_multi_inputs")


# In[7]:


convert_h5_to_sparse_csr("../input/open-problems-multimodal/train_cite_targets.h5", "train_cite_targets")


# In[8]:


convert_h5_to_sparse_csr("../input/open-problems-multimodal/train_cite_inputs.h5", "train_cite_inputs")


# In[9]:


convert_h5_to_sparse_csr("../input/open-problems-multimodal/test_multi_inputs.h5", "test_multi_inputs")


# In[10]:


convert_h5_to_sparse_csr("../input/open-problems-multimodal/test_cite_inputs.h5", "test_cite_inputs")


# # CSV -> PARQUET conversion
# For convenience, let us also convert the other files from CSV to the more efficient parquet format.
# 
# (Then just replace pd.read_csv(xxx.csv) by pd.read_parquet(xxx.parquet) to read the file into a pandas DataFrame)

# In[11]:


convert_to_parquet("../input/open-problems-multimodal/metadata.csv", "metadata")


# In[12]:


convert_to_parquet("../input/open-problems-multimodal/evaluation_ids.csv", "evaluation")


# In[13]:


convert_to_parquet("../input/open-problems-multimodal/sample_submission.csv", "sample_submission")


# In[14]:


get_ipython().system('ls -lh')


# In[ ]:




