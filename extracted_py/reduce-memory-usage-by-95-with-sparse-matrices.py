#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --quiet tables')


# In[2]:


import os
import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm as tqdm
import gc


# In[3]:


DATA_DIR = "/kaggle/input/open-problems-multimodal/"

SUBMISSON = os.path.join(DATA_DIR,"sample_submission.csv")

EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


# # Multiome Dataset

# According to https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart, Multiome dataset is way to large to fit into the 16GB memory available on Kaggle. In fact:
# - train inputs: 105942 * 228942 float32 values (97 GByte)
# - train targets: 105942 * 23418 float32 values (10 GByte)
# - test inputs: 55935 * 228942 float32 values (13 GByte)

# ## Problem
# As we can see from the competition datasets, Multiome data are instrinsically sparse. To prove this statement, we can measure the sparsity rate of the Train-Multi-Inputs dataset. As described above, the entire dataset cannot be load in memory, thus we limit our study to the first 5000 rows

# In[4]:


df = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=0, stop=5000)


# In[5]:


df.info(memory_usage='deep')


# #### Count Non-Zero Values in Each column

# In[6]:


nnz = df.astype(bool).sum()
nnz.sort_values()


# To measure the total sparsity of the DataFrame, we can extract the fraction of NNZ values over the total number of values

# In[7]:


total_nnz = nnz.sum()
total_values = df.shape[0] * df.shape[1]
total_nnz / total_values


# As we can see, the dataset is extreamly sparse, since the Number of Non-Zero values correspond to just `2%` of the entire dataset loaded. It is reasonable to state that the same behaviour holds in the rest of the dataset. 
# We are able to tackle this waste of memory by adopting a different data structure

# In[8]:


del df, nnz, total_nnz, total_values


# In[9]:


gc.collect()


# # Memory Optimization with Sparse Matrices
# Given the intrinsic sparse nature of the data in Multiome datasets, we can leverage on Sparse Matrices to optimize the space required to load data in memory. In particular, we can use [Compressed Sparse Row](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) matrices to reduce considerably the memory used.
# 
# CSR Matrix are built upon three different one-dimensional arrays:
# - Data Array: Shape: (Number Non-Zero values). It contains non-zero values that corresponds to our data.
# - Indices Array: Shape: (Number Non-Zero values). It contains the column indices
# - Indptr Array: Shape: (Number of Rows + 1). It represents the extent of each row with respect to the other two (data/indices) arrays. To access data of a particular row *i* in the matrix, we can slice the Data Array with Indptr Array as follows: `data[indptr[i]:indptr[i+1]]`. Same for Indices Array

# Since we are not able to load the entire Train-Multi-Inputs dataset in memory, we are going to manually build the three arrays by loading chunk of data at a time.

# ### Utility functions

# To speed up the computation, we compute the indptr array by exploiting Cython. In this way, we can halve the time required to compress the huge array of row indices to extract the indptr array

# In[10]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[11]:


get_ipython().run_cell_magic('cython', '', '\nimport cython\ncimport cython\ncimport numpy as np\nimport numpy as np\nfrom tqdm import tqdm, trange\n\nctypedef np.int64_t INT64_t\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ncpdef np.ndarray[INT64_t, ndim=1] create_indptr(INT64_t[:] row_indices, int start_pos, int nrows):\n    cdef int shape = row_indices.shape[0]\n    res = np.zeros(nrows, dtype=np.int64)\n    cdef INT64_t[:] res_view = res\n    \n    cdef int i\n    cdef int curr_row = 0\n    cdef int prev = row_indices[0]\n    \n    for i in range(shape):\n        if row_indices[i] != prev:\n            curr_row += 1\n            res_view[curr_row] = i\n            prev = row_indices[i]\n    # res_view[curr_row + 1] = shape\n    return res + start_pos')


# In[12]:


def create_csr_arrays(h5_file_path):
    def check_size(xs, ys, datas):
        return (xs.nbytes + ys.nbytes + datas.nbytes) * 1e-9

    print(f"\n\nProcessing File {h5_file_path}")
    pbar = tqdm()

    # Initialize Variables
    chunksize = 1000 # Keep it low
    loaded_rows = chunksize
    start = 0
    start_pos = 0
    file_pointer = 0

    # Initialize CSR arrays
    indptr = np.array([], dtype=np.int64)
    indices = np.array([], dtype=np.int32)
    data_s = np.array([], dtype=np.float32)
    
    prefix_filename = h5_file_path.split('/')[-1].replace('.h5', '')

    while chunksize == loaded_rows:

        # Check current size: if the total sum of sizes are > 7GB, then save three arrays and re-initialize them
        size_gb = check_size(indptr, indices, data_s)
        if size_gb > 7.0:
            pbar.set_description(f"Total size is {size_gb}. Saving ..")
            np.save(f"{prefix_filename}_indptr_{file_pointer}.npy", indptr)
            np.save(f"{prefix_filename}_indices_{file_pointer}.npy", indices)
            np.save(f"{prefix_filename}_data_{file_pointer}.npy", data_s)
            # Re-initialize
            indptr = np.array([], dtype=np.int64)
            indices = np.array([], dtype=np.int32)
            data_s = np.array([], dtype=np.float32)
            # Increment pointer
            file_pointer += 1

        pbar.set_description("Reading .h5 chunk")
        df = pd.read_hdf(h5_file_path, start=start, stop=start+chunksize)
        pbar.set_description("Extracting non-zero values")
        x_coords, y_coords = df.values.nonzero()
        tmp_data = df.values[df.values != 0.0]

        loaded_rows = df.shape[0]

        # Convert types
        y_coords = y_coords.astype(np.int32, copy=False)
        tmp_data = tmp_data.astype(np.float32, copy=False)

        # Compress x_coords
        pbar.set_description("Compressing rows values")
        x_coords = create_indptr(x_coords, start_pos=start_pos, nrows=loaded_rows)

        gc.collect()

        # Update variables
        pbar.set_description("Update variables")
        start_pos += y_coords.shape[0]
        start += chunksize
        # Append data at the end of each array
        indptr = np.hstack((indptr, x_coords))
        indices = np.hstack((indices, y_coords))
        data_s = np.hstack((data_s, tmp_data))

        pbar.update(loaded_rows)

    print('Done. Save last files')
    np.save(f"{prefix_filename}_indptr_{file_pointer}.npy", indptr)
    np.save(f"{prefix_filename}_indices_{file_pointer}.npy", indices)
    np.save(f"{prefix_filename}_data_{file_pointer}.npy", data_s)
    
    del indptr, indices, data_s


# In[13]:


# create_csr_arrays(FP_MULTIOME_TRAIN_INPUTS) # This will create three different arrays


# The previous command will create and save three different array in .npy format:
# - train_multi_inputs_indptr_0.npy
# - train_multi_inputs_indices_0.npy
# - train_multi_inputs_data_0.npy

# In[14]:


# indptr = np.load('train_multi_inputs_indptr_0.npy')
# indices = np.load('train_multi_inputs_indices_0.npy')
# data = np.load('train_multi_inputs_data_0.npy')


# Since indptr array has shape (Number of Rows) instead of (Number of Rows + 1), we can add the last element to the array, which corresponds to the length of indices or data arrays. 

# In[15]:


# indptr = np.append(indptr, indptr[-1] + indices[indptr[-1]:].shape)


# Eventually, we can build out csr_matrix as follows:

# In[16]:


N_ROWS = 105942
N_COLS = 228942
# csr_matrix = sps.csr_matrix((data, indices, indptr), shape=(N_ROWS, N_COLS))


# In[17]:


# sps.save_npz('train_multiome_input_sparse.npz', csr_matrix)


# In[18]:


# del csr_matrix, indices, indptr, data


# We can repeat the same process for the other Multiome Datasets, namely `train_multi_targets.h5` and `test_multi_inputs.h5` to obtain the corresponding Compressed Sparse Row matrices.
# I wrapped up these CSR matrices in the following Kaggle Dataset: https://www.kaggle.com/datasets/sbunzini/open-problems-msci-multiome-sparse-matrices

# # Compression Rate

# In[19]:


train_input = sps.load_npz('../input/open-problems-msci-multiome-sparse-matrices/train_multiome_input_sparse.npz')


# In[20]:


def get_size(sparse_m):
    size_gb = (sparse_m.indices.nbytes + sparse_m.indptr.nbytes + sparse_m.data.nbytes) * 1e-9
    return f"Size: {size_gb} GB"


# In[21]:


get_size(train_input)


# ### Memory Usage: `4.85883614 GB`

# In[22]:


# Percentage of Reduction
(1.0 - (4.85883614 / 97)) * 100


# # Reduced Memory Usage: `94.99%`
# 

# Same memory usage reduction can be applied to the other Multiome files (train_targets and test_inputs). Lots of state-of-the-art models can accept a sparse matrix as input for training, thus avoiding painful and slow iterators and speeding up the computation

# # !! Update !!
# The memory usage can be further shrinked by using float16 to represent data values and int16 to represent indices of columns. A new version of the dataset will be available with this kind of optimization which will allow to achieve a **97%** of compression
