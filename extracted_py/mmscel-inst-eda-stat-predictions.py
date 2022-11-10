#!/usr/bin/env python
# coding: utf-8

# # Multimodal Single-Cell🧬IIntegration: EDA 🔍 & simple predictions

# In[1]:


get_ipython().system(' pip install -q tables  # needed for loading HDF files')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import torch
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/open-problems-multimodal"


# # Browsing the matadata

# In[3]:


df_meta = pd.read_csv(os.path.join(PATH_DATASET, "metadata.csv")).set_index("cell_id")
display(df_meta.head())

print(f"table size: {len(df_meta)}")


# In[4]:


fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
for i, col in enumerate(["day", "donor", "technology"]):
    _= df_meta[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)


# In[5]:


fig, axarr = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
for i, col in enumerate(["cell_type", "day", "technology"]):
    _= df_meta.groupby([col, 'donor']).size().unstack().plot(
        ax=axarr[i], kind='bar', stacked=True, grid=True
    )


# In[6]:


fig, axarr = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
for i, col in enumerate(["cell_type", "donor", "technology"]):
    _= df_meta.groupby([col, 'day']).size().unstack().plot(
        ax=axarr[i], kind='bar', stacked=True, grid=True
    )


# # Browse the train dataset

# In[7]:


df_cite = pd.read_hdf(os.path.join(PATH_DATASET, "train_cite_inputs.h5")).astype(np.float16)
cols_source = list(df_cite.columns)
display(df_cite.head())


# In[8]:


cols1, cols2 = list(zip(*[c.split("_") for c in df_cite.columns]))
cols1 = sorted(set(cols1))
cols2 = sorted(set(cols2))
print(f"cols1: {len(cols1)}")
print(f"cols2: {len(cols2)}")

mx = np.zeros((len(cols1), len(cols2)))
spl = df_cite.sample(1)
for k, v in dict(spl).items():
    c1, c2 = k.split("_")
    mx[cols1.index(c1), cols2.index(c2)] = v
plt.imshow(mx)
plt.colorbar()


# In[9]:


df = pd.read_hdf(os.path.join(PATH_DATASET, "train_cite_targets.h5")).astype(np.float16)
cols_target = list(df.columns)
display(df.head())


# In[10]:


df_cite = df_cite.join(df, how='right')
df_cite = df_cite.join(df_meta, how="left")
del df

print(f"total: {len(df_cite)}")
print(f"cell_id: {len(df_cite)}")
display(df_cite.head())


# ### Meta-data details

# In[11]:


fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, col in enumerate(["day", "donor", "cell_type"]):
    _= df_cite[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)


# In[12]:


fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
for i, col in enumerate(["day", "cell_type"]):
    _= df_cite.groupby([col, 'donor']).size().unstack().plot(
        ax=axarr[i], kind='bar', stacked=True, grid=True
    )


# In[13]:


del df_cite


# ## Just a fraction of Multi dataset
# 
# Note that this dataset is too large to be leaded directly in DataFrame and crashes on Kaggle kernel

# In[14]:


path_h5 = os.path.join(PATH_DATASET, "train_multi_inputs.h5")
df_multi_ = pd.read_hdf(path_h5, start=0, stop=100)
display(df_multi_.head())


# In[15]:


cols1, cols2 = list(zip(*[c.split(":") for c in df_multi_.columns]))
cols1 = sorted(set(cols1))
cols2 = sorted(set(cols2))
print(f"cols1: {len(cols1)}")
print(f"cols2: {len(cols2)}")
del df_multi_


# In[16]:


path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")
display(pd.read_hdf(path_h5, start=0, stop=100).head())


# ### Load all indexes from chunks

# In[17]:


cell_id = []
for i in range(20):
    path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")
    df = pd.read_hdf(path_h5, start=i * 10000, stop=(i+1) * 10000)
    print(i, len(df), df["ENSG00000121410"].mean())
    if len(df) == 0:
        break
    cell_id += list(df.index)

df_multi_ = pd.DataFrame({"cell_id": cell_id}).set_index("cell_id")
display(df_multi_.head())


# In[18]:


df_multi_ = df_multi_.join(df_meta, how="left")

print(f"total: {len(df_multi_)}")
print(f"cell_id: {len(df_multi_)}")
display(df_multi_.head())


# ### Meta-data details

# In[19]:


fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, col in enumerate(["day", "donor", "cell_type"]):
    _= df_multi_[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)


# In[20]:


fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
for i, col in enumerate(["day", "cell_type"]):
    _= df_multi_.groupby([col, 'donor']).size().unstack().plot(
        ax=axarr[i], kind='bar', stacked=True, grid=True
    )


# # Show Evaluation table

# In[21]:


df_eval = pd.read_csv(os.path.join(PATH_DATASET, "evaluation_ids.csv")).set_index("row_id")
display(df_eval.head())

print(f"total: {len(df_eval)}")
print(f"cell_id: {len(df_eval['cell_id'].unique())}")
print(f"gene_id: {len(df_eval['gene_id'].unique())}")


# **NOTE** that this evaluation expect you to run predictions on the both datasets: cite & multi (as you can see bellow)
# 
# target columns for:
# - **cite**: 140 columns
# - **multi**: 23418 columns

# In[22]:


get_ipython().system(' head ../input/open-problems-multimodal/sample_submission.csv')


# # Statistic predictions 🏴‍ gene means

# In[23]:


path_h5 = os.path.join(PATH_DATASET, "train_cite_targets.h5")
col_means = dict(pd.read_hdf(path_h5).mean())


# In[24]:


col_sums = []
count = 0
for i in range(20):
    path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")
    df = pd.read_hdf(path_h5, start=i * 10000, stop=(i+1) * 10000)
    count += len(df)
    if len(df) == 0:
        break
    col_sums.append(dict(df.sum()))

df_multi_ = pd.DataFrame(col_sums)
display(df_multi_)


# In[25]:


col_means.update(dict(df_multi_.sum() / count))


# ## Map target to eval. table

# In[26]:


df_eval["target"] = df_eval["gene_id"].map(col_means)
display(df_eval)


# ## Finalize submission

# In[27]:


df_eval[["target"]].round(6).to_csv("submission.csv")

get_ipython().system(' ls -lh .')
get_ipython().system(' head submission.csv')

