#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# Focusing only on gene_id, submit the average value of gene_id.
# 
# In this case, cell_id is not used.

# # Data preparation
# 
# I referred to [@peterholderrieth](https://www.kaggle.com/peterholderrieth)'s notebook. (https://www.kaggle.com/code/peterholderrieth/getting-started-data-loading)

# In[1]:


get_ipython().system('pip install --quiet tables')


# In[2]:


import os
import pandas as pd


# In[3]:


os.listdir("/kaggle/input/open-problems-multimodal/")


# In[4]:


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


# ## Citeseq

# In[5]:


df_cite_train_y = pd.read_hdf('../input/open-problems-multimodal/train_cite_targets.h5')
df_cite_train_y.head()


# In[6]:


cite_gene_id_mean = df_cite_train_y.mean()
cite_gene_id_mean


# ## Multiome

# In[7]:


START = int(1e4)
STOP = START+10000


# In[8]:


df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
df_multi_train_y.info()


# In[9]:


multi_gene_id_mean = df_multi_train_y.mean()
multi_gene_id_mean


# ## Convert gene_id to int (to save memory)

# In[10]:


cite_gene_id_mean.index


# In[11]:


multi_gene_id_mean.index


# In[12]:


_ = list(cite_gene_id_mean.index) + list(multi_gene_id_mean.index)
gene_id = pd.DataFrame(_, columns=['gene_id'])
gene_id


# In[13]:


gene_id['gene_id_int'] = gene_id['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
gene_id['gene_id_int'].value_counts()


# # Create submit file

# In[14]:


df_sample_submission = pd.read_csv(SUBMISSON, usecols=['row_id'])
df_sample_submission.info()


# In[15]:


df_evaluation = pd.read_csv(EVALUATION_IDS, usecols=['row_id', 'gene_id'])
df_evaluation['gene_id_int'] = df_evaluation['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
df_evaluation.drop(['gene_id'], axis=1, inplace=True)
df_evaluation.info()


# In[16]:


df_sample_submission = df_sample_submission.merge(df_evaluation, how='left', on='row_id')
df_sample_submission.info()


# In[17]:


cite_gene_id_mean = pd.DataFrame(cite_gene_id_mean, columns=['target']).reset_index()
cite_gene_id_mean['gene_id_int'] = cite_gene_id_mean['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
cite_gene_id_mean.drop(['gene_id'], axis=1, inplace=True)


# In[18]:


multi_gene_id_mean = pd.DataFrame(multi_gene_id_mean, columns=['target']).reset_index()
multi_gene_id_mean['gene_id_int'] = multi_gene_id_mean['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
multi_gene_id_mean.drop(['gene_id'], axis=1, inplace=True)


# In[19]:


cite_multi_gene_id_mean = pd.concat([cite_gene_id_mean, multi_gene_id_mean])
cite_multi_gene_id_mean.info()


# In[20]:


df_sample_submission = df_sample_submission.merge(cite_multi_gene_id_mean, how='left', on='gene_id_int')
df_sample_submission.info()


# In[21]:


df_sample_submission[['row_id', 'target']].to_csv('submission.csv', index=False)

