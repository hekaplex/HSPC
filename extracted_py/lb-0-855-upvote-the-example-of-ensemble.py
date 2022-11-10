#!/usr/bin/env python
# coding: utf-8

# # The example of ensemble
# ## If the work is useful to you, don't forget to upvote !
# ## submission1.csv
# LB: 0.853 - https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart/notebook
# ## submission2.csv
# LB: 0.849 - https://www.kaggle.com/code/ravishah1/citeseq-rna-to-protein-encoder-decoder-nn
# ## submission3.csv
# LB: 0.848 - https://www.kaggle.com/code/jsmithperera/multiome-quickstart-w-sparse-m-tsvd-32
# ## Result: LB:0.855

# In[1]:


import numpy as np 
import pandas as pd 
import glob


# In[2]:


paths = ['../input/ensemble/submission1.csv','../input/ensemble/submission2.csv','../input/ensemble/submission3.csv']


# In[3]:


dfs = [pd.read_csv(x) for x in paths]


# In[4]:


pred_ensembled = 0.9 * (0.9 * dfs[0]['target']  + 0.1 * dfs[1]['target']) + 0.1 * dfs[2]['target']


# In[5]:


submit = pd.read_csv('../input/open-problems-multimodal/sample_submission.csv')


# In[6]:


submit['target'] = pred_ensembled


# In[7]:


submit


# In[8]:


submit.to_csv('3in1_ensemble.csv', index=False)

