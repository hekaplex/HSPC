#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success" style="font-size:30px">
# [LB:0.811] Normalized Ensembles for Pearson's Correlation Score Function
# </div>
# 
# 
# <div class="alert alert-block alert-danger" style="text-align:center; font-size:20px;">
#     ❤️ Dont forget to ▲upvote▲ if you find this notebook usefull!  ❤️
# </div>
# 
# 
# In this notebook I want to share a more robust ensembling method that can be used for both final multimodel ensembles and for ensembling of the same model trained on multiple folds. MSCI competition uses **correlation coefficient** as a scoring function, which affects the choice of ensembling logic. 
# 
# <span style='font-size:18px'>TL;DR: standardize your outputs per cell_id before adding base submissions to the ensemble!</span>
# 
# 
# 
# 
# 
# Lets explore the following claims:
# 
# ## Statement 1. Correlation loss is insensitive to linear transformations of predictions
# 
# You can have 2 solutions with the same score but with vastly different predictions for any given cell_id and gene_id. And this is because of this property. Quick proof. Let $X$ be our solution and $Y$ are ground truth labels. First recall that:
# $$
# corr(X, Y) = \frac {cov(X, Y)} {\sigma_X * \sigma_Y} = \frac {\mathbb{E}[(X-\mathbb{E}(X))*(Y-\mathbb{E}(Y))]} {\sigma_X * \sigma_Y}
# $$
# Now take another solution $X'=C_1+C_2*X$
# 
# $$
# corr(X', Y)=corr(C_1+C_2*X, Y) = \frac {\mathbb{E}[(C_1+C_2*X-\mathbb{E}[C_1+C_2*X])*(Y-\mathbb{E}(Y))]} {\sigma_{[C_1+C_2*X]} * \sigma_Y}\\
# =\frac {\mathbb{E}[(C_1+C_2*X-C_1+C_2*\mathbb{E}(X))*(Y-\mathbb{E}(Y))]} {C_2*\sigma_{X} * \sigma_Y}\\
# =\frac {C_2*\mathbb{E}[(X-\mathbb{E}(X))*(Y-\mathbb{E}(Y))]} {C_2*\sigma_{X} * \sigma_Y}\\
# =\frac {\mathbb{E}[(X-\mathbb{E}(X))*(Y-\mathbb{E}(Y))]} {\sigma_{X} * \sigma_Y}\\
# =corr(X, Y)
# $$
# So we see that multiplying by $C_2$ and adding $C_1$ doesn't affect the score. 
# 
# From practical standpoint this means that we could have 2 similar solutions which we want to weight with coefficients $w_1$ and $w_2$, but the difference in magnitude of these solutions could be huge (e.g. $C_2=123$) which would make correct weighting impossible. This is an unlikely scenario if MSE metric was used to train base models, but it's totally possible if models were optimized directly with correlation score loss function!
# 
# ## Statement 2. Per-cell_id standardization helps to rescale base submissions
# Under assumption that two base submissions are similar and demonstrate similar performance we could rescale them in the way that they become comparable and weighting in a regular way becomes adequate:
# 
# $$
# X'=\frac {X-\mathbb{E}X} {\sigma_X}
# $$
# 
# ## Statement 3. Weighting coefficients don't have to add up to 1!
# This is one of the benefit of the loss function that is agnostic to linear transformations. You don't have to weight base submissions as usual with $\sum_i w_i=1$. Any coefficients will do the job!
# 
# 
# ## Statement 4. Only collect predictions for one of the technologies (CITEseq, Multiome) from every base solution 
# This is another hack unrelated to the correlation score function.
# Most of public notebooks build models for a single technology (CITEseq or Multiome) and paste the rest of predictions from the best availble public notebook for the other technology. 
# This results in less control you have over base predictions. E.g. you might end up with a good ensemble for CITEseq, but all Multiome predictions would actually come from a single source notebook which is suboptimal!
# 
# In this notebook we carefully pick up only relevant predictions from every base submission. 

# <div class="alert alert-block alert-success" style="font-size:30px">
# A toy example demonstrating benefits of normalization
# </div>

# In[27]:


import numpy as np 
import pandas as pd 
import glob
from tqdm.notebook import tqdm
import os


# In[28]:


# our groud truth targets
targets = np.random.randn(100000)


# In[29]:


# submission1 = targets + some random noise
submission1 = targets + 0.5 * np.random.randn(100000)
submission1 


# In[30]:


# submission2 = targets + same amount of random noise + linear transformation
submission2 = 4 * (targets + 0.5 * np.random.randn(100000))
submission2 


# In[31]:


# correlation with target of submission1 and submission2 is quite similar
np.corrcoef(submission1, targets)


# In[32]:


np.corrcoef(submission2, targets)


# In[33]:


# Let's evaluate the standard average ensemble
np.corrcoef((submission1 + submission2) / 2, targets)


# Now let's standardise submissions first

# In[34]:


# Let's standardise first. You can see the gain of 0.92 -> 0.94 after applying normalization trick!

def std(x):
    return (x - np.mean(x)) / np.std(x)

np.corrcoef(std(submission1) + std(submission2), targets)


# <div class="alert alert-block alert-success" style="font-size:30px">
# Example of ensembling with rescaling of base solutions
# </div>

# In[35]:


SUBMISSIONS = {
    
    # LB: 0.81 https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor
    '../input/lb-t15-msci-multiome-catboostregressor/submission.csv': 1.,
    
    # LB 0.81 https://www.kaggle.com/code/sskknt/msci-citeseq-keras-quickstart-dropout
    '../input/msci-citeseq-keras-quickstart-dropout/submission.csv': 1.,         
    
    # LB: 0.809 https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart
    '../input/msci-citeseq-keras-quickstart/submission.csv': 0.7,
        
    # LB: 0.808 https://www.kaggle.com/code/fabiencrom/msci-multiome-torch-quickstart-submission
    '../input/msci-multiome-torch-quickstart-submission/submission.csv': 0.5,
    
    # LB: 0.804 https://www.kaggle.com/code/xiafire/fork-of-msci-multiome-randomsampling-sp-6b182b
    '../input/fork-of-msci-multiome-randomsampling-sp-6b182b/submission.csv': 0.3,
        
    # LB: 0.803 - https://www.kaggle.com/code/jsmithperera/multiome-quickstart-w-sparse-m-tsvd-32
    '../input/multiome-quickstart-w-sparse-m-tsvd-32/submission.csv': 0.2,
        
    # LB: 0.803 - https://www.kaggle.com/code/fabiencrom/msci-multiome-quickstart-w-sparse-matrices
    '../input/msci-multiome-quickstart-w-sparse-matrices/submission.csv': 0.2,            
    
    # LB: 0.803 - https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart/notebook
    '../input/msci-citeseq-quickstart/submission.csv': 0.2,
        
    
    # 0.797 - https://www.kaggle.com/code/ravishah1/citeseq-rna-to-protein-encoder-decoder-nn
    #'../input/citeseq-rna-to-protein-encoder-decoder-nn/submission.csv': 0.5,
        
    # LB: 0.792 - https://www.kaggle.com/code/swimmy/lgbm-baseline-msci-citeseq
    #'../input/lgbm-baseline-msci-citeseq/submission.csv': 0.2
}


# In[36]:


cell_ids = pd.read_parquet('../input/multimodal-single-cell-as-sparse-matrix/evaluation.parquet').cell_id


# In[37]:


def gen_std_submission(path, cell_ids):
    """
    Standardize submission per cell_id
    """
    df = pd.read_csv(path)
    df['cell_id'] = cell_ids    
    vals = []
    for idx, g in tqdm(df.groupby('cell_id', sort=False), desc=f'Standardizing {path}', miniters=1000):
        vals.append(std(g.target).values)
    vals = np.concatenate(vals)
    return vals


# In[38]:


def gen_ensemble(technology):
    ensemble = None
    for path in tqdm([path for path in SUBMISSIONS.keys() if technology in path], desc='Process submission'):
        weight = SUBMISSIONS[path]
        if ensemble is None:
            ensemble = gen_std_submission(path, cell_ids) * weight
        else:
            ensemble += gen_std_submission(path, cell_ids) * weight
    return ensemble


# In[ ]:


PRED_SEGMENTS = [(0, 6812820), (6812820, 65744180)]
ensemble = []
for tech, (from_idx, to_idx) in tqdm(list(zip(['citeseq', 'multiome'], PRED_SEGMENTS)), desc='Technology'):    
    ensemble.append(gen_ensemble(tech)[from_idx: to_idx])
    
    
ensemble = np.concatenate(ensemble)


# In[ ]:


df_submit = pd.read_parquet('../input/multimodal-single-cell-as-sparse-matrix/sample_submission.parquet')
df_submit['target'] = ensemble
df_submit.to_csv('submission.csv', index=False)
df_submit


# <div class="alert alert-block alert-danger" style="text-align:center; font-size:20px;">
#     ❤️ Dont forget to ▲upvote▲ if you find this notebook usefull!  ❤️
# </div>
