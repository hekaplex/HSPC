#!/usr/bin/env python
# coding: utf-8

# # Open Problems - Multimodal Single-Cell Integration - 📊 EDA

# #### I am a Data Science practitioner that is very passionate and motivated. In this notebook, I will focus on analyzing the data pertaining to the Open Problems - Multimodal Single-Cell Integration my progress step by step until I present my own conclusion for this problem.
# 
# #### If you found this Kernel insightful, please consider upvoting it 👍
# 
# #### I am all ears if you have any advice :)

# <img src= "data:image/jpeg;base64,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" alt ="Titanic" style='width: 85%; margin-left: 7.5%
# '>

# <div style="color:white;
#            display:fill;
#            background-color:#B1E1FF;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            flex-direction: row;
#            border-radius: 0.25rem;
#            border-width: 0.25rem;
#            border-style: solid;
#            border-color: #256D85">
# 
# <h1 style="padding: 2rem;
#           color:black;
#           text-align:center;
#           margin:0 auto;
#           font-size:3rem;">
#    THE NOTEBOOK CONTAINS THE FOLLOWING SECTIONS:
# </h1>
#  
# </div>

# <div id=1 style="color:white;
#            display:fill;
#            border-radius:10px;
#            background-color:#3EC70B;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.25rem;
#            border-style: solid;
#            border-color: #256D85">
# 
# <h1 style="padding: 2.5rem;
#           color:white;
#           text-align:center;
#           margin:0 auto;
#           font-size:3rem;">
#    Problem Description
# </h1>
# </div>

# <div id = 2 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6">
# 
# <h2 style="padding: 2rem;
#            color:white;
#            text-align:center;
#            margin:0 auto;">
# Objective ⛳
# </h2>
# </div>

# The Open Problems - Multimodal Single-Cell Integration Competition presents goals such as predicting how DNA, RNA, and protein measurements co-vary in single cells as bone marrow stem cells develop into more mature blood cells.
# 
# In this competition, participants must seek to develop a model trained on a subset of 300,000-cell time course dataset of CD34+ hematopoietic stem and progenitor cells (HSPC) from four human donors at five time points generated for this competition specifically. One of the main challenges is taht the test set contains data from a much later time compared to the training data
# 
# 

# <img src= "https://storage.googleapis.com/kaggle-media/competitions/Cellarity/predict.png" alt ="Titanic" style='width: 50%; margin-left: 25%
# '>

# #### Context 
# In the past decade, the advent of single-cell genomics has enabled the measurement of DNA, RNA, and proteins in single cells. These technologies allow the study of biology at an unprecedented scale and resolution. Among the outcomes have been detailed maps of early human embryonic development, the discovery of new disease-associated cell types, and cell-targeted therapeutic interventions. Moreover, with recent advances in experimental techniques it is now possible to measure multiple genomic modalities in the same cell.
# 
# While multimodal single-cell data is increasingly available, data analysis methods are still scarce. Due to the small volume of a single cell, measurements are sparse and noisy. Differences in molecular sampling depths between cells (sequencing depth) and technical effects from handling cells in batches (batch effects) can often overwhelm biological differences. When analyzing multimodal data, one must account for different feature spaces, as well as shared and unique variation between modalities and between batches. Furthermore, current pipelines for single-cell data analysis treat cells as static snapshots, even when there is an underlying dynamical biological process. Accounting for temporal dynamics alongside state changes over time is an open challenge in single-cell data science.
# 
# Generally, genetic information flows from DNA to RNA to proteins. DNA must be accessible (ATAC data) to produce RNA (GEX data), and RNA in turn is used as a template to produce protein (ADT data). These processes are regulated by feedback: for example, a protein may bind DNA to prevent the production of more RNA. This genetic regulation is the foundation for dynamic cellular processes that allow organisms to develop and adapt to changing environments. In single-cell data science, dynamic processes have been modeled by so-called pseudotime algorithms that capture the progression of the biological process. Yet, generalizing these algorithms to account for both pseudotime and real time is still an open problem.
# 
# Competition host Open Problems in Single-Cell Analysis is an open-source, community-driven effort to standardize benchmarking of single-cell methods. The core efforts of Open Problems include the formalization of existing challenges into measurable tasks, a collection of high-quality datasets, centralized benchmarking of community-contributed methods, and community-focused events that bring together diverse method developers to improve single-cell algorithms. They're excited to be partnering with Cellarity, Chan Zuckerbeg Biohub, the Chan Zuckerberg Initiative, Helmholtz Munich, and Yale to see what progress can be made in predicting changes in genetic dynamics over time through interdisciplinary collaboration.
# 
# There are approximately 37 trillion cells in the human body, all with different behaviors and functions. Understanding how a single genome gives rise to a diversity of cellular states is the key to gaining mechanistic insight into how tissues function or malfunction in health and disease. You can help solve this fundamental challenge for single-cell biology. Being able to solve the prediction problems over time may yield new insights into how gene regulation influences differentiation as blood and immune cells mature.

# <div id = 3 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6">
# 
# <h2 style="padding: 2rem;
#            color:white;
#            text-align:center;
#            margin:0 auto;">
# About the given Datasets 📁📂
# </h2>
# </div>

# #### Cell Types
# * MasP = Mast Cell Progenitor
# * MkP = Megakaryocyte Progenitor
# * NeuP = Neutrophil Progenitor
# * MoP = Monocyte Progenitor
# * EryP = Erythrocyte Progenitor
# * HSC = Hematoploetic Stem Cell
# * BP = B-Cell Progenitor

# #### metadata.csv
# * cell_id - A unique identifier for each observed cell.
# * donor - An identifier for the four cell donors.
# * day - The day of the experiment the observation was made.
# * technology - Either citeseq or multiome.
# * cell_type - One of the above cell types or else hidden.

# ##### The experimental observations are located in several large arrays and are provided in HDF5 format.
# #### HDF5 Arrays:
# * Multiome
# * CITEseq
# * Splits
# ##### We will talk about the content of these Arrays in a later section.
# 
# 

# <div id = 4 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6">
# 
# <h2 style="padding: 2rem;
#            color:white;
#            text-align:center;
#            margin:0 auto;">
# Evaluation Metrics Used 📏
# </h2>
# </div>

# ### Based on the problem, we will use the Pearson Correlation Coefficient metric (PCC).
# 
# #### PCC is a measure of linear correlation between two sets of data and gives an with range: [a, b]
# 
# <br/>
# <br/>
# <img src= "https://upload.wikimedia.org/wikipedia/commons/3/34/Correlation_coefficient.png" alt ="Titanic" style='width: 45%; margin-left: 25%
# '>
# 
# $ PCC = \frac{Cov(x, y}{\sigma_x \sigma_y}$
# 
# #### If you want to learn more about PCC: 
# https://www.analyticsvidhya.com/blog/2021/01/beginners-guide-to-pearsons-correlation-coefficient/
# 

# $$f(X,n) = X_n + X_{n-1}$$

# <div id=5 style="color:white;
#            display:fill;
#            border-radius:10px;
#            background-color:#3EC70B;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.25rem;
#            border-style: solid;
#            border-color: #256D85">
# 
# <h1 style="padding: 2.5rem;
#           color:white;
#           text-align:center;
#           margin:0 auto;
#           font-size:3rem;">
#    Work 🔨
# </h1>
# </div>

# <div id = 6 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6">
# 
# <h2 style="padding: 2rem;
#            color:white;
#            text-align:center;
#            margin:0 auto;">
# Importing relevant libraries 📚📚📚
# </h2>
# </div>

# In[1]:


get_ipython().system(' pip install -q tables  # Necessary to load hdf5 files')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from termcolor import colored
import warnings
import h5py
warnings.simplefilter('ignore')


# <div id=7 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6      ">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
# Reading Data 📖
# </h2>
# </div>

# #### Due to memory constaints, some of the data will not be evaluated.

# In[3]:


metadata = pd.read_csv('../input/open-problems-multimodal/metadata.csv').set_index('cell_id')
evaluation_ids = pd.read_csv('../input/open-problems-multimodal/evaluation_ids.csv').set_index('row_id')


# In[4]:


# test_cite_inputs = pd.read_hdf('../input/open-problems-multimodal/test_cite_inputs.h5', nrows=400).astype(np.float8)
# train_cite_inputs = pd.read_hdf('../input/open-problems-multimodal/train_cite_inputs.h5', nrows=400).astype(np.float8)
# #test_multi_inputs = pd.read_hdf('../input/open-problems-multimodal/test_multi_inputs.h5', nrows=800).astype(np.float16)
# train_multi_inputs = pd.read_hdf('../input/open-problems-multimodal/train_multi_inputs.h5', nrows=100).astype(np.float16)
train_cite_targets = pd.read_hdf('../input/open-problems-multimodal/train_cite_targets.h5', nrows=800).astype(np.float16)
# train_multi_targets = pd.read_hdf('../input/open-problems-multimodal/train_multi_targets.h5', nrows=800).astype(np.float16)


# <div style="
#             display:flex;
#             flex-direction:row">
# <div style="width:0.05%;
#             height:max;
#             background-color:#5642C5;
#             padding:0.75rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#9BA3EB;
#            font-size:70%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            flex-direction: row;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            text-align:left;
#            margin-right: auto;
#           ">
# Metadata 📁
# </h2>
# </div>
# </div>

# In[5]:


metadata.head()


# <div style="
#             display:flex;
#             flex-direction:row">
# <div style="width:0.05%;
#             height:max;
#             background-color:#5642C5;
#             padding:0.75rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#9BA3EB;
#            font-size:70%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            flex-direction: row;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            text-align:left;
#            margin-right: auto;
#           ">
# Evaluation ID's
# </h2>
# </div>
# </div>

# In[6]:


evaluation_ids.head()


# ## For now we will only cover Metadata and Evaluation ID's

# In[7]:


#train_multi_targets.head()


# <div id=7 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6      ">
# 
# <h2 style="padding: 2rem;
#               color:white;
#           text-align:center;
#           margin:0 auto;
#           ">
# Exploratory Data Analysis and Visualization 🔍 👀
# </h2>
# </div>

# <div id=9 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            background-color:#3EC70B;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.25rem;
#            border-style: solid;
#            border-color: #256D85">
# 
# <h3 style="text-align:center;
#           margin:0 auto;
#           color:white;
#            padding:1rem
#           ">
# View Data Types of Predictors and Target Variables
# </h3>
# </div>

# #### Metadata

# In[8]:


print(colored(f'DATA TYPES:', 'cyan', attrs=['bold', 'underline']))
metadata.dtypes


# #### Evaluation ID's

# In[9]:


print(colored(f'DATA TYPES:', 'cyan', attrs=['bold', 'underline']))
evaluation_ids.dtypes


# <div id=10 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            background-color:#3EC70B;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.25rem;
#            border-style: solid;
#            border-color: #256D85">
# 
# <h3 style="text-align:center;
#           margin:0 auto;
#           color:white;
#            padding:1rem
#           ">
# Analysis of Missing Values ⚠️
# </h3>
# </div>

# In[10]:


nan_cols_metadata = metadata.columns[metadata.isna().any()].tolist()
nan_cols_evaluation_ids = evaluation_ids.columns[evaluation_ids.isna().any()].tolist()
print(nan_cols_metadata, nan_cols_evaluation_ids)


# <div style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#             justify-content:center;">
# 
# <h4 style="text-align:center;
#           margin:0 auto;
#           color:black;
#           ">
# Conclusions 📍
# </h4>
# </div>
# 
# * Metadata and Evaluation ID's don't have any missing values

# <div id=11 style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            background-color:#3EC70B;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#            justify-content:center;
#            border-width: 0.25rem;
#            border-style: solid;
#            border-color: #256D85">
# 
# <h3 style="text-align:center;
#           margin:0 auto;
#           color:white;
#            padding:1rem
#           ">
#  Data Analysis of Individual Predictors 🔮
# </h3>
# </div>

# #### I used pie-charts because of inspiration from this notebook: https://www.kaggle.com/code/jirkaborovec/mmscel-inst-eda-stat-predictions

# 
# <div style="
#             display:flex;
#             flex-direction:row;
#             justify-content:left;">
# <div style="width:0.05%;
#             height:max;
#             background-color:#256D85;
#             padding:1rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem;
#             margin-left: 1rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#3EC70B;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            margin-right: auto;
#           ">
# Day
# </h2>
# </div>
# </div>

# In[11]:


fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
sns.countplot(x='day', data=metadata, palette='mako', ax=ax[0])
metadata[['day']].value_counts().plot.pie(autopct='%1.1f%%', ylabel='day', ax=ax[1])
plt.show()
print(colored(f'Day counts:', 'cyan', attrs=['bold', 'underline']))
metadata.day.value_counts().sort_index()


# <div style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#             justify-content:center;">
# 
# <h4 style="text-align:center;
#           margin:0 auto;
#           color:black;
#           ">
# Conclusions 📍
# </h4>
# </div>
# 
# * Day 4 had the most donations

# 
# <div style="
#             display:flex;
#             flex-direction:row;
#             justify-content:left;">
# <div style="width:0.05%;
#             height:max;
#             background-color:#256D85;
#             padding:1rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem;
#             margin-left: 1rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#3EC70B;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            margin-right: auto;
#           ">
# Donor 🧑
# </h2>
# </div>
# </div>

# In[12]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
sns.countplot(x='donor', data=metadata, palette='mako', ax=ax[0])
metadata[['donor']].value_counts().plot.pie(autopct='%1.1f%%', ylabel='donor', ax=ax[1])
plt.show()
print(colored(f'Donor counts:', 'cyan', attrs=['bold', 'underline']))
metadata.donor.value_counts().sort_index()


# <div style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#             justify-content:center;">
# 
# <h4 style="text-align:center;
#           margin:0 auto;
#           color:black;
#           ">
# Conclusions 📍
# </h4>
# </div>
# 
# * Donor 31800 donated the most cells

# 
# <div style="
#             display:flex;
#             flex-direction:row;
#             justify-content:left;">
# <div style="width:0.05%;
#             height:max;
#             background-color:#256D85;
#             padding:1rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem;
#             margin-left: 1rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#3EC70B;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            margin-right: auto;
#           ">
# Cell Type 🧫
# </h2>
# </div>
# </div>

# In[13]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
sns.countplot(x='cell_type', data=metadata, palette='mako', ax=ax[0])
metadata[['cell_type']].value_counts().plot.pie(autopct='%1.1f%%', ylabel='cell_type', ax=ax[1])
plt.show()
print(colored(f'Cell Type counts:', 'cyan', attrs=['bold', 'underline']))
metadata.cell_type.value_counts().sort_index()


# <div style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#             justify-content:center;">
# 
# <h4 style="text-align:center;
#           margin:0 auto;
#           color:black;
#           ">
# Conclusions 📍
# </h4>
# </div>
# 
# * The most common cell types are HSC and hidden

# 
# <div style="
#             display:flex;
#             flex-direction:row;
#             justify-content:left;">
# <div style="width:0.05%;
#             height:max;
#             background-color:#256D85;
#             padding:1rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem;
#             margin-left: 1rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#3EC70B;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            margin-right: auto;
#           ">
# Technology 👩‍💻
# </h2>
# </div>
# </div>

# In[14]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
sns.countplot(x='technology', data=metadata, palette='mako', ax=ax[0])
metadata[['technology']].value_counts().plot.pie(autopct='%1.1f%%', ylabel='technology', ax=ax[1])
plt.show()


# ### Lets have a look at citeseq and multiome and generate a plot based on teh donor

# In[15]:


metadata_cite = metadata[metadata.technology=='citeseq']
metadata_multiome = metadata[metadata.technology=='multiome']
_, ax = plt.subplots(1,2,figsize=(24,12))
sns.countplot(x='day', hue='donor', data=metadata_cite, palette='mako', ax=ax[0])
sns.countplot(x='day', hue='donor', data=metadata_multiome, palette='mako', ax=ax[1])
plt.show()


# <div style="color:white;    
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px;
#            display:flex;
#             justify-content:center;">
# 
# <h4 style="text-align:center;
#           margin:0 auto;
#           color:black;
#           ">
# Conclusions 📍
# </h4>
# </div>
# 
# * Most of the cells were sample using the Multiome kit
# * On day 4 donor 27678 did not donate using the multiome approach, whule the rest of the donors every single day used both technologies
# * For the CITEseq kit day 4 had the most samples, while for the Multiome kit day 3 seems to have the most samples

# 
# <div style="
#             display:flex;
#             flex-direction:row;
#             justify-content:left;">
# <div style="width:0.05%;
#             height:max;
#             background-color:#256D85;
#             padding:1rem;
#             border-top-left-radius:0.5rem;
#             border-bottom-left-radius: 0.5rem;
#             margin-left: 1rem">
#     
# </div>
# <div id=7 style="color:white;    
#            background-color:#3EC70B;
#            font-size:90%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            border-width: 0.2rem;
#            border-style: solid;
#            border-color: #7A86B6;
#            display: inline-block;
#            border-left: none
#                  ">
# 
# <h2 style="padding: 0.5rem;
#            color:white;
#            margin-right: auto;
#           ">
# Metadata + CITEseq KIT
# </h2>
# </div>
# </div>

# In[16]:


metadata = metadata.join(train_cite_targets, how='right')


# In[17]:


fig, axarr = plt.subplots(nrows=2, ncols=3)
for i, col in enumerate(["day", "donor", "cell_type"]):
    _= metadata[[col]].value_counts().plot.pie(ax=axarr[0][i], autopct='%1.1f%%', ylabel=col)
    sns.countplot(x=col, data=metadata, palette='mako', ax=axarr[1][i])


# <div style="color:white;
#            display:fill;
#            background-color:#F4E06D;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#            display:flex;
#            flex-direction: row;">
# 
# <h1 style="padding: 2rem;
#           color:black;
#           text-align:center;
#           margin:0 auto;
#           font-size:3rem;">
#    WORK IN PROGRESS ⚠️
# </h1>
#  
# </div>
# 
# ### Feel free to comment what you would like me to include in this Kernel. 
