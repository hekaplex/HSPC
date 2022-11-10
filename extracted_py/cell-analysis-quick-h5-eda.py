#!/usr/bin/env python
# coding: utf-8

# # Open Problems in Cell Analyis: Quick EDA

# <div style="color:white;
#        display:fill;
#        border-radius:5px;
#        background-color:#ffffe6;
#        font-size:120%;">
#     <p style="padding: 10px;
#           color:black;">
# Let's take a look at CITEseq inputs first
#     </p>
# </div>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('pip install --quiet tables')


# In[3]:


import os
os.makedirs('/kaggle/working/inputs', exist_ok=True)
# Circumvent read-only issues
get_ipython().system("cp ../input/open-problems-multimodal/train_cite_inputs.h5 '/kaggle/working/inputs'")


# In[4]:


with pd.HDFStore('/kaggle/working/inputs/train_cite_inputs.h5') as data:
    shape = data['/train_cite_inputs'].shape
    print(f"There are {shape[0]} cell IDs and {shape[1]} columns (!)")
    selected_columns = data['/train_cite_inputs'].columns[:40]
    # We select only 50 cells for starters
    df = data['/train_cite_inputs'][selected_columns].head(50)
    
df.head()


# In[5]:


get_ipython().system('pip install --quiet joypy')


# In[6]:


import pandas as pd
import joypy
import numpy as np

def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return (r, g, b)

joypy.joyplot(
              df,
    title="Cell distribution by gene",overlap=4,
              colormap=lambda x: color_gradient(x, start=(153/256, 255/256, 204/256),
                                                stop=(204/256, 102/256, 255/256)),
              linecolor='black', linewidth=.5,
             figsize=(7,12),);


# In[7]:


corr = df.corr()
plt.figure(figsize=(12,8));
sns.heatmap(corr, cmap="viridis");


# ----
# <div style="color:white;
#        display:fill;
#        border-radius:5px;
#        background-color:#ffffe6;
#        font-size:120%;">
#     <p style="padding: 10px;
#           color:black;">
# Now we get to the CITEseq targets
#     </p>
# </div>

# In[8]:


os.makedirs('/kaggle/working/labels', exist_ok=True)
get_ipython().system("cp ../input/open-problems-multimodal/train_cite_targets.h5 '/kaggle/working/labels'")


# In[9]:


with pd.HDFStore('/kaggle/working/labels/train_cite_targets.h5') as data:
    shape = data['/train_cite_targets'].shape
    print(f"There are {shape[0]} cell IDs and {shape[1]} columns (!)")
    selected_columns = data['/train_cite_targets'].columns[:40]
    # We select only 50 cells for starters
    df_targets = data['/train_cite_targets'][selected_columns].head(50)


# In[10]:


df_targets.head()


# In[11]:


joypy.joyplot(
              df_targets,
    title="Cell distribution by surface protein",overlap=4,
              colormap=lambda x: color_gradient(x, start=(153/256, 255/256, 204/256),
                                                stop=(204/256, 102/256, 255/256)),
              linecolor='black', linewidth=.5,
             figsize=(7,12),);


# In[ ]:




