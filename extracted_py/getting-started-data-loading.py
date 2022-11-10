#!/usr/bin/env python
# coding: utf-8

# # Multimodal Single-Cell Integration Competition: Data Exploration and Visualization

# ## 1. Setup Notebook

# ### 1.1. Import packages
# 
# 

# In[1]:


#If you see a urllib warning running this cell, go to "Settings" on the right hand side, 
#and turn on internet. Note, you need to be phone verified.
get_ipython().system('pip install --quiet tables')


# In[2]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1.2. Set filepaths

# In[3]:


os.listdir("/kaggle/input/open-problems-multimodal/")


# In[4]:


DATA_DIR = "/kaggle/input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


# ## 2. Load and Visualize Data

# ### 2.1. Cell Metadata

# The metadata of our dataset comes is data about the cells. To understand the different groups of cells, let's first review how the experiment was conducted (see figure below):
# 1. On the first day (*day 1*), hemapoetic stem cells are cultured in a dish with liquids that trigger the differentation of these cells into blood cells.
# 2. On subsequent *days 2,3,4,7,10* some of the cells are removed and split into two subgroups `CITE` and `MULTIOME`.
# 3. Each of these assays (technologies) gives us two readouts per single cell: 
#     1. CITEseq measures gene expression (RNA) and surface protein levels.
#     2. Multiome measures gene expression (RNA) and chromatin accessibility (via ATACseq).
# 
# This experiment was repeated for 4 different donors of hemapoetic stem cells. The metadata gives information about day, donor, cell type,
# and technology. `cell_id` is a unique cell identifier and has no meaning beyond its purpose as a cell id.
# 
# ![Dataset_Kaggle_structure_small.jpeg](attachment:8ac4d726-390d-4a7e-8a31-0be3c7df2739.jpeg)

# In[5]:


df_cell = pd.read_csv(FP_CELL_METADATA)
df_cell


# **NOTE:** the cell type is hidden for the test set of the multiome as this can reveal information about the RNA.

# **Let's split the cells by technology**

# In[6]:


df_cell_cite = df_cell[df_cell.technology=="citeseq"]
df_cell_multi = df_cell[df_cell.technology=="multiome"]


# **Number of cells per group:**
# 
# The number of cells in each group is relatively constant, around 7500 cells per donor and day.

# In[7]:


fig, axs = plt.subplots(1,2,figsize=(12,6))
df_cite_cell_dist = df_cell_cite.set_index("cell_id")[["day","donor"]].value_counts().to_frame()                .sort_values("day").reset_index()                .rename(columns={0:"# cells"})
sns.barplot(data=df_cite_cell_dist, x="day",hue="donor",y="# cells", ax=axs[0])
axs[0].set_title("Number of cells measured with CITEseq")

df_multi_cell_dist = df_cell_multi.set_index("cell_id")[["day","donor"]].value_counts().to_frame()                .sort_values("day").reset_index()                .rename(columns={0:"# cells"})
sns.barplot(data=df_multi_cell_dist, x="day",hue="donor",y="# cells", ax=axs[1])
axs[1].set_title("Number of cells measured with Multiome")
plt.show()


# ### 2.2. Citeseq

# For CITEseq, the task is to predict surface protein levels ("targets") from RNA expression levels ("inputs" of the model).

# **Inputs:** For the RNA counts, each row corresponds to a cell and each column to a gene. The column format for a gene is given by `{EnsemblID}_{GeneName}` where `EnsemblID` refers to the [Ensembl Gene ID](https://www.ebi.ac.uk/training/online/courses/ensembl-browsing-genomes/navigating-ensembl/investigating-a-gene/#:~:text=Ensembl%20gene%20IDs%20begin%20with,of%20species%20other%20than%20human) and `GeneName` to the gene name.

# In[8]:


df_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
df_cite_test_x = pd.read_hdf(FP_CITE_TEST_INPUTS)
df_cite_train_x.head()


# **Targets:** For the surface protein levels, each row corresponds to a cell and each column to a protein:

# In[9]:


df_cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
df_cite_train_y.head()


# **Donor and cell types:** The train data consists of both gene expression (RNA) and surface protein data for days 2,3,4 for donors 1-3 (donor IDs: `32606`,`13176`, and `31800`), the public test data consists of RNA for days 2,3,4 for donor 4 (donor ID: `27678`) and the private test data consists data from day 7 from all donors.

# In[10]:


train_cells = df_cite_train_x.index.to_list()    
test_cells = df_cite_test_x.index.to_list()                                                     
df_cell_cite["split"] = ""
df_cell_cite.loc[df_cell_cite.cell_id.isin(train_cells),"split"] = "train"
df_cell_cite.loc[df_cell_cite.cell_id.isin(test_cells),"split"] = "test"

df_cell_cite[["split","day","donor"]].value_counts().to_frame().sort_values(["split","day","donor"]).rename(columns={0: "n cells"})


# ### 2.3. Multiome
# 
# For the Multiome data set, the task is to predict RNA levels ("targets") from ATACseq.

# **Inputs:** for the ATACseq data, each row corresponds to a cell and each column to a fragment of a gene.
# 
# <font fontsize=20 color="red"> **NOTE**: to save memory, we only read an excerpt from the ATACseq data!

# In[11]:


START = int(1e5)
STOP = START+1000

df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS,start=START,stop=STOP)
df_multi_test_x = pd.read_hdf(FP_MULTIOME_TEST_INPUTS,start=START,stop=STOP)
df_multi_train_x.head()


# **Targets:** the RNA count data is in similar shape as the RNA count data from CITEseq:

# In[12]:


df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
df_multi_train_y.head()


# **Donor and cell types:** The train data consists of both gene expression (RNA) and ATACseq data for days 2,3,4,7 for donors 1-3 (donor IDs: `32606`,`13176`, and `31800`), the public test data consists of RNA for days 2,3,4,7 for donor 4 (donor ID: `27678`) and the private test data consists data from day 7 from all donors.
# 
# <font fontsize=20 color="red"> **NOTE**: Uncomment the below cell if you have loaded the full ATACseq data!

# In[13]:


# train_cells = df_multi_train_y.index.to_list()    
# test_cells = df_multi_test_y.index.to_list()                                                     
# df_cell_multi["split"] = ""
# df_cell_multi.loc[df_cell_multi.cell_id.isin(train_cells),"split"] = "train"
# df_cell_multi.loc[df_cell_multi.cell_id.isin(test_cells),"split"] = "test"

# df_cell_multi[["split","day","donor"]].value_counts().to_frame().sort_values(["split","day","donor"]).rename(columns={0: "n cells"})

