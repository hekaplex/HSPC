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




#!/usr/bin/env python
# coding: utf-8

# # 1. Overview
# The goal of this competition is to better understand the relationship between different modalities in cells. The goal of this notebook is to gain a better understanding of the associated data. This equips us with the knowledge needed to make good decisions about model design and data layout.
# 
# **This is a work in progress. If any aspect needs clarification, please let me know. My understanding of genetics is very limited. Feel free to point out anything that is false.**
# 
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>1.1 What do we want to learn?</b></p>
# </div>
# 
# During transcription in cells, there is a known flow of information. DNA must be accessible to produce RNA. Produced RNA is used as a template to build proteins. Therefore, one could assume that we can use knowledge about the accessibility of DNA to predict future states of RNA and that we could use knowledge about RNA to predict the concentration of proteins in the future. In this challenge, we want to learn more about this relationship between DNA, RNA, and proteins. We thus need to capture information about three distinct properties of a cell:
# * chromatin accessibility
# * gene expression
# * surface protein levels
# 
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>1.2 How are those three properties of a cell presented?</b></p>
# </div>
# 
# Before we have a look at how the information about those properties of a cell is laid out, we must note that the methods used to obtain the data do not capture all properties at once. We have two distinct methods for testing. The first one is the "10x Chromium Single Cell Multiome ATAC + Gene Expression" short "multiome" test. The second one is the "10x Genomics Single Cell Gene Expression with Feature Barcoding technology" short "citeseq" test.
# 
# With the multiome test, we can measure **chromatin accessibility and gene expression**. With the citeseq test, we can measure **gene expression and surface protein levels**.
# 
# Therefore, we will have data about chromatin accessibility and surface protein levels once (from multiome and citeseq, respectively). And we will have data about gene expression two times, one from each test. With that out of the way, let's dive into how the data is actually presented.
# 
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>1.3 Imports</b></p>
# </div>

# In[1]:


# installs
get_ipython().system('pip install --quiet tables')

# imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# set paths
DATA_DIR = "../input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


# # 2. Data
# 
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>2.1 Chromatin accessibility data</b></p>
# </div>
# 
# First we will have a look at the data about chromatin accessibility. Inspecting the corresponding HDF5 file, we see that the data is stored in one single table having dimensions (228942, 105942). Each value is stored as a 32bit float. Thus, to load the full table into memory while preserving the 32bit accuracy, we will need about 90 GB of RAM. This is quite a lot. Therefore, we will only look at a chunk of the data here and also don't load the whole dataset into memory while training or doing transformations. The other HDF5 tables also store values as 32bit floats.

# In[2]:


# Loading the whole dataset into pandas exceeds the memory,
# therefore we define start and stop values
START = int(1e4)
STOP = START+1000

df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS,start=START,stop=STOP)
df_multi_train_x.head()


# As we can see, each individual cell is identified by a cell_id. We then have 228942 columns that are named something like "STUFF:NUMBER-NUMBER". STUFF is actually the name of a chromosome, while the numbers are a range indicating where the gene starts and ends. Let's have a look at what kind of chromosomes we have:

# In[3]:


print(sorted(list({i[:i.find(':')] for i in df_multi_train_x.columns})))


# We actually find the chromosomes we expect, namely chr1-chr22, the 22 chromosomes humans have (called autosomes), and also chrX and chrY, being the gender-specific chromosomes. What about the ones starting with KI and GL? According to a quick internet search, those are unplaced genes. They most likely are part of the human genome, but we don't know yet on which chromosome they are. Noteworthy at this point is that the number of protein-coding genes in humans is estimated to be between 19.9k and 21.3k. Therefore, it looks like we have measurements of much more than just the protein-coding genes.
# 
# Next, we check the range of the values we have.

# In[4]:


# first call to min/max gives us the min/max in each column. 
# Than we min/max again to get total min/max
print(f"Values range from {df_multi_train_x.min().min()} to {df_multi_train_x.max().max()}")


# So let's summarize what we have learned about the data corresponding to the accessibility of DNA so far:
# 
# * We have chromatin accessibility measurements for approximately 106k cells in total.
# * We measure how accessible certain genes are in each cell, approximately 229k genes per cell.
# * Accessibility is given in numbers from 0.0 to ~18. We do not know the upper bound because we have not looked at all the data yet.
# * The values in our dataset use 32bit precision floats

# What else do we want to know about chromatin accessibility data?
# 
# * How many values are non-zero for each cell?
# * What is the standard deviation and what is the average non-zero value?
# 
# First, we will have a closer look at the chromatin accessibility values of each cell.

# In[5]:


# get data about non-zero values
min_cells_non_zero = df_multi_train_x.gt(0).sum(axis=1).min()
max_cells_non_zero = df_multi_train_x.gt(0).sum(axis=1).max()
sum_non_zero_values = df_multi_train_x.sum().sum()
count_non_zero_values = df_multi_train_x.gt(0).sum().sum()
average_non_zero_per_gene = df_multi_train_x[df_multi_train_x.gt(0)].count(axis = 1).mean()

print(f"Each cell has at least {min_cells_non_zero} genes with non-zero accessibility values and a maximum of {max_cells_non_zero}.")
print(f"On average there are {round(average_non_zero_per_gene)} genes with non-zero accessibility values in each cell.")
print(f"The average non-zero value is about {sum_non_zero_values / count_non_zero_values:.2f}.")

# investigate standard deviation of features
std_dev_of_genes = df_multi_train_x.std()

# ignore genes that are only accessible in a single cell
std_dev_of_genes_without_singles = std_dev_of_genes[df_multi_train_x.gt(0).sum().gt(1)]
print(f"The standard deviation of our features is between {std_dev_of_genes_without_singles.min():.2f} and {std_dev_of_genes_without_singles.max():.2f}.\nThe average standard deviation is {std_dev_of_genes_without_singles.mean():.2f}")


# That's already good information about what we can expect from our features for the first problem. To even better understand how many features we have for each sample, we will plot the number of cells per feature count.

# In[6]:


s = df_multi_train_x.gt(0).sum(axis = 1)
counts = s.groupby(lambda x: s[x] // 300).count()
counts.index = counts.index * 300

fig, ax = plt.subplots()
ax.plot(counts.index, counts.values)
ax.set_xlabel('number of accessible genes')
ax.set_ylabel('number of cells')
plt.show()


# As we can see, the majority of our cells have between 2K and 7K accessible genes.
# 
# We now have quite a good understanding of the chromatin accessibility measured with the multiome test. We continue with investigating gene expression features.

# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>2.2 Gene expression data</b></p>
# </div>
# 
# As mentioned before we have two datasets containing gene expression data. We will first look at the data from the multiome test.
# 
# ### 2.2.1 Gene expression from multiome
# We would actually be able to load the whole dataset at once, but we will only look at the part corresponding to the already seen X values for now.

# In[7]:


df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
df_multi_train_y.head()


# As we can see, we have 23418 values that our model will need to predict. But what exactly are those values?

# In[8]:


print(sorted(list({i[:10] for i in df_multi_train_y.columns})))
print(df_multi_train_y.columns.str.len().unique().item())


# As we can see, all of the features start with ENSG and then 5 zeroes. What we have here is called the Ensambl ID. The general form is ENS(species)(object type)(identifier).(version).
# ENS tells us that we are looking at an ensembl ID. The species part is empty for human genes by convention. The object type is G for gene. It looks like the identifier is always 11 decimals long. And it looks like we don't have any version specifications in our data.
# 
# We will now check for similar properties than before.

# In[9]:


print(f"Values for gene expression range from {df_multi_train_y.min().min():.2f} to {df_multi_train_y.max().max():.2f}")

# get data about non-zero values
min_cells_non_zero_y = df_multi_train_y.gt(0).sum(axis=1).min()
max_cells_non_zero_y = df_multi_train_y.gt(0).sum(axis=1).max()
sum_non_zero_values_y = df_multi_train_y.sum().sum()
count_non_zero_values_y = df_multi_train_y.gt(0).sum().sum()
average_non_zero_per_gene_y = df_multi_train_y[df_multi_train_y.gt(0)].count(axis = 1).mean()

print(f"Each cell has at least {min_cells_non_zero_y} genes with non-zero gene expression values and a maximum of {max_cells_non_zero_y}.")
print(f"On average there are {round(average_non_zero_per_gene_y)} genes with non-zero gene expression values in each cell.")
print(f"The average non-zero value for gene expression is about {sum_non_zero_values_y / count_non_zero_values_y:.2f}.")

# investigate standard deviation of features
std_dev_of_genes_y = df_multi_train_y.std()

# ignore genes that are only accessible in a single cell
std_dev_of_genes_without_singles_y = std_dev_of_genes_y[df_multi_train_y.gt(0).sum().gt(1)]
print(f"The standard deviation of gene expression values is between {std_dev_of_genes_without_singles_y.min():.2f} and {std_dev_of_genes_without_singles_y.max():.2f}.\nThe average standard deviation is {std_dev_of_genes_without_singles_y.mean():.2f}")


# We can see that the range of gene expression values is smaller than that for chromatin accessibility, but the standard deviation is higher. This might be important for the design of our model.
# 
# Even though this information will probably not influence the design of our model, let's still have a look at how many genes are expressed in cells. Just because it is interesting.

# In[10]:


s = df_multi_train_y.gt(0).sum(axis = 1)
counts = s.groupby(lambda x: s[x] // 100).count()
counts.index = counts.index * 100

fig, ax = plt.subplots()
ax.plot(counts.index, counts.values)
ax.set_xlabel('number of genes expressed')
ax.set_ylabel('number of cells')
plt.show()


# Having an overview of gene expression data obtained by the multiome test, we will now compare it to those obtained by the citeseq test.
# 
# ### 2.2.2 Gene expression from citeseq

# In[11]:


df_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS,start=START,stop=STOP)
df_cite_train_x.head()


# The first thing we notice is that the start of our gene_id looks much like what we have seen in the multiome data, but there is a new suffix. So what is it about the suffix?
# 
# Checking the Ensembl ID of gene_id on [ensembl.org](https://www.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=ENSG00000121410;r=19:58345178-58353492) (in this case for ENSG00000121410) we see that the suffix is actually the name of the gene. As we will see in the next code cell, the gene_id is unique even without this suffix, so it looks like redundant information for now.

# In[12]:


gene_ids_multiome = set(df_multi_train_y.columns)
print(f"Different Gene IDs in multiome: {len(gene_ids_multiome)}")
#for now we just keep the stem of the gene_id
gene_ids_citeseq = set([i[:i.find("_")] for i in df_cite_train_x.columns])
print(f"Different Gene IDs in citeseq: {len(gene_ids_citeseq)}")


# As mentioned, stripping of the suffix still produces unique names.
# 
# Let's check for overlap in both datasets about gene expression.

# In[13]:


print(f"Elements in Set Union: {len(gene_ids_citeseq | gene_ids_multiome)}")
print(f"Elements in Set Intersection: {len(gene_ids_citeseq & gene_ids_multiome)}")
print(f"multiome has {len(gene_ids_multiome - gene_ids_citeseq)} unique gene ids.")
print(f"Citeseq has {len(gene_ids_citeseq - gene_ids_multiome)} unique gene ids.")


# Even though we have a huge intersection, there are quite a few genes unique to each test.
# 
# We will now again get information about distribution of values:

# In[14]:


print(f"Values for gene expression range from {df_cite_train_x.min().min():.2f} to {df_cite_train_x.max().max():.2f}")

# get data about non-zero values
min_cells_non_zero_y = df_cite_train_x.gt(0).sum(axis=1).min()
max_cells_non_zero_y = df_cite_train_x.gt(0).sum(axis=1).max()
sum_non_zero_values_y = df_cite_train_x.sum().sum()
count_non_zero_values_y = df_cite_train_x.gt(0).sum().sum()
average_non_zero_per_gene_y = df_cite_train_x[df_cite_train_x.gt(0)].count(axis = 1).mean()

print(f"Each cell has at least {min_cells_non_zero_y} genes with non-zero gene expression values and a maximum of {max_cells_non_zero_y}.")
print(f"On average there are {round(average_non_zero_per_gene_y)} genes with non-zero gene expression values in each cell.")
print(f"The average non-zero value for gene expression is about {sum_non_zero_values_y / count_non_zero_values_y:.2f}.")

# investigate standard deviation of features
std_dev_of_genes_y = df_cite_train_x.std()

# ignore genes that are only accessible in a single cell
std_dev_of_genes_without_singles_y = std_dev_of_genes_y[df_cite_train_x.gt(0).sum().gt(1)]
print(f"The standard deviation of gene expression values is between {std_dev_of_genes_without_singles_y.min():.2f} and {std_dev_of_genes_without_singles_y.max():.2f}.\nThe average standard deviation is {std_dev_of_genes_without_singles_y.mean():.2f}")


# We can see that the gene expression features obtained by the citeseq and the multiome test are quite similar. Values are in the same range and also the standard deviation and average non-zero values are of comparable size. For now, we only had a look at part of the data, so it will be interesting to see if this holds for all the data. We will get to that later.
# 
# For now, let's have a look at how many genes are expressed in individual cells.

# In[15]:


s = df_cite_train_x.gt(0).sum(axis = 1)
counts = s.groupby(lambda x: s[x] // 300).count()
counts.index = counts.index * 300

fig, ax = plt.subplots()
ax.plot(counts.index, counts.values)
ax.set_xlabel('number of genes expressed')
ax.set_ylabel('number of cells')
plt.show()


# This sums up our investigation of the gene expression data obtained by the citeseq test. We saw that the data is comparable to that obtained by multiome. One key difference is that both have unique genes that are only measured in one test. Later, we will also investigate if the comparability of the data is due to preceding normalization of the raw data obtained by the tests and also address the question of what an expression value of e. g. 2.4 actually means.

# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>2.3 Surface protein level data</b></p>
# </div>
# 
# Lastly, we will have a look at the surface protein levels data gathered by citeseq.

# In[16]:


df_cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
df_cite_train_y.head()


# Compared to what we have seen so far, the number of columns in this data set is quite small. We have measurements of 140 features per cell. Most of the names start with CD, which is short for "Cluster of differentiation". CDs are used to classify surface molecules a cell expresses. This information can then be used to get an idea of what kind of cell is present, or what function this cell is supposed to serve in the body (I am not sure if my understanding here is even remotely accurate).
# 
# Let's forget about the biological view for a moment and focus on the data science centric view. As we will see in the next cell, we have no zero values in this dataset, and thus much of the computation we did before (counting non-zero values, for example) is pointless. We will look at other features:

# In[17]:


print(f"Measurements of surface protein levels range from {df_cite_train_y.min().min():.2f} to {df_cite_train_y.max().max():.2f}.")
print(f"The average value is {df_cite_train_y.mean().mean():.2f}.")
print(f"The standard deviation of surface protein levels is between {df_cite_train_y.std().min():.2f} and {df_cite_train_y.std().max():.2f}.")
print(f"The average standard deviation is {df_cite_train_y.std().mean():.2f}.")


# We would also like to know if the absence of zero values could be due to inaccuracy in measurements. The following code checks for that. ATTENTION: As of right now, the threshold is completely arbitrary and will be revised when I know how accurate the test is!

# In[18]:


threshold = 0.1
df_cite_train_y.applymap(lambda x: abs(x)).gt(threshold).sum(axis = 1)
print(f"Each cell has between {df_cite_train_y.applymap(lambda x: abs(x)).gt(threshold).sum(axis = 1).min()} and {df_cite_train_y.applymap(lambda x: abs(x)).gt(threshold).sum(axis = 1).max()} measurements with absolute values over {threshold}.")


# It does not look like inaccuracy in measurements is the reason for the absence of 0 values, but as said, this needs to be checked!
# 
# # 3 Other Data
# On top of the data we already saw, there is also a file containing metadata and two files needed to make a submission for the competition. We will first have a look at the metadata and then at the competition-related data.
# 
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>3.1 Inspection of metadata.csv</b></p>
# </div>
# 
# The Metadata is stored in the file metadata.csv. The file contains one row for each cell in the dataset and provides some additional information about that cell.

# In[19]:


df_meta = pd.read_csv(FP_CELL_METADATA).set_index("cell_id")
df_meta


# For each cell, we see the **day** column tells us on which day the test was performed. The test that was actually performed is shown in the **technology** column. Note that experiments started on day 1, therefore the first tests were performed one day after the cells were injected with Neupogen for the first time ([compare here](https://allcells.com/research-grade-tissue-products/mobilized-leukopak/)). For each of the four donors, we have a **donor** ID giving us information about the origin of the cell. Lastly, there is a **cell_type** column. The cell types are labels assigned by humans. They might be imprecise and this information is not available for test data since it would be possible to draw conclusions about surface protein levels, for example. It is not clear if we can make use of that later (maybe we can use it for creating balanced splits, but we will see).
# 
# [jirkaborovec](https://www.kaggle.com/jirkaborovec) already composed a great analysis of the metadata in this [notebook](https://www.kaggle.com/code/jirkaborovec/mmscel-inst-eda-stat-predictions). I will copy some parts here for readability and add some comments myself.

# In[20]:


fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
for i, col in enumerate(["donor", "day", "technology"]):
    _= df_meta[[col]].value_counts().plot.pie(ax=axarr[i], autopct='%1.1f%%', ylabel=col)


# As we can see, the cell data is pretty balanced. We have almost an equal number of cells from each donor (the big numbers in the first picture are the donor ids). Also, the days of the experiment were fairly balanced. The last day, day 10, only receives an 11% share of the cells. Day 10 is also the only day not present in the train data at all! Also, the train set does not contain any data from donor 27678!
# 
# We have slightly more data available for the multinome test. It is not the worst since our model also has to predict many more features for that test.
# 
# The distribution of data in general is pretty well balanced (e. g., the number of tests taken / test / day is well distributed). For a more in-depth analysis of the metadata, I highly recommend the already mentioned [notebook](https://www.kaggle.com/code/jirkaborovec/mmscel-inst-eda-stat-predictions).
# 
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>3.2 Inspection of submission helpers</b></p>
# </div>
# 
# There are two files related to result submission. evaluation_ids.csv specifies the data that needs to be submitted to the competition, and sample_submission.csv is meant as a guide for formatting the submitted data.

# # X Notes
# 
# There are still some open questions in the text we need to address. Also, we want to get an understanding of the accuracy of the values and thus how many bits we will take for storage of data in the final data format.
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

#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success" style="font-size:30px">
# [LB:0.811] Normalized Ensembles for Pearson's Correlation Score Function
# </div>
# 
# 
# <div class="alert alert-block alert-danger" style="text-align:center; font-size:20px;">
#     ?????? Dont forget to ???upvote??? if you find this notebook usefull!  ??????
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
#     ?????? Dont forget to ???upvote??? if you find this notebook usefull!  ??????
# </div>
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

#!/usr/bin/env python
# coding: utf-8

# # CITEseq LGBM + Optuna Baseline
# This notebook is based on https://www.kaggle.com/code/swimmy/lgbm-baseline-msci-citeseq.
# 
# # Please vote it if useful!
# 
# # Optuna
# Optuna is a great hyperparameter optimization framework. 
# I use Optuna to find params below:
# ```
# params = {'metric': 'mae', 'random_state': 42, 'n_estimators': 2000, 'reg_alpha': 0.03645857751758206, 'reg_lambda': 0.0025972855120393492, 'colsample_bytree': 1.0, 'subsample': 0.6, 'learning_rate': 0.013262872399411381, 'max_depth': 10, 'num_leaves': 186, 'min_child_samples': 263, 'min_data_per_groups': 46}
# ```
# And this Notebook score 0.830 in leaderboard.

# In[1]:


import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator
import warnings
warnings.simplefilter('ignore')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
import optuna

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


# In[2]:


get_ipython().system('pip install --quiet tables')


# # Loading the common metadata table
# 
# The current version of the model is so primitive that it doesn't use the metadata, but we load it anyway.

# In[3]:


df_cell = pd.read_csv(FP_CELL_METADATA)
df_cell_cite = df_cell[df_cell.technology=="citeseq"]
df_cell_multi = df_cell[df_cell.technology=="multiome"]
df_cell_cite.shape, df_cell_multi.shape


# # Cross-validation
# 
# The note I referred to had the following description, but I confirmed that 13000 rows can be rotated in memory, so I changed columns_to_use = 13000. In addition, the search is performed by changing the starting point of the line to be acquired.
# 
# Data size:
# - The training input has shape 70988\*22050 (10.6 GByte).
# - The training labels have shape 70988\*140.
# - The test input has shape 48663\*22050 (4.3 GByte).
# 
# To get a result with only 16 GByte RAM, we simplify the problem as follows:
# - We ignore the complete metadata (donors, days, cell types).
# - We drop all feature columns which are constant.
# - Of the remaining columns, we keep only the last 12000.
# - We do a PCA and keep only the 240 most important components.
# - We use PCA(copy=False), which overwrites its input in fit_transform().
# - We fit a ridge regression model with 70988\*240 inputs and 70988\*140 outputs. 

# In[4]:


get_ipython().run_cell_magic('time', '', '# Preprocessing\ncol_start = 10000\n\nclass PreprocessCiteseq(BaseEstimator, TransformerMixin):\n    columns_to_use = 13000\n    \n    @staticmethod\n    def take_column_subset(X):\n        return X[:,-(PreprocessCiteseq.columns_to_use+col_start):-col_start]\n    \n    def transform(self, X):\n        print(X.shape)\n        X = X[:,~self.all_zero_columns]\n        print(X.shape)\n        X = PreprocessCiteseq.take_column_subset(X) # use only a part of the columns\n        print(X.shape)\n        gc.collect()\n\n        X = self.pca.transform(X)\n        print(X.shape)\n        return X\n\n    def fit_transform(self, X):\n        gc.collect()\n        print(X.shape)\n        self.all_zero_columns = (X == 0).all(axis=0)\n        X = X[:,~self.all_zero_columns]\n        print(X.shape)\n        X = PreprocessCiteseq.take_column_subset(X) # use only a part of the columns\n        print(X.shape)\n        gc.collect()\n\n        self.pca = PCA(n_components=240, copy=False, random_state=1)\n        X = self.pca.fit_transform(X)\n        print(X.shape)\n        return X\n\npreprocessor = PreprocessCiteseq()\n\ncite_train_x = None\ncite_train_x = preprocessor.fit_transform(pd.read_hdf(FP_CITE_TRAIN_INPUTS).values)\n\ncite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values\nprint(cite_train_y.shape)')


# # Optuna

# In[6]:


small_train_x, small_train_y =  cite_train_x[:1000,:],cite_train_y[:1000,:]
def objective(trial):
    params = {
        'metric': 'mae', 
        'random_state': 42,
        'n_estimators': 2000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))

    model.fit(small_train_x, small_train_y)

    y_va_pred = model.predict(cite_train_x)
    mse = mean_squared_error(cite_train_y, y_va_pred)
    
    return mse


# In[ ]:


find_params = False
if find_params:
    study = optuna.create_study(
        direction='minimize', 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        study_name='small')
    study.optimize(objective, n_trials=20)


# # Modeling&Prediction
# 
# We retrain the model on all training rows, delete the training data, load the test data and compute the predictions.

# In[9]:


params = {'metric': 'mae', 'random_state': 42, 'n_estimators': 2000, 'reg_alpha': 0.03645857751758206, 'reg_lambda': 0.0025972855120393492, 'colsample_bytree': 1.0, 'subsample': 0.6, 'learning_rate': 0.013262872399411381, 'max_depth': 10, 'num_leaves': 186, 'min_child_samples': 263, 'min_data_per_groups': 46}
model = MultiOutputRegressor(lgb.LGBMRegressor(**params))

model.fit(cite_train_x, cite_train_y)

y_va_pred = model.predict(cite_train_x)
mse = mean_squared_error(cite_train_y, y_va_pred)

print(mse)


# In[ ]:


del cite_train_x, cite_train_y
gc.collect()


# In[ ]:


cite_test_x = preprocessor.transform(pd.read_hdf(FP_CITE_TEST_INPUTS).values)
test_pred = model.predict(cite_test_x)
del cite_test_x
test_pred.shape


# # Submission
# 
# We save the CITEseq predictions so that they can be merged with the Multiome predictions in the [Multiome quickstart notebook](https://www.kaggle.com/ambrosm/msci-multiome-quickstart).
# 
# The CITEseq test predictions produced by the ridge regressor have 48663 rows (i.e., cells) and 140 columns (i.e. proteins). 48663 * 140 = 6812820.
# 

# In[ ]:


with open('citeseq_pred.pickle', 'wb') as f: pickle.dump(test_pred, f) # float32 array of shape (48663, 140)


# The final submission will have 65744180 rows, of which the first 6812820 are for the CITEseq predictions and the remaining 58931360 for the Multiome predictions. 
# 
# We now read the Multiome predictions and merge the CITEseq predictions into them:

# In[ ]:


with open("../input/msci-multiome-quickstart/partial_submission_multi.pickle", 'rb') as f: submission = pickle.load(f)
submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
assert not submission.isna().any()
submission = submission.round(6) # reduce the size of the csv
submission.to_csv('submission.csv')
submission

#!/usr/bin/env python
# coding: utf-8

# # CITEseq LGBM Baseline
# 
# * This notebook will be implemented in the LGBM model using the data processed in the quick start. 
# * LGBM models usually cannot output multiple target variables, but this method can output
# 
# * The reference notes for data processing are below.
# https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart

# # Please vote if this is useful!

# In[1]:


import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator
import warnings
warnings.simplefilter('ignore')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

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


# In[2]:


get_ipython().system('pip install --quiet tables')


# # Loading the common metadata table
# 
# The current version of the model is so primitive that it doesn't use the metadata, but we load it anyway.

# In[3]:


df_cell = pd.read_csv(FP_CELL_METADATA)
df_cell_cite = df_cell[df_cell.technology=="citeseq"]
df_cell_multi = df_cell[df_cell.technology=="multiome"]
df_cell_cite.shape, df_cell_multi.shape


# # Cross-validation
# 
# The note I referred to had the following description, but I confirmed that 13000 rows can be rotated in memory, so I changed columns_to_use = 13000. In addition, the search is performed by changing the starting point of the line to be acquired.
# 
# Data size:
# - The training input has shape 70988\*22050 (10.6 GByte).
# - The training labels have shape 70988\*140.
# - The test input has shape 48663\*22050 (4.3 GByte).
# 
# To get a result with only 16 GByte RAM, we simplify the problem as follows:
# - We ignore the complete metadata (donors, days, cell types).
# - We drop all feature columns which are constant.
# - Of the remaining columns, we keep only the last 12000.
# - We do a PCA and keep only the 240 most important components.
# - We use PCA(copy=False), which overwrites its input in fit_transform().
# - We fit a ridge regression model with 70988\*240 inputs and 70988\*140 outputs. 

# In[4]:


get_ipython().run_cell_magic('time', '', '# Preprocessing\ncol_start = 10000\n\nclass PreprocessCiteseq(BaseEstimator, TransformerMixin):\n    columns_to_use = 13000\n    \n    @staticmethod\n    def take_column_subset(X):\n        return X[:,-(PreprocessCiteseq.columns_to_use+col_start):-col_start]\n    \n    def transform(self, X):\n        print(X.shape)\n        X = X[:,~self.all_zero_columns]\n        print(X.shape)\n        X = PreprocessCiteseq.take_column_subset(X) # use only a part of the columns\n        print(X.shape)\n        gc.collect()\n\n        X = self.pca.transform(X)\n        print(X.shape)\n        return X\n\n    def fit_transform(self, X):\n        gc.collect()\n        print(X.shape)\n        self.all_zero_columns = (X == 0).all(axis=0)\n        X = X[:,~self.all_zero_columns]\n        print(X.shape)\n        X = PreprocessCiteseq.take_column_subset(X) # use only a part of the columns\n        print(X.shape)\n        gc.collect()\n\n        self.pca = PCA(n_components=240, copy=False, random_state=1)\n        X = self.pca.fit_transform(X)\n#         plt.plot(self.pca.explained_variance_ratio_.cumsum())\n#         plt.title("Cumulative explained variance ratio")\n#         plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))\n#         plt.xlabel(\'PCA component\')\n#         plt.ylabel(\'Cumulative explained variance ratio\')\n#         plt.show()\n        print(X.shape)\n        return X\n\npreprocessor = PreprocessCiteseq()\n\ncite_train_x = None\ncite_train_x = preprocessor.fit_transform(pd.read_hdf(FP_CITE_TRAIN_INPUTS).values)\n\ncite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values\nprint(cite_train_y.shape)')


# # Modeling&Prediction
# 
# We retrain the model on all training rows, delete the training data, load the test data and compute the predictions.

# In[5]:


params = {
     'learning_rate': 0.1, 
     'metric': 'mae', 
     "seed": 42,
    'reg_alpha': 0.0014, 
    'reg_lambda': 0.2, 
    'colsample_bytree': 0.8, 
    'subsample': 0.5, 
    'max_depth': 10, 
    'num_leaves': 722, 
    'min_child_samples': 83, 
    }

model = MultiOutputRegressor(lgb.LGBMRegressor(**params, n_estimators=1000))

model.fit(cite_train_x, cite_train_y)

y_va_pred = model.predict(cite_train_x)
mse = mean_squared_error(cite_train_y, y_va_pred)
print(mse)
del cite_train_x, cite_train_y
gc.collect()


# In[6]:


cite_test_x = preprocessor.transform(pd.read_hdf(FP_CITE_TEST_INPUTS).values)
test_pred = model.predict(cite_test_x)
del cite_test_x
test_pred.shape


# # Submission
# 
# We save the CITEseq predictions so that they can be merged with the Multiome predictions in the [Multiome quickstart notebook](https://www.kaggle.com/ambrosm/msci-multiome-quickstart).
# 
# The CITEseq test predictions produced by the ridge regressor have 48663 rows (i.e., cells) and 140 columns (i.e. proteins). 48663 * 140 = 6812820.
# 

# In[7]:


with open('citeseq_pred.pickle', 'wb') as f: pickle.dump(test_pred, f) # float32 array of shape (48663, 140)


# The final submission will have 65744180 rows, of which the first 6812820 are for the CITEseq predictions and the remaining 58931360 for the Multiome predictions. 
# 
# We now read the Multiome predictions and merge the CITEseq predictions into them:

# In[8]:


with open("../input/msci-multiome-quickstart/partial_submission_multi.pickle", 'rb') as f: submission = pickle.load(f)
submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
assert not submission.isna().any()
submission = submission.round(6) # reduce the size of the csv
submission.to_csv('submission.csv')
submission


# In[ ]:





# In[ ]:





# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # What is about ?
# 
# EDA for Kaggle competition: Multimodal Single-Cell Integration Competition
# 
# (c) Alexander Chervov
# 
# Work in progress. We will move towards standard ( +- variations)  bionformatics analysis for single cell data. 
# 
# (For single cell RNA sequencing part - see e.g. several Scanpy tutorials: 
# https://scanpy.readthedocs.io/en/stable/tutorials.html , some of them can be found on Kaggle: 
# https://www.kaggle.com/datasets/alexandervc/scanpy-python-package-for-scrnaseq-analysis .
# If you are "R"-user - google for "Seurat") 
# 
# #### Versions:
# 
# #### 2,3,4,5 cosmetic changes
# 
# #### 1 CiteSeq targets analysis
# 
# Look on only CiteSeq targets: Proteomics  (cell surface protein markers labeled mainly CD**). CD means "Cluster of Differentiation" - see https://en.wikipedia.org/wiki/Cluster_of_differentiation . 
# As expected we see many of them quite correlate with the cell types. 
# 
# Some correlated targets are: ['CD71' ,'CD115', 'CD88'] and ['CD155', 'CD112', 'CD47', 'HLA-A-B-C', 'CD45RA', 'CD31', 'CD11a', 'CD13', 'CD29',   'CD81', 'CD18', 'CD45', 'CD49d', 'CD162']
# 

# #  Install/Import packages
# 
# 

# In[76]:


#If you see a urllib warning running this cell, go to "Settings" on the right hand side, 
#and turn on internet. Note, you need to be phone verified.
get_ipython().system('pip install --quiet tables')


# In[77]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1.2. Set filepaths

# In[78]:


os.listdir("/kaggle/input/open-problems-multimodal/")


# In[79]:


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


# ## Load Data

# In[102]:


get_ipython().run_cell_magic('time', '', "df_cite_train_y = pd.read_hdf('../input/open-problems-multimodal/train_cite_targets.h5')\nprint(df_cell['cell_type'].value_counts())\ndf_cite_train_y.head()\n\ndf_cell = pd.read_csv(FP_CELL_METADATA, index_col = 0)\nd = df_cell.join(df_cite_train_y, how = 'right')#.join()\nprint(d.shape, df_cite_train_y.shape )\n")


# ### Cell types: 
# 
#     MasP = Mast Cell Progenitor
#     MkP = Megakaryocyte Progenitor
#     NeuP = Neutrophil Progenitor
#     MoP = Monocyte Progenitor
#     EryP = Erythrocyte Progenitor
#     HSC = Hematoploetic Stem Cell
#     BP = B-Cell Progenitor

# # Dimensional reduction with PCA

# In[81]:


X = df_cite_train_y.values
list_X_column_names = list(df_cite_train_y.columns)


# In[82]:



from sklearn.decomposition import PCA
import time 
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=50)
t0 = time.time()
r = pca.fit_transform(X)
print(time.time()-t0, 'secs passed for PCA')

fig = plt.figure(figsize = (15,7) )
c = 0
# for f in ['cp_dose', 'cp_type','cp_time', 'y_sum']:
#     c+=1; fig.add_subplot(1, 4 , c) 
sns.scatterplot(x=r[:,0], y=r[:,1])# , hue = df[f]  )
#    plt.title('Colored by '+f)
plt.show()

fig = plt.figure(figsize = (15,7) )
fig.add_subplot(1, 2, 1) 
plt.plot(pca.singular_values_,'o-')
plt.title('Singular values')
fig.add_subplot(1, 2, 2) 
plt.plot(pca.explained_variance_ratio_,'o-')
plt.title('explained variance')


# In[84]:


fig = plt.figure(figsize = (20,10) )
c = 0
for f in ['day', 'donor', 'cell_type']:# , 'technology']:
    c+=1; fig.add_subplot(1, 3 , c) 
    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = d[f]  )
    plt.title('Colored by '+f)
plt.show()


# # Correlation analysis

# In[85]:


t0 = time.time()
corr_matr = np.corrcoef(X.T) # Hint - use numpy , pandas is MUCH SLOWER   (df.corr() )
print(time.time() - t0, 'seconds passed')
print(np.min(corr_matr ), 'minimal correlation' )
corr_matr_abs = np.abs( corr_matr )
print(np.mean(corr_matr_abs ), 'average absolute correlation' )
print(np.median(corr_matr_abs), 'median absolute correlation' )
print(np.min(corr_matr_abs ), 'min absolute correlation' )
print(np.std(corr_matr_abs ), 'std absolute correlation' )

v = corr_matr.flatten()
plt.figure(figsize=(14,8))
t0 = time.time()
plt.hist(v, bins = 50)
plt.title('correlation coefficients distribution')
plt.show()
print(time.time() - t0, 'seconds passed')


v.shape


# In[86]:


df_corr_matr = pd.DataFrame(corr_matr, index = list_X_column_names, columns = list_X_column_names )

#clustermap
import seaborn as sns
t0 = time.time()
sns.clustermap(np.abs(df_corr_matr),cmap='vlag');
print( np.round(time.time()- t0,1), ' seconds passed.')


# In[87]:


import igraph


# In[88]:


verbose = 0
df_stat = pd.DataFrame() # dict_save_largest_component_size = {} 
i = 0
for correlation_threshold in [0.8, 0.7, 0.6, 0.5, 0.4] :
    t0 = time.time()
    print()
    print(correlation_threshold , 'correlation_threshold ')
    corr_matr_abs_bool = corr_matr_abs > correlation_threshold
    corr_matr_abs_bool = corr_matr_abs_bool # Restrict to  genes part 
    corr_matr_abs_bool = np.triu(corr_matr_abs_bool,1) # Take upper triangular part 
    g = igraph.Graph().Adjacency(corr_matr_abs_bool.tolist())
    g.to_undirected(mode = 'collapse')
    if verbose >= 10:
        print( corr_matr_abs_bool.astype(int) )
        print('Number of nodes ', g.vcount())
        print('Number of edges ', g.ecount() )
        print('Number of weakly connected compoenents', len( g.clusters(mode='WEAK')))


    list_clusters_nodes_lists = list( g.clusters(mode='WEAK') )
    list_clusers_size = [len(t) for t in list_clusters_nodes_lists ]
    list_clusers_size = np.sort(list_clusers_size)[::-1]
    print('Top 5 cluster sizes:', list_clusers_size[:5] , 'seconds passed:', np.round(time.time()-t0 , 2))
    #dict_save_largest_component_size[correlation_threshold ] = list_clusers_size[0]
    for t  in list_clusters_nodes_lists:
        if len(t) == list_clusers_size[0]:
            print('50 Genes in largest correlated group:')
            print(np.array(list_X_column_names)[t[:50]])
    i += 1
    df_stat.loc[i,'correlation threshold'] = correlation_threshold
    df_stat.loc[i,'Largest Component Size'] = list_clusers_size[0]
    df_stat.loc[i,'Second Component Size'] = list_clusers_size[1]
    
df_stat


# # Dimensional reduction with UMAP

# In[89]:


import umap

t0 = time.time()
r = umap.UMAP().fit_transform(X)
print(time.time()-t0, 'secs passed for umap')


fig = plt.figure(figsize = (15,7) )
# c = 0
# for f in ['cp_dose', 'cp_type','cp_time','y_sum']:
#     c+=1; fig.add_subplot(1, 4 , c) 
sns.scatterplot(x=r[:,0], y=r[:,1])#  , hue = df[f]  )
# plt.show()


# In[90]:


fig = plt.figure(figsize = (20,10) )
c = 0
for f in ['day', 'donor', 'cell_type']:# , 'technology']:
    c+=1; fig.add_subplot(1, 3 , c) 
    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = d[f]  )
    plt.title('Colored by '+f)
plt.show()


#     MasP = Mast Cell Progenitor
#     MkP = Megakaryocyte Progenitor
#     NeuP = Neutrophil Progenitor
#     MoP = Monocyte Progenitor
#     EryP = Erythrocyte Progenitor
#     HSC = Hematoploetic Stem Cell
#     BP = B-Cell Progenitor

# In[91]:


# 0.7 correlation_threshold 
# Top 5 cluster sizes: [3 2 1 1 1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD71' 'CD115' 'CD88']

# 0.6 correlation_threshold 
# Top 5 cluster sizes: [14  5  2  1  1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD155' 'CD112' 'CD47' 'HLA-A-B-C' 'CD45RA' 'CD31' 'CD11a' 'CD13' 'CD29'
#  'CD81' 'CD18' 'CD45' 'CD49d' 'CD162']

import numbers

n_x_subplots = 8
palette = 'rainbow'#'viridis'

t0 = time.time()

c = 0
for f in ['CD71', 'CD115' ,'CD88'] + ['CD155', 'CD112', 'CD47', 'HLA-A-B-C' ,'CD45RA' ,'CD31' ,'CD11a', 'CD13', 'CD29',
  'CD81', 'CD18', 'CD45', 'CD49d', 'CD162']: # ['day', 'donor', 'cell_type']:# , 'technology']:
    if c % n_x_subplots == 0:
        if c > 0:
            plt.show()
        fig = plt.figure(figsize = (20,5) ); c = 0
        #plt.suptitle(str(k),fontsize = 20 )# str_data_inf + ' n_cells: ' +str(mask.sum()) + ' RED > median expression, BLUE <= median ' )# +' ' + cell_type +' ' + drug )
        
    c += 1; fig.add_subplot(1,n_x_subplots ,c)    
    
    v = d[f]
    if isinstance(v[0], numbers.Number):
        v = np.clip(v,np.percentile(v,5), np.percentile(v,95) )
        # v = (v > np.median(v)).astype(int)
    ax = sns.scatterplot(x=r[:,0], y=r[:,1] , hue =  v , palette = palette )
    plt.setp(ax.get_legend().get_texts(), fontsize=10) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=10) # for legend title    
    plt.title('Colored by '+f, fontsize = 10)
    
plt.show()
print(time.time()-t0, 'secs passed')


# In[92]:


import numbers

n_x_subplots = 8
palette = 'rainbow'#'viridis'

t0 = time.time()

c = 0
for f in d.columns: # ['day', 'donor', 'cell_type']:# , 'technology']:
    if c % n_x_subplots == 0:
        if c > 0:
            plt.show()
        fig = plt.figure(figsize = (20,5) ); c = 0
        #plt.suptitle(str(k),fontsize = 20 )# str_data_inf + ' n_cells: ' +str(mask.sum()) + ' RED > median expression, BLUE <= median ' )# +' ' + cell_type +' ' + drug )
        
    c += 1; fig.add_subplot(1,n_x_subplots ,c)    
    
    v = d[f]
    if isinstance(v[0], numbers.Number):
        v = np.clip(v,np.percentile(v,5), np.percentile(v,95) )
        # v = (v > np.median(v)).astype(int)
    ax = sns.scatterplot(x=r[:,0], y=r[:,1] , hue =  v , palette = palette )
    plt.setp(ax.get_legend().get_texts(), fontsize=10) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=10) # for legend title    
    plt.title('Colored by '+f, fontsize = 10)
    
plt.show()
print(time.time()-t0, 'secs passed')


# # Other dimensional reductions (from Sklearn)

# In[93]:


# Based on: 
# https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
# See also:
# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#



# To speed-up reduce dimensions by PCA first
X_save = X.copy( )
#r = pca.fit_transform(X)
#X = r[:1000,:20]



import umap 
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD


from collections import OrderedDict
from functools import partial
from matplotlib.ticker import NullFormatter


n_neighbors = 10
n_components = 2
# Set-up manifold methods
LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors, n_components, eigen_solver='auto')

methods = OrderedDict()
methods['PCA'] = PCA()
methods['umap'] = umap.UMAP(n_components = n_components)
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
methods['ICA'] = FastICA(n_components=n_components,         random_state=0)
methods['FA'] = FactorAnalysis(n_components=n_components, random_state=0)
#methods['LLE'] = LLE(method='standard')
#methods['Modified LLE'] = LLE(method='modified')
#methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors)
methods['NMF'] = NMF(n_components=n_components,  init='random', random_state=0) 
methods['RandProj'] = SparseRandomProjection(n_components=n_components, random_state=42)

rand_trees_embed = make_pipeline(RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5), TruncatedSVD(n_components=n_components) )
methods['RandTrees'] = rand_trees_embed
methods['LatDirAll'] = LatentDirichletAllocation(n_components=n_components,  random_state=0)
#methods['LTSA'] = LLE(method='ltsa') 
#methods['Hessian LLE'] = LLE(method='hessian') 

list_fast_methods = ['PCA','umap','FA', 'NMF','RandProj','RandTrees'] # 'ICA',
list_slow_methods = ['t-SNE','LLE','Modified LLE','Isomap','MDS','SE','LatDirAll','LTSA','Hessian LLE']

# transformer = NeighborhoodComponentsAnalysis(init='random',  n_components=2, random_state=0) # Cannot be applied since supervised - requires y 
# methods['LinDisA'] = LinearDiscriminantAnalysis(n_components=n_components)# Cannot be applied since supervised - requires y 


# Create figure
fig = plt.figure(figsize=(25, 16))

# Plot results
c = 0
for i, (label, method) in enumerate(methods.items()):
    if label not in  list_fast_methods :
        continue
        
    t0 = time.time()
    try:
        r = method.fit_transform(X)
    except:
        print('Got Exception', label )
        continue 
    t1 = time.time()
    print("%s: %.2g sec" % (label, t1 - t0))
    c+=1
    fig.add_subplot(2, 3 , c) 
    sns.scatterplot(x=r[:,0], y=r[:,1] , hue =  d['cell_type'])
    plt.title(label )
    plt.legend('')

plt.show()


# # Trimap - yet another anlogue of tsne/umap 

# In[94]:


get_ipython().system('pip install trimap ')
import trimap


# In[95]:


column4color = 'cell_type'


# In[96]:



reducer =  trimap.TRIMAP() #  umap.UMAP()

t0 = time.time()
r = reducer.fit_transform(X)
print(time.time()-t0, 'secs passed for dimension reduction')


fig = plt.figure(figsize = (15,7) )
# c = 0
# for f in ['cp_dose', 'cp_type','cp_time','y_sum']:
#     c+=1; fig.add_subplot(1, 4 , c) 
sns.scatterplot(x=r[:,0], y=r[:,1]  , hue = d[column4color]  )
plt.title(str(reducer), fontsize = 20 )

plt.show()


# # NCVis - similar to UMAP, bust faster (from Skolkovo team)

# In[97]:


get_ipython().system('pip install ncvis')
import ncvis


# In[98]:


reducer =  ncvis.NCVis() # trimap.TRIMAP() #  umap.UMAP()

t0 = time.time()
r = reducer.fit_transform(X)
print(time.time()-t0, 'secs passed for dimension reduction')


fig = plt.figure(figsize = (15,7) )
# c = 0
# for f in ['cp_dose', 'cp_type','cp_time','y_sum']:
#     c+=1; fig.add_subplot(1, 4 , c) 
sns.scatterplot(x=r[:,0], y=r[:,1]  , hue = d[column4color]  )
plt.title(str(reducer), fontsize = 20 )
plt.show()


# In[99]:


fig = plt.figure(figsize = (20,10) )
c = 0
for f in ['day', 'donor', 'cell_type']:# , 'technology']:
    c+=1; fig.add_subplot(1, 3 , c) 
    sns.scatterplot(x=r[:,0], y=r[:,1] , hue = d[f]  )
    plt.title('Colored by '+f)
plt.show()


#     MasP = Mast Cell Progenitor
#     MkP = Megakaryocyte Progenitor
#     NeuP = Neutrophil Progenitor
#     MoP = Monocyte Progenitor
#     EryP = Erythrocyte Progenitor
#     HSC = Hematoploetic Stem Cell
#     BP = B-Cell Progenitor

# In[100]:


# 0.7 correlation_threshold 
# Top 5 cluster sizes: [3 2 1 1 1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD71' 'CD115' 'CD88']

# 0.6 correlation_threshold 
# Top 5 cluster sizes: [14  5  2  1  1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD155' 'CD112' 'CD47' 'HLA-A-B-C' 'CD45RA' 'CD31' 'CD11a' 'CD13' 'CD29'
#  'CD81' 'CD18' 'CD45' 'CD49d' 'CD162']

import numbers

n_x_subplots = 8
palette = 'rainbow'#'viridis'

t0 = time.time()

c = 0
for f in ['CD71', 'CD115' ,'CD88'] + ['CD155', 'CD112', 'CD47', 'HLA-A-B-C' ,'CD45RA' ,'CD31' ,'CD11a', 'CD13', 'CD29',
  'CD81', 'CD18', 'CD45', 'CD49d']:# , 'CD162']: # ['day', 'donor', 'cell_type']:# , 'technology']:
    if c % n_x_subplots == 0:
        if c > 0:
            plt.show()
        fig = plt.figure(figsize = (20,5) ); c = 0
        #plt.suptitle(str(k),fontsize = 20 )# str_data_inf + ' n_cells: ' +str(mask.sum()) + ' RED > median expression, BLUE <= median ' )# +' ' + cell_type +' ' + drug )
        
    c += 1; fig.add_subplot(1,n_x_subplots ,c)    
    
    v = d[f]
    if isinstance(v[0], numbers.Number):
        v = np.clip(v,np.percentile(v,5), np.percentile(v,95) )
        # v = (v > np.median(v)).astype(int)
    ax = sns.scatterplot(x=r[:,0], y=r[:,1] , hue =  v , palette = palette )
    plt.setp(ax.get_legend().get_texts(), fontsize=10) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=10) # for legend title    
    plt.title('Colored by '+f, fontsize = 10)
    
plt.show()
print(time.time()-t0, 'secs passed')


# In[104]:



# 0.7 correlation_threshold 
# Top 5 cluster sizes: [3 2 1 1 1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD71' 'CD115' 'CD88']

# 0.6 correlation_threshold 
# Top 5 cluster sizes: [14  5  2  1  1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD155' 'CD112' 'CD47' 'HLA-A-B-C' 'CD45RA' 'CD31' 'CD11a' 'CD13' 'CD29'
#  'CD81' 'CD18' 'CD45' 'CD49d' 'CD162']

# CD69 ?????????????????????????????? ?? ?? ?????????????? ?????????? ??????????????????. ?????????? ?????????? ?????? ??????????????????????????. ?????? ???? ?????????? ???????? ?????????????????? CD25 ?? CD71

# https://link.springer.com/article/10.1007/BF01305907
    
import numbers

n_x_subplots = 4
palette = 'rainbow'#'viridis'

t0 = time.time()

c = 0
for f in ['CD69', 'CD25' ,'CD71','cell_type']:# , 'CD162']: # ['day', 'donor', 'cell_type']:# , 'technology']:
    if c % n_x_subplots == 0:
        if c > 0:
            plt.show()
        fig = plt.figure(figsize = (20,5) ); c = 0
        #plt.suptitle(str(k),fontsize = 20 )# str_data_inf + ' n_cells: ' +str(mask.sum()) + ' RED > median expression, BLUE <= median ' )# +' ' + cell_type +' ' + drug )
        
    c += 1; fig.add_subplot(1,n_x_subplots ,c)    
    
    v = d[f]
    if isinstance(v[0], numbers.Number):
        v = np.clip(v,np.percentile(v,5), np.percentile(v,95) )
        # v = (v > np.median(v)).astype(int)
    ax = sns.scatterplot(x=r[:,0], y=r[:,1] , hue =  v , palette = palette )
    plt.setp(ax.get_legend().get_texts(), fontsize=10) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=10) # for legend title    
    plt.title('Colored by '+f, fontsize = 10)
    
plt.show()
print(time.time()-t0, 'secs passed')


#     MasP = Mast Cell Progenitor
#     MkP = Megakaryocyte Progenitor
#     NeuP = Neutrophil Progenitor
#     MoP = Monocyte Progenitor
#     EryP = Erythrocyte Progenitor
#     HSC = Hematoploetic Stem Cell
#     BP = B-Cell Progenitor

# In[106]:



# 0.7 correlation_threshold 
# Top 5 cluster sizes: [3 2 1 1 1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD71' 'CD115' 'CD88']

# 0.6 correlation_threshold 
# Top 5 cluster sizes: [14  5  2  1  1] seconds passed: 0.0
# 50 Genes in largest correlated group:
# ['CD155' 'CD112' 'CD47' 'HLA-A-B-C' 'CD45RA' 'CD31' 'CD11a' 'CD13' 'CD29'
#  'CD81' 'CD18' 'CD45' 'CD49d' 'CD162']

# CD69 ?????????????????????????????? ?? ?? ?????????????? ?????????? ??????????????????. ?????????? ?????????? ?????? ??????????????????????????. ?????? ???? ?????????? ???????? ?????????????????? CD25 ?? CD71

# https://link.springer.com/article/10.1007/BF01305907
    
import numbers

n_x_subplots = 5
palette = 'rainbow'#'viridis'

t0 = time.time()

c = 0
for f in ['CD3', 'CD4', 'CD8', 'CD25','cell_type']: #  ['CD69', 'CD25' ,'CD71','cell_type']:# , 'CD162']: # ['day', 'donor', 'cell_type']:# , 'technology']:
    if f not in d.columns: continue 
    if c % n_x_subplots == 0:
        if c > 0:
            plt.show()
        fig = plt.figure(figsize = (20,5) ); c = 0
        #plt.suptitle(str(k),fontsize = 20 )# str_data_inf + ' n_cells: ' +str(mask.sum()) + ' RED > median expression, BLUE <= median ' )# +' ' + cell_type +' ' + drug )
        
    c += 1; fig.add_subplot(1,n_x_subplots ,c)    
    
    v = d[f]
    if isinstance(v[0], numbers.Number):
        v = np.clip(v,np.percentile(v,5), np.percentile(v,95) )
        # v = (v > np.median(v)).astype(int)
    ax = sns.scatterplot(x=r[:,0], y=r[:,1] , hue =  v , palette = palette )
    plt.setp(ax.get_legend().get_texts(), fontsize=10) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=10) # for legend title    
    plt.title('Colored by '+f, fontsize = 10)
    
plt.show()
print(time.time()-t0, 'secs passed')


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # Multimodal Single-Cell????IIntegration: EDA ???? & simple predictions

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


# # Statistic predictions ??????? gene means

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

#!/usr/bin/env python
# coding: utf-8

# # CITEseq Keras Quickstart
# 
# This notebook shows how to tune and cross-validate a Keras model for the CITEseq part of the *Multimodal Single-Cell Integration* competition.
# 
# It does not show the EDA - see the separate notebook [MSCI EDA which makes sense ??????????????????????????????](https://www.kaggle.com/ambrosm/msci-eda-which-makes-sense).
# 
# The CITEseq predictions of the Keras model are then concatenated with the Multiome predictions of @jsmithperera's [Multiome Quickstart w/ Sparse M + tSVD = 32](https://www.kaggle.com/code/jsmithperera/multiome-quickstart-w-sparse-m-tsvd-32) to a complete submission file.
# 
# ## Summary
# 
# The CITEseq part of the competition has sizeable datasets, when compared to the standard 16 GByte RAM of Kaggle notebooks:
# - The training input has shape 70988\*22050 (10.6 GByte).
# - The training labels have shape 70988\*140.
# - The test input has shape 48663\*22050 (4.3 GByte).
# 
# Our solution strategy has five elements:
# 1. **Dimensionality reduction:** To reduce the size of the 10.6 GByte input data, we project the 22050 features to a space with only 64 dimensions by applying a truncated SVD. To these 64 dimensions, we add 144 features whose names shows their importance.
# 2. **The model:** The model is a sequential dense network with four hidden layers.
# 3. **The loss function:** The competition is scored by the average Pearson correlation coefficient between the predictions and the ground truth. As this scoring function is differentiable, we can directly use it as loss function for a neural network. This gives neural networks an advantage in comparison to algorithms which use mean squared error as a surrogate loss. 
# 3. **Hyperparameter tuning with KerasTuner:** We tune the hyperparameters with [KerasTuner](https://keras.io/keras_tuner/). 
# 4. **Cross-validation:** Submitting unvalidated models and relying only on the public leaderboard is bad practice. The model in this notebook is fully cross-validated with a 3-fold GroupKFold.
# 

# In[1]:


import os, gc, pickle, datetime, scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.utils import plot_model
import keras_tuner

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

TUNE = False
SUBMIT = True


# A little trick to save time with pip: If the module is already installed (after a restart of the notebook, for instance), pip wastes 10 seconds by checking whether a newer version exists. We can skip this check by testing for the presence of the module in a simple if statement.

# In[2]:


get_ipython().run_cell_magic('time', '', '# If you see a warning "Failed to establish a new connection" running this cell,\n# go to "Settings" on the right hand side, \n# and turn on internet. Note, you need to be phone verified.\n# We need this library to read HDF files.\nif not os.path.exists(\'/opt/conda/lib/python3.7/site-packages/tables\'):\n    !pip install --quiet tables\n    ')


# # The scoring function
# 
# This competition has a special metric: For every row, it computes the Pearson correlation between y_true and y_pred, and then all these correlation coefficients are averaged. We implement two variants of the metric: The first one is for numpy arrays, the second one for tensors - thanks to @lucasmorin for the [original tensor implementation](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/347595).

# In[3]:


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)

def negative_correlation_loss(y_true, y_pred):
    """Negative correlation loss function for Keras
    
    Precondition:
    y_true.mean(axis=1) == 0
    y_true.std(axis=1) == 1
    
    Returns:
    -1 = perfect positive correlation
    1 = totally negative correlation
    """
    my = K.mean(tf.convert_to_tensor(y_pred), axis=1)
    my = tf.tile(tf.expand_dims(my, axis=1), (1, y_true.shape[1]))
    ym = y_pred - my
    r_num = K.sum(tf.multiply(y_true, ym), axis=1)
    r_den = tf.sqrt(K.sum(K.square(ym), axis=1) * float(y_true.shape[-1]))
    r = tf.reduce_mean(r_num / r_den)
    return - r


# # Data loading and preprocessing
# 
# The metadata is used only for the `GroupKFold`: 

# In[4]:


metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="citeseq"]
metadata_df.shape


# We now define two sets of features:
# - `constant_cols` is the set of all features which are constant in the train or test datset. These columns will be discarded immediately after loading.
# - `important_cols` is the set of all features whose name matches the name of a target protein. If a gene is named 'ENSG00000114013_CD86', it should be related to a protein named 'CD86'. These features will be used for the model unchanged, that is, they don't undergo dimensionality reduction. 

# In[5]:


# constant_cols = list(X.columns[(X == 0).all(axis=0).values]) + list(X_test.columns[(X_test == 0).all(axis=0).values])
constant_cols = ['ENSG00000003137_CYP26B1', 'ENSG00000004848_ARX', 'ENSG00000006606_CCL26', 'ENSG00000010379_SLC6A13', 'ENSG00000010932_FMO1', 'ENSG00000017427_IGF1', 'ENSG00000022355_GABRA1', 'ENSG00000041982_TNC', 'ENSG00000060709_RIMBP2', 'ENSG00000064886_CHI3L2', 'ENSG00000065717_TLE2', 'ENSG00000067798_NAV3', 'ENSG00000069535_MAOB', 'ENSG00000073598_FNDC8', 'ENSG00000074219_TEAD2', 'ENSG00000074964_ARHGEF10L', 'ENSG00000077264_PAK3', 'ENSG00000078053_AMPH', 'ENSG00000082684_SEMA5B', 'ENSG00000083857_FAT1', 'ENSG00000084628_NKAIN1', 'ENSG00000084734_GCKR', 'ENSG00000086967_MYBPC2', 'ENSG00000087258_GNAO1', 'ENSG00000089505_CMTM1', 'ENSG00000091129_NRCAM', 'ENSG00000091986_CCDC80', 'ENSG00000092377_TBL1Y', 'ENSG00000092969_TGFB2', 'ENSG00000095397_WHRN', 'ENSG00000095970_TREM2', 'ENSG00000099715_PCDH11Y', 'ENSG00000100197_CYP2D6', 'ENSG00000100218_RSPH14', 'ENSG00000100311_PDGFB', 'ENSG00000100362_PVALB', 'ENSG00000100373_UPK3A', 'ENSG00000100625_SIX4', 'ENSG00000100867_DHRS2', 'ENSG00000100985_MMP9', 'ENSG00000101197_BIRC7', 'ENSG00000101298_SNPH', 'ENSG00000102387_TAF7L', 'ENSG00000103034_NDRG4', 'ENSG00000104059_FAM189A1', 'ENSG00000104112_SCG3', 'ENSG00000104313_EYA1', 'ENSG00000104892_KLC3', 'ENSG00000105088_OLFM2', 'ENSG00000105261_OVOL3', 'ENSG00000105290_APLP1', 'ENSG00000105507_CABP5', 'ENSG00000105642_KCNN1', 'ENSG00000105694_ELOCP28', 'ENSG00000105707_HPN', 'ENSG00000105894_PTN', 'ENSG00000106018_VIPR2', 'ENSG00000106541_AGR2', 'ENSG00000107317_PTGDS', 'ENSG00000108688_CCL7', 'ENSG00000108702_CCL1', 'ENSG00000108947_EFNB3', 'ENSG00000109193_SULT1E1', 'ENSG00000109794_FAM149A', 'ENSG00000109832_DDX25', 'ENSG00000110195_FOLR1', 'ENSG00000110375_UPK2', 'ENSG00000110436_SLC1A2', 'ENSG00000111339_ART4', 'ENSG00000111863_ADTRP', 'ENSG00000112761_WISP3', 'ENSG00000112852_PCDHB2', 'ENSG00000114251_WNT5A', 'ENSG00000114279_FGF12', 'ENSG00000114455_HHLA2', 'ENSG00000114757_PEX5L', 'ENSG00000115155_OTOF', 'ENSG00000115266_APC2', 'ENSG00000115297_TLX2', 'ENSG00000115590_IL1R2', 'ENSG00000115844_DLX2', 'ENSG00000116194_ANGPTL1', 'ENSG00000116661_FBXO2', 'ENSG00000116774_OLFML3', 'ENSG00000117322_CR2', 'ENSG00000117971_CHRNB4', 'ENSG00000118322_ATP10B', 'ENSG00000118402_ELOVL4', 'ENSG00000118520_ARG1', 'ENSG00000118946_PCDH17', 'ENSG00000118972_FGF23', 'ENSG00000119771_KLHL29', 'ENSG00000120549_KIAA1217', 'ENSG00000121316_PLBD1', 'ENSG00000121905_HPCA', 'ENSG00000122224_LY9', 'ENSG00000124194_GDAP1L1', 'ENSG00000124440_HIF3A', 'ENSG00000124657_OR2B6', 'ENSG00000125462_C1orf61', 'ENSG00000125895_TMEM74B', 'ENSG00000126838_PZP', 'ENSG00000128422_KRT17', 'ENSG00000128918_ALDH1A2', 'ENSG00000129170_CSRP3', 'ENSG00000129214_SHBG', 'ENSG00000129673_AANAT', 'ENSG00000129910_CDH15', 'ENSG00000130294_KIF1A', 'ENSG00000130307_USHBP1', 'ENSG00000130545_CRB3', 'ENSG00000131019_ULBP3', 'ENSG00000131044_TTLL9', 'ENSG00000131183_SLC34A1', 'ENSG00000131386_GALNT15', 'ENSG00000131400_NAPSA', 'ENSG00000131914_LIN28A', 'ENSG00000131941_RHPN2', 'ENSG00000131951_LRRC9', 'ENSG00000132170_PPARG', 'ENSG00000132681_ATP1A4', 'ENSG00000132958_TPTE2', 'ENSG00000133454_MYO18B', 'ENSG00000134545_KLRC1', 'ENSG00000134853_PDGFRA', 'ENSG00000135083_CCNJL', 'ENSG00000135100_HNF1A', 'ENSG00000135116_HRK', 'ENSG00000135312_HTR1B', 'ENSG00000135324_MRAP2', 'ENSG00000135436_FAM186B', 'ENSG00000135472_FAIM2', 'ENSG00000135898_GPR55', 'ENSG00000135929_CYP27A1', 'ENSG00000136002_ARHGEF4', 'ENSG00000136099_PCDH8', 'ENSG00000136274_NACAD', 'ENSG00000137078_SIT1', 'ENSG00000137142_IGFBPL1', 'ENSG00000137473_TTC29', 'ENSG00000137474_MYO7A', 'ENSG00000137491_SLCO2B1', 'ENSG00000137691_CFAP300', 'ENSG00000137731_FXYD2', 'ENSG00000137747_TMPRSS13', 'ENSG00000137878_GCOM1', 'ENSG00000138411_HECW2', 'ENSG00000138741_TRPC3', 'ENSG00000138769_CDKL2', 'ENSG00000138823_MTTP', 'ENSG00000139908_TSSK4', 'ENSG00000140832_MARVELD3', 'ENSG00000142178_SIK1', 'ENSG00000142538_PTH2', 'ENSG00000142910_TINAGL1', 'ENSG00000143217_NECTIN4', 'ENSG00000143858_SYT2', 'ENSG00000144130_NT5DC4', 'ENSG00000144214_LYG1', 'ENSG00000144290_SLC4A10', 'ENSG00000144366_GULP1', 'ENSG00000144583_MARCH4', 'ENSG00000144771_LRTM1', 'ENSG00000144891_AGTR1', 'ENSG00000145087_STXBP5L', 'ENSG00000145107_TM4SF19', 'ENSG00000146197_SCUBE3', 'ENSG00000146966_DENND2A', 'ENSG00000147082_CCNB3', 'ENSG00000147614_ATP6V0D2', 'ENSG00000147642_SYBU', 'ENSG00000147869_CER1', 'ENSG00000149403_GRIK4', 'ENSG00000149596_JPH2', 'ENSG00000150630_VEGFC', 'ENSG00000150722_PPP1R1C', 'ENSG00000151631_AKR1C6P', 'ENSG00000151704_KCNJ1', 'ENSG00000152154_TMEM178A', 'ENSG00000152292_SH2D6', 'ENSG00000152315_KCNK13', 'ENSG00000152503_TRIM36', 'ENSG00000153253_SCN3A', 'ENSG00000153902_LGI4', 'ENSG00000153930_ANKFN1', 'ENSG00000154040_CABYR', 'ENSG00000154118_JPH3', 'ENSG00000154175_ABI3BP', 'ENSG00000154645_CHODL', 'ENSG00000157060_SHCBP1L', 'ENSG00000157087_ATP2B2', 'ENSG00000157152_SYN2', 'ENSG00000157168_NRG1', 'ENSG00000157680_DGKI', 'ENSG00000158246_TENT5B', 'ENSG00000158477_CD1A', 'ENSG00000158481_CD1C', 'ENSG00000158488_CD1E', 'ENSG00000159189_C1QC', 'ENSG00000159217_IGF2BP1', 'ENSG00000160683_CXCR5', 'ENSG00000160801_PTH1R', 'ENSG00000160973_FOXH1', 'ENSG00000161594_KLHL10', 'ENSG00000162409_PRKAA2', 'ENSG00000162840_MT2P1', 'ENSG00000162873_KLHDC8A', 'ENSG00000162944_RFTN2', 'ENSG00000162949_CAPN13', 'ENSG00000163116_STPG2', 'ENSG00000163288_GABRB1', 'ENSG00000163531_NFASC', 'ENSG00000163618_CADPS', 'ENSG00000163637_PRICKLE2', 'ENSG00000163735_CXCL5', 'ENSG00000163873_GRIK3', 'ENSG00000163898_LIPH', 'ENSG00000164061_BSN', 'ENSG00000164078_MST1R', 'ENSG00000164123_C4orf45', 'ENSG00000164690_SHH', 'ENSG00000164761_TNFRSF11B', 'ENSG00000164821_DEFA4', 'ENSG00000164845_FAM86FP', 'ENSG00000164867_NOS3', 'ENSG00000166073_GPR176', 'ENSG00000166148_AVPR1A', 'ENSG00000166250_CLMP', 'ENSG00000166257_SCN3B', 'ENSG00000166268_MYRFL', 'ENSG00000166523_CLEC4E', 'ENSG00000166535_A2ML1', 'ENSG00000166819_PLIN1', 'ENSG00000166928_MS4A14', 'ENSG00000167210_LOXHD1', 'ENSG00000167306_MYO5B', 'ENSG00000167634_NLRP7', 'ENSG00000167748_KLK1', 'ENSG00000167889_MGAT5B', 'ENSG00000168140_VASN', 'ENSG00000168546_GFRA2', 'ENSG00000168646_AXIN2', 'ENSG00000168955_TM4SF20', 'ENSG00000168993_CPLX1', 'ENSG00000169075_Z99496.1', 'ENSG00000169194_IL13', 'ENSG00000169246_NPIPB3', 'ENSG00000169884_WNT10B', 'ENSG00000169900_PYDC1', 'ENSG00000170074_FAM153A', 'ENSG00000170075_GPR37L1', 'ENSG00000170289_CNGB3', 'ENSG00000170356_OR2A20P', 'ENSG00000170537_TMC7', 'ENSG00000170689_HOXB9', 'ENSG00000170827_CELP', 'ENSG00000171346_KRT15', 'ENSG00000171368_TPPP', 'ENSG00000171501_OR1N2', 'ENSG00000171532_NEUROD2', 'ENSG00000171611_PTCRA', 'ENSG00000171873_ADRA1D', 'ENSG00000171916_LGALS9C', 'ENSG00000172005_MAL', 'ENSG00000172987_HPSE2', 'ENSG00000173068_BNC2', 'ENSG00000173077_DEC1', 'ENSG00000173210_ABLIM3', 'ENSG00000173267_SNCG', 'ENSG00000173369_C1QB', 'ENSG00000173372_C1QA', 'ENSG00000173391_OLR1', 'ENSG00000173626_TRAPPC3L', 'ENSG00000173698_ADGRG2', 'ENSG00000173868_PHOSPHO1', 'ENSG00000174407_MIR1-1HG', 'ENSG00000174807_CD248', 'ENSG00000175206_NPPA', 'ENSG00000175746_C15orf54', 'ENSG00000175985_PLEKHD1', 'ENSG00000176043_AC007160.1', 'ENSG00000176399_DMRTA1', 'ENSG00000176510_OR10AC1', 'ENSG00000176697_BDNF', 'ENSG00000176826_FKBP9P1', 'ENSG00000176988_FMR1NB', 'ENSG00000177324_BEND2', 'ENSG00000177335_C8orf31', 'ENSG00000177535_OR2B11', 'ENSG00000177614_PGBD5', 'ENSG00000177707_NECTIN3', 'ENSG00000178033_CALHM5', 'ENSG00000178175_ZNF366', 'ENSG00000178462_TUBAL3', 'ENSG00000178732_GP5', 'ENSG00000178750_STX19', 'ENSG00000179058_C9orf50', 'ENSG00000179101_AL590139.1', 'ENSG00000179388_EGR3', 'ENSG00000179611_DGKZP1', 'ENSG00000179899_PHC1P1', 'ENSG00000179934_CCR8', 'ENSG00000180537_RNF182', 'ENSG00000180712_LINC02363', 'ENSG00000180988_OR52N2', 'ENSG00000181001_OR52N1', 'ENSG00000181616_OR52H1', 'ENSG00000181634_TNFSF15', 'ENSG00000182021_AL591379.1', 'ENSG00000182230_FAM153B', 'ENSG00000182853_VMO1', 'ENSG00000183090_FREM3', 'ENSG00000183562_AC131971.1', 'ENSG00000183615_FAM167B', 'ENSG00000183625_CCR3', 'ENSG00000183770_FOXL2', 'ENSG00000183779_ZNF703', 'ENSG00000183831_ANKRD45', 'ENSG00000183844_FAM3B', 'ENSG00000183960_KCNH8', 'ENSG00000184106_TREML3P', 'ENSG00000184227_ACOT1', 'ENSG00000184363_PKP3', 'ENSG00000184434_LRRC19', 'ENSG00000184454_NCMAP', 'ENSG00000184571_PIWIL3', 'ENSG00000184702_SEPT5', 'ENSG00000184908_CLCNKB', 'ENSG00000184923_NUTM2A', 'ENSG00000185070_FLRT2', 'ENSG00000185156_MFSD6L', 'ENSG00000185567_AHNAK2', 'ENSG00000185686_PRAME', 'ENSG00000186190_BPIFB3', 'ENSG00000186191_BPIFB4', 'ENSG00000186231_KLHL32', 'ENSG00000186431_FCAR', 'ENSG00000186715_MST1L', 'ENSG00000187116_LILRA5', 'ENSG00000187185_AC092118.1', 'ENSG00000187268_FAM9C', 'ENSG00000187554_TLR5', 'ENSG00000187867_PALM3', 'ENSG00000188153_COL4A5', 'ENSG00000188158_NHS', 'ENSG00000188163_FAM166A', 'ENSG00000188316_ENO4', 'ENSG00000188959_C9orf152', 'ENSG00000189013_KIR2DL4', 'ENSG00000189409_MMP23B', 'ENSG00000196092_PAX5', 'ENSG00000196260_SFTA2', 'ENSG00000197358_BNIP3P1', 'ENSG00000197446_CYP2F1', 'ENSG00000197540_GZMM', 'ENSG00000198049_AVPR1B', 'ENSG00000198134_AC007537.1', 'ENSG00000198156_NPIPB6', 'ENSG00000198221_AFDN-DT', 'ENSG00000198626_RYR2', 'ENSG00000198759_EGFL6', 'ENSG00000198822_GRM3', 'ENSG00000198963_RORB', 'ENSG00000199090_MIR326', 'ENSG00000199753_SNORD104', 'ENSG00000199787_RF00406', 'ENSG00000199872_RNU6-942P', 'ENSG00000200075_RF00402', 'ENSG00000200296_RNU1-83P', 'ENSG00000200683_RNU6-379P', 'ENSG00000201044_RNU6-268P', 'ENSG00000201343_RF00019', 'ENSG00000201564_RN7SKP50', 'ENSG00000201616_RNU1-91P', 'ENSG00000201737_RNU1-133P', 'ENSG00000202048_SNORD114-20', 'ENSG00000202415_RN7SKP269', 'ENSG00000203395_AC015969.1', 'ENSG00000203721_LINC00862', 'ENSG00000203727_SAMD5', 'ENSG00000203737_GPR52', 'ENSG00000203783_PRR9', 'ENSG00000203867_RBM20', 'ENSG00000203907_OOEP', 'ENSG00000203999_LINC01270', 'ENSG00000204010_IFIT1B', 'ENSG00000204044_SLC12A5-AS1', 'ENSG00000204091_TDRG1', 'ENSG00000204121_ECEL1P1', 'ENSG00000204165_CXorf65', 'ENSG00000204173_LRRC37A5P', 'ENSG00000204248_COL11A2', 'ENSG00000204424_LY6G6F', 'ENSG00000204539_CDSN', 'ENSG00000204583_LRCOL1', 'ENSG00000204677_FAM153C', 'ENSG00000204709_LINC01556', 'ENSG00000204711_C9orf135', 'ENSG00000204792_LINC01291', 'ENSG00000204850_AC011484.1', 'ENSG00000204851_PNMA8B', 'ENSG00000204909_SPINK9', 'ENSG00000205037_AC134312.1', 'ENSG00000205038_PKHD1L1', 'ENSG00000205089_CCNI2', 'ENSG00000205106_DKFZp779M0652', 'ENSG00000205364_MT1M', 'ENSG00000205502_C2CD4B', 'ENSG00000205746_AC126755.1', 'ENSG00000205856_C22orf42', 'ENSG00000206052_DOK6', 'ENSG00000206579_XKR4', 'ENSG00000206645_RF00019', 'ENSG00000206786_RNU6-701P', 'ENSG00000206846_RF00019', 'ENSG00000206848_RNU6-890P', 'ENSG00000207088_SNORA7B', 'ENSG00000207181_SNORA14B', 'ENSG00000207234_RNU6-125P', 'ENSG00000207326_RF00019', 'ENSG00000207359_RNU6-925P', 'ENSG00000211677_IGLC2', 'ENSG00000211699_TRGV3', 'ENSG00000211895_IGHA1', 'ENSG00000212385_RNU6-817P', 'ENSG00000212391_RF00554', 'ENSG00000212607_SNORA3B', 'ENSG00000212829_RPS26P3', 'ENSG00000213083_AC010731.1', 'ENSG00000213216_AC007066.1', 'ENSG00000213222_AC093724.1', 'ENSG00000213228_RPL12P38', 'ENSG00000213250_RBMS2P1', 'ENSG00000213272_RPL7AP9', 'ENSG00000213303_AC008481.1', 'ENSG00000213402_PTPRCAP', 'ENSG00000213471_TTLL13P', 'ENSG00000213588_ZBTB9', 'ENSG00000213609_RPL7AP50', 'ENSG00000213757_AC020898.1', 'ENSG00000213931_HBE1', 'ENSG00000213950_RPS10P2', 'ENSG00000213994_AL157395.1', 'ENSG00000214787_MS4A4E', 'ENSG00000214866_DCDC2C', 'ENSG00000214908_AL353678.1', 'ENSG00000214975_PPIAP29', 'ENSG00000215198_AL353795.1', 'ENSG00000215208_KRT18P60', 'ENSG00000215218_UBE2QL1', 'ENSG00000215297_AL354941.1', 'ENSG00000215464_AP000354.1', 'ENSG00000215483_LINC00598', 'ENSG00000215817_ZC3H11B', 'ENSG00000215861_AC245297.1', 'ENSG00000215910_C1orf167', 'ENSG00000216475_AL024474.1', 'ENSG00000217195_AL513475.1', 'ENSG00000217414_DDX18P3', 'ENSG00000217512_AL356776.1', 'ENSG00000218351_RPS3AP23', 'ENSG00000218418_AL591135.1', 'ENSG00000218749_AL033519.1', 'ENSG00000218766_AL450338.1', 'ENSG00000218792_HSPD1P16', 'ENSG00000219249_AMZ2P2', 'ENSG00000219395_HSPA8P15', 'ENSG00000219410_AC125494.1', 'ENSG00000219932_RPL12P8', 'ENSG00000220091_LAP3P1', 'ENSG00000220237_RPS24P12', 'ENSG00000220494_YAP1P1', 'ENSG00000221102_SNORA11B', 'ENSG00000221887_HMSD', 'ENSG00000222276_RNU2-33P', 'ENSG00000222370_SNORA36B', 'ENSG00000222421_RF00019', 'ENSG00000222431_RNU6-141P', 'ENSG00000223342_AL158817.1', 'ENSG00000223379_AL391987.3', 'ENSG00000223403_MEG9', 'ENSG00000223519_KIF28P', 'ENSG00000223576_AL355001.1', 'ENSG00000223668_EEF1A1P24', 'ENSG00000223741_PSMD4P1', 'ENSG00000223779_AC239800.1', 'ENSG00000223783_LINC01983', 'ENSG00000223784_LINP1', 'ENSG00000223855_HRAT92', 'ENSG00000223884_AC068481.1', 'ENSG00000223899_SEC13P1', 'ENSG00000224067_AL354877.1', 'ENSG00000224072_AL139811.1', 'ENSG00000224081_SLC44A3-AS1', 'ENSG00000224099_AC104823.1', 'ENSG00000224116_INHBA-AS1', 'ENSG00000224137_LINC01857', 'ENSG00000224155_AC073136.2', 'ENSG00000224321_RPL12P14', 'ENSG00000224402_OR6D1P', 'ENSG00000224479_AC104162.1', 'ENSG00000224599_BMS1P12', 'ENSG00000224689_ZNF812P', 'ENSG00000224848_AL589843.1', 'ENSG00000224908_TIMM8BP2', 'ENSG00000224957_LINC01266', 'ENSG00000224959_AC017002.1', 'ENSG00000224988_AL158207.1', 'ENSG00000224993_RPL29P12', 'ENSG00000225096_AL445250.1', 'ENSG00000225101_OR52K3P', 'ENSG00000225107_AC092484.1', 'ENSG00000225187_AC073283.1', 'ENSG00000225313_AL513327.1', 'ENSG00000225345_SNX18P3', 'ENSG00000225393_BX571846.1', 'ENSG00000225422_RBMS1P1', 'ENSG00000225423_TNPO1P1', 'ENSG00000225531_AL807761.2', 'ENSG00000225554_AL359764.1', 'ENSG00000225650_EIF2S2P5', 'ENSG00000225674_IPO7P2', 'ENSG00000225807_AC069281.1', 'ENSG00000226010_AL355852.1', 'ENSG00000226084_AC113935.1', 'ENSG00000226251_AL451060.1', 'ENSG00000226383_LINC01876', 'ENSG00000226491_FTOP1', 'ENSG00000226501_USF1P1', 'ENSG00000226545_AL357552.1', 'ENSG00000226564_FTH1P20', 'ENSG00000226617_RPL21P110', 'ENSG00000226647_AL365356.1', 'ENSG00000226800_CACTIN-AS1', 'ENSG00000226913_BSN-DT', 'ENSG00000226948_RPS4XP2', 'ENSG00000226970_AL450063.1', 'ENSG00000227006_AL136988.2', 'ENSG00000227051_C14orf132', 'ENSG00000227072_AL353706.1', 'ENSG00000227110_LMCD1-AS1', 'ENSG00000227192_AL023581.2', 'ENSG00000227198_C6orf47-AS1', 'ENSG00000227207_RPL31P12', 'ENSG00000227477_STK4-AS1', 'ENSG00000227541_SFR1P1', 'ENSG00000227590_ATP5MC1P5', 'ENSG00000227649_MTND6P32', 'ENSG00000227682_ATP5F1AP2', 'ENSG00000227740_AL513329.1', 'ENSG00000227742_CALR4P', 'ENSG00000228097_MTATP6P11', 'ENSG00000228140_AL031283.1', 'ENSG00000228175_GEMIN8P4', 'ENSG00000228212_OFD1P17', 'ENSG00000228232_GAPDHP1', 'ENSG00000228317_AL158070.1', 'ENSG00000228413_AC024937.1', 'ENSG00000228430_AL162726.3', 'ENSG00000228501_RPL15P18', 'ENSG00000228550_AC073583.1', 'ENSG00000228655_AC096558.1', 'ENSG00000228727_SAPCD1', 'ENSG00000228826_AL592494.1', 'ENSG00000228839_PIK3IP1-AS1', 'ENSG00000228863_AL121985.1', 'ENSG00000229066_AC093459.1', 'ENSG00000229150_CRYGEP', 'ENSG00000229154_KCNQ5-AS1', 'ENSG00000229163_NAP1L1P2', 'ENSG00000229236_TTTY10', 'ENSG00000229274_AL662860.1', 'ENSG00000229308_AC010737.1', 'ENSG00000229326_AC069154.1', 'ENSG00000229372_SZT2-AS1', 'ENSG00000229444_AL451062.1', 'ENSG00000229567_AL139421.1', 'ENSG00000229703_CR589904.1', 'ENSG00000229742_AC092809.1', 'ENSG00000229758_DYNLT3P2', 'ENSG00000229839_AC018462.1', 'ENSG00000229847_EMX2OS', 'ENSG00000229853_AL034418.1', 'ENSG00000229918_DOCK9-AS1', 'ENSG00000229953_AL590666.2', 'ENSG00000229992_HMGB3P9', 'ENSG00000230063_AL360091.2', 'ENSG00000230064_AL772161.1', 'ENSG00000230138_AC119428.2', 'ENSG00000230149_AL021707.3', 'ENSG00000230289_AL358781.2', 'ENSG00000230295_GTF2IP23', 'ENSG00000230479_AP000695.1', 'ENSG00000230508_RPL19P21', 'ENSG00000230519_HMGB1P49', 'ENSG00000230534_AL392046.1', 'ENSG00000230563_AL121757.1', 'ENSG00000230721_AL049597.1', 'ENSG00000230772_VN1R108P', 'ENSG00000230777_RPS29P5', 'ENSG00000230799_AC007279.1', 'ENSG00000230813_AL356583.3', 'ENSG00000230815_AL807757.1', 'ENSG00000230872_MFSD13B', 'ENSG00000230910_AL391807.1', 'ENSG00000230912_AL021707.4', 'ENSG00000230968_AC084149.2', 'ENSG00000230993_RPL12P15', 'ENSG00000231265_TRERNA1', 'ENSG00000231307_RPS3P2', 'ENSG00000231407_AL354732.1', 'ENSG00000231449_AC097359.1', 'ENSG00000231507_LINC01353', 'ENSG00000231531_HINT1P1', 'ENSG00000231548_OR55B1P', 'ENSG00000231731_AC010976.1', 'ENSG00000231742_LINC01273', 'ENSG00000231788_RPL31P50', 'ENSG00000231830_AC245140.1', 'ENSG00000231927_AC093734.1', 'ENSG00000231993_EP300-AS1', 'ENSG00000232027_AL671986.1', 'ENSG00000232028_AC007391.1', 'ENSG00000232065_LINC01063', 'ENSG00000232133_IMPDH1P10', 'ENSG00000232139_LINC00867', 'ENSG00000232273_FTH1P1', 'ENSG00000232333_RPS27AP2', 'ENSG00000232466_AL356133.1', 'ENSG00000232500_AP005273.1', 'ENSG00000232530_LIF-AS1', 'ENSG00000232568_RPL23AP35', 'ENSG00000232578_AC093311.1', 'ENSG00000232606_LINC01412', 'ENSG00000232654_FAM136BP', 'ENSG00000232656_IDI2-AS1', 'ENSG00000232719_AC007272.1', 'ENSG00000232803_SLCO4A1-AS1', 'ENSG00000232987_LINC01219', 'ENSG00000233025_CRYZP1', 'ENSG00000233093_LINC00892', 'ENSG00000233099_AC095030.1', 'ENSG00000233401_PRKAR1AP1', 'ENSG00000233427_AL009181.1', 'ENSG00000233540_DNM3-IT1', 'ENSG00000233674_AL451062.2', 'ENSG00000233825_AL391839.2', 'ENSG00000233862_AC016907.2', 'ENSG00000233994_GDI2P2', 'ENSG00000234026_AL157834.2', 'ENSG00000234106_SRP14P2', 'ENSG00000234145_NAP1L4P3', 'ENSG00000234174_AC016683.1', 'ENSG00000234271_Z98752.2', 'ENSG00000234425_AL138930.1', 'ENSG00000234488_AC096664.2', 'ENSG00000234630_AC245060.2', 'ENSG00000234645_YWHAEP5', 'ENSG00000234718_AC007161.1', 'ENSG00000234810_AL603840.1', 'ENSG00000235045_RPL7P8', 'ENSG00000235072_AC012074.1', 'ENSG00000235214_FAM83C-AS1', 'ENSG00000235288_AC099329.1', 'ENSG00000235376_RPEL1', 'ENSG00000235429_AC083875.1', 'ENSG00000235472_EIF4A1P7', 'ENSG00000235478_LINC01664', 'ENSG00000235531_MSC-AS1', 'ENSG00000235640_AC092646.2', 'ENSG00000235677_NPM1P26', 'ENSG00000235683_AC018442.1', 'ENSG00000235701_PCBP2P1', 'ENSG00000235740_PHACTR2-AS1', 'ENSG00000235774_AC023347.1', 'ENSG00000235802_HCFC1-AS1', 'ENSG00000235917_MTCO2P11', 'ENSG00000235958_UBOX5-AS1', 'ENSG00000236032_OR5H14', 'ENSG00000236180_AL445669.2', 'ENSG00000236254_MTND4P14', 'ENSG00000236283_AC019197.1', 'ENSG00000236290_EEF1GP7', 'ENSG00000236317_AC104333.2', 'ENSG00000236364_AL358115.1', 'ENSG00000236457_AC090617.1', 'ENSG00000236564_YWHAQP5', 'ENSG00000236671_PRKG1-AS1', 'ENSG00000236680_AL356000.1', 'ENSG00000236682_AC068282.1', 'ENSG00000236711_SMAD9-IT1', 'ENSG00000236806_RPL7AP15', 'ENSG00000236869_ZKSCAN7-AS1', 'ENSG00000236886_AC007563.2', 'ENSG00000236915_AL356270.1', 'ENSG00000236936_AL031005.1', 'ENSG00000237057_LINC02087', 'ENSG00000237101_AC092809.4', 'ENSG00000237276_ANO7L1', 'ENSG00000237317_AL022400.1', 'ENSG00000237387_AL022329.2', 'ENSG00000237618_BTBD7P2', 'ENSG00000237685_AL139039.3', 'ENSG00000237757_EEF1A1P30', 'ENSG00000237766_GGTA2P', 'ENSG00000237798_AC010894.4', 'ENSG00000238015_AC104837.2', 'ENSG00000238133_MAP3K20-AS1', 'ENSG00000238259_AC067940.1', 'ENSG00000238324_RN7SKP198', 'ENSG00000238358_AC004969.1', 'ENSG00000239219_AC008040.1', 'ENSG00000239316_RN7SL11P', 'ENSG00000239474_KLHL41', 'ENSG00000239527_RPS23P7', 'ENSG00000239642_MEIKIN', 'ENSG00000239650_GUSBP4', 'ENSG00000239686_AL158801.1', 'ENSG00000239701_AC006512.1', 'ENSG00000239705_AL354710.2', 'ENSG00000239797_RPL21P39', 'ENSG00000239830_RPS4XP22', 'ENSG00000239930_AP001625.3', 'ENSG00000240086_AC092969.1', 'ENSG00000240087_RPSAP12', 'ENSG00000240183_RN7SL297P', 'ENSG00000240219_AL512306.2', 'ENSG00000240498_CDKN2B-AS1', 'ENSG00000240809_AC026877.1', 'ENSG00000240993_RN7SL459P', 'ENSG00000241111_PRICKLE2-AS1', 'ENSG00000241135_LINC00881', 'ENSG00000241319_SETP6', 'ENSG00000241570_PAQR9-AS1', 'ENSG00000241631_RN7SL316P', 'ENSG00000241932_AC092324.1', 'ENSG00000241933_DENND6A-DT', 'ENSG00000242060_RPS3AP49', 'ENSG00000242107_LINC01100', 'ENSG00000242175_RN7SL127P', 'ENSG00000242431_AC107398.1', 'ENSG00000242551_POU5F1P6', 'ENSG00000242571_RPL21P11', 'ENSG00000242641_LINC00971', 'ENSG00000242747_AC090515.1', 'ENSG00000242992_FTH1P4', 'ENSG00000243055_GK-AS1', 'ENSG00000243498_UBA52P5', 'ENSG00000243592_RPL17P22', 'ENSG00000243709_LEFTY1', 'ENSG00000243830_AC092865.1', 'ENSG00000243836_WDR86-AS1', 'ENSG00000243961_PARAL1', 'ENSG00000244021_AC093591.1', 'ENSG00000244097_RPS4XP17', 'ENSG00000244151_AC010973.2', 'ENSG00000244183_PPIAP71', 'ENSG00000244242_IFITM10', 'ENSG00000244245_AC133134.1', 'ENSG00000244251_AC013356.1', 'ENSG00000244355_LY6G6D', 'ENSG00000244357_RN7SL145P', 'ENSG00000244476_ERVFRD-1', 'ENSG00000244482_LILRA6', 'ENSG00000244585_RPL12P33', 'ENSG00000244618_RN7SL334P', 'ENSG00000244703_CD46P1', 'ENSG00000245261_AL133375.1', 'ENSG00000245482_AC046130.1', 'ENSG00000246363_LINC02458', 'ENSG00000246863_AC012377.1', 'ENSG00000247199_AC091948.1', 'ENSG00000248121_SMURF2P1', 'ENSG00000248155_CR545473.1', 'ENSG00000248223_AC026785.2', 'ENSG00000248485_PCP4L1', 'ENSG00000248690_HAS2-AS1', 'ENSG00000248884_AC010280.2', 'ENSG00000248936_AC027607.1', 'ENSG00000249140_PRDX2P3', 'ENSG00000249363_AC011411.1', 'ENSG00000249381_LINC00500', 'ENSG00000249456_AL731577.2', 'ENSG00000249492_AC114956.3', 'ENSG00000249574_AC226118.1', 'ENSG00000249614_LINC02503', 'ENSG00000249691_AC026117.1', 'ENSG00000249695_AC026369.1', 'ENSG00000249803_AC112178.1', 'ENSG00000249825_CTD-2201I18.1', 'ENSG00000249848_AC112673.1', 'ENSG00000249850_KRT18P31', 'ENSG00000249884_RNF103-CHMP3', 'ENSG00000249978_TRGV7', 'ENSG00000250130_AC090519.1', 'ENSG00000250148_KRT8P31', 'ENSG00000250332_AC010460.3', 'ENSG00000250334_LINC00989', 'ENSG00000250539_KRT8P33', 'ENSG00000250548_LINC01303', 'ENSG00000250608_AC010210.1', 'ENSG00000250635_CXXC5-AS1', 'ENSG00000250645_AC010442.2', 'ENSG00000250733_C8orf17', 'ENSG00000250853_RNF138P1', 'ENSG00000250902_SMAD1-AS1', 'ENSG00000250950_AC093752.2', 'ENSG00000250982_GAPDHP35', 'ENSG00000251129_LINC02506', 'ENSG00000251152_AC025539.1', 'ENSG00000251250_AC091951.3', 'ENSG00000251288_AC018797.3', 'ENSG00000251468_AC135352.1', 'ENSG00000251537_AC005324.3', 'ENSG00000251538_LINC02201', 'ENSG00000251584_AC096751.2', 'ENSG00000251676_SNHG27', 'ENSG00000251916_RNU1-61P', 'ENSG00000252759_RF00019', 'ENSG00000253256_AC134043.1', 'ENSG00000253305_PCDHGB6', 'ENSG00000253394_LINC00534', 'ENSG00000253490_LINC02099', 'ENSG00000253537_PCDHGA7', 'ENSG00000253629_AP000426.1', 'ENSG00000253651_SOD1P3', 'ENSG00000253730_AC015909.2', 'ENSG00000253734_LINC01289', 'ENSG00000253767_PCDHGA8', 'ENSG00000253853_AC246817.1', 'ENSG00000253873_PCDHGA11', 'ENSG00000254028_AC083843.1', 'ENSG00000254048_AC105150.1', 'ENSG00000254054_AC087273.2', 'ENSG00000254122_PCDHGB7', 'ENSG00000254248_AC068189.1', 'ENSG00000254680_AC079329.1', 'ENSG00000254708_AL139174.1', 'ENSG00000254780_AC023232.1', 'ENSG00000254810_AP001189.3', 'ENSG00000254812_AC067930.3', 'ENSG00000254842_LINC02551', 'ENSG00000254846_AL355075.1', 'ENSG00000254862_AC100771.2', 'ENSG00000254897_AP003035.1', 'ENSG00000255002_LINC02324', 'ENSG00000255074_AC018523.1', 'ENSG00000255102_AP005436.1', 'ENSG00000255156_RNY1P9', 'ENSG00000255158_AC131934.1', 'ENSG00000255222_SETP17', 'ENSG00000255256_AL136146.2', 'ENSG00000255367_AC127526.2', 'ENSG00000255418_AC090092.1', 'ENSG00000255443_CD44-AS1', 'ENSG00000255446_AP003064.2', 'ENSG00000255479_AP001189.6', 'ENSG00000255487_AC087362.2', 'ENSG00000255867_DENND5B-AS1', 'ENSG00000255871_AC007529.1', 'ENSG00000256029_SNHG28', 'ENSG00000256571_AC079866.2', 'ENSG00000256588_AC027544.2', 'ENSG00000256712_AC134349.1', 'ENSG00000256746_AC018410.1', 'ENSG00000256813_AP000777.3', 'ENSG00000256967_AC018653.3', 'ENSG00000256968_SNRPEP2', 'ENSG00000257074_RPL29P33', 'ENSG00000257120_AL356756.1', 'ENSG00000257146_AC079905.2', 'ENSG00000257195_HNRNPA1P50', 'ENSG00000257327_AC012555.1', 'ENSG00000257345_LINC02413', 'ENSG00000257379_AC023509.1', 'ENSG00000257386_AC025257.1', 'ENSG00000257431_AC089998.1', 'ENSG00000257715_AC007298.1', 'ENSG00000257838_OTOAP1', 'ENSG00000257987_TEX49', 'ENSG00000258084_AC128707.1', 'ENSG00000258090_AC093014.1', 'ENSG00000258177_AC008149.1', 'ENSG00000258357_AC023161.2', 'ENSG00000258410_AC087386.1', 'ENSG00000258498_DIO3OS', 'ENSG00000258504_AL157871.1', 'ENSG00000258512_LINC00239', 'ENSG00000258867_LINC01146', 'ENSG00000258886_HIGD1AP17', 'ENSG00000259032_ENSAP2', 'ENSG00000259100_AL157791.1', 'ENSG00000259294_AC005096.1', 'ENSG00000259327_AC023906.3', 'ENSG00000259345_AC013652.1', 'ENSG00000259377_AC026770.1', 'ENSG00000259380_AC087473.1', 'ENSG00000259442_AC105339.3', 'ENSG00000259461_ANP32BP3', 'ENSG00000259556_AC090971.3', 'ENSG00000259569_AC013489.2', 'ENSG00000259617_AC020661.3', 'ENSG00000259684_AC084756.1', 'ENSG00000259719_LINC02284', 'ENSG00000259954_IL21R-AS1', 'ENSG00000259986_AC103876.1', 'ENSG00000260135_MMP2-AS1', 'ENSG00000260206_AC105020.2', 'ENSG00000260235_AC105020.3', 'ENSG00000260269_AC105036.3', 'ENSG00000260394_Z92544.1', 'ENSG00000260425_AL031709.1', 'ENSG00000260447_AC009065.3', 'ENSG00000260615_RPL23AP97', 'ENSG00000260871_AC093510.2', 'ENSG00000260877_AP005233.2', 'ENSG00000260979_AC022167.3', 'ENSG00000261051_AC107021.2', 'ENSG00000261113_AC009034.1', 'ENSG00000261168_AL592424.1', 'ENSG00000261253_AC137932.2', 'ENSG00000261269_AC093278.2', 'ENSG00000261552_AC109460.4', 'ENSG00000261572_AC097639.1', 'ENSG00000261602_AC092115.2', 'ENSG00000261630_AC007496.2', 'ENSG00000261644_AC007728.2', 'ENSG00000261734_AC116096.1', 'ENSG00000261773_AC244090.2', 'ENSG00000261837_AC046158.2', 'ENSG00000261838_AC092718.6', 'ENSG00000261888_AC144831.1', 'ENSG00000262061_AC129507.1', 'ENSG00000262097_LINC02185', 'ENSG00000262372_CR936218.1', 'ENSG00000262406_MMP12', 'ENSG00000262580_AC087741.1', 'ENSG00000262772_LINC01977', 'ENSG00000262833_AC016245.1', 'ENSG00000263006_ROCK1P1', 'ENSG00000263011_AC108134.4', 'ENSG00000263155_MYZAP', 'ENSG00000263393_AC011825.2', 'ENSG00000263426_RN7SL471P', 'ENSG00000263503_MAPK8IP1P2', 'ENSG00000263595_RN7SL823P', 'ENSG00000263878_DLGAP1-AS4', 'ENSG00000263940_RN7SL275P', 'ENSG00000264019_AC018521.2', 'ENSG00000264031_ABHD15-AS1', 'ENSG00000264044_AC005726.2', 'ENSG00000264070_DND1P1', 'ENSG00000264188_AC106037.1', 'ENSG00000264269_AC016866.1', 'ENSG00000264339_AP001020.1', 'ENSG00000264434_AC110603.1', 'ENSG00000264714_KIAA0895LP1', 'ENSG00000265010_AC087301.1', 'ENSG00000265073_AC010761.2', 'ENSG00000265107_GJA5', 'ENSG00000265179_AP000894.2', 'ENSG00000265218_AC103810.2', 'ENSG00000265334_AC130324.2', 'ENSG00000265439_RN7SL811P', 'ENSG00000265531_FCGR1CP', 'ENSG00000265845_AC024267.4', 'ENSG00000265907_AP000919.2', 'ENSG00000265942_RN7SL577P', 'ENSG00000266256_LINC00683', 'ENSG00000266456_AP001178.3', 'ENSG00000266733_TBC1D29', 'ENSG00000266835_GAPLINC', 'ENSG00000266844_AC093330.1', 'ENSG00000266903_AC243964.2', 'ENSG00000266944_AC005262.1', 'ENSG00000266946_MRPL37P1', 'ENSG00000266947_AC022916.1', 'ENSG00000267034_AC010980.2', 'ENSG00000267044_AC005757.1', 'ENSG00000267147_LINC01842', 'ENSG00000267175_AC105094.2', 'ENSG00000267191_AC006213.3', 'ENSG00000267275_AC020911.2', 'ENSG00000267288_AC138150.2', 'ENSG00000267313_KC6', 'ENSG00000267316_AC090409.2', 'ENSG00000267323_SLC25A1P5', 'ENSG00000267345_AC010632.1', 'ENSG00000267387_AC020931.1', 'ENSG00000267395_DM1-AS', 'ENSG00000267429_AC006116.6', 'ENSG00000267452_LINC02073', 'ENSG00000267491_AC100788.1', 'ENSG00000267529_AP005131.4', 'ENSG00000267554_AC015911.8', 'ENSG00000267601_AC022966.1', 'ENSG00000267638_AC023855.1', 'ENSG00000267665_AC021683.3', 'ENSG00000267681_AC135721.1', 'ENSG00000267703_AC020917.2', 'ENSG00000267731_AC005332.2', 'ENSG00000267733_AP005264.5', 'ENSG00000267750_RUNDC3A-AS1', 'ENSG00000267890_AC010624.2', 'ENSG00000267898_AC026803.2', 'ENSG00000267927_AC010320.1', 'ENSG00000268070_AC006539.2', 'ENSG00000268355_AC243960.3', 'ENSG00000268416_AC010329.1', 'ENSG00000268520_AC008750.5', 'ENSG00000268636_AC011495.2', 'ENSG00000268696_ZNF723', 'ENSG00000268777_AC020914.1', 'ENSG00000268849_SIGLEC22P', 'ENSG00000268903_AL627309.6', 'ENSG00000268983_AC005253.2', 'ENSG00000269019_HOMER3-AS1', 'ENSG00000269067_ZNF728', 'ENSG00000269103_RF00017', 'ENSG00000269274_AC078899.4', 'ENSG00000269288_AC092070.3', 'ENSG00000269352_PTOV1-AS2', 'ENSG00000269400_AC008734.2', 'ENSG00000269506_AC110792.2', 'ENSG00000269653_AC011479.3', 'ENSG00000269881_AC004754.1', 'ENSG00000269926_DDIT4-AS1', 'ENSG00000270048_AC068790.4', 'ENSG00000270050_AL035427.1', 'ENSG00000270503_YTHDF2P1', 'ENSG00000270706_PRMT1P1', 'ENSG00000270765_GAS2L2', 'ENSG00000270882_HIST2H4A', 'ENSG00000270906_MTND4P35', 'ENSG00000271013_LRRC37A9P', 'ENSG00000271129_AC009027.1', 'ENSG00000271259_AC010201.1', 'ENSG00000271524_BNIP3P17', 'ENSG00000271543_AC021443.1', 'ENSG00000271743_AF287957.1', 'ENSG00000271792_AC008667.4', 'ENSG00000271868_AC114810.1', 'ENSG00000271973_AC141002.1', 'ENSG00000271984_AL008726.1', 'ENSG00000271996_AC019080.4', 'ENSG00000272070_AC005618.1', 'ENSG00000272138_LINC01607', 'ENSG00000272150_NBPF25P', 'ENSG00000272265_AC034236.3', 'ENSG00000272279_AL512329.2', 'ENSG00000272473_AC006273.1', 'ENSG00000272510_AL121992.3', 'ENSG00000272582_AL031587.3', 'ENSG00000272695_GAS6-DT', 'ENSG00000272732_AC004982.1', 'ENSG00000272770_AC005696.2', 'ENSG00000272788_AP000864.1', 'ENSG00000272824_AC245100.7', 'ENSG00000272825_AL844908.1', 'ENSG00000272848_AL135910.1', 'ENSG00000272916_AC022400.6', 'ENSG00000273133_AC116651.1', 'ENSG00000273177_AC092954.2', 'ENSG00000273212_AC000068.2', 'ENSG00000273218_AC005776.2', 'ENSG00000273245_AC092653.1', 'ENSG00000273274_ZBTB8B', 'ENSG00000273312_AL121749.1', 'ENSG00000273325_AL008723.3', 'ENSG00000273369_AC096586.2', 'ENSG00000273474_AL157392.4', 'ENSG00000273599_AL731571.1', 'ENSG00000273724_AC106782.5', 'ENSG00000273870_AL138721.1', 'ENSG00000273920_AC103858.2', 'ENSG00000274023_AL360169.2', 'ENSG00000274029_AC069209.1', 'ENSG00000274114_ALOX15P1', 'ENSG00000274124_AC074029.3', 'ENSG00000274139_AC090164.2', 'ENSG00000274281_AC022929.2', 'ENSG00000274308_AC244093.1', 'ENSG00000274373_AC148476.1', 'ENSG00000274386_TMEM269', 'ENSG00000274403_AC090510.2', 'ENSG00000274570_SPDYE10P', 'ENSG00000274670_AC137590.2', 'ENSG00000274723_AC079906.1', 'ENSG00000274742_RF00017', 'ENSG00000274798_AC025166.1', 'ENSG00000274911_AL627230.2', 'ENSG00000275106_AC025594.2', 'ENSG00000275197_AC092794.2', 'ENSG00000275302_CCL4', 'ENSG00000275348_AC096861.1', 'ENSG00000275367_AC092111.1', 'ENSG00000275489_C17orf98', 'ENSG00000275527_AC100835.2', 'ENSG00000275995_AC109809.1', 'ENSG00000276070_CCL4L2', 'ENSG00000276255_AL136379.1', 'ENSG00000276282_AC022960.2', 'ENSG00000276547_PCDHGB5', 'ENSG00000276704_AL442067.2', 'ENSG00000276952_AL121772.3', 'ENSG00000276984_AL023881.1', 'ENSG00000276997_AL513314.2', 'ENSG00000277117_FP565260.3', 'ENSG00000277152_AC110048.2', 'ENSG00000277186_AC131212.1', 'ENSG00000277229_AC084781.1', 'ENSG00000277496_AL357033.4', 'ENSG00000277504_AC010536.3', 'ENSG00000277531_PNMA8C', 'ENSG00000278041_AL133325.3', 'ENSG00000278344_AC063943.1', 'ENSG00000278467_AC138393.3', 'ENSG00000278513_AC091046.2', 'ENSG00000278621_AC037198.2', 'ENSG00000278713_AC120114.2', 'ENSG00000278716_AC133540.1', 'ENSG00000278746_RN7SL660P', 'ENSG00000278774_RF00004', 'ENSG00000279091_AC026523.2', 'ENSG00000279130_AC091925.1', 'ENSG00000279141_LINC01451', 'ENSG00000279161_AC093503.3', 'ENSG00000279187_AC027601.5', 'ENSG00000279263_OR2L8', 'ENSG00000279315_AL158212.4', 'ENSG00000279319_AC105074.1', 'ENSG00000279332_AC090772.4', 'ENSG00000279339_AC100788.2', 'ENSG00000279365_AP000695.3', 'ENSG00000279378_AC009159.4', 'ENSG00000279384_AC080188.2', 'ENSG00000279404_AC008739.5', 'ENSG00000279417_AC019322.4', 'ENSG00000279444_AC135584.1', 'ENSG00000279486_OR2AG1', 'ENSG00000279530_AC092881.1', 'ENSG00000279590_AC005786.4', 'ENSG00000279619_AC020907.5', 'ENSG00000279633_AL137918.1', 'ENSG00000279636_LINC00216', 'ENSG00000279672_AP006621.5', 'ENSG00000279690_AP000280.1', 'ENSG00000279727_LINC02033', 'ENSG00000279861_AC073548.1', 'ENSG00000279913_AP001962.1', 'ENSG00000279970_AC023024.2', 'ENSG00000280055_TMEM75', 'ENSG00000280057_AL022069.2', 'ENSG00000280135_AL096816.1', 'ENSG00000280310_AC092437.1', 'ENSG00000280422_AC115284.2', 'ENSG00000280432_AP000962.2', 'ENSG00000280693_SH3PXD2A-AS1', 'ENSG00000281490_CICP14', 'ENSG00000281530_AC004461.2', 'ENSG00000281571_AC241585.2', 'ENSG00000282772_AL358790.1', 'ENSG00000282989_AP001206.1', 'ENSG00000282996_AC022021.1', 'ENSG00000283023_FRG1GP', 'ENSG00000283031_AC009242.1', 'ENSG00000283097_AL159152.1', 'ENSG00000283141_AL157832.3', 'ENSG00000283209_AC106858.1', 'ENSG00000283538_AC005972.3', 'ENSG00000284240_AC099062.1', 'ENSG00000284512_AC092718.8', 'ENSG00000284657_AL031432.5', 'ENSG00000284664_AL161756.3', 'ENSG00000284931_AC104389.5', 'ENSG00000285016_AC017002.6', 'ENSG00000285117_AC068724.4', 'ENSG00000285162_AC004593.3', 'ENSG00000285210_AL136382.1', 'ENSG00000285215_AC241377.4', 'ENSG00000285292_AC021097.2', 'ENSG00000285498_AC104389.6', 'ENSG00000285534_AL163541.1', 'ENSG00000285577_AC019127.1', 'ENSG00000285611_AC007132.1', 'ENSG00000285629_AL031847.2', 'ENSG00000285641_AL358472.6', 'ENSG00000285649_AL357079.2', 'ENSG00000285650_AL157827.2', 'ENSG00000285662_AL731733.1', 'ENSG00000285672_AL160396.2', 'ENSG00000285763_AL358777.1', 'ENSG00000285865_AC010285.3', 'ENSG00000285879_AC018628.2']
print('Constant cols:', len(constant_cols))

# important_cols = []
# for y_col in Y.columns:
#     important_cols += [x_col for x_col in X.columns if y_col in x_col]
# print(important_cols)
important_cols = ['ENSG00000114013_CD86', 'ENSG00000120217_CD274', 'ENSG00000196776_CD47', 'ENSG00000117091_CD48', 'ENSG00000101017_CD40', 'ENSG00000102245_CD40LG', 'ENSG00000169442_CD52', 'ENSG00000117528_ABCD3', 'ENSG00000168014_C2CD3', 'ENSG00000167851_CD300A', 'ENSG00000167850_CD300C', 'ENSG00000186407_CD300E', 'ENSG00000178789_CD300LB', 'ENSG00000186074_CD300LF', 'ENSG00000241399_CD302', 'ENSG00000167775_CD320', 'ENSG00000105383_CD33', 'ENSG00000174059_CD34', 'ENSG00000135218_CD36', 'ENSG00000104894_CD37', 'ENSG00000004468_CD38', 'ENSG00000167286_CD3D', 'ENSG00000198851_CD3E', 'ENSG00000117877_CD3EAP', 'ENSG00000074696_HACD3', 'ENSG00000015676_NUDCD3', 'ENSG00000161714_PLCD3', 'ENSG00000132300_PTCD3', 'ENSG00000082014_SMARCD3', 'ENSG00000121594_CD80', 'ENSG00000110651_CD81', 'ENSG00000238184_CD81-AS1', 'ENSG00000085117_CD82', 'ENSG00000112149_CD83', 'ENSG00000066294_CD84', 'ENSG00000114013_CD86', 'ENSG00000172116_CD8B', 'ENSG00000254126_CD8B2', 'ENSG00000177455_CD19', 'ENSG00000105383_CD33', 'ENSG00000173762_CD7', 'ENSG00000125726_CD70', 'ENSG00000137101_CD72', 'ENSG00000019582_CD74', 'ENSG00000105369_CD79A', 'ENSG00000007312_CD79B', 'ENSG00000090470_PDCD7', 'ENSG00000119688_ABCD4', 'ENSG00000010610_CD4', 'ENSG00000101017_CD40', 'ENSG00000102245_CD40LG', 'ENSG00000026508_CD44', 'ENSG00000117335_CD46', 'ENSG00000196776_CD47', 'ENSG00000117091_CD48', 'ENSG00000188921_HACD4', 'ENSG00000150593_PDCD4', 'ENSG00000203497_PDCD4-AS1', 'ENSG00000115556_PLCD4', 'ENSG00000026508_CD44', 'ENSG00000170458_CD14', 'ENSG00000117281_CD160', 'ENSG00000177575_CD163', 'ENSG00000135535_CD164', 'ENSG00000091972_CD200', 'ENSG00000163606_CD200R1', 'ENSG00000206531_CD200R1L', 'ENSG00000182685_BRICD5', 'ENSG00000111731_C2CD5', 'ENSG00000169442_CD52', 'ENSG00000143119_CD53', 'ENSG00000196352_CD55', 'ENSG00000116815_CD58', 'ENSG00000085063_CD59', 'ENSG00000105185_PDCD5', 'ENSG00000255909_PDCD5P1', 'ENSG00000145284_SCD5', 'ENSG00000167775_CD320', 'ENSG00000110848_CD69', 'ENSG00000139187_KLRG1', 'ENSG00000139193_CD27', 'ENSG00000215039_CD27-AS1', 'ENSG00000120217_CD274', 'ENSG00000103855_CD276', 'ENSG00000204287_HLA-DRA', 'ENSG00000196126_HLA-DRB1', 'ENSG00000198502_HLA-DRB5', 'ENSG00000229391_HLA-DRB6', 'ENSG00000116815_CD58', 'ENSG00000168329_CX3CR1', 'ENSG00000272398_CD24', 'ENSG00000122223_CD244', 'ENSG00000198821_CD247', 'ENSG00000122223_CD244', 'ENSG00000177575_CD163', 'ENSG00000112149_CD83', 'ENSG00000185963_BICD2', 'ENSG00000157617_C2CD2', 'ENSG00000172375_C2CD2L', 'ENSG00000116824_CD2', 'ENSG00000091972_CD200', 'ENSG00000163606_CD200R1', 'ENSG00000206531_CD200R1L', 'ENSG00000012124_CD22', 'ENSG00000150637_CD226', 'ENSG00000272398_CD24', 'ENSG00000122223_CD244', 'ENSG00000198821_CD247', 'ENSG00000139193_CD27', 'ENSG00000215039_CD27-AS1', 'ENSG00000120217_CD274', 'ENSG00000103855_CD276', 'ENSG00000198087_CD2AP', 'ENSG00000169217_CD2BP2', 'ENSG00000144554_FANCD2', 'ENSG00000206527_HACD2', 'ENSG00000170584_NUDCD2', 'ENSG00000071994_PDCD2', 'ENSG00000126249_PDCD2L', 'ENSG00000049883_PTCD2', 'ENSG00000186193_SAPCD2', 'ENSG00000108604_SMARCD2', 'ENSG00000185561_TLCD2', 'ENSG00000075035_WSCD2', 'ENSG00000150637_CD226', 'ENSG00000110651_CD81', 'ENSG00000238184_CD81-AS1', 'ENSG00000134061_CD180', 'ENSG00000004468_CD38', 'ENSG00000012124_CD22', 'ENSG00000150637_CD226', 'ENSG00000135404_CD63', 'ENSG00000135218_CD36', 'ENSG00000137101_CD72', 'ENSG00000125810_CD93', 'ENSG00000010278_CD9', 'ENSG00000125810_CD93', 'ENSG00000153283_CD96', 'ENSG00000002586_CD99', 'ENSG00000102181_CD99L2', 'ENSG00000223773_CD99P1', 'ENSG00000204592_HLA-E', 'ENSG00000085117_CD82', 'ENSG00000134256_CD101']
print('Important cols:', len(important_cols))


# We read train and test datasets, keep the important columns and convert the rest to sparse matrices.

# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Read train and convert to sparse matrix\nX = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=constant_cols)\ncell_index = X.index\nmeta = metadata_df.reindex(cell_index)\nX0 = X[important_cols].values\nprint(f"Original X shape: {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")\ngc.collect()\nX = scipy.sparse.csr_matrix(X.values)\ngc.collect()\n\n# Read test and convert to sparse matrix\nXt = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=constant_cols)\ncell_index_test = Xt.index\nmeta_test = metadata_df.reindex(cell_index_test)\nX0t = Xt[important_cols].values\nprint(f"Original Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")\ngc.collect()\nXt = scipy.sparse.csr_matrix(Xt.values)')


# We apply the truncated SVD to train and test together. The truncated SVD is memory-efficient. We concatenate the SVD output (64 components) with the 144 important features and get the arrays `X` and `Xt`, which will be the input to the Keras model. 

# In[7]:


get_ipython().run_cell_magic('time', '', '\n# Apply the singular value decomposition\nboth = scipy.sparse.vstack([X, Xt])\nassert both.shape[0] == 119651\nprint(f"Shape of both before SVD: {both.shape}")\nsvd = TruncatedSVD(n_components=64, random_state=1) # 512 is possible\nboth = svd.fit_transform(both)\nprint(f"Shape of both after SVD:  {both.shape}")\n    \n# Hstack the svd output with the important features\nX = both[:70988]\nXt = both[70988:]\ndel both\nX = np.hstack([X, X0])\nXt = np.hstack([Xt, X0t])\nprint(f"Reduced X shape:  {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")\nprint(f"Reduced Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")')


# Finally, we read the target array `Y`:

# In[8]:


# Read Y
Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
y_columns = list(Y.columns)
Y = Y.values

# Normalize the targets row-wise: This doesn't change the correlations,
# and negative_correlation_loss depends on it
Y -= Y.mean(axis=1).reshape(-1, 1)
Y /= Y.std(axis=1).reshape(-1, 1)
    
print(f"Y shape: {str(Y.shape):14} {Y.size*4/1024/1024/1024:2.3f} GByte")


# # The model
# 
# Our model is a sequential network consisting of a few dense layers. The hyperparameters will be tuned with KerasTuner.
# 
# We use the `negative_correlation_loss` defined above as loss function.

# In[9]:


LR_START = 0.01
BATCH_SIZE = 256

def my_model(hp, n_inputs=X.shape[1]):
    """Sequential neural network
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    reg1 = hp.Float("reg1", min_value=1e-8, max_value=1e-4, sampling="log")
    reg2 = hp.Float("reg2", min_value=1e-10, max_value=1e-5, sampling="log")
    
    inputs = Input(shape=(n_inputs, ))
    x0 = Dense(hp.Choice('units1', [64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(inputs)
    x1 = Dense(hp.Choice('units2', [64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(x0)
    x2 = Dense(hp.Choice('units3', [32, 64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(x1)
    x3 = Dense(hp.Choice('units4', [32, 64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(x2)
    x = Concatenate()([x0, x1, x2, x3])
    x = Dense(Y.shape[1], kernel_regularizer=tf.keras.regularizers.l2(reg2),
              #activation=activation,
             )(x)
    regressor = Model(inputs, x)
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
                      metrics=[negative_correlation_loss],
                      loss=negative_correlation_loss
                     )
    
    return regressor

display(plot_model(my_model(keras_tuner.HyperParameters()), show_layer_names=False, show_shapes=True, dpi=72))


# # Tuning with KerasTuner
# 
# Now we let [KerasTuner](https://keras.io/keras_tuner/) optimize the hyperparameters. The tunable hyperparameters are:
# - the sizes of the hidden layers
# - the regularization factors
# 
# If you want to save time, you can either set `max_trials` to a lower value or skip tuning completely and set `best_hp.values` manually. If you don't want to see all the output of the tuner, you can set `verbose` to 0 in the call to `tuner.search()`.

# In[10]:


get_ipython().run_cell_magic('time', '', 'if TUNE:\n    tuner = keras_tuner.BayesianOptimization(\n        my_model,\n        overwrite=True,\n        objective=keras_tuner.Objective("val_negative_correlation_loss", direction="min"),\n        max_trials=100,\n        directory=\'/kaggle/temp\',\n        seed=1)\n    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, \n                           patience=4, verbose=0)\n    es = EarlyStopping(monitor="val_loss",\n                       patience=12, \n                       verbose=0,\n                       mode="min", \n                       restore_best_weights=True)\n    callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]\n    X_tr, X_va, y_tr, y_va = train_test_split(X, Y, test_size=0.2, random_state=10)\n    tuner.search(X_tr, y_tr,\n                 epochs=1000,\n                 validation_data=(X_va, y_va),\n                 batch_size=BATCH_SIZE,\n                 callbacks=callbacks, verbose=2)\n    del X_tr, X_va, y_tr, y_va, lr, es, callbacks')


# In[11]:


if TUNE:
    tuner.results_summary()
    
    # Table of the 10 best trials
    display(pd.DataFrame([hp.values for hp in tuner.get_best_hyperparameters(10)]))
    
    # Keep the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]


# In[12]:


# Hyperparameters can be set manually
if not TUNE:
    best_hp = keras_tuner.HyperParameters()
    best_hp.values = {'reg1': 8e-6,
                      'reg2': 2e-6,
                      'units1': 256,
                      'units2': 256,
                      'units3': 256,
                      'units4': 128}
    


# # Cross-validation
# 
# For cross-validation of the tuned model, we create three folds. In every fold, we train on the data of two donors and predict the third one. This scheme mimics the situation of the public leaderboard, where we train on three donors and predict the fourth one (see [EDA](https://www.kaggle.com/ambrosm/msci-eda-which-makes-sense)). 
# 
# The models are saved so that we can use them to compute the test predictions later.

# In[13]:


get_ipython().run_cell_magic('time', '', '# Cross-validation\nVERBOSE = 0 # set to 2 for more output, set to 0 for less output\nEPOCHS = 1000\nN_SPLITS = 3\n\nnp.random.seed(1)\ntf.random.set_seed(1)\n\nkf = GroupKFold(n_splits=N_SPLITS)\nscore_list = []\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(X, groups=meta.donor)):\n    start_time = datetime.datetime.now()\n    model = None\n    gc.collect()\n    X_tr = X[idx_tr]\n    y_tr = Y[idx_tr]\n    X_va = X[idx_va]\n    y_va = Y[idx_va]\n\n    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, \n                           patience=4, verbose=VERBOSE)\n    es = EarlyStopping(monitor="val_loss",\n                       patience=12, \n                       verbose=0,\n                       mode="min", \n                       restore_best_weights=True)\n    callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]\n\n    # Construct and compile the model\n    model = my_model(best_hp, X_tr.shape[1])\n\n    # Train the model\n    history = model.fit(X_tr, y_tr, \n                        validation_data=(X_va, y_va), \n                        epochs=EPOCHS,\n                        verbose=VERBOSE,\n                        batch_size=BATCH_SIZE,\n                        shuffle=True,\n                        callbacks=callbacks)\n    del X_tr, y_tr\n    if SUBMIT:\n        model.save(f"/kaggle/temp/model_{fold}")\n    history = history.history\n    callbacks, lr = None, None\n    \n    # We validate the model\n    y_va_pred = model.predict(X_va, batch_size=len(X_va))\n    corrscore = correlation_score(y_va, y_va_pred)\n\n    print(f"Fold {fold}: {es.stopped_epoch:3} epochs, corr =  {corrscore:.5f}")\n    del es, X_va#, y_va, y_va_pred\n    score_list.append(corrscore)\n\n# Show overall score\nprint(f"{Fore.GREEN}{Style.BRIGHT}Average  corr = {np.array(score_list).mean():.5f}{Style.RESET_ALL}")')


# Cross-validation shows us the average correlation between predictions and ground truth. The histogram additionally shows how the correlations of the cells are distributed. While most correlations are around 0.9, there exist a few predictions with negative correlations.

# In[14]:


corr_list = []
for i in range(len(y_va)):
    corr_list.append(np.corrcoef(y_va[i], y_va_pred[i])[1, 0])
plt.figure(figsize=(10, 4))
plt.hist(corr_list, bins=100, density=True, color='lightgreen')
plt.title('Distribution of correlations')
plt.xlabel('Correlation')
plt.ylabel('Density')
plt.show()


# # Prediction and submission
# 
# We ensemble the test predictions of all Keras models. 
# 
# It has been pointed out in several discussion posts that the first 7476 rows of test (day 2, donor 27678) are identical to the first 7476 rows of train (day 2, donor 32606):
# - [CITEseq data: same RNA expression matrices from different donors in day2?](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349867) (@gwentea)
# -[Data contamination between CITEseq train/test datasets?](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349833) (@aglaros)
# - [Leak in public test set](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349867) (@psilogram)
# 
# These rows belong to the public test set; the private leaderboard is not affected. We copy the 7476 rows from the training targets into the test predictions.
# 
# At the end we concatenate the CITEseq predictions with @jsmithperera's predictions of the [Multiome Quickstart w/ Sparse M + tSVD = 32](https://www.kaggle.com/code/jsmithperera/multiome-quickstart-w-sparse-m-tsvd-32) notebook to get a complete submission.
# 

# In[15]:


if SUBMIT:
    test_pred = np.zeros((len(Xt), 140), dtype=np.float32)
    for fold in range(N_SPLITS):
        print(f"Predicting with fold {fold}")
        model = load_model(f"/kaggle/temp/model_{fold}",
                           custom_objects={'negative_correlation_loss': negative_correlation_loss})
        test_pred += model.predict(Xt)
    
    # Copy the targets for the data leak
    test_pred[:7476] = Y[:7476]

    #with open("../input/msci-multiome-quickstart/partial_submission_multi.pickle", 'rb') as f: submission = pickle.load(f)
    submission = pd.read_csv('../input/multiome-quickstart-w-sparse-m-tsvd-32/submission.csv',
                             index_col='row_id', squeeze=True)
    submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
    assert not submission.isna().any()
    submission.to_csv('submission.csv')
    display(submission)


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # CITEseq Quickstart
# 
# This notebook shows how to implement a lightgbm model and prediction for the CITEseq part of the *Multimodal Single-Cell Integration* competition without running out of memory.
# 
# It does not show the EDA - see the separate notebook [MSCI EDA which makes sense ??????????????????????????????](https://www.kaggle.com/ambrosm/msci-eda-which-makes-sense).
# 
# The predictions of this notebook are merged with those of Fabien Crom's [Multiome notebook](https://www.kaggle.com/code/fabiencrom/msci-multiome-quickstart-w-sparse-matrices).
# 
# 
# ## Summary
# 
# The CITEseq part of the competition has sizeable datasets, when compared to the standard 16 GByte RAM of Kaggle notebooks:
# - The training input has shape 70988\*22050 (6.3 GByte).
# - The training labels have shape 70988\*140.
# - The test input has shape 48663\*22050 (4.3 GByte).
# 
# Our solution strategy has four elements:
# 1. **Dimensionality reduction:** To get rid of the 11 GByte data, we first convert the data to a sparse matrix (because most of the matrix entries are zero) and then project them to 512 dimensions by applying a truncated singular value decomposition (SVD).
# 2. **Domain knowledge:** The column names of the data reveal which features are most important.
# 3. **Gradient boosting:** We fit 140 LightGBM models to the data (because there are 140 targets).
# 4. **Cross-validation:** Submitting unvalidated models and relying only on the public leaderboard is bad practice. The model in this notebook is fully cross-validated with a 3-fold GroupKFold.
# 
# The code contains some tricks to deal with the memory restrictions, among them:
# - Discarding constant features
# - Using sparse matrices
# - Projecting the data to a lower-dimensional subspace
# - Using `TruncatedSVD` for the projection, a memory-efficient implementation of the singular value decomposition
# 

# In[1]:


import os, gc, pickle, scipy.sparse, lightgbm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

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

CROSS_VALIDATE = True
SUBMIT = True


# A little trick to save time: If the tables module is already installed (after a restart of the notebook, for instance), pip wastes 10 seconds by checking whether a newer version exists. We can skip this check by testing for the presence of the module in a simple if statement:

# In[2]:


get_ipython().run_cell_magic('time', '', '# If you see a warning "Failed to establish a new connection" running this cell,\n# go to "Settings" on the right hand side, \n# and turn on internet. Note, you need to be phone verified.\n# We need this library to read HDF files.\nif not os.path.exists(\'/opt/conda/lib/python3.7/site-packages/tables\'):\n    !pip install --quiet tables')


# # The scoring function
# 
# This competition has a special metric: For every row, it computes the Pearson correlation between y_true and y_pred, and then all these correlation coefficients are averaged.

# In[3]:


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


# # Data loading and preprocessing
# 
# We first load the metadata, which we only use for the `GroupKFold` operation.

# In[4]:


metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="citeseq"]
metadata_df.shape


# We now define two sets of features:
# - `constant_cols` is the set of all features which are constant in the train or test datset. These columns will be discarded immediately after loading.
# - `important_cols` is the set of all features whose name matches the name of a target protein. If a gene is named 'ENSG00000114013_CD86', it should be related to a protein named 'CD86'. These features will be used for the model unchanged, that is, they don't undergo dimensionality reduction. 

# In[5]:


# constant_cols = list(X.columns[(X == 0).all(axis=0).values]) + list(X_test.columns[(X_test == 0).all(axis=0).values])
constant_cols = ['ENSG00000003137_CYP26B1', 'ENSG00000004848_ARX', 'ENSG00000006606_CCL26', 'ENSG00000010379_SLC6A13', 'ENSG00000010932_FMO1', 'ENSG00000017427_IGF1', 'ENSG00000022355_GABRA1', 'ENSG00000041982_TNC', 'ENSG00000060709_RIMBP2', 'ENSG00000064886_CHI3L2', 'ENSG00000065717_TLE2', 'ENSG00000067798_NAV3', 'ENSG00000069535_MAOB', 'ENSG00000073598_FNDC8', 'ENSG00000074219_TEAD2', 'ENSG00000074964_ARHGEF10L', 'ENSG00000077264_PAK3', 'ENSG00000078053_AMPH', 'ENSG00000082684_SEMA5B', 'ENSG00000083857_FAT1', 'ENSG00000084628_NKAIN1', 'ENSG00000084734_GCKR', 'ENSG00000086967_MYBPC2', 'ENSG00000087258_GNAO1', 'ENSG00000089505_CMTM1', 'ENSG00000091129_NRCAM', 'ENSG00000091986_CCDC80', 'ENSG00000092377_TBL1Y', 'ENSG00000092969_TGFB2', 'ENSG00000095397_WHRN', 'ENSG00000095970_TREM2', 'ENSG00000099715_PCDH11Y', 'ENSG00000100197_CYP2D6', 'ENSG00000100218_RSPH14', 'ENSG00000100311_PDGFB', 'ENSG00000100362_PVALB', 'ENSG00000100373_UPK3A', 'ENSG00000100625_SIX4', 'ENSG00000100867_DHRS2', 'ENSG00000100985_MMP9', 'ENSG00000101197_BIRC7', 'ENSG00000101298_SNPH', 'ENSG00000102387_TAF7L', 'ENSG00000103034_NDRG4', 'ENSG00000104059_FAM189A1', 'ENSG00000104112_SCG3', 'ENSG00000104313_EYA1', 'ENSG00000104892_KLC3', 'ENSG00000105088_OLFM2', 'ENSG00000105261_OVOL3', 'ENSG00000105290_APLP1', 'ENSG00000105507_CABP5', 'ENSG00000105642_KCNN1', 'ENSG00000105694_ELOCP28', 'ENSG00000105707_HPN', 'ENSG00000105894_PTN', 'ENSG00000106018_VIPR2', 'ENSG00000106541_AGR2', 'ENSG00000107317_PTGDS', 'ENSG00000108688_CCL7', 'ENSG00000108702_CCL1', 'ENSG00000108947_EFNB3', 'ENSG00000109193_SULT1E1', 'ENSG00000109794_FAM149A', 'ENSG00000109832_DDX25', 'ENSG00000110195_FOLR1', 'ENSG00000110375_UPK2', 'ENSG00000110436_SLC1A2', 'ENSG00000111339_ART4', 'ENSG00000111863_ADTRP', 'ENSG00000112761_WISP3', 'ENSG00000112852_PCDHB2', 'ENSG00000114251_WNT5A', 'ENSG00000114279_FGF12', 'ENSG00000114455_HHLA2', 'ENSG00000114757_PEX5L', 'ENSG00000115155_OTOF', 'ENSG00000115266_APC2', 'ENSG00000115297_TLX2', 'ENSG00000115590_IL1R2', 'ENSG00000115844_DLX2', 'ENSG00000116194_ANGPTL1', 'ENSG00000116661_FBXO2', 'ENSG00000116774_OLFML3', 'ENSG00000117322_CR2', 'ENSG00000117971_CHRNB4', 'ENSG00000118322_ATP10B', 'ENSG00000118402_ELOVL4', 'ENSG00000118520_ARG1', 'ENSG00000118946_PCDH17', 'ENSG00000118972_FGF23', 'ENSG00000119771_KLHL29', 'ENSG00000120549_KIAA1217', 'ENSG00000121316_PLBD1', 'ENSG00000121905_HPCA', 'ENSG00000122224_LY9', 'ENSG00000124194_GDAP1L1', 'ENSG00000124440_HIF3A', 'ENSG00000124657_OR2B6', 'ENSG00000125462_C1orf61', 'ENSG00000125895_TMEM74B', 'ENSG00000126838_PZP', 'ENSG00000128422_KRT17', 'ENSG00000128918_ALDH1A2', 'ENSG00000129170_CSRP3', 'ENSG00000129214_SHBG', 'ENSG00000129673_AANAT', 'ENSG00000129910_CDH15', 'ENSG00000130294_KIF1A', 'ENSG00000130307_USHBP1', 'ENSG00000130545_CRB3', 'ENSG00000131019_ULBP3', 'ENSG00000131044_TTLL9', 'ENSG00000131183_SLC34A1', 'ENSG00000131386_GALNT15', 'ENSG00000131400_NAPSA', 'ENSG00000131914_LIN28A', 'ENSG00000131941_RHPN2', 'ENSG00000131951_LRRC9', 'ENSG00000132170_PPARG', 'ENSG00000132681_ATP1A4', 'ENSG00000132958_TPTE2', 'ENSG00000133454_MYO18B', 'ENSG00000134545_KLRC1', 'ENSG00000134853_PDGFRA', 'ENSG00000135083_CCNJL', 'ENSG00000135100_HNF1A', 'ENSG00000135116_HRK', 'ENSG00000135312_HTR1B', 'ENSG00000135324_MRAP2', 'ENSG00000135436_FAM186B', 'ENSG00000135472_FAIM2', 'ENSG00000135898_GPR55', 'ENSG00000135929_CYP27A1', 'ENSG00000136002_ARHGEF4', 'ENSG00000136099_PCDH8', 'ENSG00000136274_NACAD', 'ENSG00000137078_SIT1', 'ENSG00000137142_IGFBPL1', 'ENSG00000137473_TTC29', 'ENSG00000137474_MYO7A', 'ENSG00000137491_SLCO2B1', 'ENSG00000137691_CFAP300', 'ENSG00000137731_FXYD2', 'ENSG00000137747_TMPRSS13', 'ENSG00000137878_GCOM1', 'ENSG00000138411_HECW2', 'ENSG00000138741_TRPC3', 'ENSG00000138769_CDKL2', 'ENSG00000138823_MTTP', 'ENSG00000139908_TSSK4', 'ENSG00000140832_MARVELD3', 'ENSG00000142178_SIK1', 'ENSG00000142538_PTH2', 'ENSG00000142910_TINAGL1', 'ENSG00000143217_NECTIN4', 'ENSG00000143858_SYT2', 'ENSG00000144130_NT5DC4', 'ENSG00000144214_LYG1', 'ENSG00000144290_SLC4A10', 'ENSG00000144366_GULP1', 'ENSG00000144583_MARCH4', 'ENSG00000144771_LRTM1', 'ENSG00000144891_AGTR1', 'ENSG00000145087_STXBP5L', 'ENSG00000145107_TM4SF19', 'ENSG00000146197_SCUBE3', 'ENSG00000146966_DENND2A', 'ENSG00000147082_CCNB3', 'ENSG00000147614_ATP6V0D2', 'ENSG00000147642_SYBU', 'ENSG00000147869_CER1', 'ENSG00000149403_GRIK4', 'ENSG00000149596_JPH2', 'ENSG00000150630_VEGFC', 'ENSG00000150722_PPP1R1C', 'ENSG00000151631_AKR1C6P', 'ENSG00000151704_KCNJ1', 'ENSG00000152154_TMEM178A', 'ENSG00000152292_SH2D6', 'ENSG00000152315_KCNK13', 'ENSG00000152503_TRIM36', 'ENSG00000153253_SCN3A', 'ENSG00000153902_LGI4', 'ENSG00000153930_ANKFN1', 'ENSG00000154040_CABYR', 'ENSG00000154118_JPH3', 'ENSG00000154175_ABI3BP', 'ENSG00000154645_CHODL', 'ENSG00000157060_SHCBP1L', 'ENSG00000157087_ATP2B2', 'ENSG00000157152_SYN2', 'ENSG00000157168_NRG1', 'ENSG00000157680_DGKI', 'ENSG00000158246_TENT5B', 'ENSG00000158477_CD1A', 'ENSG00000158481_CD1C', 'ENSG00000158488_CD1E', 'ENSG00000159189_C1QC', 'ENSG00000159217_IGF2BP1', 'ENSG00000160683_CXCR5', 'ENSG00000160801_PTH1R', 'ENSG00000160973_FOXH1', 'ENSG00000161594_KLHL10', 'ENSG00000162409_PRKAA2', 'ENSG00000162840_MT2P1', 'ENSG00000162873_KLHDC8A', 'ENSG00000162944_RFTN2', 'ENSG00000162949_CAPN13', 'ENSG00000163116_STPG2', 'ENSG00000163288_GABRB1', 'ENSG00000163531_NFASC', 'ENSG00000163618_CADPS', 'ENSG00000163637_PRICKLE2', 'ENSG00000163735_CXCL5', 'ENSG00000163873_GRIK3', 'ENSG00000163898_LIPH', 'ENSG00000164061_BSN', 'ENSG00000164078_MST1R', 'ENSG00000164123_C4orf45', 'ENSG00000164690_SHH', 'ENSG00000164761_TNFRSF11B', 'ENSG00000164821_DEFA4', 'ENSG00000164845_FAM86FP', 'ENSG00000164867_NOS3', 'ENSG00000166073_GPR176', 'ENSG00000166148_AVPR1A', 'ENSG00000166250_CLMP', 'ENSG00000166257_SCN3B', 'ENSG00000166268_MYRFL', 'ENSG00000166523_CLEC4E', 'ENSG00000166535_A2ML1', 'ENSG00000166819_PLIN1', 'ENSG00000166928_MS4A14', 'ENSG00000167210_LOXHD1', 'ENSG00000167306_MYO5B', 'ENSG00000167634_NLRP7', 'ENSG00000167748_KLK1', 'ENSG00000167889_MGAT5B', 'ENSG00000168140_VASN', 'ENSG00000168546_GFRA2', 'ENSG00000168646_AXIN2', 'ENSG00000168955_TM4SF20', 'ENSG00000168993_CPLX1', 'ENSG00000169075_Z99496.1', 'ENSG00000169194_IL13', 'ENSG00000169246_NPIPB3', 'ENSG00000169884_WNT10B', 'ENSG00000169900_PYDC1', 'ENSG00000170074_FAM153A', 'ENSG00000170075_GPR37L1', 'ENSG00000170289_CNGB3', 'ENSG00000170356_OR2A20P', 'ENSG00000170537_TMC7', 'ENSG00000170689_HOXB9', 'ENSG00000170827_CELP', 'ENSG00000171346_KRT15', 'ENSG00000171368_TPPP', 'ENSG00000171501_OR1N2', 'ENSG00000171532_NEUROD2', 'ENSG00000171611_PTCRA', 'ENSG00000171873_ADRA1D', 'ENSG00000171916_LGALS9C', 'ENSG00000172005_MAL', 'ENSG00000172987_HPSE2', 'ENSG00000173068_BNC2', 'ENSG00000173077_DEC1', 'ENSG00000173210_ABLIM3', 'ENSG00000173267_SNCG', 'ENSG00000173369_C1QB', 'ENSG00000173372_C1QA', 'ENSG00000173391_OLR1', 'ENSG00000173626_TRAPPC3L', 'ENSG00000173698_ADGRG2', 'ENSG00000173868_PHOSPHO1', 'ENSG00000174407_MIR1-1HG', 'ENSG00000174807_CD248', 'ENSG00000175206_NPPA', 'ENSG00000175746_C15orf54', 'ENSG00000175985_PLEKHD1', 'ENSG00000176043_AC007160.1', 'ENSG00000176399_DMRTA1', 'ENSG00000176510_OR10AC1', 'ENSG00000176697_BDNF', 'ENSG00000176826_FKBP9P1', 'ENSG00000176988_FMR1NB', 'ENSG00000177324_BEND2', 'ENSG00000177335_C8orf31', 'ENSG00000177535_OR2B11', 'ENSG00000177614_PGBD5', 'ENSG00000177707_NECTIN3', 'ENSG00000178033_CALHM5', 'ENSG00000178175_ZNF366', 'ENSG00000178462_TUBAL3', 'ENSG00000178732_GP5', 'ENSG00000178750_STX19', 'ENSG00000179058_C9orf50', 'ENSG00000179101_AL590139.1', 'ENSG00000179388_EGR3', 'ENSG00000179611_DGKZP1', 'ENSG00000179899_PHC1P1', 'ENSG00000179934_CCR8', 'ENSG00000180537_RNF182', 'ENSG00000180712_LINC02363', 'ENSG00000180988_OR52N2', 'ENSG00000181001_OR52N1', 'ENSG00000181616_OR52H1', 'ENSG00000181634_TNFSF15', 'ENSG00000182021_AL591379.1', 'ENSG00000182230_FAM153B', 'ENSG00000182853_VMO1', 'ENSG00000183090_FREM3', 'ENSG00000183562_AC131971.1', 'ENSG00000183615_FAM167B', 'ENSG00000183625_CCR3', 'ENSG00000183770_FOXL2', 'ENSG00000183779_ZNF703', 'ENSG00000183831_ANKRD45', 'ENSG00000183844_FAM3B', 'ENSG00000183960_KCNH8', 'ENSG00000184106_TREML3P', 'ENSG00000184227_ACOT1', 'ENSG00000184363_PKP3', 'ENSG00000184434_LRRC19', 'ENSG00000184454_NCMAP', 'ENSG00000184571_PIWIL3', 'ENSG00000184702_SEPT5', 'ENSG00000184908_CLCNKB', 'ENSG00000184923_NUTM2A', 'ENSG00000185070_FLRT2', 'ENSG00000185156_MFSD6L', 'ENSG00000185567_AHNAK2', 'ENSG00000185686_PRAME', 'ENSG00000186190_BPIFB3', 'ENSG00000186191_BPIFB4', 'ENSG00000186231_KLHL32', 'ENSG00000186431_FCAR', 'ENSG00000186715_MST1L', 'ENSG00000187116_LILRA5', 'ENSG00000187185_AC092118.1', 'ENSG00000187268_FAM9C', 'ENSG00000187554_TLR5', 'ENSG00000187867_PALM3', 'ENSG00000188153_COL4A5', 'ENSG00000188158_NHS', 'ENSG00000188163_FAM166A', 'ENSG00000188316_ENO4', 'ENSG00000188959_C9orf152', 'ENSG00000189013_KIR2DL4', 'ENSG00000189409_MMP23B', 'ENSG00000196092_PAX5', 'ENSG00000196260_SFTA2', 'ENSG00000197358_BNIP3P1', 'ENSG00000197446_CYP2F1', 'ENSG00000197540_GZMM', 'ENSG00000198049_AVPR1B', 'ENSG00000198134_AC007537.1', 'ENSG00000198156_NPIPB6', 'ENSG00000198221_AFDN-DT', 'ENSG00000198626_RYR2', 'ENSG00000198759_EGFL6', 'ENSG00000198822_GRM3', 'ENSG00000198963_RORB', 'ENSG00000199090_MIR326', 'ENSG00000199753_SNORD104', 'ENSG00000199787_RF00406', 'ENSG00000199872_RNU6-942P', 'ENSG00000200075_RF00402', 'ENSG00000200296_RNU1-83P', 'ENSG00000200683_RNU6-379P', 'ENSG00000201044_RNU6-268P', 'ENSG00000201343_RF00019', 'ENSG00000201564_RN7SKP50', 'ENSG00000201616_RNU1-91P', 'ENSG00000201737_RNU1-133P', 'ENSG00000202048_SNORD114-20', 'ENSG00000202415_RN7SKP269', 'ENSG00000203395_AC015969.1', 'ENSG00000203721_LINC00862', 'ENSG00000203727_SAMD5', 'ENSG00000203737_GPR52', 'ENSG00000203783_PRR9', 'ENSG00000203867_RBM20', 'ENSG00000203907_OOEP', 'ENSG00000203999_LINC01270', 'ENSG00000204010_IFIT1B', 'ENSG00000204044_SLC12A5-AS1', 'ENSG00000204091_TDRG1', 'ENSG00000204121_ECEL1P1', 'ENSG00000204165_CXorf65', 'ENSG00000204173_LRRC37A5P', 'ENSG00000204248_COL11A2', 'ENSG00000204424_LY6G6F', 'ENSG00000204539_CDSN', 'ENSG00000204583_LRCOL1', 'ENSG00000204677_FAM153C', 'ENSG00000204709_LINC01556', 'ENSG00000204711_C9orf135', 'ENSG00000204792_LINC01291', 'ENSG00000204850_AC011484.1', 'ENSG00000204851_PNMA8B', 'ENSG00000204909_SPINK9', 'ENSG00000205037_AC134312.1', 'ENSG00000205038_PKHD1L1', 'ENSG00000205089_CCNI2', 'ENSG00000205106_DKFZp779M0652', 'ENSG00000205364_MT1M', 'ENSG00000205502_C2CD4B', 'ENSG00000205746_AC126755.1', 'ENSG00000205856_C22orf42', 'ENSG00000206052_DOK6', 'ENSG00000206579_XKR4', 'ENSG00000206645_RF00019', 'ENSG00000206786_RNU6-701P', 'ENSG00000206846_RF00019', 'ENSG00000206848_RNU6-890P', 'ENSG00000207088_SNORA7B', 'ENSG00000207181_SNORA14B', 'ENSG00000207234_RNU6-125P', 'ENSG00000207326_RF00019', 'ENSG00000207359_RNU6-925P', 'ENSG00000211677_IGLC2', 'ENSG00000211699_TRGV3', 'ENSG00000211895_IGHA1', 'ENSG00000212385_RNU6-817P', 'ENSG00000212391_RF00554', 'ENSG00000212607_SNORA3B', 'ENSG00000212829_RPS26P3', 'ENSG00000213083_AC010731.1', 'ENSG00000213216_AC007066.1', 'ENSG00000213222_AC093724.1', 'ENSG00000213228_RPL12P38', 'ENSG00000213250_RBMS2P1', 'ENSG00000213272_RPL7AP9', 'ENSG00000213303_AC008481.1', 'ENSG00000213402_PTPRCAP', 'ENSG00000213471_TTLL13P', 'ENSG00000213588_ZBTB9', 'ENSG00000213609_RPL7AP50', 'ENSG00000213757_AC020898.1', 'ENSG00000213931_HBE1', 'ENSG00000213950_RPS10P2', 'ENSG00000213994_AL157395.1', 'ENSG00000214787_MS4A4E', 'ENSG00000214866_DCDC2C', 'ENSG00000214908_AL353678.1', 'ENSG00000214975_PPIAP29', 'ENSG00000215198_AL353795.1', 'ENSG00000215208_KRT18P60', 'ENSG00000215218_UBE2QL1', 'ENSG00000215297_AL354941.1', 'ENSG00000215464_AP000354.1', 'ENSG00000215483_LINC00598', 'ENSG00000215817_ZC3H11B', 'ENSG00000215861_AC245297.1', 'ENSG00000215910_C1orf167', 'ENSG00000216475_AL024474.1', 'ENSG00000217195_AL513475.1', 'ENSG00000217414_DDX18P3', 'ENSG00000217512_AL356776.1', 'ENSG00000218351_RPS3AP23', 'ENSG00000218418_AL591135.1', 'ENSG00000218749_AL033519.1', 'ENSG00000218766_AL450338.1', 'ENSG00000218792_HSPD1P16', 'ENSG00000219249_AMZ2P2', 'ENSG00000219395_HSPA8P15', 'ENSG00000219410_AC125494.1', 'ENSG00000219932_RPL12P8', 'ENSG00000220091_LAP3P1', 'ENSG00000220237_RPS24P12', 'ENSG00000220494_YAP1P1', 'ENSG00000221102_SNORA11B', 'ENSG00000221887_HMSD', 'ENSG00000222276_RNU2-33P', 'ENSG00000222370_SNORA36B', 'ENSG00000222421_RF00019', 'ENSG00000222431_RNU6-141P', 'ENSG00000223342_AL158817.1', 'ENSG00000223379_AL391987.3', 'ENSG00000223403_MEG9', 'ENSG00000223519_KIF28P', 'ENSG00000223576_AL355001.1', 'ENSG00000223668_EEF1A1P24', 'ENSG00000223741_PSMD4P1', 'ENSG00000223779_AC239800.1', 'ENSG00000223783_LINC01983', 'ENSG00000223784_LINP1', 'ENSG00000223855_HRAT92', 'ENSG00000223884_AC068481.1', 'ENSG00000223899_SEC13P1', 'ENSG00000224067_AL354877.1', 'ENSG00000224072_AL139811.1', 'ENSG00000224081_SLC44A3-AS1', 'ENSG00000224099_AC104823.1', 'ENSG00000224116_INHBA-AS1', 'ENSG00000224137_LINC01857', 'ENSG00000224155_AC073136.2', 'ENSG00000224321_RPL12P14', 'ENSG00000224402_OR6D1P', 'ENSG00000224479_AC104162.1', 'ENSG00000224599_BMS1P12', 'ENSG00000224689_ZNF812P', 'ENSG00000224848_AL589843.1', 'ENSG00000224908_TIMM8BP2', 'ENSG00000224957_LINC01266', 'ENSG00000224959_AC017002.1', 'ENSG00000224988_AL158207.1', 'ENSG00000224993_RPL29P12', 'ENSG00000225096_AL445250.1', 'ENSG00000225101_OR52K3P', 'ENSG00000225107_AC092484.1', 'ENSG00000225187_AC073283.1', 'ENSG00000225313_AL513327.1', 'ENSG00000225345_SNX18P3', 'ENSG00000225393_BX571846.1', 'ENSG00000225422_RBMS1P1', 'ENSG00000225423_TNPO1P1', 'ENSG00000225531_AL807761.2', 'ENSG00000225554_AL359764.1', 'ENSG00000225650_EIF2S2P5', 'ENSG00000225674_IPO7P2', 'ENSG00000225807_AC069281.1', 'ENSG00000226010_AL355852.1', 'ENSG00000226084_AC113935.1', 'ENSG00000226251_AL451060.1', 'ENSG00000226383_LINC01876', 'ENSG00000226491_FTOP1', 'ENSG00000226501_USF1P1', 'ENSG00000226545_AL357552.1', 'ENSG00000226564_FTH1P20', 'ENSG00000226617_RPL21P110', 'ENSG00000226647_AL365356.1', 'ENSG00000226800_CACTIN-AS1', 'ENSG00000226913_BSN-DT', 'ENSG00000226948_RPS4XP2', 'ENSG00000226970_AL450063.1', 'ENSG00000227006_AL136988.2', 'ENSG00000227051_C14orf132', 'ENSG00000227072_AL353706.1', 'ENSG00000227110_LMCD1-AS1', 'ENSG00000227192_AL023581.2', 'ENSG00000227198_C6orf47-AS1', 'ENSG00000227207_RPL31P12', 'ENSG00000227477_STK4-AS1', 'ENSG00000227541_SFR1P1', 'ENSG00000227590_ATP5MC1P5', 'ENSG00000227649_MTND6P32', 'ENSG00000227682_ATP5F1AP2', 'ENSG00000227740_AL513329.1', 'ENSG00000227742_CALR4P', 'ENSG00000228097_MTATP6P11', 'ENSG00000228140_AL031283.1', 'ENSG00000228175_GEMIN8P4', 'ENSG00000228212_OFD1P17', 'ENSG00000228232_GAPDHP1', 'ENSG00000228317_AL158070.1', 'ENSG00000228413_AC024937.1', 'ENSG00000228430_AL162726.3', 'ENSG00000228501_RPL15P18', 'ENSG00000228550_AC073583.1', 'ENSG00000228655_AC096558.1', 'ENSG00000228727_SAPCD1', 'ENSG00000228826_AL592494.1', 'ENSG00000228839_PIK3IP1-AS1', 'ENSG00000228863_AL121985.1', 'ENSG00000229066_AC093459.1', 'ENSG00000229150_CRYGEP', 'ENSG00000229154_KCNQ5-AS1', 'ENSG00000229163_NAP1L1P2', 'ENSG00000229236_TTTY10', 'ENSG00000229274_AL662860.1', 'ENSG00000229308_AC010737.1', 'ENSG00000229326_AC069154.1', 'ENSG00000229372_SZT2-AS1', 'ENSG00000229444_AL451062.1', 'ENSG00000229567_AL139421.1', 'ENSG00000229703_CR589904.1', 'ENSG00000229742_AC092809.1', 'ENSG00000229758_DYNLT3P2', 'ENSG00000229839_AC018462.1', 'ENSG00000229847_EMX2OS', 'ENSG00000229853_AL034418.1', 'ENSG00000229918_DOCK9-AS1', 'ENSG00000229953_AL590666.2', 'ENSG00000229992_HMGB3P9', 'ENSG00000230063_AL360091.2', 'ENSG00000230064_AL772161.1', 'ENSG00000230138_AC119428.2', 'ENSG00000230149_AL021707.3', 'ENSG00000230289_AL358781.2', 'ENSG00000230295_GTF2IP23', 'ENSG00000230479_AP000695.1', 'ENSG00000230508_RPL19P21', 'ENSG00000230519_HMGB1P49', 'ENSG00000230534_AL392046.1', 'ENSG00000230563_AL121757.1', 'ENSG00000230721_AL049597.1', 'ENSG00000230772_VN1R108P', 'ENSG00000230777_RPS29P5', 'ENSG00000230799_AC007279.1', 'ENSG00000230813_AL356583.3', 'ENSG00000230815_AL807757.1', 'ENSG00000230872_MFSD13B', 'ENSG00000230910_AL391807.1', 'ENSG00000230912_AL021707.4', 'ENSG00000230968_AC084149.2', 'ENSG00000230993_RPL12P15', 'ENSG00000231265_TRERNA1', 'ENSG00000231307_RPS3P2', 'ENSG00000231407_AL354732.1', 'ENSG00000231449_AC097359.1', 'ENSG00000231507_LINC01353', 'ENSG00000231531_HINT1P1', 'ENSG00000231548_OR55B1P', 'ENSG00000231731_AC010976.1', 'ENSG00000231742_LINC01273', 'ENSG00000231788_RPL31P50', 'ENSG00000231830_AC245140.1', 'ENSG00000231927_AC093734.1', 'ENSG00000231993_EP300-AS1', 'ENSG00000232027_AL671986.1', 'ENSG00000232028_AC007391.1', 'ENSG00000232065_LINC01063', 'ENSG00000232133_IMPDH1P10', 'ENSG00000232139_LINC00867', 'ENSG00000232273_FTH1P1', 'ENSG00000232333_RPS27AP2', 'ENSG00000232466_AL356133.1', 'ENSG00000232500_AP005273.1', 'ENSG00000232530_LIF-AS1', 'ENSG00000232568_RPL23AP35', 'ENSG00000232578_AC093311.1', 'ENSG00000232606_LINC01412', 'ENSG00000232654_FAM136BP', 'ENSG00000232656_IDI2-AS1', 'ENSG00000232719_AC007272.1', 'ENSG00000232803_SLCO4A1-AS1', 'ENSG00000232987_LINC01219', 'ENSG00000233025_CRYZP1', 'ENSG00000233093_LINC00892', 'ENSG00000233099_AC095030.1', 'ENSG00000233401_PRKAR1AP1', 'ENSG00000233427_AL009181.1', 'ENSG00000233540_DNM3-IT1', 'ENSG00000233674_AL451062.2', 'ENSG00000233825_AL391839.2', 'ENSG00000233862_AC016907.2', 'ENSG00000233994_GDI2P2', 'ENSG00000234026_AL157834.2', 'ENSG00000234106_SRP14P2', 'ENSG00000234145_NAP1L4P3', 'ENSG00000234174_AC016683.1', 'ENSG00000234271_Z98752.2', 'ENSG00000234425_AL138930.1', 'ENSG00000234488_AC096664.2', 'ENSG00000234630_AC245060.2', 'ENSG00000234645_YWHAEP5', 'ENSG00000234718_AC007161.1', 'ENSG00000234810_AL603840.1', 'ENSG00000235045_RPL7P8', 'ENSG00000235072_AC012074.1', 'ENSG00000235214_FAM83C-AS1', 'ENSG00000235288_AC099329.1', 'ENSG00000235376_RPEL1', 'ENSG00000235429_AC083875.1', 'ENSG00000235472_EIF4A1P7', 'ENSG00000235478_LINC01664', 'ENSG00000235531_MSC-AS1', 'ENSG00000235640_AC092646.2', 'ENSG00000235677_NPM1P26', 'ENSG00000235683_AC018442.1', 'ENSG00000235701_PCBP2P1', 'ENSG00000235740_PHACTR2-AS1', 'ENSG00000235774_AC023347.1', 'ENSG00000235802_HCFC1-AS1', 'ENSG00000235917_MTCO2P11', 'ENSG00000235958_UBOX5-AS1', 'ENSG00000236032_OR5H14', 'ENSG00000236180_AL445669.2', 'ENSG00000236254_MTND4P14', 'ENSG00000236283_AC019197.1', 'ENSG00000236290_EEF1GP7', 'ENSG00000236317_AC104333.2', 'ENSG00000236364_AL358115.1', 'ENSG00000236457_AC090617.1', 'ENSG00000236564_YWHAQP5', 'ENSG00000236671_PRKG1-AS1', 'ENSG00000236680_AL356000.1', 'ENSG00000236682_AC068282.1', 'ENSG00000236711_SMAD9-IT1', 'ENSG00000236806_RPL7AP15', 'ENSG00000236869_ZKSCAN7-AS1', 'ENSG00000236886_AC007563.2', 'ENSG00000236915_AL356270.1', 'ENSG00000236936_AL031005.1', 'ENSG00000237057_LINC02087', 'ENSG00000237101_AC092809.4', 'ENSG00000237276_ANO7L1', 'ENSG00000237317_AL022400.1', 'ENSG00000237387_AL022329.2', 'ENSG00000237618_BTBD7P2', 'ENSG00000237685_AL139039.3', 'ENSG00000237757_EEF1A1P30', 'ENSG00000237766_GGTA2P', 'ENSG00000237798_AC010894.4', 'ENSG00000238015_AC104837.2', 'ENSG00000238133_MAP3K20-AS1', 'ENSG00000238259_AC067940.1', 'ENSG00000238324_RN7SKP198', 'ENSG00000238358_AC004969.1', 'ENSG00000239219_AC008040.1', 'ENSG00000239316_RN7SL11P', 'ENSG00000239474_KLHL41', 'ENSG00000239527_RPS23P7', 'ENSG00000239642_MEIKIN', 'ENSG00000239650_GUSBP4', 'ENSG00000239686_AL158801.1', 'ENSG00000239701_AC006512.1', 'ENSG00000239705_AL354710.2', 'ENSG00000239797_RPL21P39', 'ENSG00000239830_RPS4XP22', 'ENSG00000239930_AP001625.3', 'ENSG00000240086_AC092969.1', 'ENSG00000240087_RPSAP12', 'ENSG00000240183_RN7SL297P', 'ENSG00000240219_AL512306.2', 'ENSG00000240498_CDKN2B-AS1', 'ENSG00000240809_AC026877.1', 'ENSG00000240993_RN7SL459P', 'ENSG00000241111_PRICKLE2-AS1', 'ENSG00000241135_LINC00881', 'ENSG00000241319_SETP6', 'ENSG00000241570_PAQR9-AS1', 'ENSG00000241631_RN7SL316P', 'ENSG00000241932_AC092324.1', 'ENSG00000241933_DENND6A-DT', 'ENSG00000242060_RPS3AP49', 'ENSG00000242107_LINC01100', 'ENSG00000242175_RN7SL127P', 'ENSG00000242431_AC107398.1', 'ENSG00000242551_POU5F1P6', 'ENSG00000242571_RPL21P11', 'ENSG00000242641_LINC00971', 'ENSG00000242747_AC090515.1', 'ENSG00000242992_FTH1P4', 'ENSG00000243055_GK-AS1', 'ENSG00000243498_UBA52P5', 'ENSG00000243592_RPL17P22', 'ENSG00000243709_LEFTY1', 'ENSG00000243830_AC092865.1', 'ENSG00000243836_WDR86-AS1', 'ENSG00000243961_PARAL1', 'ENSG00000244021_AC093591.1', 'ENSG00000244097_RPS4XP17', 'ENSG00000244151_AC010973.2', 'ENSG00000244183_PPIAP71', 'ENSG00000244242_IFITM10', 'ENSG00000244245_AC133134.1', 'ENSG00000244251_AC013356.1', 'ENSG00000244355_LY6G6D', 'ENSG00000244357_RN7SL145P', 'ENSG00000244476_ERVFRD-1', 'ENSG00000244482_LILRA6', 'ENSG00000244585_RPL12P33', 'ENSG00000244618_RN7SL334P', 'ENSG00000244703_CD46P1', 'ENSG00000245261_AL133375.1', 'ENSG00000245482_AC046130.1', 'ENSG00000246363_LINC02458', 'ENSG00000246863_AC012377.1', 'ENSG00000247199_AC091948.1', 'ENSG00000248121_SMURF2P1', 'ENSG00000248155_CR545473.1', 'ENSG00000248223_AC026785.2', 'ENSG00000248485_PCP4L1', 'ENSG00000248690_HAS2-AS1', 'ENSG00000248884_AC010280.2', 'ENSG00000248936_AC027607.1', 'ENSG00000249140_PRDX2P3', 'ENSG00000249363_AC011411.1', 'ENSG00000249381_LINC00500', 'ENSG00000249456_AL731577.2', 'ENSG00000249492_AC114956.3', 'ENSG00000249574_AC226118.1', 'ENSG00000249614_LINC02503', 'ENSG00000249691_AC026117.1', 'ENSG00000249695_AC026369.1', 'ENSG00000249803_AC112178.1', 'ENSG00000249825_CTD-2201I18.1', 'ENSG00000249848_AC112673.1', 'ENSG00000249850_KRT18P31', 'ENSG00000249884_RNF103-CHMP3', 'ENSG00000249978_TRGV7', 'ENSG00000250130_AC090519.1', 'ENSG00000250148_KRT8P31', 'ENSG00000250332_AC010460.3', 'ENSG00000250334_LINC00989', 'ENSG00000250539_KRT8P33', 'ENSG00000250548_LINC01303', 'ENSG00000250608_AC010210.1', 'ENSG00000250635_CXXC5-AS1', 'ENSG00000250645_AC010442.2', 'ENSG00000250733_C8orf17', 'ENSG00000250853_RNF138P1', 'ENSG00000250902_SMAD1-AS1', 'ENSG00000250950_AC093752.2', 'ENSG00000250982_GAPDHP35', 'ENSG00000251129_LINC02506', 'ENSG00000251152_AC025539.1', 'ENSG00000251250_AC091951.3', 'ENSG00000251288_AC018797.3', 'ENSG00000251468_AC135352.1', 'ENSG00000251537_AC005324.3', 'ENSG00000251538_LINC02201', 'ENSG00000251584_AC096751.2', 'ENSG00000251676_SNHG27', 'ENSG00000251916_RNU1-61P', 'ENSG00000252759_RF00019', 'ENSG00000253256_AC134043.1', 'ENSG00000253305_PCDHGB6', 'ENSG00000253394_LINC00534', 'ENSG00000253490_LINC02099', 'ENSG00000253537_PCDHGA7', 'ENSG00000253629_AP000426.1', 'ENSG00000253651_SOD1P3', 'ENSG00000253730_AC015909.2', 'ENSG00000253734_LINC01289', 'ENSG00000253767_PCDHGA8', 'ENSG00000253853_AC246817.1', 'ENSG00000253873_PCDHGA11', 'ENSG00000254028_AC083843.1', 'ENSG00000254048_AC105150.1', 'ENSG00000254054_AC087273.2', 'ENSG00000254122_PCDHGB7', 'ENSG00000254248_AC068189.1', 'ENSG00000254680_AC079329.1', 'ENSG00000254708_AL139174.1', 'ENSG00000254780_AC023232.1', 'ENSG00000254810_AP001189.3', 'ENSG00000254812_AC067930.3', 'ENSG00000254842_LINC02551', 'ENSG00000254846_AL355075.1', 'ENSG00000254862_AC100771.2', 'ENSG00000254897_AP003035.1', 'ENSG00000255002_LINC02324', 'ENSG00000255074_AC018523.1', 'ENSG00000255102_AP005436.1', 'ENSG00000255156_RNY1P9', 'ENSG00000255158_AC131934.1', 'ENSG00000255222_SETP17', 'ENSG00000255256_AL136146.2', 'ENSG00000255367_AC127526.2', 'ENSG00000255418_AC090092.1', 'ENSG00000255443_CD44-AS1', 'ENSG00000255446_AP003064.2', 'ENSG00000255479_AP001189.6', 'ENSG00000255487_AC087362.2', 'ENSG00000255867_DENND5B-AS1', 'ENSG00000255871_AC007529.1', 'ENSG00000256029_SNHG28', 'ENSG00000256571_AC079866.2', 'ENSG00000256588_AC027544.2', 'ENSG00000256712_AC134349.1', 'ENSG00000256746_AC018410.1', 'ENSG00000256813_AP000777.3', 'ENSG00000256967_AC018653.3', 'ENSG00000256968_SNRPEP2', 'ENSG00000257074_RPL29P33', 'ENSG00000257120_AL356756.1', 'ENSG00000257146_AC079905.2', 'ENSG00000257195_HNRNPA1P50', 'ENSG00000257327_AC012555.1', 'ENSG00000257345_LINC02413', 'ENSG00000257379_AC023509.1', 'ENSG00000257386_AC025257.1', 'ENSG00000257431_AC089998.1', 'ENSG00000257715_AC007298.1', 'ENSG00000257838_OTOAP1', 'ENSG00000257987_TEX49', 'ENSG00000258084_AC128707.1', 'ENSG00000258090_AC093014.1', 'ENSG00000258177_AC008149.1', 'ENSG00000258357_AC023161.2', 'ENSG00000258410_AC087386.1', 'ENSG00000258498_DIO3OS', 'ENSG00000258504_AL157871.1', 'ENSG00000258512_LINC00239', 'ENSG00000258867_LINC01146', 'ENSG00000258886_HIGD1AP17', 'ENSG00000259032_ENSAP2', 'ENSG00000259100_AL157791.1', 'ENSG00000259294_AC005096.1', 'ENSG00000259327_AC023906.3', 'ENSG00000259345_AC013652.1', 'ENSG00000259377_AC026770.1', 'ENSG00000259380_AC087473.1', 'ENSG00000259442_AC105339.3', 'ENSG00000259461_ANP32BP3', 'ENSG00000259556_AC090971.3', 'ENSG00000259569_AC013489.2', 'ENSG00000259617_AC020661.3', 'ENSG00000259684_AC084756.1', 'ENSG00000259719_LINC02284', 'ENSG00000259954_IL21R-AS1', 'ENSG00000259986_AC103876.1', 'ENSG00000260135_MMP2-AS1', 'ENSG00000260206_AC105020.2', 'ENSG00000260235_AC105020.3', 'ENSG00000260269_AC105036.3', 'ENSG00000260394_Z92544.1', 'ENSG00000260425_AL031709.1', 'ENSG00000260447_AC009065.3', 'ENSG00000260615_RPL23AP97', 'ENSG00000260871_AC093510.2', 'ENSG00000260877_AP005233.2', 'ENSG00000260979_AC022167.3', 'ENSG00000261051_AC107021.2', 'ENSG00000261113_AC009034.1', 'ENSG00000261168_AL592424.1', 'ENSG00000261253_AC137932.2', 'ENSG00000261269_AC093278.2', 'ENSG00000261552_AC109460.4', 'ENSG00000261572_AC097639.1', 'ENSG00000261602_AC092115.2', 'ENSG00000261630_AC007496.2', 'ENSG00000261644_AC007728.2', 'ENSG00000261734_AC116096.1', 'ENSG00000261773_AC244090.2', 'ENSG00000261837_AC046158.2', 'ENSG00000261838_AC092718.6', 'ENSG00000261888_AC144831.1', 'ENSG00000262061_AC129507.1', 'ENSG00000262097_LINC02185', 'ENSG00000262372_CR936218.1', 'ENSG00000262406_MMP12', 'ENSG00000262580_AC087741.1', 'ENSG00000262772_LINC01977', 'ENSG00000262833_AC016245.1', 'ENSG00000263006_ROCK1P1', 'ENSG00000263011_AC108134.4', 'ENSG00000263155_MYZAP', 'ENSG00000263393_AC011825.2', 'ENSG00000263426_RN7SL471P', 'ENSG00000263503_MAPK8IP1P2', 'ENSG00000263595_RN7SL823P', 'ENSG00000263878_DLGAP1-AS4', 'ENSG00000263940_RN7SL275P', 'ENSG00000264019_AC018521.2', 'ENSG00000264031_ABHD15-AS1', 'ENSG00000264044_AC005726.2', 'ENSG00000264070_DND1P1', 'ENSG00000264188_AC106037.1', 'ENSG00000264269_AC016866.1', 'ENSG00000264339_AP001020.1', 'ENSG00000264434_AC110603.1', 'ENSG00000264714_KIAA0895LP1', 'ENSG00000265010_AC087301.1', 'ENSG00000265073_AC010761.2', 'ENSG00000265107_GJA5', 'ENSG00000265179_AP000894.2', 'ENSG00000265218_AC103810.2', 'ENSG00000265334_AC130324.2', 'ENSG00000265439_RN7SL811P', 'ENSG00000265531_FCGR1CP', 'ENSG00000265845_AC024267.4', 'ENSG00000265907_AP000919.2', 'ENSG00000265942_RN7SL577P', 'ENSG00000266256_LINC00683', 'ENSG00000266456_AP001178.3', 'ENSG00000266733_TBC1D29', 'ENSG00000266835_GAPLINC', 'ENSG00000266844_AC093330.1', 'ENSG00000266903_AC243964.2', 'ENSG00000266944_AC005262.1', 'ENSG00000266946_MRPL37P1', 'ENSG00000266947_AC022916.1', 'ENSG00000267034_AC010980.2', 'ENSG00000267044_AC005757.1', 'ENSG00000267147_LINC01842', 'ENSG00000267175_AC105094.2', 'ENSG00000267191_AC006213.3', 'ENSG00000267275_AC020911.2', 'ENSG00000267288_AC138150.2', 'ENSG00000267313_KC6', 'ENSG00000267316_AC090409.2', 'ENSG00000267323_SLC25A1P5', 'ENSG00000267345_AC010632.1', 'ENSG00000267387_AC020931.1', 'ENSG00000267395_DM1-AS', 'ENSG00000267429_AC006116.6', 'ENSG00000267452_LINC02073', 'ENSG00000267491_AC100788.1', 'ENSG00000267529_AP005131.4', 'ENSG00000267554_AC015911.8', 'ENSG00000267601_AC022966.1', 'ENSG00000267638_AC023855.1', 'ENSG00000267665_AC021683.3', 'ENSG00000267681_AC135721.1', 'ENSG00000267703_AC020917.2', 'ENSG00000267731_AC005332.2', 'ENSG00000267733_AP005264.5', 'ENSG00000267750_RUNDC3A-AS1', 'ENSG00000267890_AC010624.2', 'ENSG00000267898_AC026803.2', 'ENSG00000267927_AC010320.1', 'ENSG00000268070_AC006539.2', 'ENSG00000268355_AC243960.3', 'ENSG00000268416_AC010329.1', 'ENSG00000268520_AC008750.5', 'ENSG00000268636_AC011495.2', 'ENSG00000268696_ZNF723', 'ENSG00000268777_AC020914.1', 'ENSG00000268849_SIGLEC22P', 'ENSG00000268903_AL627309.6', 'ENSG00000268983_AC005253.2', 'ENSG00000269019_HOMER3-AS1', 'ENSG00000269067_ZNF728', 'ENSG00000269103_RF00017', 'ENSG00000269274_AC078899.4', 'ENSG00000269288_AC092070.3', 'ENSG00000269352_PTOV1-AS2', 'ENSG00000269400_AC008734.2', 'ENSG00000269506_AC110792.2', 'ENSG00000269653_AC011479.3', 'ENSG00000269881_AC004754.1', 'ENSG00000269926_DDIT4-AS1', 'ENSG00000270048_AC068790.4', 'ENSG00000270050_AL035427.1', 'ENSG00000270503_YTHDF2P1', 'ENSG00000270706_PRMT1P1', 'ENSG00000270765_GAS2L2', 'ENSG00000270882_HIST2H4A', 'ENSG00000270906_MTND4P35', 'ENSG00000271013_LRRC37A9P', 'ENSG00000271129_AC009027.1', 'ENSG00000271259_AC010201.1', 'ENSG00000271524_BNIP3P17', 'ENSG00000271543_AC021443.1', 'ENSG00000271743_AF287957.1', 'ENSG00000271792_AC008667.4', 'ENSG00000271868_AC114810.1', 'ENSG00000271973_AC141002.1', 'ENSG00000271984_AL008726.1', 'ENSG00000271996_AC019080.4', 'ENSG00000272070_AC005618.1', 'ENSG00000272138_LINC01607', 'ENSG00000272150_NBPF25P', 'ENSG00000272265_AC034236.3', 'ENSG00000272279_AL512329.2', 'ENSG00000272473_AC006273.1', 'ENSG00000272510_AL121992.3', 'ENSG00000272582_AL031587.3', 'ENSG00000272695_GAS6-DT', 'ENSG00000272732_AC004982.1', 'ENSG00000272770_AC005696.2', 'ENSG00000272788_AP000864.1', 'ENSG00000272824_AC245100.7', 'ENSG00000272825_AL844908.1', 'ENSG00000272848_AL135910.1', 'ENSG00000272916_AC022400.6', 'ENSG00000273133_AC116651.1', 'ENSG00000273177_AC092954.2', 'ENSG00000273212_AC000068.2', 'ENSG00000273218_AC005776.2', 'ENSG00000273245_AC092653.1', 'ENSG00000273274_ZBTB8B', 'ENSG00000273312_AL121749.1', 'ENSG00000273325_AL008723.3', 'ENSG00000273369_AC096586.2', 'ENSG00000273474_AL157392.4', 'ENSG00000273599_AL731571.1', 'ENSG00000273724_AC106782.5', 'ENSG00000273870_AL138721.1', 'ENSG00000273920_AC103858.2', 'ENSG00000274023_AL360169.2', 'ENSG00000274029_AC069209.1', 'ENSG00000274114_ALOX15P1', 'ENSG00000274124_AC074029.3', 'ENSG00000274139_AC090164.2', 'ENSG00000274281_AC022929.2', 'ENSG00000274308_AC244093.1', 'ENSG00000274373_AC148476.1', 'ENSG00000274386_TMEM269', 'ENSG00000274403_AC090510.2', 'ENSG00000274570_SPDYE10P', 'ENSG00000274670_AC137590.2', 'ENSG00000274723_AC079906.1', 'ENSG00000274742_RF00017', 'ENSG00000274798_AC025166.1', 'ENSG00000274911_AL627230.2', 'ENSG00000275106_AC025594.2', 'ENSG00000275197_AC092794.2', 'ENSG00000275302_CCL4', 'ENSG00000275348_AC096861.1', 'ENSG00000275367_AC092111.1', 'ENSG00000275489_C17orf98', 'ENSG00000275527_AC100835.2', 'ENSG00000275995_AC109809.1', 'ENSG00000276070_CCL4L2', 'ENSG00000276255_AL136379.1', 'ENSG00000276282_AC022960.2', 'ENSG00000276547_PCDHGB5', 'ENSG00000276704_AL442067.2', 'ENSG00000276952_AL121772.3', 'ENSG00000276984_AL023881.1', 'ENSG00000276997_AL513314.2', 'ENSG00000277117_FP565260.3', 'ENSG00000277152_AC110048.2', 'ENSG00000277186_AC131212.1', 'ENSG00000277229_AC084781.1', 'ENSG00000277496_AL357033.4', 'ENSG00000277504_AC010536.3', 'ENSG00000277531_PNMA8C', 'ENSG00000278041_AL133325.3', 'ENSG00000278344_AC063943.1', 'ENSG00000278467_AC138393.3', 'ENSG00000278513_AC091046.2', 'ENSG00000278621_AC037198.2', 'ENSG00000278713_AC120114.2', 'ENSG00000278716_AC133540.1', 'ENSG00000278746_RN7SL660P', 'ENSG00000278774_RF00004', 'ENSG00000279091_AC026523.2', 'ENSG00000279130_AC091925.1', 'ENSG00000279141_LINC01451', 'ENSG00000279161_AC093503.3', 'ENSG00000279187_AC027601.5', 'ENSG00000279263_OR2L8', 'ENSG00000279315_AL158212.4', 'ENSG00000279319_AC105074.1', 'ENSG00000279332_AC090772.4', 'ENSG00000279339_AC100788.2', 'ENSG00000279365_AP000695.3', 'ENSG00000279378_AC009159.4', 'ENSG00000279384_AC080188.2', 'ENSG00000279404_AC008739.5', 'ENSG00000279417_AC019322.4', 'ENSG00000279444_AC135584.1', 'ENSG00000279486_OR2AG1', 'ENSG00000279530_AC092881.1', 'ENSG00000279590_AC005786.4', 'ENSG00000279619_AC020907.5', 'ENSG00000279633_AL137918.1', 'ENSG00000279636_LINC00216', 'ENSG00000279672_AP006621.5', 'ENSG00000279690_AP000280.1', 'ENSG00000279727_LINC02033', 'ENSG00000279861_AC073548.1', 'ENSG00000279913_AP001962.1', 'ENSG00000279970_AC023024.2', 'ENSG00000280055_TMEM75', 'ENSG00000280057_AL022069.2', 'ENSG00000280135_AL096816.1', 'ENSG00000280310_AC092437.1', 'ENSG00000280422_AC115284.2', 'ENSG00000280432_AP000962.2', 'ENSG00000280693_SH3PXD2A-AS1', 'ENSG00000281490_CICP14', 'ENSG00000281530_AC004461.2', 'ENSG00000281571_AC241585.2', 'ENSG00000282772_AL358790.1', 'ENSG00000282989_AP001206.1', 'ENSG00000282996_AC022021.1', 'ENSG00000283023_FRG1GP', 'ENSG00000283031_AC009242.1', 'ENSG00000283097_AL159152.1', 'ENSG00000283141_AL157832.3', 'ENSG00000283209_AC106858.1', 'ENSG00000283538_AC005972.3', 'ENSG00000284240_AC099062.1', 'ENSG00000284512_AC092718.8', 'ENSG00000284657_AL031432.5', 'ENSG00000284664_AL161756.3', 'ENSG00000284931_AC104389.5', 'ENSG00000285016_AC017002.6', 'ENSG00000285117_AC068724.4', 'ENSG00000285162_AC004593.3', 'ENSG00000285210_AL136382.1', 'ENSG00000285215_AC241377.4', 'ENSG00000285292_AC021097.2', 'ENSG00000285498_AC104389.6', 'ENSG00000285534_AL163541.1', 'ENSG00000285577_AC019127.1', 'ENSG00000285611_AC007132.1', 'ENSG00000285629_AL031847.2', 'ENSG00000285641_AL358472.6', 'ENSG00000285649_AL357079.2', 'ENSG00000285650_AL157827.2', 'ENSG00000285662_AL731733.1', 'ENSG00000285672_AL160396.2', 'ENSG00000285763_AL358777.1', 'ENSG00000285865_AC010285.3', 'ENSG00000285879_AC018628.2']
print('Constant cols:', len(constant_cols))

# important_cols = []
# for y_col in Y.columns:
#     important_cols += [x_col for x_col in X.columns if y_col in x_col]
# print(important_cols)
important_cols = ['ENSG00000114013_CD86', 'ENSG00000120217_CD274', 'ENSG00000196776_CD47', 'ENSG00000117091_CD48', 'ENSG00000101017_CD40', 'ENSG00000102245_CD40LG', 'ENSG00000169442_CD52', 'ENSG00000117528_ABCD3', 'ENSG00000168014_C2CD3', 'ENSG00000167851_CD300A', 'ENSG00000167850_CD300C', 'ENSG00000186407_CD300E', 'ENSG00000178789_CD300LB', 'ENSG00000186074_CD300LF', 'ENSG00000241399_CD302', 'ENSG00000167775_CD320', 'ENSG00000105383_CD33', 'ENSG00000174059_CD34', 'ENSG00000135218_CD36', 'ENSG00000104894_CD37', 'ENSG00000004468_CD38', 'ENSG00000167286_CD3D', 'ENSG00000198851_CD3E', 'ENSG00000117877_CD3EAP', 'ENSG00000074696_HACD3', 'ENSG00000015676_NUDCD3', 'ENSG00000161714_PLCD3', 'ENSG00000132300_PTCD3', 'ENSG00000082014_SMARCD3', 'ENSG00000121594_CD80', 'ENSG00000110651_CD81', 'ENSG00000238184_CD81-AS1', 'ENSG00000085117_CD82', 'ENSG00000112149_CD83', 'ENSG00000066294_CD84', 'ENSG00000114013_CD86', 'ENSG00000172116_CD8B', 'ENSG00000254126_CD8B2', 'ENSG00000177455_CD19', 'ENSG00000105383_CD33', 'ENSG00000173762_CD7', 'ENSG00000125726_CD70', 'ENSG00000137101_CD72', 'ENSG00000019582_CD74', 'ENSG00000105369_CD79A', 'ENSG00000007312_CD79B', 'ENSG00000090470_PDCD7', 'ENSG00000119688_ABCD4', 'ENSG00000010610_CD4', 'ENSG00000101017_CD40', 'ENSG00000102245_CD40LG', 'ENSG00000026508_CD44', 'ENSG00000117335_CD46', 'ENSG00000196776_CD47', 'ENSG00000117091_CD48', 'ENSG00000188921_HACD4', 'ENSG00000150593_PDCD4', 'ENSG00000203497_PDCD4-AS1', 'ENSG00000115556_PLCD4', 'ENSG00000026508_CD44', 'ENSG00000170458_CD14', 'ENSG00000117281_CD160', 'ENSG00000177575_CD163', 'ENSG00000135535_CD164', 'ENSG00000091972_CD200', 'ENSG00000163606_CD200R1', 'ENSG00000206531_CD200R1L', 'ENSG00000182685_BRICD5', 'ENSG00000111731_C2CD5', 'ENSG00000169442_CD52', 'ENSG00000143119_CD53', 'ENSG00000196352_CD55', 'ENSG00000116815_CD58', 'ENSG00000085063_CD59', 'ENSG00000105185_PDCD5', 'ENSG00000255909_PDCD5P1', 'ENSG00000145284_SCD5', 'ENSG00000167775_CD320', 'ENSG00000110848_CD69', 'ENSG00000139187_KLRG1', 'ENSG00000139193_CD27', 'ENSG00000215039_CD27-AS1', 'ENSG00000120217_CD274', 'ENSG00000103855_CD276', 'ENSG00000204287_HLA-DRA', 'ENSG00000196126_HLA-DRB1', 'ENSG00000198502_HLA-DRB5', 'ENSG00000229391_HLA-DRB6', 'ENSG00000116815_CD58', 'ENSG00000168329_CX3CR1', 'ENSG00000272398_CD24', 'ENSG00000122223_CD244', 'ENSG00000198821_CD247', 'ENSG00000122223_CD244', 'ENSG00000177575_CD163', 'ENSG00000112149_CD83', 'ENSG00000185963_BICD2', 'ENSG00000157617_C2CD2', 'ENSG00000172375_C2CD2L', 'ENSG00000116824_CD2', 'ENSG00000091972_CD200', 'ENSG00000163606_CD200R1', 'ENSG00000206531_CD200R1L', 'ENSG00000012124_CD22', 'ENSG00000150637_CD226', 'ENSG00000272398_CD24', 'ENSG00000122223_CD244', 'ENSG00000198821_CD247', 'ENSG00000139193_CD27', 'ENSG00000215039_CD27-AS1', 'ENSG00000120217_CD274', 'ENSG00000103855_CD276', 'ENSG00000198087_CD2AP', 'ENSG00000169217_CD2BP2', 'ENSG00000144554_FANCD2', 'ENSG00000206527_HACD2', 'ENSG00000170584_NUDCD2', 'ENSG00000071994_PDCD2', 'ENSG00000126249_PDCD2L', 'ENSG00000049883_PTCD2', 'ENSG00000186193_SAPCD2', 'ENSG00000108604_SMARCD2', 'ENSG00000185561_TLCD2', 'ENSG00000075035_WSCD2', 'ENSG00000150637_CD226', 'ENSG00000110651_CD81', 'ENSG00000238184_CD81-AS1', 'ENSG00000134061_CD180', 'ENSG00000004468_CD38', 'ENSG00000012124_CD22', 'ENSG00000150637_CD226', 'ENSG00000135404_CD63', 'ENSG00000135218_CD36', 'ENSG00000137101_CD72', 'ENSG00000125810_CD93', 'ENSG00000010278_CD9', 'ENSG00000125810_CD93', 'ENSG00000153283_CD96', 'ENSG00000002586_CD99', 'ENSG00000102181_CD99L2', 'ENSG00000223773_CD99P1', 'ENSG00000204592_HLA-E', 'ENSG00000085117_CD82', 'ENSG00000134256_CD101']
print('Important cols:', len(important_cols))


# We read train and test datasets, keep the important columns and convert the rest to sparse matrices.

# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Read train and convert to sparse matrix\nX = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=constant_cols)\ncell_index = X.index\nmeta = metadata_df.reindex(cell_index)\nX0 = X[important_cols].values\nprint(f"Original X shape: {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")\ngc.collect()\nX = scipy.sparse.csr_matrix(X.values)\ngc.collect()\n\n# Read test and convert to sparse matrix\nXt = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=constant_cols)\ncell_index_test = Xt.index\nmeta_test = metadata_df.reindex(cell_index_test)\nX0t = Xt[important_cols].values\nprint(f"Original Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")\ngc.collect()\nXt = scipy.sparse.csr_matrix(Xt.values)')


# We apply the truncated SVD to train and test together. The truncated SVD can take an hour, but it is memory-efficient. We concatenate the SVD output with the important features and get the arrays `X` and `Xt`, which will be the input to the LightGBM model. 

# In[7]:


get_ipython().run_cell_magic('time', '', '\n# Apply the singular value decomposition\nboth = scipy.sparse.vstack([X, Xt])\nassert both.shape[0] == 119651\nprint(f"Shape of both before SVD: {both.shape}")\nsvd = TruncatedSVD(n_components=512, random_state=1) # 512\nboth = svd.fit_transform(both)\nprint(f"Shape of both after SVD:  {both.shape}")\n\n# Hstack the svd output with the important features\nX = both[:70988]\nXt = both[70988:]\ndel both\nX = np.hstack([X, X0])\nXt = np.hstack([Xt, X0t])\nprint(f"Reduced X shape:  {str(X.shape):14} {X.size*4/1024/1024/1024:2.3f} GByte")\nprint(f"Reduced Xt shape: {str(Xt.shape):14} {Xt.size*4/1024/1024/1024:2.3f} GByte")')


# Finally, we read the target array `Y`:

# In[8]:


Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
y_columns = list(Y.columns)
Y = Y.values

print(f"Y shape: {str(Y.shape):14} {Y.size*4/1024/1024/1024:2.3f} GByte")


# # LightGBM parameters

# In[9]:


lightgbm_params = {
     'learning_rate': 0.1, 
     'max_depth': 10, 
     'num_leaves': 200,
     'min_child_samples': 250,
     'colsample_bytree': 0.8, 
     'subsample': 0.6, 
     "seed": 1,
    }


# # Cross-validation
# 
# For cross-validation, we create three folds. In every fold, we train on the data of two donors and predict the third one. This scheme mimics the situation of the public leaderboard, where we train on three donors and predict the fourth one (see [EDA](https://www.kaggle.com/ambrosm/msci-eda-which-makes-sense)). 
# 
# As we want to predict 140 targets, we fit 140 LightGBM models in a loop. We could use `sklearn.multioutput.MultiOutputRegressor` for this purpose, but an explicit loop is more flexible.
# 
# The cross-validation takes some time. If you want to save time, you have three options:
# - Run only one fold of three (uncomment the break at the end of the loop)
# - Predict only a subset of the 140 targets (set `y_cols` to a low value, such as 3 or 10)
# - Skip the cross-validation completely by setting `CROSS_VALIDATE` to False

# In[10]:


get_ipython().run_cell_magic('time', '', '# Cross-validation with LGBMRegressor in a loop\n\nif CROSS_VALIDATE:\n    y_cols = Y.shape[1] # set this to a small number for a quick test\n    n_estimators = 300\n\n    kf = GroupKFold(n_splits=3)\n    score_list = []\n    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, groups=meta.donor)):\n        model = None\n        gc.collect()\n        X_tr = X[idx_tr]\n        y_tr = Y[:,:y_cols][idx_tr]\n        X_va = X[idx_va]\n        y_va = Y[:,:y_cols][idx_va]\n\n        models, va_preds = [], []\n        for i in range(y_cols):\n            #print(f"Training column {i:3} for validation")\n            model = lightgbm.LGBMRegressor(n_estimators=n_estimators, **lightgbm_params)\n            # models.append(model) # not needed\n            model.fit(X_tr, y_tr[:,i].copy())\n            va_preds.append(model.predict(X_va))\n        y_va_pred = np.column_stack(va_preds) # concatenate the 140 predictions\n        del va_preds\n\n        del X_tr, y_tr, X_va\n        gc.collect()\n\n        # We validate the model (mse and correlation over all 140 columns)\n        mse = mean_squared_error(y_va, y_va_pred)\n        corrscore = correlation_score(y_va, y_va_pred)\n        \n        del y_va\n\n        print(f"Fold {fold} {X.shape[1]:4}: mse = {mse:.5f}, corr =  {corrscore:.5f}")\n        score_list.append((mse, corrscore))\n        break # We only need the first fold\n\n    if len(score_list) > 1:\n        # Show overall score\n        result_df = pd.DataFrame(score_list, columns=[\'mse\', \'corrscore\'])\n        print(f"{Fore.GREEN}{Style.BRIGHT}Average LGBM mse = {result_df.mse.mean():.5f}; corr = {result_df.corrscore.mean():.5f}{Style.RESET_ALL}")')


# # Retraining
# 
# We retrain the model on all training rows and compute the predictions.

# In[11]:


if SUBMIT:
    te_preds = []
    n_estimators = 300
    y_cols = Y.shape[1]
    for i in range(y_cols):
        #print(f"Training column {i:3} for test")
        model = lightgbm.LGBMRegressor(n_estimators=n_estimators,
                                       **lightgbm_params
                                      )
        model.fit(X, Y[:,i].copy())
        te_preds.append(model.predict(Xt))
    test_pred = np.column_stack(te_preds)
    del te_preds

    print(f"test_pred shape: {str(test_pred.shape):14}")


# # The data leak
# 
# It has been pointed out in several discussion posts that the first 7476 rows of test are identical to the first 7476 rows of train:
# - [CITEseq data: same RNA expression matrices from different donors in day2?](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349867) (@gwentea)
# -[Data contamination between CITEseq train/test datasets?](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349833) (@aglaros)
# - [Leak in public test set](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349867) (@psilogram)
# 
# These rows belong to the public test set; the private leaderboard is not affected. We copy the 7476 rows from the training targets into the test predictions:

# In[12]:


test_pred[:7476] = Y[:7476]


# # Submission
# 
# The CITEseq test predictions have 48663 rows (i.e., cells) and 140 columns (i.e. proteins). 48663 * 140 = 6812820. The final submission will have 65744180 rows, of which the first 6812820 are for the CITEseq predictions and the remaining 58931360 for the Multiome predictions. 
# 
# We now read the Multiome predictions from Fabien Crom's notebook and merge the CITEseq predictions into them:

# In[13]:


if SUBMIT:
    #with open("../input/msci-multiome-quickstart/partial_submission_multi.pickle", 'rb') as f: submission = pickle.load(f)
    submission = pd.read_csv('../input/msci-multiome-quickstart-w-sparse-matrices/submission.csv',
                             index_col='row_id', squeeze=True)
    submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
    assert not submission.isna().any()
    submission.to_csv('submission.csv')
    display(submission)
    


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # MSCI - CITEseq - TF / Keras Baseline
# 
# Simple keras nn baseline that I intend to improve over time to match competitive models.
# 
# Now with the multiome part: https://www.kaggle.com/code/lucasmorin/msci-multiome-tf-keras-nn-baseline

# # Imports
# 
# Import base libraries, graphic libraries and modelling librairies (sklearn for Cross-validation, TF/Keras for modelling).

# In[1]:


import numpy as np, pandas as pd
import glob, os, gc

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt, seaborn as sns

from sklearn import preprocessing, model_selection
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import tensorflow_probability as tfp

#set backend as float16 
K.set_floatx('float16')
tf.keras.mixed_precision.set_global_policy('mixed_float16')

DEBUG = True
TEST = False


# (needed tor ead hdf files)

# In[2]:


get_ipython().system('pip install tables')


# # read data
# 
# Reading data, adding celltype as categorical integer; perform 10% sampling if DEBUG mode is enabled.

# In[3]:


get_ipython().run_cell_magic('time', '', '\ntrain = pd.read_hdf("/kaggle/input/open-problems-multimodal/train_cite_inputs.h5").astype(\'float16\')\n\nmeta_data = pd.read_csv(\'../input/open-problems-multimodal/metadata.csv\')\ntrain_meta_data = meta_data.set_index(\'cell_id\').loc[train.index]\n\ntrain = train.values\ntrain_cat =  train_meta_data.cell_type.values\n\nlabels = pd.read_hdf("/kaggle/input/open-problems-multimodal/train_cite_targets.h5").astype(\'float16\').values')


# In[4]:


map_cat = { 'BP':0, 'EryP':1, 'HSC':2, 'MasP':3, 'MkP':4, 'MoP':5, 'NeuP':6 }
train_cat = np.array([map_cat[t] for t in train_cat])


# In[5]:


if DEBUG:
    idx = np.random.randint(0, train.shape[0], int(train.shape[0]/10))
    train = train[idx]
    train_cat = train_cat[idx]
    labels = labels[idx]


# # Custom Loss
# 
# I implemented the needed correlation as a custom metric. To have a decreasing loss, the standard approach is to consider 1- corr instead of corr. 
# Using only 1-corr as a metric is problematic as their might be a problem with scale. As the metric is independant of scale, the scale of the output can drift uncontrollably an cause overflow errors (exacerbated by the usage of float16). One solution is to add a bit of MSE loss. The final loss is 1 - corr + lambda * MSE where lambda is a small hand-tuned hyper-parameter.

# In[6]:


lam = 0.03

def correlation_metric(y_true, y_pred):
    x = tf.convert_to_tensor(y_true)
    y = tf.convert_to_tensor(y_pred)
    mx = K.mean(x,axis=1)
    my = K.mean(y,axis=1)
    mx = tf.tile(tf.expand_dims(mx,axis=1),(1,x.shape[1]))
    my = tf.tile(tf.expand_dims(my,axis=1),(1,x.shape[1]))
    xm, ym = (x-mx)/100, (y-my)/100
    r_num = K.sum(tf.multiply(xm,ym),axis=1)
    r_den = tf.sqrt(tf.multiply(K.sum(K.square(xm),axis=1), K.sum(K.square(ym),axis=1)))
    r = tf.reduce_mean(r_num / r_den)
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def correlation_loss(y_true, y_pred):
    return 1 - correlation_metric(y_true, y_pred) + lam * tf.keras.losses.MeanSquaredError()(tf.convert_to_tensor(y_true),tf.convert_to_tensor(y_pred))


# # Model

# I start with a very vanilla MLP; I try to add a cell-type embedding layer. 
# To avoid too much drift, I scale each layer with batchnorm.
# I also add some noise to make the learning more robust.
# I initially chose 'relu' as the activation function that seems well suited to handle sparse data; 'selu' is usually better than 'relu'.

# In[7]:


hidden_units = (256,128,64)
cell_embedding_size = 2
noise = 0.1

def base_model():
    
    num_input = keras.Input(shape=(train.shape[1],), name='num_data')
    
    cat_input = keras.Input(shape=(1,), name='cell_id')

    cell_embedded = keras.layers.Embedding(8, cell_embedding_size, input_length=1)(cat_input)
    cell_flattened = keras.layers.Flatten()(cell_embedded)
    
    out = keras.layers.Concatenate()([cell_flattened, num_input])

    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.GaussianNoise(noise)(out)
    
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='selu', kernel_regularizer = tf.keras.regularizers.L2(l2=0.01))(out)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.GaussianNoise(noise)(out)
        
    out = keras.layers.Dense(labels.shape[1], activation='selu', name='prediction')(out)

    model = keras.Model(
        inputs = [num_input, cat_input],
        outputs = out,
    )
    
    return model


# # Training
# 
# General training loop; Data is split accordingly to CV. Then I train the model with some basic callbacks. 
# Then the model is evaluated out of sample (we can check that the tf corr metric match the numpy implementation).

# In[8]:


gc.collect()

epochs = 3 if DEBUG else 1000
n_folds = 2 if DEBUG else (2 if TEST else 3)
n_seeds = 2 if DEBUG else (2 if TEST else 3)

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_correlation_metric', min_delta=1e-05, patience=5, verbose=1,
    mode='max', restore_best_weights = True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_correlation_metric', factor=0.2, patience=3, verbose=1,
    mode='max')

kf = model_selection.ShuffleSplit(n_splits=n_folds, random_state=2020, test_size = 0.4)

df_scores = []

for fold, (cal_index, val_index) in enumerate(kf.split(range(len(train)))):
    print(f'CV {fold}/{n_folds}')
    
    X_train = train[cal_index, :]
    X_train_cat = train_cat[cal_index]
    y_train = labels[cal_index, :]
    
    X_test = train[val_index, :]
    X_test_cat = train_cat[val_index]
    y_test = labels[val_index, :]
    
    
    for seed in range(n_seeds):
        print(f'Fold: {str(fold)} - seed: {str(seed)}')
        key = str(fold)+'-'+str(seed)
    
        model = base_model()

        model.compile(
            keras.optimizers.Adam(learning_rate=1e-4),
            loss = correlation_loss,
            metrics = correlation_metric,
        )

        model.fit([X_train,X_train_cat], 
                  y_train, 
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([X_test,X_test_cat], y_test),
                  callbacks=[es, plateau],
                  shuffle=True,
                  verbose = 1)

        output_test = model.predict([X_test, X_test_cat])
        score = np.mean([np.corrcoef(y_test[i],output_test[i])[0,1] for i in range(len(y_test))])
        print(f'Fold: {str(fold)} - seed: {str(seed)}: {score:.2%}')

        df_scores.append((fold, seed, score))
        model.save(f'model_cite_nn_{key}')
    
    tf.keras.backend.clear_session()
    del  X_train, X_train_cat, y_train, X_test, X_test_cat, y_test
    gc.collect()


# In[9]:


del train, labels
gc.collect


# # Results

# In[10]:


df_results = pd.DataFrame(df_scores,columns=['fold','seed','score']).pivot(index='fold',columns='seed',values='score')

df_results.loc['seed_mean']= df_results.mean(numeric_only=True, axis=0)
df_results.loc[:,'fold_mean'] = df_results.mean(numeric_only=True, axis=1)
df_results


# # Submission
# 
# Loading and preparing test data. Inference on test data. Constitution of the first part of the submission.

# In[11]:


get_ipython().run_cell_magic('time', '', "\nevaluation_ids = pd.read_csv('../input/open-problems-multimodal/evaluation_ids.csv').set_index('row_id')\nunique_ids = np.unique(evaluation_ids.cell_id)\nsubmission = pd.Series(name='target', index=pd.MultiIndex.from_frame(evaluation_ids), dtype=np.float16)\n\ndel evaluation_ids\ngc.collect()")


# In[12]:


get_ipython().run_cell_magic('time', '', '\ntest = pd.read_hdf("/kaggle/input/open-problems-multimodal/test_cite_inputs.h5").astype(\'float16\')\nmeta_data = pd.read_csv(\'../input/open-problems-multimodal/metadata.csv\')\ntest_meta_data = meta_data.set_index(\'cell_id\').loc[test.index]\n\ntest = test.values\ntest_cat =  test_meta_data.cell_type.values\n\nmap_cat = { \'BP\':0, \'EryP\':1, \'HSC\':2, \'MasP\':3, \'MkP\':4, \'MoP\':5, \'NeuP\':6}\ntest_cat = np.array([map_cat[t] for t in test_cat])')


# In[13]:


gc.collect()

all_preds = []

for fold in range(n_folds):
    for seed in range(n_seeds):
        print(f'Preds - Fold: {str(fold)} - seed: {str(seed)}')
        key = str(fold)+'-'+str(seed)
        
        model_cite = tf.keras.models.load_model(f'./model_cite_nn_{key}/', compile=False)

        cite_pred = model_cite.predict([test, test_cat])
        cite_pred = cite_pred.ravel()
        len_cite_raveled = len(cite_pred)
        all_preds.append(cite_pred)


# In[14]:


del test, test_cat, cite_pred
gc.collect()


# In[15]:


submission.iloc[:len_cite_raveled] = np.nanmean(np.array(all_preds),axis=0)


# In[16]:


submission.to_csv('submission_cite.csv')
submission.head()


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # EDA for the Multimodal Single-Cell Integration Competition

# In[1]:


import os, gc, scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style
from sklearn.decomposition import TruncatedSVD

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


# A little trick to save time with pip: On the first run of the notebook, we need to install the `tables` module with pip. If the module is already installed (after a restart of the notebook, for instance), pip wastes 10 seconds by checking whether a newer version exists. We can skip this check by testing for the presence of the module in a simple if statement.

# In[2]:


get_ipython().run_cell_magic('time', '', '# If you see a warning "Failed to establish a new connection" running this cell,\n# go to "Settings" on the right hand side, \n# and turn on internet. Note, you need to be phone verified.\n# We need this library to read HDF files.\nif not os.path.exists(\'/opt/conda/lib/python3.7/site-packages/tables\'):\n    !pip install --quiet tables')


# # The metadata table
# 
# The metadata table (which describes training and test data) shows us:
# - There is data about 281528 unique cells.
# - The cells belong to five days, four donors, eight cell types (including one type named 'hidden'), and two technologies.
# - The metadata table has no missing values.
# 
# **Insight:** 
# - Every cell is used only on a single day and then discarded. There are no time series over single cells.
# - The two technologies do not share cells. It looks like we may create two completely independent models, one per technology, even if they share the same four donors. It's two Kaggle competitions in one!
# - As the models are independent, it is a good idea to work with two separate notebooks, one for CITEseq, the other one for Multiome.
# - Donor and cell_type are categorical features, which can be one-hot encoded. 

# In[3]:


df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
display(df_meta)
if not df_meta.index.duplicated().any(): print('All cell_ids are unique.')
if not df_meta.isna().any().any(): print('There are no missing values.')
    


# In[4]:


_, axs = plt.subplots(2, 2, figsize=(11, 6))
for col, ax in zip(['day', 'donor', 'cell_type', 'technology'], axs.ravel()):
    vc = df_meta[col].astype(str).value_counts()
    if col == 'day':
        vc.sort_index(key = lambda x : x.astype(int), ascending=False, inplace=True)
    else:
        vc.sort_index(ascending=False, inplace=True)
    ax.barh(vc.index, vc, color=['MediumSeaGreen'])
    ax.set_ylabel(col)
    ax.set_xlabel('# cells')
plt.tight_layout(h_pad=4, w_pad=4)
plt.suptitle('Metadata distribution', y=1.04, fontsize=20)
plt.show()


# The CITEseq measurements took place on four days, the Multiome measurements on five (except that there are no measurements for donor 27678 on day 4. For every combination of day, donor and technology, there are around 8000 cells:

# In[5]:


# From https://www.kaggle.com/code/peterholderrieth/getting-started-data-loading
df_meta_cite = df_meta[df_meta.technology=="citeseq"]
df_meta_multi = df_meta[df_meta.technology=="multiome"]

fig, axs = plt.subplots(1,2,figsize=(12,6))
df_cite_cell_dist = df_meta_cite[["day","donor"]].value_counts().to_frame()                .sort_values("day").reset_index()                .rename(columns={0:"# cells"})
sns.barplot(data=df_cite_cell_dist, x="day",hue="donor",y="# cells", ax=axs[0])
axs[0].set_title(f"{len(df_meta_cite)} cells measured with CITEseq")

df_multi_cell_dist = df_meta_multi[["day","donor"]].value_counts().to_frame()                .sort_values("day").reset_index()                .rename(columns={0:"# cells"})
sns.barplot(data=df_multi_cell_dist, x="day",hue="donor",y="# cells", ax=axs[1])
axs[1].set_title(f"{len(df_meta_multi)} cells measured with Multiome")
plt.suptitle('# Cells per day, donor and technology', y=1.04, fontsize=20)
plt.show()
print('Average:', round(len(df_meta) / 35))


# A diagram (taken from the [competition homepage](https://www.kaggle.com/competitions/open-problems-multimodal/data)) illustrates the relationships:
# - In the CITEseq competition, we train on data for three donors and three days. For the public leaderboard, we predict the fourth donor; for the private leaderboard, we predict another day for all donors.
# - In the Multiome competition, we train on three donors and four days. For the public leaderboard, we predict the fourth donor; for the private leaderboard, we predict a fifth day.
# 
# **Insight:** As the data is grouped and we have to predict unseen groups (the last donor or the last day), we might choose `GroupKFold` as cross-validation scheme.
# 
# ![Diagram](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4308072%2F23e8c1f6faea1453998544cdc116a20e%2FNeurIPS%202022%20-%20Frame%204.jpg?generation=1660755395301873&alt=media)

# # CITEseq inputs
# 
# We start by looking at CITEseq, which has the smaller datafiles than Multiome and is more tractable.
# 
# The CITEseq input files contain 70988 samples (i.e., cells) for train and 48663 samples for test. 70988 + 48663 = 119651, which matches the number of rows in the CITEseq metadata table. No values are missing.
# 
# The input data corresponds to RNA expression levels for 22050 genes (there are 22050 columns).
# 
# The data have dtype float32, which means we need 119651 * 22050 * 4 = 10553218200 bytes = 10.6 GByte of RAM just for the features (train and test) without the targets. 
# 
# Originally, these RNA expression levels were counts (i.e., nonnegative integers), but they have been normalized and log1p-transformed. With the log1p transformation, the data remain nonnegative.
# 
# Most columns have a minimum of zero, which means that for most genes there are cells which didn't express this gene (the count was 0). In fact, 78 % of all table entries are zero, and for some columns, the count is always zero.
# 
# **Insight:**
# - This is big data. Make sure you don't waste RAM and use efficient algorithms.
# - Perhaps we should first load only the training data into RAM, fit one or more models, and then delete the training data from RAM before loading the test data.
# - The columns which are zero for every cell should be dropped before modeling.

# In[6]:


get_ipython().run_cell_magic('time', '', '# Analyze train and test features\ndf_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS)\ndisplay(df_cite_train_x.head())\nprint(\'Shape:\', df_cite_train_x.shape)\nprint("Missing values:", df_cite_train_x.isna().sum().sum())\nprint("Genes which never occur in train:", (df_cite_train_x == 0).all(axis=0).sum())\nprint(f"Zero entries in train: {(df_cite_train_x == 0).sum().sum() / df_cite_train_x.size:.0%}")\ncite_gene_names = list(df_cite_train_x.columns)')


# The distribution of the zeros can be visualized with the pyplot `spy()` function. This function plots a black dot for every nonzero entry of the array. The resulting image shows us that the differences between the columns are substantial: some columns are almost white (i.e., they contain mostly zeros), others are dark (i.e., they contain a lot of nonzero values).
# 
# The rows of the matrix look homogeneous.
# 
# **Insight:** Maybe we can exploit the column differences for feature selection.

# In[7]:


plt.figure(figsize=(10, 4))
plt.spy(df_cite_train_x[:5000])
plt.show()


# The histogram shows some artefacts because the data originally were integers. We don't show the zeros in the histogram, because with 78 % zeros, the histogram would have such a high peak at zero that we couldn't see anything else.
# 
# **Insight:** The feature values are either 0 or between 2.9 and 12 - the distribution is far away from normal. If we apply statistical tests, this fact has to be taken into consideration.

# In[8]:


get_ipython().run_cell_magic('time', '', 'nonzeros = df_cite_train_x.values.ravel()\nnonzeros = nonzeros[nonzeros != 0] # comment this line if you want to see the peak at zero\nplt.figure(figsize=(16, 4))\nplt.gca().set_facecolor(\'#0057b8\')\nplt.hist(nonzeros, bins=500, density=True, color=\'#ffd700\')\nprint(\'Minimum nonzero value:\', nonzeros.min())\ndel nonzeros\nplt.title("Histogram of nonzero RNA expression levels in train")\nplt.xlabel("log1p-transformed expression count")\nplt.ylabel("density")\nplt.show()')


# In[9]:


_, axs = plt.subplots(5, 4, figsize=(16, 16))
for col, ax in zip(df_cite_train_x.columns[:20], axs.ravel()):
    nonzeros = df_cite_train_x[col].values
    nonzeros = nonzeros[nonzeros != 0] # comment this line if you want to see the peak at zero
    ax.hist(nonzeros, bins=100, density=True)
    ax.set_title(col)
plt.tight_layout(h_pad=2)
plt.suptitle('Histograms of nonzero RNA expression levels for selected features', fontsize=20, y=1.04)
plt.show()
del nonzeros


# As we have seen that most entries of df_cite_train_x are zero and memory is scarce, we convert the numpy array to a [compressed sparse row](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) (CSR) matrix:

# In[10]:


cell_index = df_cite_train_x.index
meta = df_meta_cite.reindex(cell_index)
gc.collect()
df_cite_train_x = scipy.sparse.csr_matrix(df_cite_train_x.values)


# We have freed enough memory to analyze the test data; afterwards we convert the test data to a CSR matrix as well.

# In[11]:


df_cite_test_x = pd.read_hdf(FP_CITE_TEST_INPUTS)
print('Shape of CITEseq test:', df_cite_test_x.shape)
print("Missing values:", df_cite_test_x.isna().sum().sum())
print("Genes which never occur in test: ", (df_cite_test_x == 0).all(axis=0).sum())
print(f"Zero entries in test:  {(df_cite_test_x == 0).sum().sum() / df_cite_test_x.size:.0%}")


gc.collect()
cell_index_test = df_cite_test_x.index
meta_test = df_meta_cite.reindex(cell_index_test)
df_cite_test_x = scipy.sparse.csr_matrix(df_cite_test_x.values)


# # The data leak
# 
# It has been pointed out in several discussion posts (the first one was [CITEseq data: same RNA expression matrices from different donors in day2?](https://www.kaggle.com/competitions/open-problems-multimodal/discussion/349867) by @gwentea) that the first 7476 rows of test (day 2, donor 27678) are identical to the first 7476 rows of train (day 2, donor 32606):
# 

# In[12]:


print('Data leak:', (df_cite_train_x[:7476] == df_cite_test_x[:7476]).toarray().all())


# **Insight:**
# - Some mistake happened when the data was prepared; it can't be that 7476 cells of one donor return exactly the same measurements as 7476 cells of another donor.
# - These rows belong to the public test set; the private test set is not affected.
# - To get a good public leaderboard score, we'll copy the 7476 rows from the training targets into the test predictions.

# # The distributions of train and test in feature space
# 
# For the next diagrams, we need to project the data (train and test together) to two dimensions. Usually I'd do that with a PCA, but the scikit-learn PCA implementation needs too much memory. Fortunately, TruncatedSVD does a similar projection and uses much less memory:

# In[13]:


# Concatenate train and test for the SVD
both = scipy.sparse.vstack([df_cite_train_x, df_cite_test_x])
print(f"Shape of both before SVD: {both.shape}")

# Project to two dimensions
svd = TruncatedSVD(n_components=2, random_state=1)
both = svd.fit_transform(both)
print(f"Shape of both after SVD:  {both.shape}")

# Separate train and test
X = both[:df_cite_train_x.shape[0]]
Xt = both[df_cite_train_x.shape[0]:]


# The scatterplots below show the extent of the data for every day and every donor (SVD projection to two dimensions). The nine diagrams with orange dots make up the training data. The three diagrams with orange-red dots below are the public test set and the four dark red diagrams at the right are the private test set.
# 
# The gray area marks the complete training data (union of the nine orange diagrams), and the black area behind marks the test data.
# 
# We see that the distributions differ. In particular the day 2 distribution (left column of diagrams) is much less wide than the others.
# 
# **Insight:**
# - For the public test we predict a previously unseen donor, for the private test a previously unseen day. This situation suggests that we use a GroupKFold for cross-validation, although its unclear whether we should group on day, donor, or both. 
# - Every small diagram consists of only 8000 samples and the data is noisy. It will be difficult not to overfit.
# - As the public test set is three times smaller than the training dataset, we must not rely on the public leaderboard to evaluate models. 
# - Day 7 (private test) covers areas of the feature space which occur neither in train nor in public test. We'll need a model which can extrapolate to this area, and the public leaderboard will give us no clue about the quality of this extrapolation.
# - The diagrams confirm the data leak described above: The lower two diagrams of day 2 are identical.

# In[14]:


# Scatterplot for every day and donor
_, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12, 11))
for donor, axrow in zip([13176, 31800, 32606, 27678], axs):
    for day, ax in zip([2, 3, 4, 7], axrow):
        ax.scatter(Xt[:,0], Xt[:,1], s=1, c='k')
        ax.scatter(X[:,0], X[:,1], s=1, c='lightgray')
        if day != 7 and donor != 27678: # train
            temp = X[(meta.donor == donor) & (meta.day == day)]
            ax.scatter(temp[:,0], temp[:,1], s=1, c='orange')
        else: # test
            temp = Xt[(meta_test.donor == donor) & (meta_test.day == day)]
            ax.scatter(temp[:,0], temp[:,1], s=1, c='darkred' if day == 7 else 'orangered')
        ax.set_title(f'Donor {donor} day {day}')
        ax.set_aspect('equal')
plt.suptitle('CITEseq features, projected to the first two SVD components', y=0.95, fontsize=20)
plt.show()


# In[15]:


df_cite_train_x, df_cite_test_x, X, Xt = None, None, None, None # release the memory


# # CITEseq targets (surface protein levels)
# 
# The CITEseq output (target) file is much smaller - it has 70988 rows like the training input file, but only 140 columns. The 140 columns correspond to 140 proteins.
# 
# The targets are dsb-normalized surface protein levels. We plot the histograms of a few selected columns and see that the distributions vary: some columns are normally distributed, some columns are multimodal, some have other shapes, and there seem to be outliers.
# 
# **Insight:**
# - This is a multi-output regression problem with 140 outputs. We won't be able to apply some standard methods of single-output regression - e.g., we can't compute the correlation between every feature and the target. 
# - As the targets are so diverse, a one-size-fits-all approach might not give the best results.

# In[16]:


df_cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
display(df_cite_train_y.head())
print('Output shape:', df_cite_train_y.shape)

_, axs = plt.subplots(5, 4, figsize=(16, 16))
for col, ax in zip(['CD86', 'CD270', 'CD48', 'CD8', 'CD7', 'CD14', 'CD62L', 'CD54', 'CD42b', 'CD2', 'CD18', 'CD36', 'CD328', 'CD224', 'CD35', 'CD57', 'TCRVd2', 'HLA-E', 'CD82', 'CD101'], axs.ravel()):
    ax.hist(df_cite_train_y[col], bins=100, density=True)
    ax.set_title(col)
plt.tight_layout(h_pad=2)
plt.suptitle('Selected target histograms (surface protein levels)', fontsize=20, y=1.04)
plt.show()

cite_protein_names = list(df_cite_train_y.columns)


# A projection of the CITEseq targets to two dimensions again shows that the groups have different distributions.

# In[17]:


svd = TruncatedSVD(n_components=2, random_state=1)
X = svd.fit_transform(df_cite_train_y)

# Scatterplot for every day and donor
_, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12, 11))
for donor, axrow in zip([13176, 31800, 32606, 27678], axs):
    for day, ax in zip([2, 3, 4, 7], axrow):
        if day != 7 and donor != 27678: # train
            ax.scatter(X[:,0], X[:,1], s=1, c='lightgray')
            temp = X[(meta.donor == donor) & (meta.day == day)]
            ax.scatter(temp[:,0], temp[:,1], s=1, c='orange')
        else: # test
            ax.text(50, -25, '?', fontsize=100, color='gray', ha='center')
        ax.set_title(f'Donor {donor} day {day}')
        ax.set_aspect('equal')
plt.suptitle('CITEseq target, projected to the first two SVD components', y=0.95, fontsize=20)
plt.show()


# In[18]:


df_cite_train_x, df_cite_train_y, X, svd = None, None, None, None # release the memory


# # Name matching
# 
# The CITEseq task has genes as input and proteins as output. Genes encode proteins, and it is more or less known which genes encode which proteins. This information is encoded in the column names: The input dataframe has the genes as column names, and the target dataframe has the proteins as column names. According to the naming convention, the gene names contain the protein name as suffix after a '_'. 
# 
# If we match the input column names with the target column names, we find 151 genes which encode a target protein (see the table below). It doesn't matter that some proteins are encoded by more than one gene (e.g., rows 146 and 147 of the table). We may assume that these 151 features will have a high feature importance in our models. 
# 
# **Insight:** If we apply dimensionality reduction (PCA, SVD, whatever) to the 22050 features, we should make sure that we don't reduce away the 151 features which encode the proteins we want to predict.

# In[19]:


matching_names = []
for protein in cite_protein_names:
    matching_names += [(gene, protein) for gene in cite_gene_names if protein in gene]
pd.DataFrame(matching_names, columns=['Gene', 'Protein'])


# # Multiome input
# 
# The Multiome dataset is much larger than the CITEseq part and way too large to fit into 16 GByte RAM:
# - train inputs:  105942 * 228942 float32 values (97 GByte)
# - train targets: 105942 *  23418 float32 values (10 GByte)
# - test inputs:    55935 * 228942 float32 values (13 GByte)
# 
# For this EDA, we read all the data to check for missing, zero and negative values, we plot a histogram and we look at the Y chromosome, but we don't analyze much more.
# 
# No values are missing.
# 
# The data consists of ATAC-seq peak counts transformed with [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html). They are all nonnegative. In the sample we are looking at, 98 % of the entries are zero, and thousands of columns are always zero.
# 
# **Insight:**
# - The Multiome data is even bigger data than the CITEseq data. Most of us can't afford notebooks with 97 GByte RAM.
# - Dimensionality reduction and feature selection will be key to get successful models.
# - Maybe we can do something with sparse array data structures.
# - Maybe we can train on a subset of the rows (and sacrifice some precision).
# - Maybe we can convert the data to float16.
# - We should never have training and test data in memory at the same time.
# - We may even want to look for algorithms which don't need all the training data in RAM at the same time. Neural networks are a candidate.
# - The columns which are zero for every cell should be dropped before modeling.
# - The training set has n_features > n_samples. We'll use algorithms which can deal with more features than samples.
# 

# In[20]:


get_ipython().run_cell_magic('time', '', 'bins = 100\n\ndef analyze_multiome_x(filename):\n    start = 0\n    chunksize = 5000\n    total_rows = 0\n    maximum_x = 0\n\n    while True:\n        X = pd.read_hdf(filename, start=start, stop=start+chunksize)\n        if X.isna().any().any(): print(\'There are missing values.\')\n        if (X < 0).any().any(): print(\'There are negative values.\')\n        total_rows += len(X)\n        print(total_rows, \'rows read\')\n\n        donors = df_meta_multi.donor.reindex(X.index) # metadata: donor of cell\n        chrY_cols = [f for f in X.columns if \'chrY\' in f]\n        maximum_x = max(maximum_x, X[chrY_cols].values.ravel().max())\n        for donor in [13176, 31800, 32606, 27678]:\n            hist, _ = np.histogram(X[chrY_cols][donors == donor].values.ravel(), bins=bins, range=(0, 15))\n            chrY_histo[donor] += hist\n\n        if len(X) < chunksize: break\n        start += chunksize\n    display(X.head(3))\n    print(f"Zero entries in {filename}: {(X == 0).sum().sum() / X.size:.0%}")\n\nchrY_histo = dict()\nfor donor in [13176, 31800, 32606, 27678]:\n    chrY_histo[donor] = np.zeros((bins, ), int)\n\n# Look at the training data\nanalyze_multiome_x(FP_MULTIOME_TRAIN_INPUTS)')


# In the histogram, we again hide the peak for the 98 % zero values:

# In[21]:


df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=0, stop=5000)
nonzeros = df_multi_train_x.values.ravel()
nonzeros = nonzeros[nonzeros != 0] # comment this line if you want to see the peak at zero
plt.figure(figsize=(16, 4))
plt.gca().set_facecolor('#0057b8')
plt.hist(nonzeros, bins=500, density=True, color='#ffd700')
del nonzeros
plt.title("Histogram of nonzero feature values (subset)")
plt.xlabel("TFIDF-transformed peak count")
plt.ylabel("density")
plt.show()

del df_multi_train_x # free the memory


# We do the same checks for the test data:

# In[22]:


get_ipython().run_cell_magic('time', '', '# Look at the test data\nanalyze_multiome_x(FP_MULTIOME_TEST_INPUTS)')


# # The Y chromosome
# 
# Let's plot a histogram of the nonzero values of the Y chromosome for every donor. 

# In[23]:


plt.rcParams['savefig.facecolor'] = "1.0"
_, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(14, 4))
for donor, ax in zip([13176, 31800, 32606, 27678], axs):
    ax.set_title(f"Donor {donor} {'(test)' if donor == 27678 else ''}", fontsize=16)
    total = chrY_histo[donor].sum()
    ax.fill_between(range(bins-1), chrY_histo[donor][1:] / total, color='limegreen')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle("Histogram of nonzero Y chromosome accessibility", y=0.95, fontsize=20)
plt.tight_layout()
plt.show()


# The histograms of the Y chromosomes illustrate the diversity of the donors: Donor 13176 seems to have (almost) no Y chromosome (maybe the few nonzero values are measuring errors). It appears that the donors are one woman and three men.
# 
# **Insight:**
# - If we assume that the true values of donor 13176 are all zero, these data show us the magnitude of the measurement error.
# - Maybe we can create a new (binary) feature for the presence of the Y chromosome.
# - The diagram reminds us that the donors are different and that our model should be robust to these differences.

# # Multiome target
# 
# The Multiome targets (RNA count data) are in similar shape as the CITEseq inputs: They have 105942 rows and 23418 columns. All targets are nonnegative and no values are missing.
# 

# In[24]:


get_ipython().run_cell_magic('time', '', "start = 0\nchunksize = 10000\ntotal_rows = 0\nwhile True:\n    df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=start+chunksize)\n    if df_multi_train_y.isna().any().any(): print('There are missing values.')\n    if (df_multi_train_y < 0).any().any(): print('There are negative values.')\n    total_rows += len(df_multi_train_y)\n    print(total_rows, 'rows read')\n    if len(df_multi_train_y) < chunksize: break\n    start += chunksize\n    \ndisplay(df_multi_train_y.head())")


# In[25]:


get_ipython().run_cell_magic('time', '', 'nonzeros = df_multi_train_y.values.ravel()\nnonzeros = nonzeros[nonzeros != 0]\nplt.figure(figsize=(16, 4))\nplt.gca().set_facecolor(\'#0057b8\')\nplt.hist(nonzeros, bins=500, density=True, color=\'#ffd700\')\ndel nonzeros\nplt.title("Histogram of nonzero target values (based on a subset of the rows)")\nplt.xlabel("log1p-transformed expression count")\nplt.ylabel("density")\nplt.show()\n\ndf_multi_train_y = None # release the memory')


# # Summary
# 
# - This is a big data competition. The more RAM you have, the better. In any case, you'll have to treat RAM as a scarce resource and write memory-efficient code.
# - As there are two competitions in one, you can choose to participate in only one of them (and merge your predictions with somebody else's predictions for the other part). CITEseq is easier to begin with.
# - Make sure you decide for a good cross-validation scheme so that you avoid the shakedown at the end of the competition.
# 
# After this EDA, you may want to look at some models which generate predictions in spite of having only 16 GB RAM:
# - [MSCI CITEseq Keras Quickstart](https://www.kaggle.com/code/ambrosm/msci-citeseq-keras-quickstart)
# - [MSCI CITEseq Quickstart](https://www.kaggle.com/ambrosm/msci-citeseq-quickstart) (using LightGBM)
# - [MSCI Multiome Quickstart](https://www.kaggle.com/ambrosm/msci-multiome-quickstart) (using ridge regression)

# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # Multiome Quickstart With Sparse Matrices
# 
# This notebook is mostly for demonstrating the utility of sparse matrices in this competition. (Especially for the Multiome dataset).
# 
# As the Multiome dataset is  very sparse (about 98% of cells are zeros), it benefits greatly from being encoded as sparse matrices. 
# 
# This notebook is largely based on [this notebook](https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart) by AmbrosM. It is a nice first attempt at handling Multiome data, and I thought it would informative for kagglers to be able to contrast directly the performances of sparse vs dense representations. 
# 
# Mostly, the differences with AmbrosM's notebooks are:
# - We use a representation of the data in sparse CSR format, which let us load all of the training data in memory (using less than 8GB memory instead of the >90GB it would take to represent the data in a dense format)
# - We perform PCA (actually, TruncatedSVD) on the totality of the training data (while AmbrosM's notebook had to work with a subset of 6000 rows and 4000 columns). 
# - We keep 16 components (vs 4 in AmbrosM's notebook)
# - We apply Ridge regression on 50000 rows (vs 6000 in AmbrosM's notebook)
# - Despite using much more data, this notebook should run in a bit more than 10 minutes (vs >1h for AmbrosM's notebook)
# 
# The competition data is pre-encoded as sparse matrices in [this dataset](https://www.kaggle.com/datasets/fabiencrom/multimodal-single-cell-as-sparse-matrix) generated by [this notebook](https://www.kaggle.com/code/fabiencrom/multimodal-single-cell-creating-sparse-data/).
# 
# Since we will only generate the multiome predictions in this notebook, I am taking the CITEseq predictions from [this notebook](https://www.kaggle.com/code/vuonglam/lgbm-baseline-optuna-drop-constant-cite-task) by VuongLam, which is the public notebook with the best score at the time I am publishing.
# 

# In[1]:


import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

import scipy
import scipy.sparse


# # The scoring function (from AmbrosM)
# 
# This competition has a special metric: For every row, it computes the Pearson correlation between y_true and y_pred, and then all these correlation coefficients are averaged.

# In[2]:


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape: raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


# # Preprocessing and cross-validation
# 
# We first load all of the training input data for Multiome. It should take less than a minute.

# In[3]:


get_ipython().run_cell_magic('time', '', 'train_inputs = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_values.sparse.npz")')


# ## PCA / TruncatedSVD
# It is not possible to directly apply PCA to a sparse matrix, because PCA has to first "center" the data, which destroys the sparsity. This is why we apply `TruncatedSVD` instead (which is pretty much "PCA without centering"). It might be better to normalize the data a bit more here, but we will keep it simple.

# In[4]:


get_ipython().run_cell_magic('time', '', 'pca = TruncatedSVD(n_components=16, random_state=1)\ntrain_inputs = pca.fit_transform(train_inputs)')


# ## Random row selection and conversion of the target data to a dense matrix
# 
# Unfortunately, although sklearn's `Ridge` regressor do accept sparse matrices as input, it does not accept sparse matrices as target values. This means we will have to convert the targets to a dense format. Although we could fit in memory both the dense target data and the sparse input data, the Ridge regression process would then lack memory. Therefore, from now on, we will work with a subset of 50 000 rows from the training data.

# In[5]:


np.random.seed(42)
all_row_indices = np.arange(train_inputs.shape[0])
np.random.shuffle(all_row_indices)
selected_rows_indices = all_row_indices[:50000]


# In[6]:


train_inputs = train_inputs[selected_rows_indices]


# In[7]:


get_ipython().run_cell_magic('time', '', 'train_target = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_values.sparse.npz")')


# In[8]:


train_target = train_target[selected_rows_indices]
train_target = train_target.todense()
gc.collect()


# ## KFold Ridge regression
# `sklearn` complains that we should use array instead of matrices. Unfortunately, the old `scipy` version available on kaggle do not provide sparse arrays; only sparse matrices. So we suppress the warnings.

# In[9]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# This Kfold ridge regression code is mostly taken from AmbrosM's [notebook](https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart). Note that `sklearn`'s `Ridge` handles sparse matrices transparently. I found [this blog post](https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/) that list the other algorithms of `sklearn` that accept sparse matrices.

# In[10]:


get_ipython().run_cell_magic('time', '', '# Cross-validation\n\nkf = KFold(n_splits=5, shuffle=True, random_state=1)\nscore_list = []\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train_inputs)):\n    model = None\n    gc.collect()\n    X_tr = train_inputs[idx_tr] # creates a copy, https://numpy.org/doc/stable/user/basics.copies.html\n    y_tr = train_target[idx_tr]\n    del idx_tr\n\n    model = Ridge(copy_X=False)\n    model.fit(X_tr, y_tr)\n    del X_tr, y_tr\n    gc.collect()\n\n    # We validate the model\n    X_va = train_inputs[idx_va]\n    y_va = train_target[idx_va]\n    del idx_va\n    y_va_pred = model.predict(X_va)\n    mse = mean_squared_error(y_va, y_va_pred)\n    corrscore = correlation_score(y_va, y_va_pred)\n    del X_va, y_va\n\n    print(f"Fold {fold}: mse = {mse:.5f}, corr =  {corrscore:.3f}")\n    score_list.append((mse, corrscore))\n\n# Show overall score\nresult_df = pd.DataFrame(score_list, columns=[\'mse\', \'corrscore\'])\nprint(f"{Fore.GREEN}{Style.BRIGHT}{train_inputs.shape} Average  mse = {result_df.mse.mean():.5f}; corr = {result_df.corrscore.mean():.3f}{Style.RESET_ALL}")')


# # Retraining
# 

# In[11]:


# We retrain the model and then delete the training data, which is no longer needed
model, score_list, result_df = None, None, None # free the RAM occupied by the old model
gc.collect()
model = Ridge(copy_X=False) # we overwrite the training data
model.fit(train_inputs, train_target)


# In[12]:


del train_inputs, train_target # free the RAM
_ = gc.collect()


# # Predicting

# In[13]:


get_ipython().run_cell_magic('time', '', 'multi_test_x = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_values.sparse.npz")\nmulti_test_x = pca.transform(multi_test_x)\ntest_pred = model.predict(multi_test_x)\ndel multi_test_x\ngc.collect()')


# # Creating submission
# 
# We load the cells that will have to appear in submission.

# In[14]:


get_ipython().run_cell_magic('time', '', '# Read the table of rows and columns required for submission\neval_ids = pd.read_parquet("../input/multimodal-single-cell-as-sparse-matrix/evaluation.parquet")\n\n# Convert the string columns to more efficient categorical types\n#eval_ids.cell_id = eval_ids.cell_id.apply(lambda s: int(s, base=16))\n\neval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())\neval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())')


# In[15]:


# Prepare an empty series which will be filled with predictions
submission = pd.Series(name='target',
                       index=pd.MultiIndex.from_frame(eval_ids), 
                       dtype=np.float32)
submission


# We load the `index`  and `columns` of the original dataframe, as we need them to make the submission.

# In[16]:


get_ipython().run_cell_magic('time', '', 'y_columns = np.load("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_idxcol.npz",\n                   allow_pickle=True)["columns"]\n\ntest_index = np.load("../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz",\n                    allow_pickle=True)["index"]')


# We assign the predicted values to the correct row in the submission file.

# In[17]:


cell_dict = dict((k,v) for v,k in enumerate(test_index)) 
assert len(cell_dict)  == len(test_index)

gene_dict = dict((k,v) for v,k in enumerate(y_columns))
assert len(gene_dict) == len(y_columns)


# In[18]:


eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))

valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)


# In[19]:


submission.iloc[valid_multi_rows] = test_pred[eval_ids_cell_num[valid_multi_rows].to_numpy(),
eval_ids_gene_num[valid_multi_rows].to_numpy()]


# In[20]:


del eval_ids_cell_num, eval_ids_gene_num, valid_multi_rows, eval_ids, test_index, y_columns
gc.collect()


# In[21]:


submission


# # Merging with CITEseq predictions
# 
# We use the CITEseq predictions from [this notebook](https://www.kaggle.com/code/vuonglam/lgbm-baseline-optuna-drop-constant-cite-task) by VuongLam.

# In[22]:


submission.reset_index(drop=True, inplace=True)
submission.index.name = 'row_id'
# with open("partial_submission_multi.pickle", 'wb') as f:
#     pickle.dump(submission, f)
# submission


# In[23]:


cite_submission = pd.read_csv("../input/lgbm-baseline-optuna-drop-constant-cite-task/submission.csv")
cite_submission = cite_submission.set_index("row_id")
cite_submission = cite_submission["target"]


# In[24]:


submission[submission.isnull()] = cite_submission[submission.isnull()]


# In[25]:


submission


# In[26]:


submission.isnull().any()


# In[27]:


submission.to_csv("submission.csv")


# In[28]:


get_ipython().system('head submission.csv')

#!/usr/bin/env python
# coding: utf-8

# # Multiome Quickstart
# 
# This notebook shows how to cross-validate a baseline model and create a submission for the Multiome part of the *Multimodal Single-Cell Integration* competition without running out of memory.
# 
# It does not show the EDA - see the separate notebook [MSCI EDA which makes sense ??????????????????????????????](https://www.kaggle.com/ambrosm/msci-eda-which-makes-sense).
# 
# The baseline model for the other part of the competition (CITEseq) is [here](https://www.kaggle.com/ambrosm/msci-citeseq-quickstart).

# In[1]:


import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

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


# In[2]:


# If you see a warning "Failed to establish a new connection" running this cell,
# go to "Settings" on the right hand side, 
# and turn on internet. Note, you need to be phone verified.
# We need this library to read HDF files.
get_ipython().system('pip install --quiet tables')


# # Loading the common metadata table
# 
# The current version of the model is so primitive that it doesn't use the metadata, but we load it anyway.

# In[3]:


df_cell = pd.read_csv(FP_CELL_METADATA)
df_cell_cite = df_cell[df_cell.technology=="citeseq"]
df_cell_multi = df_cell[df_cell.technology=="multiome"]
df_cell_cite.shape, df_cell_multi.shape


# # The scoring function
# 
# This competition has a special metric: For every row, it computes the Pearson correlation between y_true and y_pred, and then all these correlation coefficients are averaged.

# In[4]:


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape: raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


# # Preprocessing and cross-validation
# 
# The Multiome dataset is way too large to fit into 16 GByte RAM:
# - train inputs:  105942 * 228942 float32 values (97 GByte)
# - train targets: 105942 *  23418 float32 values (10 GByte)
# - test inputs:    55935 * 228942 float32 values (13 GByte)
# 
# To get a result with only 16 GByte RAM, we simplify the problem as follows:
# - We ignore the complete metadata (donors, days, cell types).
# - We read only 6000 rows of the training data.
# - We drop all feature columns which are constant.
# - Of the remaining columns, we keep only 4000.
# - We do a PCA and keep only the 4 most important components.
# - We fit a ridge regression model with 6000\*4 inputs and 6000\*23418 targets.

# In[5]:


#%%time
# Preprocessing

class PreprocessMultiome(BaseEstimator, TransformerMixin):
    columns_to_use = slice(10000, 14000)
    
    @staticmethod
    def take_column_subset(X):
        return X[:,PreprocessMultiome.columns_to_use]
    
    def transform(self, X):
        print(X.shape)
        X = X[:,~self.all_zero_columns]
        print(X.shape)
        X = PreprocessMultiome.take_column_subset(X) # use only a part of the columns
        print(X.shape)
        gc.collect()

        X = self.pca.transform(X)
        print(X.shape)
        return X

    def fit_transform(self, X):
        print(X.shape)
        self.all_zero_columns = (X == 0).all(axis=0)
        X = X[:,~self.all_zero_columns]
        print(X.shape)
        X = PreprocessMultiome.take_column_subset(X) # use only a part of the columns
        print(X.shape)
        gc.collect()

        self.pca = PCA(n_components=4, copy=False, random_state=1)
        X = self.pca.fit_transform(X)
        plt.plot(self.pca.explained_variance_ratio_.cumsum())
        plt.title("Cumulative explained variance ratio")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('PCA component')
        plt.ylabel('Cumulative explained variance ratio')
        plt.show()
        print(X.shape)
        return X

preprocessor = PreprocessMultiome()

multi_train_x = None
start, stop = 0, 6000
multi_train_x = preprocessor.fit_transform(pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=start, stop=stop).values)

multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=stop)
y_columns = multi_train_y.columns
multi_train_y = multi_train_y.values
print(multi_train_y.shape)


# In[6]:


get_ipython().run_cell_magic('time', '', '# Cross-validation\n\nkf = KFold(n_splits=5, shuffle=True, random_state=1)\nscore_list = []\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(multi_train_x)):\n    model = None\n    gc.collect()\n    X_tr = multi_train_x[idx_tr] # creates a copy, https://numpy.org/doc/stable/user/basics.copies.html\n    y_tr = multi_train_y[idx_tr]\n    del idx_tr\n\n    model = Ridge(copy_X=False)\n    model.fit(X_tr, y_tr)\n    del X_tr, y_tr\n    gc.collect()\n\n    # We validate the model\n    X_va = multi_train_x[idx_va]\n    y_va = multi_train_y[idx_va]\n    del idx_va\n    y_va_pred = model.predict(X_va)\n    mse = mean_squared_error(y_va, y_va_pred)\n    corrscore = correlation_score(y_va, y_va_pred)\n    del X_va, y_va\n\n    print(f"Fold {fold}: mse = {mse:.5f}, corr =  {corrscore:.3f}")\n    score_list.append((mse, corrscore))\n\n# Show overall score\nresult_df = pd.DataFrame(score_list, columns=[\'mse\', \'corrscore\'])\nprint(f"{Fore.GREEN}{Style.BRIGHT}{multi_train_x.shape} Average  mse = {result_df.mse.mean():.5f}; corr = {result_df.corrscore.mean():.3f}{Style.RESET_ALL}")')


# By the way, this ridge regression is not much better than DummyRegressor, which scores `mse = 2.01718; corr = 0.679`.

# # Retraining
# 

# In[7]:


# We retrain the model and then delete the training data, which is no longer needed
model, score_list, result_df = None, None, None # free the RAM occupied by the old model
gc.collect()
model = Ridge(copy_X=False) # we overwrite the training data
model.fit(multi_train_x, multi_train_y)
del multi_train_x, multi_train_y # free the RAM
_ = gc.collect()


# The final submission will contain 65744180 predictions, of which the first 6812820 are CITEseq predictions and the remaining 58931360 are Multiome. 
# 
# The Multiome test predictions have 55935 rows and 23418 columns. 55935 \* 23418 = 1???309???885???830 predictions. We'll only submit 4.5 % of these predictions. According to the data description, this subset was created by sampling 30 % of the Multiome rows, and for each row, 15 % of the columns (i.e., 16780 rows and 3512 columns per row). Consequently, when reading the test data, we can immediately drop 70 % of the rows and keep only the remaining 16780.
# 
# The eval_ids table specifies which predictions are required for the submission file.

# In[8]:


get_ipython().run_cell_magic('time', '', "# Read the table of rows and columns required for submission\neval_ids = pd.read_csv(FP_EVALUATION_IDS, index_col='row_id')\n\n# Convert the string columns to more efficient categorical types\n#eval_ids.cell_id = eval_ids.cell_id.apply(lambda s: int(s, base=16))\neval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())\neval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())\ndisplay(eval_ids)\n\n# Create the set of needed cell_ids\ncell_id_set = set(eval_ids.cell_id)\n\n# Convert the string gene_ids to a more efficient categorical dtype\ny_columns = pd.CategoricalIndex(y_columns, dtype=eval_ids.gene_id.dtype, name='gene_id')")


# In[9]:


# Prepare an empty series which will be filled with predictions
submission = pd.Series(name='target',
                       index=pd.MultiIndex.from_frame(eval_ids), 
                       dtype=np.float32)
submission


# We now compute the predictions in chunks of 5000 rows and match them with the eval_ids table row by row. The matching is very slow, but space-efficient.

# In[10]:


get_ipython().run_cell_magic('time', '', "# Process the test data in chunks of 5000 rows\n\nstart = 0\nchunksize = 5000\ntotal_rows = 0\nwhile True:\n    multi_test_x = None # Free the memory if necessary\n    gc.collect()\n    # Read the 5000 rows and select the 30 % subset which is needed for the submission\n    multi_test_x = pd.read_hdf(FP_MULTIOME_TEST_INPUTS, start=start, stop=start+chunksize)\n    rows_read = len(multi_test_x)\n    needed_row_mask = multi_test_x.index.isin(cell_id_set)\n    multi_test_x = multi_test_x.loc[needed_row_mask]\n    \n    # Keep the index (the cell_ids) for later\n    multi_test_index = multi_test_x.index\n    \n    # Predict\n    multi_test_x = multi_test_x.values\n    multi_test_x = preprocessor.transform(multi_test_x)\n    test_pred = model.predict(multi_test_x)\n    \n    # Convert the predictions to a dataframe so that they can be matched with eval_ids\n    test_pred = pd.DataFrame(test_pred,\n                             index=pd.CategoricalIndex(multi_test_index,\n                                                       dtype=eval_ids.cell_id.dtype,\n                                                       name='cell_id'),\n                             columns=y_columns)\n    gc.collect()\n    \n    # Fill the predictions into the submission series row by row\n    for i, (index, row) in enumerate(test_pred.iterrows()):\n        row = row.reindex(eval_ids.gene_id[eval_ids.cell_id == index])\n        submission.loc[index] = row.values\n    print('na:', submission.isna().sum())\n\n    #test_pred_list.append(test_pred)\n    total_rows += len(multi_test_x)\n    print(total_rows)\n    if rows_read < chunksize: break # this was the last chunk\n    start += chunksize\n    \ndel multi_test_x, multi_test_index, needed_row_mask")


# # Submission
# 
# As we don't yet have the CITEseq predictions, we save the partial predictions so that they can be used in the [CITEseq notebook](https://www.kaggle.com/ambrosm/msci-citeseq-quickstart).

# In[11]:


submission.reset_index(drop=True, inplace=True)
submission.index.name = 'row_id'
with open("partial_submission_multi.pickle", 'wb') as f: pickle.dump(submission, f)
submission


# In[ ]:




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




#!/usr/bin/env python
# coding: utf-8

# # Multiome with Torch
# This notebooks is to help competitors that would like to apply Deep Neural Networks to the MSCI data.
# It is focused on the more challenging Multiome part of the data (but it is trivial to adapt it to to the CITEseq data)
# 
# The main challenge here is that the Multiome data is very large while Kaggle  GPU machines only have 13GB RAM + 16GB GPU Memory.
# 
# I found it is actually possible to store all of the dataset in GPU memory using sparse tensor formats. This uses ~12GB on the GPU, leaving only ~4GB for the model parameters and the forward/backward computation. Given that we have only ~100K training examples, I do not expect we will need very large models, so I feel 4GB is actually enough.
# 
# If 4GB is not enough, the other option is to leave the dataset in RAM and load the batches on demand to the GPU (which is what is more classically done). In that case, however, we have only ~1GB RAM left, and will suffer a small performance penalty from having to load the batches to the GPU. But we will have the whole 16GB available for training a complex model. Yet another option is to apply dimensionality reduction to the data beforehand (e.g. with PCA/TruncatedSVD), although I like more the idea of using the raw data and letting the network do its own dimensionality reduction.
# 
# The competition data is pre-encoded as sparse matrices in [this dataset](https://www.kaggle.com/datasets/fabiencrom/multimodal-single-cell-as-sparse-matrix) generated by [this notebook](https://www.kaggle.com/code/fabiencrom/multimodal-single-cell-creating-sparse-data/).
# 
# The model used here is just a very simple MLP. In the current version, I add a `Softplus` activation at the end, considering the values we have to predict are all positives (although I am not sure it will really work better that way).
# 
# In the current version, I also directly optimize the competition metric (row-wise Pearson correlation). Although it does not seem to perform much better than using a simpler Mean Square Error Loss.
# 
# This notebook will train 5 models over 5 folds. The final submission is created in [this notebook](https://www.kaggle.com/fabiencrom/msci-multiome-torch-quickstart-submission).
# 
# So far I did not get results better than the one obtained by the much simpler PCA+Ridge Regression method (that you can find in [this notebook](https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart) as initially proposed by AmbrosM or in [this notebook](https://www.kaggle.com/code/fabiencrom/msci-multiome-quickstart-w-sparse-matrices) for a version using sparse matrices for better results). But I expect it will perform better after working on the architecture/hyperparameters. In any case, I think a deep learning model will be a part of any winning submission.

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


# # Score and loss functions
# We can use either a classic Mean Square Error loss (nn.MSELoss) or use a loss that will optimize directly the competition metric.

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


# # Config
# We put the configuration dict at the beginning of the notebook, so that it is easier to find and modify

# In[3]:


config = dict(
    layers = [128, 128, 128],
    patience = 4,
    max_epochs = 20,
    criterion = correl_loss, #nn.MSELoss(),
    
    n_folds = 5,
    folds_to_train = [0, 1, 2, 3, 4],
    kfold_random_state = 42,
    
    optimizerparams = dict(
     lr=1e-3, 
     weight_decay=1e-2
    ),
    
    head="softplus"
    
)

INPUT_SIZE = 228942
OUTPUT_SIZE = 23418


# # Utility functions for loading and batching the sparse data in device memory
# There are a few challenges here:
# - If we directly try to create a torch sparse tensor before moving it to memory, we will get an OOM error
# - Torch CSR tensors cannot be moved to the gpu; so we make our own TorchCSR class that will contain the csr format information
# - torch gpu operations are only compatible with COO tensors (not CSR), so we need some functions to create batches of COO tensors from the TorchCSR objects

# In[4]:


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
    start_pts = th_indptr[start]
    end_pts = th_indptr[end]
    coo_data = th_data[start_pts: end_pts]
    coo_col = th_indices[start_pts: end_pts]
    coo_row = torch.repeat_interleave(torch.arange(end-start, device=device), th_indptr[start+1:end+1] - th_indptr[start:end])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [end-start, th_shape[1]])
    return coo_batch


# # GPU memory DataLoader
# We create a dataloader that will work with the in-device TorchCSR tensor.
# This should ensure the fastest training speed.

# In[5]:


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
        
        self.nb_examples = len(self.train_idx) if self.train_idx is not None else len(train_inputs)
        
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
                tgt_batch = make_coo_batch_slice(self.train_targets, i*self.batch_size, (i+1)*self.batch_size)
            else:
                idx_batch = idx_array[slc]
                inp_batch = make_coo_batch(self.train_inputs, idx_batch)
                tgt_batch = make_coo_batch(self.train_targets, idx_batch)
            yield inp_batch, tgt_batch
            
            
    def __len__(self):
        return self.nb_batches


# # Simple Model: MLP

# In[6]:


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


# # Training functions

# In[7]:


def train_fn(model, optimizer, criterion, dl_train):

    loss_list = []
    model.train()
    for inpt, tgt in tqdm(dl_train):
        mb_size = inpt.shape[0]
        tgt = tgt.to_dense()

        optimizer.zero_grad()
        pred = model(inpt)

        loss = criterion(pred, tgt)
        loss_list.append(loss.detach())
        loss.backward()
        optimizer.step()
    avg_loss = sum(loss_list).cpu().item()/len(loss_list)
    
    return {"loss":avg_loss}


# In[8]:


def valid_fn(model, criterion, dl_valid):
    loss_list = []
    all_preds = []
    all_tgts = []
    partial_correlation_scores = []
    model.eval()
    for inpt, tgt in tqdm(dl_valid):
        mb_size = inpt.shape[0]
        tgt = tgt.to_dense()
        with torch.no_grad():
            pred = model(inpt)
        loss = criterion(pred, tgt)
        loss_list.append(loss.detach())
        
        partial_correlation_scores.append(partial_correlation_score_torch_faster(tgt, pred))

    avg_loss = sum(loss_list).cpu().item()/len(loss_list)
    
    partial_correlation_scores = torch.cat(partial_correlation_scores)

    score = torch.sum(partial_correlation_scores).cpu().item()/len(partial_correlation_scores) #correlation_score_torch(all_tgts, all_preds)
    
    return {"loss":avg_loss, "score":score}


# In[9]:


def train_model(model, optimizer, dl_train, dl_valid, save_prefix):

    criterion = config["criterion"]
    
    save_params_filename = save_prefix+"_best_params.pth"
    save_config_filename = save_prefix+"_config.pkl"
    best_score = None

    for epoch in range(config["max_epochs"]):
        log_train = train_fn(model, optimizer, criterion, dl_train)
        log_valid = valid_fn(model, criterion, dl_valid)

        print(log_train)
        print(log_valid)
        
        score = log_valid["score"]
        if best_score is None or score > best_score:
            best_score = score
            patience = config["patience"]
            best_params = copy.deepcopy(model.state_dict())
        else:
            patience -= 1
        
        if patience < 0:
            print("out of patience")
            break


    torch.save(best_params, save_params_filename)
    pickle.dump(config,open(save_config_filename, "wb"))
    


# In[10]:


def train_one_fold(num_fold):
    
    train_idx, valid_idx = FOLDS_LIST[num_fold]
    
    train_idx = torch.from_numpy(train_idx).to(device)
    valid_idx = torch.from_numpy(valid_idx).to(device)
    
    
    dl_train = DataLoaderCOO(train_inputs, train_targets, train_idx=train_idx,
                batch_size=512, shuffle=True, drop_last=True)
    dl_valid = DataLoaderCOO(train_inputs, train_targets, train_idx=valid_idx,
                batch_size=512, shuffle=False, drop_last=False)
    
    model =  build_model()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizerparams"])
    
    train_model(model, optimizer, dl_train, dl_valid, save_prefix="f%i"%num_fold)


# # Load Data

# In[11]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"machine has {torch.cuda.device_count()} cuda devices")
    print(f"model of first cuda device is {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")


# In[12]:


get_ipython().run_cell_magic('time', '', 'train_inputs = scipy.sparse.load_npz(\n    "../input/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_values.sparse.npz")')


# We will normalize the input by dividing each column by its max value. This is the simplest reasonable option. Centering the data (i.e. substracting the mean, would destroy the sparsity here)

# In[13]:


max_inputs = train_inputs.max(axis=0)
max_inputs = max_inputs.todense()+1e-10
np.savez("max_inputs.npz", max_inputs = max_inputs)
max_inputs = torch.from_numpy(max_inputs)[0].to(device)


# In[14]:


get_ipython().run_cell_magic('time', '', 'train_inputs = load_csr_data_to_gpu(train_inputs)\ngc.collect()')


# In[15]:


train_inputs.data[...] /= max_inputs[train_inputs.indices.long()]


# In[16]:


torch.max(train_inputs.data)


# In[17]:


get_ipython().run_cell_magic('time', '', 'train_targets = scipy.sparse.load_npz(\n    "../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_values.sparse.npz")')


# In[18]:


get_ipython().run_cell_magic('time', '', 'train_targets = load_csr_data_to_gpu(train_targets)\ngc.collect()')


# In[19]:


assert INPUT_SIZE == train_inputs.shape[1]
assert OUTPUT_SIZE == train_targets.shape[1]

NB_EXAMPLES = train_inputs.shape[0]
assert NB_EXAMPLES == train_targets.shape[0]

print(INPUT_SIZE, OUTPUT_SIZE, NB_EXAMPLES)


# # Training
# We use a rather naive kfold split here, which might not be optimal for this competition.

# In[20]:


kfold = KFold(n_splits=config["n_folds"], shuffle=True, random_state=config["kfold_random_state"])
FOLDS_LIST = list(kfold.split(range(train_inputs.shape[0])))


# In[21]:


for num_fold in config["folds_to_train"]:
    train_one_fold(num_fold)


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success" style="font-size:30px">
# ???? PyTorch Swiss Army Knife for MSCI Competition ????
# </div>
# 
# <div class="alert alert-block alert-danger" style="text-align:center; font-size:20px;">
#     ?????? Dont forget to ???upvote??? if you find this notebook usefull!  ??????
# </div>
# 
# This notebook includes a set of tools to build your deep learning solution for MSCI-2022 competition.
# This is your one-stop shop to build a winning deep learning model.
# Hope you'll find it useful!
# 
# Here is the list of features implemented in this notebook:
# 
# ### Training both Multiome and CITEseq Regressors
# There is no much difference between Multiome and CITEseq problems apart from the scale. With minibatch training both problmes can be soled within a single framework.
# * set `CFG=CFG_MULTIOME_SVD` or `CFG=CFG_MULTIOME_SPARSE` to train your Multiome regressor
# * set `CFG=CFG_CITESEQ_SVD` or `CFG=CFG_CITESEQ_SPARSE` to train your CITEseq regressor
# 
# 
# ### Both SVD-compressed and Raw Features are Supported
# * SVD-compressed data is loaded from [this notebook](https://www.kaggle.com/code/vslaykovsky/multiome-citeseq-svd-transforms), where TruncatedSVD is used to project raw features to 512 dimensional space. SVD features are concatenated with cell type features in `MSCIDatasetSVD` class
# * Raw data is loaded to memory as sparse matrices and is lazily uncomressed and concatenated with cell_id features in the `MSCIDatasetSparse` class.
# 
# ### Both Kaggle and Custom Training is Supported
# This notebook can be easily customised for local training or training on more powerful machines like Colab/google cloud/AWS.
# just fill out your constants in `# local run` section to get started.
# 
# ### K-fold Training
# Training on multiple folds enables cross-validation score.
# Enable `TRAIN=True` for training
# 
# ### Accurate Correlation Metric/Loss Function
# `CorrError` metric is implemented to match competition requirements. This can be used for both training and evaluation of your solution.
# 
# ### Optuna Hyperparameter Optimization
# Optimize your hyperparameters with Optuna. Multiple hyperparameters are already supported by `MSCIModel`. You are free to add more parameters of course!
# Enable Optuna with `OPTUNA=True`
# 
# ### Ensembling of k-fold models
# Submission is generated from K models that come from k-fold training. Implementation is memory-efficient. Only a single copy of predictions (the largest matrix) is loaded in memory at any time.
# 
# ### Wandb Logging
# Train with Wandb and track your scores even in background execution! Here is the list of implemented metrics:
# * `eval_score` - correlation score on evaluation set
# * `eval_mse` - MSE score on evaluation set
# * `lr` - learning rate.
# * `epoch` - epoch
# * `train_score` - correlation score on training set
# * `train_loss` - training loss (most of the time == `train_score`)
# * `train_epoch_mse` - epoch-average MSE on training set
# 
# ### Patching you Predictions Into the Best Public Solution
# You can always patch your model's outputs into the best public solution if you only generate predictions for a single technology (CITEseq or Multiome).

# ## Model diagram
# 
# This is a simplified diagram of the model generated by pytorchviz. 
# You can see sequences of Linear->LayerNorm->SiLU(ReLU) layers here
# 
# <img src="https://images2.imgbox.com/be/27/9vy3PmRH_o.png" alt="image host"/>

# # Configuration
# 
# Set `CFG` to the right configuration to train a model

# In[1]:


get_ipython().system('pip install -q torchviz')


# In[2]:


import gc
import os.path

import numpy as np
import optuna
import pandas as pd
import torch
import wandb
from optuna.study import StudyDirection
from scipy import sparse
from tqdm.notebook import tqdm
import copy
from torchviz import make_dot


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    import kaggle_secrets
    IS_KAGGLE = True
except:
    IS_KAGGLE = False

if IS_KAGGLE:
    # kaggle run
    MSCI_ROOT = '../input/open-problems-multimodal'
    SPARSE_ROOT = '../input/multimodal-single-cell-as-sparse-matrix'
    SVD_ROOT = '../input/multiome-citeseq-sv-transforms-ds'
    MODELS_ROOT = '../input/multiomekfoldmodels'
    PREDICTIONS_ROOT = '.'
    SUBMISSIONS_ROOT = '../input/lb-0-858-normalized-ensembles-for-pearson-s-r/'
else:
    # local run
    MSCI_ROOT = '/mnt/msci'
    SPARSE_ROOT = f'data/sparse'
    SVD_ROOT = f'data/svd'
    MODELS_ROOT = 'models'
    PREDICTIONS_ROOT = '.'
    SUBMISSIONS_ROOT = 'data/sub'


META_FILE = f'{SPARSE_ROOT}/metadata.parquet'

N_FOLDS = 5

# Enable/disable parts of the notebook
TRAIN = False  # training
CROSS_VALIDATE = False
OPTUNA = False # hyperparameters search with Optuna
OPTUNA_N_TRIALS = 30
PREDICT = True
SUBMISSION = False

SUBMISSION_FOR_PATCHING = f'{SUBMISSIONS_ROOT}/submission.csv'
PATCH_CITESEQ = False
PATCH_MULTI = True


# In[3]:


CFG_MULTIOME_SVD = {
    'TECHNOLOGY': 'Multiome',
    'MSE_LOSS': False,
    'SCHEDULER': 'onecycle',
    'SKIP_CONNECTION': False,

    'TRAIN_INPUTS_VALUES_NPZ': f'{SVD_ROOT}/train_multi_inputs.npz',

    'TRAIN_TARGETS_VALUES_NPZ': f'{SPARSE_ROOT}/train_multi_targets_values.sparse.npz',
    'TRAIN_TARGETS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_multi_targets_idxcol.npz',
    'TRAIN_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_multi_inputs_idxcol.npz',

    'TEST_INPUTS_VALUES_NPZ': f'{SVD_ROOT}/test_multi_inputs.npz',
    'TEST_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/test_multi_inputs_idxcol.npz',

    'WANDB_PROJECT': 'MSCI-MULTI-SVD',
    'NUM_WORKERS': 1,
    'BATCH_SIZE': 1024,
    'EPOCHS': 20,
    'MAX_LR': 0.001,
    'N_LAYERS': 5,
    'DROPOUT': False,
    'HIDDEN_SIZE': 1024,
    'ADAMW': False,
    'WEIGHT_DECAY': 0.05,
    'ACTIVATION': torch.nn.SiLU,
    'N_FEATURES': 520,
    'N_TARGETS': 23418,
}

CFG_MULTIOME_SPARSE = {
    'TECHNOLOGY': 'Multiome',
    'MSE_LOSS': False,
    'SCHEDULER': 'onecycle',
    'SKIP_CONNECTION': False,

    'TRAIN_INPUTS_VALUES_NPZ': f'{SPARSE_ROOT}/train_multi_inputs_values.sparse.npz',
    'TRAIN_TARGETS_VALUES_NPZ': f'{SPARSE_ROOT}/train_multi_targets_values.sparse.npz',
    'TRAIN_TARGETS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_multi_targets_idxcol.npz',
    'TRAIN_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_multi_inputs_idxcol.npz',

    'TEST_INPUTS_VALUES_NPZ': f'{SPARSE_ROOT}/test_multi_inputs_values.sparse.npz',
    'TEST_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/test_multi_inputs_idxcol.npz',

    'WANDB_PROJECT': 'MSCI-MULTI-Sparse',
    'NUM_WORKERS': 1,
    'BATCH_SIZE': 64,
    'EPOCHS': 7,
    'WEIGHT_DECAY': 0.05,
    'MAX_LR': 0.0001,
    'ADAMW': True,
    'N_LAYERS': 7,
    'DROPOUT': False,
    'HIDDEN_SIZE': 1024,
    'ACTIVATION': torch.nn.SiLU,
    'N_FEATURES': 228950,
    'N_TARGETS': 23418,
}

CFG_CITESEQ_SVD = {
    'TECHNOLOGY': 'CITEseq',
    'MSE_LOSS': False,
    'SCHEDULER': 'onecycle',

    'TRAIN_INPUTS_VALUES_NPZ': f'{SVD_ROOT}/train_cite_inputs.npz',
    
    'TRAIN_TARGETS_VALUES_NPZ': f'{SPARSE_ROOT}/train_cite_targets_values.sparse.npz',
    'TRAIN_TARGETS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_cite_targets_idxcol.npz',
    'TRAIN_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_cite_inputs_idxcol.npz',
    
    'TEST_INPUTS_VALUES_NPZ': f'{SVD_ROOT}/test_cite_inputs.npz',
    'TEST_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/test_cite_inputs_idxcol.npz',
    
    'SKIP_CONNECTION': False,

    'WANDB_PROJECT': 'MSCI-CITE-SVD',
    'NUM_WORKERS': 2,
    'BATCH_SIZE': 1024,
    'EPOCHS': 10,
    'ADAMW': False,
    'WEIGHT_DECAY': 0.05,
    'MAX_LR': 0.003,
    'N_LAYERS': 7,
    'DROPOUT': False,
    'HIDDEN_SIZE': 1024,
    'ACTIVATION': torch.nn.SiLU,
    'N_FEATURES': 520,
    'N_TARGETS': 140,
}

CFG_CITESEQ_SPARSE = {
    'TECHNOLOGY': 'CITEseq',
    'MSE_LOSS': False,
    'SCHEDULER': 'onecycle',

    'TRAIN_INPUTS_VALUES_NPZ': f'{SPARSE_ROOT}/train_cite_inputs_values.sparse.npz',
    
    'TRAIN_TARGETS_VALUES_NPZ': f'{SPARSE_ROOT}/train_cite_targets_values.sparse.npz',
    'TRAIN_TARGETS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_cite_targets_idxcol.npz',
    'TRAIN_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/train_cite_inputs_idxcol.npz',
    
    'TEST_INPUTS_VALUES_NPZ': f'{SPARSE_ROOT}/test_cite_inputs_values.sparse.npz',
    'TEST_INPUTS_IDXCOL_NPZ': f'{SPARSE_ROOT}/test_cite_inputs_idxcol.npz',

    'SKIP_CONNECTION': False,

    'WANDB_PROJECT': 'MSCI-CITE-Sparse',
    'NUM_WORKERS': 2,
    'BATCH_SIZE': 256,
    'EPOCHS': 10,
    'WEIGHT_DECAY': 0.05,
    'MAX_LR': 0.0001,
    'ADAMW': True,
    'N_LAYERS': 7,
    'DROPOUT': False,
    'HIDDEN_SIZE': 1024,
    'ACTIVATION': torch.nn.SiLU,
    'N_FEATURES': 22058,
    'N_TARGETS': 140,
}


# Choose any of the above configurations to train your model!
CFG = CFG_MULTIOME_SVD


# # Datasets
# 
# Both SVD and raw sparse featurers are supported in MSCIDatasetSVD and MSCIDatasetSparse

# In[4]:


def load_meta():
    df_meta = pd.read_parquet(META_FILE).set_index('cell_id')
    df_meta = pd.get_dummies(df_meta['cell_type'])
    return df_meta

# quick test
load_meta().shape


# In[5]:


class MSCIDatasetSVD(torch.utils.data.Dataset):

    def __init__(self, df_meta, input_index, input_svd, targets=None):
        cell_type = df_meta.loc[input_index].values
        self.data = np.concatenate([cell_type, input_svd[:len(cell_type)]], axis=1)
        self.targets = targets

    def __getitem__(self, item):
        if self.targets is not None:
            return self.data[item], np.asarray(self.targets[item].todense())[0]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


class MSCIDatasetSparse(torch.utils.data.Dataset):

    def __init__(self, df_meta, input_index, input_sparse, targets=None):
        self.cell_type = df_meta.loc[input_index].values.astype('float32')
        self.data = input_sparse
        self.targets = targets

    def __getitem__(self, item):
        if self.targets is not None:
            return np.concatenate([
                self.cell_type[item],
                np.asarray(self.data[item].todense())[0]
            ]), np.asarray(self.targets[item].todense())[0]
        else:
            return np.concatenate([
                self.cell_type[item],
                np.asarray(self.data[item].todense())[0]
            ])

    def __len__(self):
        return len(self.cell_type)


def load_dataset():
    df_meta = load_meta()
    if 'sparse.npz' in CFG['TRAIN_INPUTS_VALUES_NPZ']:
        ds_data = MSCIDatasetSparse(
            df_meta,
            np.load(CFG['TRAIN_INPUTS_IDXCOL_NPZ'], allow_pickle=True)['index'],
            sparse.load_npz(CFG['TRAIN_INPUTS_VALUES_NPZ']),
            sparse.load_npz(CFG['TRAIN_TARGETS_VALUES_NPZ'])
        )
    else:
        ds_data = MSCIDatasetSVD(
            df_meta,
            np.load(CFG['TRAIN_INPUTS_IDXCOL_NPZ'], allow_pickle=True)['index'],
            np.load(CFG['TRAIN_INPUTS_VALUES_NPZ'])['values'],
            sparse.load_npz(CFG['TRAIN_TARGETS_VALUES_NPZ'])
        )
    return ds_data


# # Model
# 
# A set of dense layers with LayerNorm. LayerNorm and SiLU result in better convergence

# In[6]:


get_ipython().system('apt install graphviz')


# In[7]:


class MSCIModel(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        input_size, hidden_size, n_layers, output_size, activation, dropout, skip_connection = cfg['N_FEATURES'], cfg['HIDDEN_SIZE'], cfg['N_LAYERS'], cfg['N_TARGETS'], cfg['ACTIVATION'], cfg['DROPOUT'], cfg['SKIP_CONNECTION']

        self.skip_connection = skip_connection
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            activation(),
        )
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
                activation(),
            )
            for _ in range(n_layers)]
        )

        self.output = torch.nn.Sequential(
            *(
                    [torch.nn.Dropout(0.1)] if dropout else [] +
                    [
                        torch.nn.Linear(hidden_size, output_size),
                        torch.nn.LayerNorm(output_size),
                        torch.nn.ReLU(),
                    ]
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            if self.skip_connection:
                x = block(x) + x
            else:
                x = block(x)
        x = self.output(x)
        return x


# quick test

cfg = copy.copy(CFG)
cfg['N_LAYERS'] = 1
m = MSCIModel(cfg)
print(m)
print(m.forward(torch.randn(2, cfg['N_FEATURES'])))


x = torch.randn(2, cfg['N_FEATURES']).requires_grad_(True)
y = m(x)   
make_dot(y, params=dict(list(m.named_parameters()) + [('x', x)]))


# # Training
# 
# Differentiable correlation error function.
# We test that it's correct by comparing to `torch.corrcoef`

# In[8]:


class CorrError():
    def __init__(self, reduction='mean', normalize=True):
        self.reduction, self.normalize = reduction, normalize

    def __call__(self, y, y_target):
        y = y - torch.mean(y, dim=1).unsqueeze(1)
        y_target = y_target - torch.mean(y_target, dim=1).unsqueeze(1)
        loss = -torch.sum(y * y_target, dim=1) / (y_target.shape[-1] - 1)  # minus because we want gradient ascend
        if self.normalize:
            s1 = torch.sqrt(torch.sum(y * y, dim=1) / (y.shape[-1] - 1))
            s2 = torch.sqrt(torch.sum(y_target * y_target, dim=1) / (y_target.shape[-1] - 1))
            loss = loss / s1 / s2
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss


# quick test
a = torch.tensor([[0, 1, 1., 0.1, 0.3, 0.4]])
b = torch.tensor([[0, 0, 1., 10, 10, -13]])

corr1 = CorrError()(a, b).item()
corr2 = torch.corrcoef(torch.stack([a[0], b[0]]))[0, 1].item()
assert abs(-corr1 - corr2) < 1e-5

CorrError(reduction='none')(torch.randn(2, 3), torch.randn(2, 3))


# In[9]:


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def adamw_optimizer(model, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)


# quick test
adamw_optimizer(torch.nn.Linear(3, 4), 0.001).step()


# In[10]:


def evaluate(model, ds_eval, cfg, batch_progress):
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'])

    with torch.no_grad():
        model.eval()
        with tqdm(dl_eval, miniters=100, desc='Batch', disable=not batch_progress) as progress:
            scores = []
            mses = []
            for batch_idx, (X, y) in enumerate(progress):
                y_pred = model.forward(X.to(DEVICE))
                score = CorrError()(y_pred.detach(), y.to(DEVICE)).item()
                progress.set_description(f'Eval Loss: {score:02f}', refresh=False)
                scores.append(score)
                mses.append(torch.nn.MSELoss()(y_pred.detach(), y.to(DEVICE)).item())

            score = np.mean(scores)
            mses = np.mean(mses)
            model.train()
            return score, mses


# In[11]:


def train(cfg, ds_train, ds_eval, wandb_run=None, batch_progress=True, store_best=True, name=''):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['NUM_WORKERS'])

    model = MSCIModel(cfg)
    model.to(DEVICE)
    model.train()
    best_score = 1.

    if cfg['ADAMW']:
        optim = adamw_optimizer(model, cfg['WEIGHT_DECAY'])
    else:
        optim = torch.optim.Adam(model.parameters(), lr=(cfg['MAX_LR']))

    if cfg['MSE_LOSS']:
        criterion = torch.nn.MSELoss()
    else:
        criterion = CorrError()

    if cfg['SCHEDULER'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=(cfg['MAX_LR']), epochs=cfg['EPOCHS'],
                                                        steps_per_epoch=len(dl_train))
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.3)

    with tqdm(range(cfg['EPOCHS']), desc='Epoch') as epoch_progress:
        for epoch in epoch_progress:
            # ************** train cycle **************
            mses = []
            scores = []
            with tqdm(dl_train, miniters=100, desc='Batch', disable=not batch_progress) as progress:
                for batch_idx, (X, y) in enumerate(progress):
                    y_pred = model.forward(X.to(DEVICE))
                    loss = criterion(y_pred, y.to(DEVICE))
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    if cfg['SCHEDULER'] == 'onecycle':
                        scheduler.step()

                    score = CorrError()(y_pred.detach(), y.to(DEVICE)).item()
                    scores.append(score)
                    mses.append(torch.nn.MSELoss()(y_pred.detach(), y.to(DEVICE)).item())
                    progress.set_description(f'Loss: {score:02f}', refresh=False)
                    if wandb_run is not None:
                        wandb_run.log({'lr': float(scheduler.get_last_lr()[0]),
                                       'train_score': score,
                                       'train_loss': loss.item(),
                                       'epoch': epoch})
            if wandb_run is not None:
                wandb_run.log({'train_epoch_score': np.mean(scores), 'train_epoch_mse': np.mean(mses)})
            if cfg['SCHEDULER'] != 'onecycle':
                scheduler.step()

            # ************** eval cycle **************
            score, mses = evaluate(model, ds_eval, cfg, batch_progress)
            if wandb_run is not None:
                wandb_run.log({'eval_score': score, 'eval_mse': mses})
            if score < best_score:
                best_score = score
                if store_best:
                    get_ipython().system('mkdir -p {MODELS_ROOT}')
                    torch.save(model.state_dict(), f'{cfg["WANDB_PROJECT"]}-{name}.pth')
            epoch_progress.set_description(f'Epochs, eval loss:{score:.03f}')
    return best_score


# In[12]:


def kfold_split(ds):
    fold_sizes = [len(ds) // N_FOLDS] * (N_FOLDS - 1) + [len(ds) // N_FOLDS + len(ds) % N_FOLDS]
    ds_folds = torch.utils.data.random_split(ds, fold_sizes, generator=torch.Generator().manual_seed(42))
    for fold in range(N_FOLDS):
        yield torch.utils.data.ConcatDataset(ds_folds[:fold] + ds_folds[fold + 1:]), ds_folds[fold]

# quick test
for ds_train, ds_test in kfold_split(list(range(10))):
    print([i for i in ds_train], [i for i in ds_test])


# In[13]:


if TRAIN:
    ds_data = load_dataset()
    for fold, (ds_train, ds_eval) in enumerate(tqdm(kfold_split(ds_data), desc='Train fold', total=N_FOLDS)):
        with wandb.init(project=CFG['WANDB_PROJECT'], name=f'pytorch-{fold}') as run:
            train(CFG, ds_train, ds_eval, wandb_run=run, batch_progress=False, name=str(fold))


# Wandb output for Multiome SVD training:
# 
# <img src="https://images2.imgbox.com/d2/98/Dm7OTXD3_o.png" alt="image host"/>

# # Cross-validation

# In[14]:


def load_model(fname):
    model = MSCIModel(CFG)
    model.load_state_dict(torch.load(fname))
    return model


# In[15]:


if CROSS_VALIDATE:
    ds = load_dataset()
    scores = []
    for fold, (_, ds_eval) in enumerate(tqdm(kfold_split(ds), desc='Evaluating Folds', total=N_FOLDS)):
        model = load_model(f'{MODELS_ROOT}/{CFG["WANDB_PROJECT"]}-{fold}.pth').to(DEVICE)
        scores, mses = evaluate(model, ds_eval, CFG, batch_progress=False)
        del model, ds_eval
        gc.collect()
    print('CV score:', -np.mean(scores))


# # Optuna Hyperparameters Tuning

# In[16]:


import copy


def objective(trial: optuna.trial.Trial, ds_data):
    is_adamw = trial.suggest_categorical('adamw', [True, False])
    weight_decay = 0.
    if is_adamw:
        weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.1, log=True)
    max_lr = trial.suggest_float('max_lr', 0.0001, 0.01, log=True)
    n_layers = trial.suggest_int('n_layers', 2, 8, log=True)
    hidden_size = trial.suggest_int('hidden_size', 128, 4096, log=True)
    activation = eval(trial.suggest_categorical('activation', ['torch.nn.SiLU', 'torch.nn.GELU', 'torch.nn.ReLU']))
    dropout = trial.suggest_categorical('dropout', [True, False])
    skip_connection = trial.suggest_categorical('skip', [True, False])

    ds_train, ds_test = next(iter(kfold_split(ds_data)))
    cfg = copy.copy(CFG)
    cfg.update({
        'ADAMW': is_adamw,
        'WEIGHT_DECAY': weight_decay,
        'MAX_LR': max_lr,
        'N_LAYERS': n_layers,
        'HIDDEN_SIZE': hidden_size,
        'ACTIVATION': activation,
        'DROPOUT': dropout,
        'SKIP_CONNECTION': skip_connection,
        'EPOCHS': 5,
    })
    return train(cfg, ds_train, ds_test, wandb_run=None, batch_progress=False, store_best=False)

def run_optuna():
    ds_data = load_dataset()
    study = optuna.create_study(direction=StudyDirection.MINIMIZE)
    study.optimize(lambda trial: objective(trial, ds_data), n_trials=OPTUNA_N_TRIALS)
    print(study.best_params)

if OPTUNA:
    run_optuna()


# # Prediction

# In[17]:


def load_ds_test():
    df_meta = load_meta()
    if 'sparse.npz' in CFG['TEST_INPUTS_VALUES_NPZ']:
        ds_test = MSCIDatasetSparse(
            df_meta,
            np.load(CFG['TEST_INPUTS_IDXCOL_NPZ'], allow_pickle=True)['index'],
            sparse.load_npz(CFG['TEST_INPUTS_VALUES_NPZ']),
        )
    else:        
        ds_test = MSCIDatasetSVD(
            df_meta,
            np.load(CFG['TEST_INPUTS_IDXCOL_NPZ'], allow_pickle=True)['index'],
            np.load(CFG['TEST_INPUTS_VALUES_NPZ'])['values'],
        )
    return ds_test


# In[18]:


def predict(models, ds):
    with torch.no_grad():
        dl_eval = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=1)
        preds = []
        with tqdm(dl_eval, miniters=100, desc='Predict') as progress:
            for batch_idx, (X) in enumerate(progress):
                pred = None
                for model in models:
                    model.eval()
                    pred_fold = model.forward(X.to(DEVICE))
                    if pred is None:
                        pred = pred_fold / len(models)
                    else:
                        pred += pred_fold / len(models)
                preds.append(pred)
        preds = torch.concat(preds)                
        return preds


# In[19]:


def predict_nfold():
    ds_test = load_ds_test()
    models = [load_model(f'{MODELS_ROOT}/{CFG["WANDB_PROJECT"]}-{fold}.pth').to(DEVICE) for fold in tqdm(range(N_FOLDS), desc='Loading models')]
    preds = predict(models, ds_test)
    del models
    del ds_test
    gc.collect()
    return preds.cpu()
    

def predict_and_save():
    preds = predict_nfold()
    np.save(CFG["TECHNOLOGY"], preds)
    
if PREDICT:
    predict_and_save()
    gc.collect()    


# # Submission

# In[20]:


def gen_preds(df_eval, pred_file, train_targets_idxcol, test_inputs_idxcol):
    cite_pred = np.load(pred_file)
    cols = np.load(train_targets_idxcol, allow_pickle=True)['columns']
    cols_idx = dict(zip(cols, range(len(cols))))
    cells = np.load(test_inputs_idxcol, allow_pickle=True)['index']
    cells_idx = dict(zip(cells, range(len(cells))))
    return cite_pred[df_eval.cell_id.map(cells_idx), df_eval.gene_id.map(cols_idx)]


# In[21]:


def gen_submission():
    df_meta = pd.read_parquet(META_FILE)
    df_evaluation = pd.read_parquet(f'{SPARSE_ROOT}/evaluation.parquet')
    print('Loaded', f'{SPARSE_ROOT}/evaluation.parquet', df_evaluation.shape)

    # citeseq
    df_cite_eval = df_evaluation[df_evaluation.cell_id.isin(df_meta.query('technology == "citeseq"').cell_id)]
    if os.path.exists(f'CITEseq.npy'):
        cite_eval_pred = gen_preds(df_cite_eval, f'{PREDICTIONS_ROOT}/CITEseq.npy', f'{SPARSE_ROOT}/train_cite_targets_idxcol.npz', f'{SPARSE_ROOT}/test_cite_inputs_idxcol.npz')
        print('Loaded citeseq prediction', cite_eval_pred.shape)
    else:
        cite_eval_pred = np.zeros(df_cite_eval.shape[0])
        print('Empty citeseq prediction', cite_eval_pred.shape)
    del df_cite_eval
    gc.collect()

    # multiome
    df_multi_eval = df_evaluation[df_evaluation.cell_id.isin(df_meta.query('technology == "multiome"').cell_id)]
    if os.path.exists(f'Multiome.npy'):
        multi_eval_pred = gen_preds(df_multi_eval, f'{PREDICTIONS_ROOT}/Multiome.npy', f'{SPARSE_ROOT}/train_multi_targets_idxcol.npz', f'{SPARSE_ROOT}/test_multi_inputs_idxcol.npz')
        print("Loaded Multiome predictions", multi_eval_pred.shape)
    else:
        multi_eval_pred = np.zeros(df_multi_eval.shape[0])
        print("Empty Multiome predictions", multi_eval_pred.shape)
    del df_multi_eval, df_meta
    gc.collect()

    if SUBMISSION:    
        print('Generating pure submission')
        df_evaluation['target'] = np.concatenate([cite_eval_pred, multi_eval_pred])
        df_evaluation[['row_id', 'target']].to_csv('submission.csv', index=False)
    elif SUBMISSION_FOR_PATCHING is not None:
        del df_evaluation
        gc.collect()
        df_sub = pd.read_csv(SUBMISSION_FOR_PATCHING)
        print('Generating patched submission from', SUBMISSION_FOR_PATCHING, df_sub.shape)
        if PATCH_CITESEQ:
            preds = np.concatenate([cite_eval_pred, df_sub['target'].tail(len(multi_eval_pred)).values])
            print("Patching CITEseq data", preds.shape)
            df_sub['target'] = preds
        else:
            multi_patch = np.concatenate([df_sub['target'].head(len(cite_eval_pred)).values, multi_eval_pred])
            print("Patching Multiome data", multi_patch.shape)
            df_sub['target'] = multi_patch
        df_sub.to_csv('submission.csv', index=False)
    print('Done')
                

gen_submission()


# <div class="alert alert-block alert-danger" style="text-align:center; font-size:20px;">
#     ?????? Dont forget to ???upvote??? if you find this notebook usefull!  ??????
# </div>
#!/usr/bin/env python
# coding: utf-8

# The aim for this notebook is the create a central log (meta resource) of all the information and resources I've come across in the discussions, other notebooks, in my own discovery. I tend to find it best for me to keep track of all the many nooks one can store information on Kaggle in one place. 
# 
# The following notebook has tried to aggregate important papers / resources that will help in understanding how this one might go about completing this challenge. I've also provided where these resources have been taken at the bottom. 

# # Insightful notebooks
# 
# ## EDA 
# * MSCI EDA which makes sense - https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/notebook
# * Complette EDA of MmSCel Integration Data - https://www.kaggle.com/code/leohash/complete-eda-of-mmscel-integration-data
# 
# ## Solutions 
# * MSCI CITEseq Quickstart - https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart
# 

# ## Experiments
# ### CITE-seq
# Key to defining cell type/states indentity through modules of genes uniquely expressed
# 
# ![citeseq.com](https://citeseq.files.wordpress.com/2018/02/figure1.png?w=700)
#  * Protein (CITE-seq): Enables leveraging legacy markers used over the last decades by immunologists to define cell spectrums 
#  
#  
#  ### ATAC-seq 
#  Key to defining immune cell states transitional states best defined by up- and down- regulation of critical transcription factors (usually poorly captured transcriptionally) 
#  
#  ![ATAC-seq transposition reaction ](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41596-022-00692-9/MediaObjects/41596_2022_692_Fig1_HTML.png)
# 
# **Steps:** 
# 1. Nuclei are isolated from cells
#     * maintaining the chromatin structure and any associated DNA-binding proteins, including nucleosomes and TFs, intact
# 2. This chromatin is then exposed to the Tn5 transposase, 
#     * which acts as a homodimer to simultaneously fragment the chromatin and insert sequences containing PCR handles that enable downstream amplification with i5/P5 and i7/P7 ATAC-seq adapters.
#     * Only fragments that receive an i5/P5 adapter at one end and an i7/P7 adapter at the other end will be properly amplified and sequenced.
# 3. Sequence analysis of the library fragments, genomic regions enriched for many Tn5 transposition events are designated as peaks of chromatin accessibility, or ATAC-seq peaks
# 
# * This is the test kit that was usedin these experiments:  Multinome - ATAC + Gene Expression: https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression

# # Coding Tips
# 
# ## Reading in partial h5 files
# When you only want to read a large file partially using `pandas` you can do it as follows by using the `start` and `stop` parameters in `pd.read_hdf(..)`
# 
# ```python
# targets = pd.read_hdf('../input/targets.h5', start=0, stop=6000)
# ```

# # Data 
# **Multiome_train_inputs** 
# * Columns - gene id (228942 genes)
# * Rows - Cells (105942 cells) 
# * Values - ATAC signal falling in the range of 0 - 16.9336 
# 
# **Multiome_train_targets** 

# # Dictionary 
# * ATAC: assay for transposase-accessible chromatin 
# * bi-CCA: bi-order canonical correlation analysis 
# * BCC:  Basal cell carcinoma 
# * CITE-seq: Cellular indexing of transcriptomes and epitopes by sequencing
# * CLOUD-HSPCs: ???continuum of low-primed undifferentiated haematopoietic stem and progenitor cells??? 
# * scRNA-seq: single-cell RNA sequencing  
# * scATAC-seq: single-cell assay for transposase-accessible chromatin using sequencing aka **chromatin accessibility**
# * PBMC:  human peripheral blood mononuclear cells
# * HSC: haematopoietic stem cells [wiki](https://en.wikipedia.org/wiki/Hematopoietic_stem_cell)

# # Data Challenges 
# Single cell analysis has a unique set of challenges. Following are papers and posts that I've found to be helpful in identifying those challenges.
# * [Computational challenges of cell cycle analysis using single cell
# transcriptomics](https://arxiv.org/pdf/2208.05229.pdf)

# # Papers
# * [Integrated analysis of multimodal single-cell data](https://www.sciencedirect.com/science/article/pii/S0092867421005833)
#     * The simultaneous measurement of multiple modalities represents an exciting frontier for single-cell genomics and necessitates computational methods that can define cellular states based on multimodal data. Here, we introduce **???weighted-nearest neighbor??? analysis**, an unsupervised framework to learn the relative utility of each data type in each cell, enabling an integrative analysis of multiple modalities. We apply our procedure to a CITE-seq dataset of 211,000 human peripheral blood mononuclear cells (PBMCs) with panels extending to 228 antibodies to construct a multimodal reference atlas of the circulating immune system. **Multimodal analysis substantially improves our ability to resolve cell states, allowing us to identify and validate previously unreported lymphoid subpopulations.** Moreover, we demonstrate how to leverage this reference to rapidly map new datasets and to interpret immune responses to vaccination and coronavirus disease 2019 (COVID-19). Our approach represents a broadly applicable strategy to analyze single-cell multimodal datasets and to look beyond the transcriptome toward a unified and multimodal definition of cellular identity.
# * [New horizons in the stormy sea of multimodal single-cell data integration](https://www.sciencedirect.com/science/article/abs/pii/S1097276521010741)
#     * We review steps and challenges toward this goal. Single-cell transcriptomics is now a mature technology, and methods to measure proteins, lipids, small-molecule metabolites, and other molecular phenotypes at the single-cell level are rapidly developing. Integrating these single-cell readouts so that each cell has measurements of multiple types of data, e.g., transcriptomes, proteomes, and metabolomes, is expected to allow identification of highly specific cellular subpopulations and to provide the basis for inferring causal biological mechanisms.
# * [Computation principles and challenges in single-cell data integration](https://www.nature.com/articles/s41587-021-00895-7)
#     * The development of single-cell multimodal assays provides a powerful tool for investigating multiple dimensions of cellular heterogeneity, enabling new insights into development, tissue homeostasis and disease. **A key challenge in the analysis of single-cell multimodal data is to devise appropriate strategies for tying together data across different modalities.** The term ???data integration??? has been used to describe this task, encompassing a broad collection of approaches ranging from batch correction of individual omics datasets to association of chromatin accessibility and genetic variation with transcription. Although existing integration strategies exploit similar mathematical ideas, they typically have distinct goals and rely on different principles and assumptions. Consequently, new definitions and concepts are needed to contextualize existing methods and to enable development of new methods.
# * [Diagonal integration of multimodal single-cell data: potential pitfalls and paths forward](https://www.nature.com/articles/s41467-022-31104-x)
#     * Diagonal integration of multimodal single-cell data emerges as a trending topic. However, empowering diagonal methods for novel biological discoveries requires bridging huge gaps. Here, we comment on **potential risks and future directions of diagonal integration for multimodal single-cell data**
# * [Bi-order multimodal integration of single-cell data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02679-x)
#     * Integration of single-cell multiomics profiles generated by different single-cell technologies from the same biological sample is still challenging. Previous approaches based on shared features have only provided approximate solutions. Here, we present **a novel mathematical solution named bi-order canonical correlation analysis (bi-CCA), which extends the widely used CCA approach to iteratively align the rows and the columns between data matrices.** Bi-CCA is generally applicable to combinations of any two single-cell modalities. Validations using co-assayed ground truth data and application to a CAR-NK study and a fetal muscle atlas demonstrate its capability in generating accurate multimodal co-embeddings and discovering cellular identity.
# * [Multimodal single-cell approaches shed light on T cell heterogeneity](https://www.sciencedirect.com/science/article/pii/S0952791519300469)
#     * Single-cell methods have revolutionized the study of T cell biology by enabling the identification and characterization of individual cells. This has led to a deeper understanding of T cell heterogeneity by generating functionally relevant measurements ??? like gene expression, surface markers, chromatin accessibility, T cell receptor sequences ??? in individual cells. While these methods are independently valuable, they can be augmented when applied jointly, either on separate cells from the same sample or on the same cells. **Multimodal approaches are already being deployed to characterize T cells in diverse disease contexts and demonstrate the value of having multiple insights into a cell???s function.** But, these data sets pose new statistical challenges for integration and joint analysis.
# * [Cobolt: integrative analysis of multimodal single-cell sequencing data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02556-z)
#     * A growing number of single-cell sequencing platforms enable joint profiling of multiple omics from the same cells. We present **Cobolt, a novel method that not only allows for analyzing the data from joint-modality platforms, but provides a coherent framework for the integration of multiple datasets measured on different modalities.** We demonstrate its performance on multi-modality data of gene expression and chromatin accessibility and illustrate the integration abilities of Cobolt by jointly analyzing this multi-modality data with single-cell RNA-seq and ATAC-seq datasets.
# * [Human haematopoietic stem cell lineage commitment is a continuous process](https://www.nature.com/articles/ncb3493)
#     * Blood formation is believed to occur through stepwise progression of haematopoietic stem cells (HSCs) following a tree-like hierarchy of oligo-, bi- and unipotent progenitors. However, this model is based on the analysis of predefined flow-sorted cell populations. Here we integrated flow cytometric, transcriptomic and functional data at single-cell resolution to quantitatively map early differentiation of human HSCs towards lineage commitment. During homeostasis, individual HSCs gradually acquire lineage biases along multiple directions without passing through discrete hierarchically organized progenitor populations. Instead, unilineage-restricted cells emerge directly from a ???continuum of low-primed undifferentiated haematopoietic stem and progenitor cells??? (CLOUD-HSPCs). **Distinct gene expression modules operate in a combinatorial manner to control stemness, early lineage priming and the subsequent progression into all major branches of haematopoiesis.** These data reveal a continuous landscape of human steady-state haematopoiesis downstream of HSCs and provide a basis for the understanding of haematopoietic malignancies.
# * [Normalizing and denoising protein expression data from droplet-based single cell profiling](https://www.nature.com/articles/s41467-022-29356-8)
#     * Multimodal single-cell profiling methods that measure protein expression with oligo-conjugated antibodies hold promise for comprehensive dissection of cellular heterogeneity, yet the resulting protein counts have substantial technical noise that can mask biological variations. Here we integrate experiments and computational analyses to reveal two major noise sources and develop a method called ???dsb??? (denoised and scaled by background) to normalize and denoise droplet-based protein expression data. **We discover that protein-specific noise originates from unbound antibodies encapsulated during droplet generation; this noise can thus be accurately estimated and corrected by utilizing protein levels in empty droplets**. We also find that isotype control antibodies and the background protein population average in each cell exhibit significant correlations across single cells, we thus use their shared variance to correct for cell-to-cell technical noise in each cell. We validate these findings by analyzing the performance of dsb in eight independent datasets spanning multiple technologies, including CITE-seq, ASAP-seq, and TEA-seq. Compared to existing normalization methods, our approach improves downstream analyses by better unmasking biologically meaningful cell populations. Our method is available as an open-source R package that interfaces easily with existing single cell software platforms such as Seurat, Bioconductor, and Scanpy.
# * [BABEL enables cross-modality translation between multiomic profiles at single-cell resolution](https://pubmed.ncbi.nlm.nih.gov/33827925/)
#     * Simultaneous profiling of multiomic modalities within a single cell is a grand challenge for single-cell biology. While there have been impressive technical innovations demonstrating feasibility-for example, generating paired measurements of single-cell transcriptome (single-cell RNA sequencing (scRNA-seq) and chromatin accessibility (single-cell assay for transposase-accessible chromatin using sequencing (scATAC-seq))-widespread application of joint profiling is challenging due to its experimental complexity, noise, and cost. Here, we introduce BABEL, a deep learning method that translates between the transcriptome and chromatin profiles of a single cell. **Leveraging an interoperable neural network model, BABEL can predict single-cell expression directly from a cell's scATAC-seq and vice versa after training on relevant data. This makes it possible to computationally synthesize paired multiomic measurements when only one modality is experimentally available. Across several paired single-cell ATAC and gene expression datasets in human and mouse, we validate that BABEL accurately translates between these modalities for individual cells.** BABEL also generalizes well to cell types within new biological contexts not seen during training. Starting from scATAC-seq of patient-derived basal cell carcinoma (BCC), BABEL generated single-cell expression that enabled fine-grained classification of complex cell states, despite having never seen BCC data. These predictions are comparable to analyses of experimental BCC scRNA-seq data for diverse cell types related to BABEL's training data. We further show that BABEL can incorporate additional single-cell data modalities, such as protein epitope profiling, thus enabling translation across chromatin, RNA, and protein. BABEL offers a powerful approach for data exploration and hypothesis generation.
# * [Current best practices in single-cell RNA-seq analysis: a tutorial](https://www.embopress.org/doi/full/10.15252/msb.20188746)
#     * Single-cell RNA-seq has enabled gene expression to be studied at an unprecedented resolution. The promise of this technology is attracting a growing user base for single-cell analysis methods. As more analysis tools are becoming available, it is becoming increasingly difficult to navigate this landscape and produce an up-to-date workflow to analyse one's data. Here, **we detail the steps of a typical single-cell RNA-seq analysis, including pre-processing (quality control, normalization, data correction, feature selection, and dimensionality reduction) and cell- and gene-level downstream analysis.** We formulate current best-practice recommendations for these steps based on independent comparison studies. We have integrated these best-practice recommendations into a workflow, which we apply to a public dataset to further illustrate how these steps work in practice. Our documented case study can be found at https://www.github.com/theislab/single-cell-tutorial. This review will serve as a workflow tutorial for new entrants into the field, and help established users update their analysis pipelines.
# 
# ### Methods 
# * [Chromatin accessibility profiling by ATAC-seq](https://www.nature.com/articles/s41596-022-00692-9)
#     * The assay for transposase-accessible chromatin using sequencing (ATAC-seq) provides a simple and scalable way to detect the unique chromatin landscape associated with a cell type and how it may be altered by perturbation or disease. ATAC-seq requires a relatively small number of input cells and does not require a priori knowledge of the epigenetic marks or transcription factors governing the dynamics of the system. Here we describe an updated and optimized protocol for ATAC-seq, called Omni-ATAC, that is applicable across a broad range of cell and tissue types. The ATAC-seq workflow has five main steps: sample preparation, transposition, library preparation, sequencing and data analysis. This protocol details the steps to generate and sequence ATAC-seq libraries, with recommendations for sample preparation and downstream bioinformatic analysis. ATAC-seq libraries for roughly 12 samples can be generated in 10 h by someone familiar with basic molecular biology, and downstream sequencing analysis can be implemented using benchmarked pipelines by someone with basic bioinformatics skills and with access to a high-performance computing environment.
# 
# ## Preprint
# * [Computational challenges in cell cycle analysis using single cell transcriptomics](https://arxiv.org/abs/2208.05229) 
# * [Multimodal single-cell chromatin analysis with Signac](https://www.biorxiv.org/content/10.1101/2020.11.09.373613v1.abstract) 
# * [MultiVI: deep generative model for the integration of multimodal-data](https://www.biorxiv.org/content/10.1101/2021.08.20.457057v1)

# # Experimental Details
# * Cell Lines Used: https://allcells.com/research-grade-tissue-products/mobilized-leukopak/
# * Multignome - ATAC + Gene Expression: https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression
#     * Chromatin accessibility to predict gene expression
# * CITESeq - Single Cell Gene Expression: https://support.10xgenomics.com/permalink/getting-started-single-cell-gene-expression-with-feature-barcoding-technology
#     * Cell Surface Reagent - https://www.biolegend.com/en-gb/products/totalseq-b-human-universal-cocktail-v1dot0-20960

# # Additional Information 
# * [EBI Ensemble Id Information](https://www.ebi.ac.uk/training/online/courses/ensembl-browsing-genomes/navigating-ensembl/investigating-a-gene/#:~:text=Ensembl%20gene%20IDs%20begin%20with,of%20species%20other%20than%20human)
# * [Eleven Grand Challenges in Single-Cell Data Science](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1926-6)

# # Kaggle Notebooks (Background)
# * [scRNA-seq ????: Differential Expression with scVI](https://www.kaggle.com/code/hiramcho/scrna-seq-differential-expression-with-scvi/notebook)
# * [scRNA-seq ????: Scanpy & SCMER for Feature Selection](https://www.kaggle.com/code/hiramcho/scrna-seq-scanpy-scmer-for-feature-selection/notebook)
# * [scRNA-seq ????: scGAE with Spektral and RAPIDS](https://www.kaggle.com/code/hiramcho/scrna-seq-scgae-with-spektral-and-rapids/notebook)
# * [scATAC-seq ????: Feature Importance with TabNet](https://www.kaggle.com/code/hiramcho/scatac-seq-feature-importance-with-tabnet/notebook)
# * [scATAC-seq ????: EpiScanpy & PeakVI](https://www.kaggle.com/code/hiramcho/scatac-seq-episcanpy-peakvi)
# 
# 
# # Kaggle 
# * [MSCI CITEseq Quickstart](https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart)
# * [Data Loading - Getting Started](https://www.kaggle.com/code/peterholderrieth/getting-started-data-loading)
# * [MmSCel????Inst: EDA ???? & Stat. ????????????? predictions](https://www.kaggle.com/code/jirkaborovec/mmscel-inst-eda-stat-predictions)
# * [MultiSCI- ???? EDA](https://www.kaggle.com/code/vicsonsam/multisci-eda)
# * [???? Cell Analysis - quick h5 EDA](https://www.kaggle.com/code/queyrusi/cell-analysis-quick-h5-eda)
# * [Multimodal Single-Cell Integration](https://www.kaggle.com/code/erivanoliveirajr/multimodal-single-cell-integration)
# * [MSCI - CITEseq - TF/Keras NN Custom loss](https://www.kaggle.com/code/lucasmorin/msci-citeseq-tf-keras-nn-custom-loss)
# * [Complete EDA of MmSCel Integration Data](https://www.kaggle.com/code/leohash/complete-eda-of-mmscel-integration-data)
# * [???Tune LGBM Only - Final???CITE Task](https://www.kaggle.com/code/vuonglam/tune-lgbm-only-final-cite-task)
# * [MSCI EDA which makes sense ??????????????????????????????](https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/data)
# * [MSCI Multiome Quickstart w/ Sparse Matrices](https://www.kaggle.com/code/fabiencrom/msci-multiome-quickstart-w-sparse-matrices) - 0.847
# * [???LGBM Baseline???MSCI CITEseq](https://www.kaggle.com/code/swimmy/lgbm-baseline-msci-citeseq) - 0.824
# * [MSCI Multiome Quickstart](https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart)
# * [Simple Submission - Average by gene_id](https://www.kaggle.com/code/shuntarotanaka/simple-submission-average-by-gene-id) 0.741
# * [Reduce Memory Usage by 95% with Sparse Matrices](https://www.kaggle.com/code/sbunzini/reduce-memory-usage-by-95-with-sparse-matrices)
# 

# # External Notebooks / Packages 
# * [KNN Solution](https://github.com/adavoudi/msci_knn)

# # Learning Resources
# * [MIA: Multimodal Single-cell data, open benchmarks, and a NeurIPS 2021](https://www.biolegend.com/en-gb/products/totalseq-b-human-universal-cocktail-v1dot0-20960) - *video* 
# * [Open Problems in Single Cells Analysis](https://openproblems.bio/neurips_docs/data/about_multimodal/)
#     * Open problems in scAnalysis - 

# # Potentially Useful Packages (Python)
# * [muon](https://muon.readthedocs.io/en/latest/api/generated/muon.atac.pp.tfidf.html?highlight=tfidf) - muon is a Python framework for multimodal omics analysis. While there are many features that muon brings to the table, there are three key areas that its functionality is focused on.
# * [scanpy](https://scanpy.readthedocs.io/en/stable/index.html) - Scanpy is a scalable toolkit for analyzing single-cell gene expression data built jointly with anndata. It includes preprocessing, visualization, clustering, trajectory inference and differential expression testing. The Python-based implementation efficiently deals with datasets of more than one million cells.
# * [anndata](https://anndata.readthedocs.io/en/latest/#) - nndata is a Python package for handling annotated data matrices in memory and on disk, positioned between pandas and xarray. anndata offers a broad range of computationally efficient features including, among others, sparse data support, lazy operations, and a PyTorch interface.
# * [Xarray](https://docs.xarray.dev/en/v0.9.2/dask.html) - xarray (formerly xray) is an open source project and Python package that aims to bring the labeled data power of pandas to the physical sciences, by providing N-dimensional variants of the core pandas data structures. **This will help split up the large dataset**
# * [ivis](https://bering-ivis.readthedocs.io/en/latest/index.html) - ivis is a machine learning library for reducing dimensionality of very large datasets using Siamese Neural Networks. ivis preserves global data structures in a low-dimensional space, adds new data points to existing embeddings using a parametric mapping function, and scales linearly to millions of observations. The algorithm is described in detail in Structure-preserving visualisation of high dimensional single-cell datasets.
# * [epiScanpy](https://episcanpy.readthedocs.io/en/latest/) - EpiScanpy is a toolkit to analyse single-cell open chromatin (scATAC-seq) and single-cell DNA methylation (for example scBS-seq) data. EpiScanpy is the epigenomic extension of the very popular scRNA-seq analysis tool Scanpy (Genome Biology, 2018). For more information, read scanpy documentation.

# # Intro to Filetypes
# 

# # Last Year Competitions
# * (Novel team solution of ADT2GEX task in predict modality - [Presentation](https://drive.google.com/file/d/1aQss-KyfYlzdrBQcH5joiXMlTwpG5gdf/view) 
# * [Code / Methodology](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods)

# # Appendix 
# 
# ## Cell Types 
# * MasP = Mast Cell Progenitor
# * MkP = Megakaryocyte Progenitor
# * NeuP = Neutrophil Progenitor
# * MoP = Monocyte Progenitor
# * EryP = Erythrocyte Progenitor
# * HSC = Hematoploetic Stem Cell
# * BP = B-Cell Progenitor

# # Thanks 
# At this point I just have aggregated the information from various notebooks and discussions as a way to keep track of all of the various notebooks 
# * Thomas Shelby - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344686
# * Daniel Burkhardt - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344607
# * Kaggle Data Details the team at Cellarity - https://www.kaggle.com/competitions/open-problems-multimodal/data
# * Peter Holderrieth - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/345958
# * Mar??lia Prat - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346686
# * Alireza - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346894
# * Jiwei Liu - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/348792
# * AMBROSM - https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/notebook
# * Lennard Henze - https://www.kaggle.com/code/leohash/complete-eda-of-mmscel-integration-data

# In[ ]:




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




#!/usr/bin/env python
# coding: utf-8

# inspired by : https://www.kaggle.com/code/jirkaborovec/mmscel-inst-eda-stat-predictions/notebook?scriptVersionId=103611408

# In[1]:


get_ipython().system(' pip install -q tables  # needed for loading HDF files')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import pandas as pd
from collections import Counter

PATH_DATASET = "/kaggle/input/open-problems-multimodal"

class MyDict(dict):
    def __missing__(self, key):
        return key


# In[3]:


df_meta = pd.read_csv(os.path.join(PATH_DATASET, "metadata.csv"))
display(df_meta.head())
print(f"table size: {len(df_meta)}")


# In[4]:


donors = list(df_meta.donor.unique())[1::]
days = list(df_meta.day.unique())
cell_typedic = dict(zip(df_meta.cell_type.unique(), range(1,9)))
cells =list(df_meta.cell_type.unique())[0:-1:]


# In[5]:


df_meta['cell_type'] = df_meta['cell_type'].map(cell_typedic)


# In[6]:


df_eval = pd.read_csv(os.path.join(PATH_DATASET, "evaluation_ids.csv"))
display(df_eval.head())
      
print(f"total: {len(df_eval)}")
print(f"cell_id: {len(df_eval['cell_id'].unique())}")
print(f"gene_id: {len(df_eval['gene_id'].unique())}")


# In[7]:


df_eval = df_eval.merge(df_meta[['cell_id', 'day', 'donor', 'cell_type', 'technology']], how = 'left', on = 'cell_id').set_index("cell_id")
df_meta = df_meta.set_index("cell_id")
df_eval['target'] = 0 
df_eval


# In[8]:


df = pd.read_hdf(os.path.join(PATH_DATASET, "train_cite_targets.h5")).astype(np.float16)
cols_target = list(df.columns)
df.head()


# In[9]:


df = df.join(df_meta[['day','donor','cell_type']], how="left")
print(f"total: {len(df)}")
print(f"cell_id: {len(df)}")
df.head()


# In[10]:


protein_pred = pd.DataFrame()
for cell in range(1,8):
    for donor in donors:
        df_p = df[(df.donor == donor) & (df.cell_type == cell)].groupby(['day']).aggregate('mean').reset_index()
        if len(df_p) < 3:
            df_p = df[(df.cell_type == cell)].groupby(['day']).aggregate('mean').reset_index()
            df_p.donor = donor
        protein_pred = pd.concat([protein_pred, df_p])
protein_pred = protein_pred.reset_index(drop = True)
protein_pred


# In[11]:


for cell in range(1,8):
    for day in days:
        df_p = df[(df.day == day) & (df.cell_type == cell)].groupby(['day']).aggregate('mean').reset_index()
        df_p.cell_type = cell
        df_p.donor = 27678 # o faltante
        protein_pred = pd.concat([protein_pred, df_p])
protein_pred = protein_pred.reset_index(drop = True)
protein_pred


# In[12]:


for donor in donors:
    for day in days:
        df_p = df[(df.day == day) & (df.donor == donor)].groupby(['day']).aggregate('mean').reset_index()
        df_p.cell_type = 8 #interesse faltante
        df_p.donor = donor
        protein_pred = pd.concat([protein_pred, df_p])
protein_pred = protein_pred.reset_index(drop = True)
protein_pred


# In[13]:


for day in days:
    df_p = df[(df.day == day)].groupby(['day']).aggregate('mean').reset_index()
    df_p.cell_type = 8 #interesse faltante
    df_p.donor = 27678 #interessante faltante
    protein_pred = pd.concat([protein_pred, df_p])
protein_pred = protein_pred.reset_index(drop = True)
protein_pred


# In[14]:


up_donors = [27678] + donors
for cell in range(1,9):
    for donor in up_donors:
        lista = [7]
        x = protein_pred.loc[(protein_pred.donor == donor) & (protein_pred.cell_type == cell)][['day']]
        x = np.array([valor[0] for valor in x.values])
        for feature in protein_pred.drop(columns = ['day', 'donor', 'cell_type']).columns:
            y = protein_pred.loc[(protein_pred.donor == donor) & (protein_pred.cell_type == cell)][[f'{feature}']].mean()[0]
            #y = np.array([valor[0] for valor in y.values])
            #modelo_7 = np.poly1d(np.polyfit(x.astype('float64') ,y.astype('float64') , 2))
            #lista.append(modelo_7(np.array([7]))[0])
            lista.append(y)

        lista.append(donor)
        lista.append(cell)
        df_novo = pd.DataFrame([lista], columns = protein_pred.columns)
        protein_pred = pd.concat([protein_pred, df_novo])  
protein_pred = protein_pred.reset_index(drop = True)
protein_pred


# In[15]:


protein_pred['technology'] = 'citeseq'
protein_pred


# In[16]:


get_ipython().run_cell_magic('time', '', 'df_final = pd.DataFrame()\nfor cell in range(1,8):\n    for donor in donors:\n        df_temp = pd.DataFrame()\n        for day in [2,3,4,7]:\n            col_sums_2 = []\n            count_2 = 0  \n            for i in range(11):\n                path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")\n                df = pd.read_hdf(path_h5, start=i * 10000, stop=(i+1) * 10000)\n                df = df.join(df_meta[[\'day\',\'donor\',\'cell_type\']][(df_meta.day == day) & (df_meta.donor == donor) & (df_meta.cell_type == cell)], how="inner")\n                count_2 += len(df)\n                col_sums_2.append(dict(df.sum()))\n\n            df_multi_ = pd.DataFrame(col_sums_2)\n\n            df = pd.DataFrame((df_multi_.sum() / count_2)).T\n            df_temp = pd.concat([df_temp, df])\n        df_final = pd.concat([df_final, df_temp])\ndf_final = df_final.reset_index(drop = True)\ndf_final')


# In[17]:


get_ipython().run_cell_magic('time', '', 'for cell in range(1,8):\n    for day in [2,3,4,7]:\n        col_sums_2 = []\n        count_2 = 0  \n        for i in range(11):\n            path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")\n            df = pd.read_hdf(path_h5, start=i * 10000, stop=(i+1) * 10000)\n            df = df.join(df_meta[[\'day\', \'donor\',\'cell_type\']][(df_meta.day == day) & (df_meta.cell_type == cell)], how="inner")\n            count_2 += len(df)\n            col_sums_2.append(dict(df.sum()))\n\n        df_multi_ = pd.DataFrame(col_sums_2)\n        df = pd.DataFrame((df_multi_.sum() / count_2)).T\n        df.cell_type = cell\n        df.donor = 27678 # o faltante\n        df_final = pd.concat([df_final, df])\ndf_final = df_final.reset_index(drop = True)\ndf_final')


# In[18]:


get_ipython().run_cell_magic('time', '', 'for donor in donors:\n    for day in [2,3,4,7]:\n        col_sums_2 = []\n        count_2 = 0  \n        for i in range(11):\n            path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")\n            df = pd.read_hdf(path_h5, start=i * 10000, stop=(i+1) * 10000)\n            df = df.join(df_meta[[\'day\', \'donor\',\'cell_type\']][(df_meta.day == day) & (df_meta.donor == donor)], how="inner")\n            count_2 += len(df)\n            col_sums_2.append(dict(df.sum()))\n\n        df_multi_ = pd.DataFrame(col_sums_2)\n        df = pd.DataFrame((df_multi_.sum() / count_2)).T\n        df.cell_type = 8 #interesse faltante\n        df.donor = donor\n        df_final = pd.concat([df_final, df])\ndf_final = df_final.reset_index(drop = True)\ndf_final')


# In[19]:


get_ipython().run_cell_magic('time', '', 'for day in [2,3,4,7]:\n    col_sums_2 = []\n    count_2 = 0  \n    for i in range(11):\n        path_h5 = os.path.join(PATH_DATASET, "train_multi_targets.h5")\n        df = pd.read_hdf(path_h5, start=i * 10000, stop=(i+1) * 10000)\n        df = df.join(df_meta[[\'day\', \'donor\',\'cell_type\']][(df_meta.day == day)], how="inner")\n        count_2 += len(df)\n        col_sums_2.append(dict(df.sum()))\n\n    df_multi_ = pd.DataFrame(col_sums_2)\n    df = pd.DataFrame((df_multi_.sum() / count_2)).T\n    df.cell_type = 8 #interesse faltante\n    df.donor = 27678 #interesse faltante\n    df_final = pd.concat([df_final, df])\ndf_final = df_final.reset_index(drop = True)\ndf_final')


# In[20]:


get_ipython().run_cell_magic('time', '', "for cell in range(1,9):\n    for donor in up_donors:\n        lista = []\n        x = df_final.loc[(df_final.donor == donor) & (df_final.cell_type == cell)][['day']]\n        x = np.array([valor[0] for valor in x.values])\n        for feature in df_final.drop(columns = ['day', 'donor', 'cell_type']).columns:\n            y = df_final.loc[(df_final.donor == donor) & (df_final.cell_type == cell)][[f'{feature}']].mean()[0]\n            #y = np.array([valor[0] for valor in y.values])\n            #modelo_10 = np.poly1d(np.polyfit(x.astype('float64') ,y.astype('float64') , 3))\n            #result = modelo_10(np.array([10]))[0]\n            #if result >=1:\n            #    lista.append(np.log(result))\n            #else:\n            #    lista.append(0)\n            lista.append(y)\n                \n        lista.append(10)\n        lista.append(donor)\n        lista.append(cell)\n        df_novo = pd.DataFrame([lista], columns = df_final.columns)\n        df_final = pd.concat([df_final, df_novo])\ndf_final = df_final.reset_index(drop = True)\ndf_final")


# In[21]:


df_final[df_final < 0] = 0


# In[22]:


df_final['technology'] = 'multiome'
df_final


# In[23]:


get_ipython().run_cell_magic('time', '', "for cell in range(1,9):\n    for donor in up_donors:\n        for dia in [2, 3, 4, 7, 10]:\n            dic = dict()\n            dic.update(dict(df_final[(df_final.day == dia) & (df_final.donor == donor) & (df_final.cell_type == cell) & (df_final.technology == 'multiome')].drop(columns = ['day', 'donor', 'technology']).mean()))\n            col_day = MyDict(dic)\n            df_eval['target'].loc[(df_eval.day == dia) & (df_eval.donor == donor) & (df_eval.cell_type == cell) & (df_eval.technology == 'multiome')] = df_eval.loc[(df_eval.day == dia) & (df_eval.donor == donor) & (df_eval.cell_type == cell) & (df_eval.technology == 'multiome')]['gene_id'].map(col_day)")


# In[24]:


df_eval


# In[25]:


get_ipython().run_cell_magic('time', '', "for cell in range(1,9):\n    for donor in up_donors:\n        for dia in [2, 3, 4, 7]:\n            dic = dict()\n            dic.update(dict(protein_pred[(protein_pred.day == dia) & (protein_pred.donor == donor) & (protein_pred.cell_type == cell) & (protein_pred.technology == 'citeseq')].drop(columns = ['day', 'donor', 'technology']).mean()))\n            col_day = MyDict(dic)\n            df_eval['target'].loc[(df_eval.day == dia) & (df_eval.donor == donor) & (df_eval.cell_type == cell) & (df_eval.technology == 'citeseq')] = df_eval.loc[(df_eval.day == dia) & (df_eval.donor == donor) & (df_eval.cell_type == cell) & (df_eval.technology == 'citeseq')]['gene_id'].map(col_day)")


# In[26]:


df_eval


# In[27]:


df_eval = df_eval.set_index('row_id')

print(f"total: {len(df_eval)}")
print(f"gene_id: {len(df_eval['gene_id'].unique())}")
df_eval[["target"]].round(6).to_csv("submission.csv")

get_ipython().system(' ls -lh .')
get_ipython().system(' head submission.csv')

#!/usr/bin/env python
# coding: utf-8

# # Open Problems - Multimodal Single-Cell Integration - ???? EDA

# #### I am a Data Science practitioner that is very passionate and motivated. In this notebook, I will focus on analyzing the data pertaining to the Open Problems - Multimodal Single-Cell Integration my progress step by step until I present my own conclusion for this problem.
# 
# #### If you found this Kernel insightful, please consider upvoting it ????
# 
# #### I am all ears if you have any advice :)

# <img src= "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUSExMVFhUVFRUVFRcVFRcVFxcVFRUXFhUVFxYYHSggGBolHRUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OFRAQFy0dHR0tLS0tLS0tLS0tLSstLS0rNy0rLS0tLS0tLS03LS0tKy0tLS0tLS0tKy0tLS0rLS0tK//AABEIAGcB6gMBIgACEQEDEQH/xAAaAAADAQEBAQAAAAAAAAAAAAAAAQIDBAUH/8QAMxAAAgIABAQFBAAGAgMAAAAAAAECEQMhMVESQWFxBBOBkfChsdHhIjJCUsHxBRRicrL/xAAZAQEBAQEBAQAAAAAAAAAAAAAAAQIDBAX/xAAeEQEBAQEAAgMBAQAAAAAAAAAAARECAzESIVFBYf/aAAwDAQACEQMRAD8A+iqXcL+MVitHVD4Vt7Crr7hQ7AXC/wDQrHY79e4CsOIGl2FwvuEUFkWOwHQcW4kxkBWwcRNApbhVMhx2KrYXEBNhZUo2Zu0BVpnPi+ET/l9ja0wcqLLZ6Trmde3nTi1kxWelKms8zlxfCc4+x057l9vP34bPufbnsLJdrJhZ0cF2FkWFkF2Fk2FgXYWRYWBdhZNhYF2FkWAwXYWRY7CqsdkWFgXYWTYWBVjsiwsC7CyLAC7CybCwLsVk2AF2FkhOfDlrJ6L8ktxvji93ItySVv0FPxDXTZc/0YJ0+J5yen66DWT3k/oZz9fQ44nMyKcnq83yW37KS3zf27k1XWT+ewLZev5ZXbnlpKd5DWXV/YmK5LXmzfCw6Ja6+hh4fN6m8UJFozalppDQJFIiaEOhOQWyJqOFizHfRjUuoeJPENSKJcV2AdgTwPkJt8wLsPmRCkOwq79Sa2ALCJY+Id+pNAVYMhoFIB6FcViTE0A6BSJUhvMiplh7EcTWppdA0mBn2GpbkyjQuLcC5xT1OTF8LWh0LoNSE6sTrjnr285xaFZ6M4J9OpzYvh/iOs8kvt5+vBZ6YWFg4COmyuNlns7CxAGVWFk2MCrAmwAqwJsdhTsdkhYRQCsLAqwJsYU7AmxgMBABQ0rElzeSIc70yjvzfYzbjv4/Devu/UVLErKOb5vlFEJVpm3vq++y6Av7Yr8Lq3zZV8OUc5PVmHt55kmQadZv2Q7UerevX9EOSiur92TBXn7/AIK68xcf9s0iuS05vcUY32+5uqQtdNxWHGjRMyTs3hAzU1UEaImwSsiariBIaiO9iIaiVQkh8ATUCcUMCPMjh2DifcsQEqSGDRHDsyhuKE4sanuPIIix8Q2vUlw2YDAhvcdlFX6irYVgQLQakO9yJR2CLZOnVfUniKUgaaYmthNc0Cn6MimpEyjsUyboKza9PsJvf3NWrM9ABSKTM+Hb2FxejILnhpmE8F/7NlIosthZK4ZR9CaO6Ub69GYzwNnXR/k3O3Lrw81zisuScdUTxI6Tpxvg/KACgaL8oxfF1AAhlYvNn8OwJsZUOwsQWDTHZNhZGpLf4qwI4th8Mui7mb1HXnw9X/FpC8xf0rif0RLgv6nf0Q7vRZeyM3rXp48PPP8Aoa5yd/8AyvyNJvPRb832XIVLu/p7CnidSO63OlUTKU0vmbMpYuw8OHN6msz21FwXN/OiN4RvtsSo1my0m+iJa3uL4+S9zTDwx4eEbR6GdNOMUilbEkWrMoaVFJ7AoloGkolIVhxERYEX19gyAmwFYWV5jEKwsBgICKCHHYoConjrUq7EyHHYC2vUhw2Gp7lMDLi3HZUl6mbhsBXEHzoZqRVlRTz11IlGirC/UgmMxvMUoXoZ8TQGnFXVFIhSsVVmgKaGpWKMgaIpSj8+aiu9Rp0DWwGbw2tCYzNEwaTIoUhsycGuqCOJ8YGleqMMTwyemRtYF2xXn4mE46/r3J4mel9TnxPDJ6Zfb9Gp1+sfFzeYNYhOLCUdV68jPjRvEb8fUfF2MLQ/UK3vogvp9jFDsEkaegW9vsRfULI20t7oXd+xm2hPFQxWqa5ITnuc8sYyeJZqcmumeMYObegRwm9TojBL58sv1GonCw/9m6dZLN/NRwwm+i+p0YWEloYtbiMPB5v52OiMQKim+hm1VVuWkwUEiuIhpxgXZnxAgmtOILIsfHsgatDozz7D4dwmrtbhxozeJBc17oP+xDcYmp4ugcZIWV59XYWZjsYauwsjiHZMXVWFk2FhdUILABNEU1pmaWIBRnZTREoWSpNa+4DnG9fczkmuqN0yWgMrGpClDbJ7EX6FRrYPZmaZSYEyg0EcQtMmUE9CIbQlKtTO2i1KxhrTUnT58sjTQqMyKd3qJ5fM/wBjaEmFCYpQTG1sTfzkBm4NfsI4n+ma2KUEwpWNmTw2tH6MSxayeQw1r9TnxfCRen8L6aexupDsS2L7eXieExI8rXT8HO5tapo9ythTz/mRueT9Z+LxPNDzup6k/CQf7SZlL/jtlF+6N/OHxrg84PPOx+Af9i90NeFl/YvdD5QyuJTb0LjhyZ2xwJ7JeqLXhpc2vS3+B82pHHHw25rHCSy+x1x8Oubb+dC00skZvTUjCGC+y+ptDBS/JSt6IuOC+ZnWk8RUYN9DSkhPEC6aikPjM2yfM2GJrYXGu5n3fp+hYmKoouJrZSf+iu79zz5+Ik+fDH6v8Gfmbflv15Gvgnyek8aK52Zz8bsvVnnYmKorP2OOeJKfb6G+fHrHXkx34/8AystI5vtl+Tmlf82JJvu/sjCM6ygs9zaGBWcnmdPjOfThfJa0hit/yqur/BrUtyodFXcunuZtTa9ABAcWzAQEDAQrBqrDiJsYXVWFkDsmLqrGQmFkXViYrACHGtPYqMxkyiBTInHcSnWpdg1hJNdUKzdoynh7ZPYupgUhmV7jTKmtG7yZnPDazRSY0QZxxC2hSgn3M3aB9xopbl3Zkp2NoLKtoLJUhsi6KCxWOyAsTQMQVDwdnRLlJaq+xrY7Ayhjo1UxSgnqjN4Gza+oXW1IXlbMx4ZLmn9BqclyGLrXy5bhwSIWM9mNY72+5cXVeXMfkS3J89h5svlBdaLwu7K8uKMqk9/nsHDWrS+dAa2eMlov8EPEb/RmpL+lN/YHB83XRFkPkHiJE8TfT5saRw0unV6gpf2r1ZU1Pl1nJjWfRfV9kDpZt31ei7I5sbGvnS35v8IshrTExqyjm+fT/wBn/hHPKSTt/wAUvt2Rm8S8lkvnv8seHh323+a/Y3Izott/Pn+BYuIo9WLHx1HJHJLLOWvJG5HProPPOWn3BfxdIoVNvPN8o/k2jHPPNrlyRtwvWqwYv+lUt3qzohFLqyG936foqNvTJfX3M0aSnu/RE8f/AI/X9GuH4c28kzsa+NdFhYAc2hYMAIoCwAAsAAKLAAICwsAAdhYARRYDANEyNOwgIi4yBgAWInHcxmnHqgAsSlZakAGmVA5bgBlWc8Lb2IjiABYlaJ2FgBLCKsTACNCwUgAKBMACgLAADjH5gADQpoPNQAXDQsXZD4pdhgF0uBvWXsChFcr7jACuIni2AApVvn9icTFr/CEBYOPGx983y2+dSHF6y71t3EB0/jDZQ5v2+cuhh4jxHJABrn7Z6rllKusmSlz57/gYHSOHdbYOG30X1ZrB8o+/4ACfqT+OjC8PzebOlJIAOVuunqJl4hLkR/23sgA3OY43ydP/2Q==" alt ="Titanic" style='width: 85%; margin-left: 7.5%
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
# Objective ???
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
# About the given Datasets ????????
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
# Evaluation Metrics Used ????
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
#    Work ????
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
# Importing relevant libraries ????????????
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
# Reading Data ????
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
# Metadata ????
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
# Exploratory Data Analysis and Visualization ???? ????
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
# Analysis of Missing Values ??????
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
# Conclusions ????
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
#  Data Analysis of Individual Predictors ????
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
# Conclusions ????
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
# Donor ????
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
# Conclusions ????
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
# Cell Type ????
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
# Conclusions ????
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
# Technology ???????????
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
# Conclusions ????
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
#    WORK IN PROGRESS ??????
# </h1>
#  
# </div>
# 
# ### Feel free to comment what you would like me to include in this Kernel. 
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
        #??Append data at the end of each array
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
#??indices = np.load('train_multi_inputs_indices_0.npy')
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


#??sps.save_npz('train_multiome_input_sparse.npz', csr_matrix)


# In[18]:


#??del csr_matrix, indices, indptr, data


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

#!/usr/bin/env python
# coding: utf-8

# # CITEseq LGBM Baseline
# 
# * This notebook will be implemented in the LGBM model using the data processed in the quick start. 
# * LGBM models usually cannot output multiple target variables, but this method can output
# 
# * The reference notes for data processing are below.
# https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart

# # Please vote if this is useful!

# In[1]:


import os, gc, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore, Back, Style
from matplotlib.ticker import MaxNLocator
import warnings
warnings.simplefilter('ignore')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

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


# In[2]:


import optuna


# In[3]:


get_ipython().system('pip install --quiet tables')


# # Loading the common metadata table
# 
# The current version of the model is so primitive that it doesn't use the metadata, but we load it anyway.

# In[4]:


# df_cell = pd.read_csv(FP_CELL_METADATA)
# df_cell_cite = df_cell[df_cell.technology=="citeseq"]
# df_cell_multi = df_cell[df_cell.technology=="multiome"]
# df_cell_cite.shape, df_cell_multi.shape


# # Cross-validation
# To get a result with only 16 GByte RAM, we simplify the problem as follows:
# - We ignore the complete metadata (donors, days, cell types).
# - We drop all feature columns which are constant.
# - We do a PCA and keep only the 512 most important components.
# - We use PCA(copy=False), which overwrites its input in fit_transform().
# - We fit a ridge regression model with 70988\*512 inputs and 70988\*140 outputs. 

# In[5]:


get_ipython().run_cell_magic('time', '', 'class Preprocess(BaseEstimator, TransformerMixin):\n    def transform(self, X):\n        print(X.shape)\n        gc.collect()\n        X = self.pca.transform(X)\n        print(X.shape)\n        return X\n\n    def fit_transform(self, X):\n        print(X.shape)\n        gc.collect()\n        self.pca = PCA(n_components=512, copy=False, random_state=42)\n        X = self.pca.fit_transform(X)\n        print(X.shape)\n        return X')


# # Modeling&Prediction Cite
# 
# We retrain the model on all training rows, delete the training data, load the test data and compute the predictions.

# In[6]:


cnam_cite_start = ['ENSG00000000003_TSPAN6', 'ENSG00000000419_DPM1', 'ENSG00000000457_SCYL3', 'ENSG00000000460_C1orf112', 'ENSG00000000938_FGR', 'ENSG00000000971_CFH', 'ENSG00000001036_FUCA2', 'ENSG00000001084_GCLC', 'ENSG00000001167_NFYA', 'ENSG00000001460_STPG1', 'ENSG00000001461_NIPAL3', 'ENSG00000001497_LAS1L', 'ENSG00000001561_ENPP4', 'ENSG00000001617_SEMA3F', 'ENSG00000001629_ANKIB1', 'ENSG00000001630_CYP51A1', 'ENSG00000001631_KRIT1', 'ENSG00000002016_RAD52', 'ENSG00000002330_BAD', 'ENSG00000002549_LAP3', 'ENSG00000002586_CD99', 'ENSG00000002587_HS3ST1', 'ENSG00000002726_AOC1', 'ENSG00000002822_MAD1L1', 'ENSG00000002834_LASP1', 'ENSG00000002919_SNX11', 'ENSG00000002933_TMEM176A', 'ENSG00000003056_M6PR', 'ENSG00000003096_KLHL13', 'ENSG00000003147_ICA1', 'ENSG00000003249_DBNDD1', 'ENSG00000003393_ALS2', 'ENSG00000003400_CASP10', 'ENSG00000003402_CFLAR', 'ENSG00000003436_TFPI', 'ENSG00000003509_NDUFAF7', 'ENSG00000003756_RBM5', 'ENSG00000003987_MTMR7', 'ENSG00000003989_SLC7A2', 'ENSG00000004059_ARF5', 'ENSG00000004139_SARM1', 'ENSG00000004142_POLDIP2', 'ENSG00000004399_PLXND1', 'ENSG00000004455_AK2', 'ENSG00000004468_CD38', 'ENSG00000004478_FKBP4', 'ENSG00000004487_KDM1A', 'ENSG00000004534_RBM6', 'ENSG00000004660_CAMKK1', 'ENSG00000004700_RECQL', 'ENSG00000004766_VPS50', 'ENSG00000004776_HSPB6', 'ENSG00000004777_ARHGAP33', 'ENSG00000004779_NDUFAB1', 'ENSG00000004799_PDK4', 'ENSG00000004809_SLC22A16', 'ENSG00000004838_ZMYND10', 'ENSG00000004864_SLC25A13', 'ENSG00000004866_ST7', 'ENSG00000004897_CDC27', 'ENSG00000004939_SLC4A1', 'ENSG00000004961_HCCS', 'ENSG00000004975_DVL2', 'ENSG00000005007_UPF1', 'ENSG00000005020_SKAP2', 'ENSG00000005022_SLC25A5', 'ENSG00000005059_MCUB', 'ENSG00000005073_HOXA11', 'ENSG00000005075_POLR2J', 'ENSG00000005100_DHX33', 'ENSG00000005108_THSD7A', 'ENSG00000005156_LIG3', 'ENSG00000005175_RPAP3', 'ENSG00000005187_ACSM3', 'ENSG00000005189_REXO5', 'ENSG00000005194_CIAPIN1', 'ENSG00000005206_SPPL2B', 'ENSG00000005238_FAM214B', 'ENSG00000005243_COPZ2', 'ENSG00000005249_PRKAR2B', 'ENSG00000005302_MSL3', 'ENSG00000005339_CREBBP', 'ENSG00000005379_TSPOAP1', 'ENSG00000005381_MPO', 'ENSG00000005436_GCFC2', 'ENSG00000005448_WDR54', 'ENSG00000005469_CROT', 'ENSG00000005471_ABCB4', 'ENSG00000005483_KMT2E', 'ENSG00000005486_RHBDD2', 'ENSG00000005700_IBTK', 'ENSG00000005801_ZNF195', 'ENSG00000005810_MYCBP2', 'ENSG00000005812_FBXL3', 'ENSG00000005844_ITGAL', 'ENSG00000005882_PDK2', 'ENSG00000005884_ITGA3', 'ENSG00000005889_ZFX', 'ENSG00000005893_LAMP2', 'ENSG00000005961_ITGA2B', 'ENSG00000006007_GDE1', 'ENSG00000006015_REX1BD', 'ENSG00000006016_CRLF1', 'ENSG00000006025_OSBPL7', 'ENSG00000006042_TMEM98', 'ENSG00000006047_YBX2', 'ENSG00000006062_MAP3K14', 'ENSG00000006125_AP2B1', 'ENSG00000006194_ZNF263', 'ENSG00000006282_SPATA20', 'ENSG00000006327_TNFRSF12A', 'ENSG00000006432_MAP3K9', 'ENSG00000006451_RALA', 'ENSG00000006453_BAIAP2L1', 'ENSG00000006459_KDM7A', 'ENSG00000006468_ETV1', 'ENSG00000006530_AGK', 'ENSG00000006534_ALDH3B1', 'ENSG00000006555_TTC22', 'ENSG00000006576_PHTF2', 'ENSG00000006607_FARP2', 'ENSG00000006625_GGCT', 'ENSG00000006634_DBF4', 'ENSG00000006638_TBXA2R', 'ENSG00000006652_IFRD1', 'ENSG00000006659_LGALS14', 'ENSG00000006695_COX10', 'ENSG00000006704_GTF2IRD1', 'ENSG00000006712_PAF1', 'ENSG00000006715_VPS41', 'ENSG00000006740_ARHGAP44', 'ENSG00000006744_ELAC2', 'ENSG00000006747_SCIN', 'ENSG00000006756_ARSD', 'ENSG00000006757_PNPLA4', 'ENSG00000006831_ADIPOR2', 'ENSG00000006837_CDKL3', 'ENSG00000007038_PRSS21', 'ENSG00000007047_MARK4', 'ENSG00000007062_PROM1', 'ENSG00000007080_CCDC124', 'ENSG00000007129_CEACAM21', 'ENSG00000007168_PAFAH1B1', 'ENSG00000007202_KIAA0100', 'ENSG00000007237_GAS7', 'ENSG00000007255_TRAPPC6A', 'ENSG00000007264_MATK', 'ENSG00000007312_CD79B', 'ENSG00000007314_SCN4A', 'ENSG00000007341_ST7L', 'ENSG00000007372_PAX6', 'ENSG00000007376_RPUSD1', 'ENSG00000007384_RHBDF1', 'ENSG00000007392_LUC7L', 'ENSG00000007402_CACNA2D2', 'ENSG00000007516_BAIAP3', 'ENSG00000007520_TSR3', 'ENSG00000007541_PIGQ', 'ENSG00000007545_CRAMP1', 'ENSG00000007866_TEAD3', 'ENSG00000007923_DNAJC11', 'ENSG00000007944_MYLIP', 'ENSG00000007968_E2F2', 'ENSG00000008018_PSMB1', 'ENSG00000008056_SYN1', 'ENSG00000008083_JARID2', 'ENSG00000008086_CDKL5', 'ENSG00000008128_CDK11A', 'ENSG00000008130_NADK', 'ENSG00000008226_DLEC1', 'ENSG00000008256_CYTH3', 'ENSG00000008277_ADAM22', 'ENSG00000008282_SYPL1', 'ENSG00000008283_CYB561', 'ENSG00000008294_SPAG9', 'ENSG00000008300_CELSR3', 'ENSG00000008311_AASS', 'ENSG00000008323_PLEKHG6', 'ENSG00000008324_SS18L2', 'ENSG00000008382_MPND', 'ENSG00000008394_MGST1', 'ENSG00000008405_CRY1', 'ENSG00000008441_NFIX', 'ENSG00000008513_ST3GAL1', 'ENSG00000008516_MMP25', 'ENSG00000008517_IL32', 'ENSG00000008710_PKD1', 'ENSG00000008735_MAPK8IP2', 'ENSG00000008838_MED24', 'ENSG00000008853_RHOBTB2', 'ENSG00000008869_HEATR5B', 'ENSG00000008952_SEC62', 'ENSG00000008988_RPS20', 'ENSG00000009307_CSDE1', 'ENSG00000009335_UBE3C', 'ENSG00000009413_REV3L', 'ENSG00000009724_MASP2', 'ENSG00000009765_IYD', 'ENSG00000009780_FAM76A', 'ENSG00000009790_TRAF3IP3', 'ENSG00000009830_POMT2', 'ENSG00000009844_VTA1', 'ENSG00000009950_MLXIPL', 'ENSG00000009954_BAZ1B', 'ENSG00000010017_RANBP9', 'ENSG00000010030_ETV7', 'ENSG00000010072_SPRTN', 'ENSG00000010165_METTL13', 'ENSG00000010219_DYRK4', 'ENSG00000010244_ZNF207', 'ENSG00000010256_UQCRC1', 'ENSG00000010270_STARD3NL', 'ENSG00000010278_CD9', 'ENSG00000010292_NCAPD2', 'ENSG00000010295_IFFO1', 'ENSG00000010310_GIPR', 'ENSG00000010318_PHF7', 'ENSG00000010319_SEMA3G', 'ENSG00000010322_NISCH', 'ENSG00000010327_STAB1', 'ENSG00000010361_FUZ', 'ENSG00000010404_IDS', 'ENSG00000010438_PRSS3', 'ENSG00000010539_ZNF200', 'ENSG00000010610_CD4', 'ENSG00000010626_LRRC23', 'ENSG00000010671_BTK', 'ENSG00000010704_HFE', 'ENSG00000010803_SCMH1', 'ENSG00000010810_FYN', 'ENSG00000010818_HIVEP2', 'ENSG00000011007_ELOA', 'ENSG00000011009_LYPLA2', 'ENSG00000011021_CLCN6', 'ENSG00000011028_MRC2', 'ENSG00000011052_NME1-NME2', 'ENSG00000011105_TSPAN9', 'ENSG00000011114_BTBD7', 'ENSG00000011132_APBA3', 'ENSG00000011143_MKS1', 'ENSG00000011198_ABHD5', 'ENSG00000011201_ANOS1', 'ENSG00000011243_AKAP8L', 'ENSG00000011258_MBTD1', 'ENSG00000011260_UTP18', 'ENSG00000011275_RNF216', 'ENSG00000011295_TTC19', 'ENSG00000011304_PTBP1', 'ENSG00000011332_DPF1', 'ENSG00000011376_LARS2', 'ENSG00000011405_PIK3C2A', 'ENSG00000011422_PLAUR', 'ENSG00000011426_ANLN', 'ENSG00000011451_WIZ', 'ENSG00000011454_RABGAP1', 'ENSG00000011478_QPCTL', 'ENSG00000011485_PPP5C', 'ENSG00000011523_CEP68', 'ENSG00000011566_MAP4K3', 'ENSG00000011590_ZBTB32', 'ENSG00000011600_TYROBP', 'ENSG00000011638_TMEM159', 'ENSG00000012048_BRCA1', 'ENSG00000012061_ERCC1', 'ENSG00000012124_CD22', 'ENSG00000012171_SEMA3B', 'ENSG00000012174_MBTPS2', 'ENSG00000012211_PRICKLE3', 'ENSG00000012223_LTF', 'ENSG00000012232_EXTL3', 'ENSG00000012660_ELOVL5', 'ENSG00000012779_ALOX5', 'ENSG00000012817_KDM5D', 'ENSG00000012822_CALCOCO1', 'ENSG00000012963_UBR7', 'ENSG00000012983_MAP4K5', 'ENSG00000013016_EHD3', 'ENSG00000013275_PSMC4', 'ENSG00000013288_MAN2B2', 'ENSG00000013306_SLC25A39', 'ENSG00000013364_MVP', 'ENSG00000013374_NUB1', 'ENSG00000013375_PGM3', 'ENSG00000013392_RWDD2A', 'ENSG00000013441_CLK1', 'ENSG00000013503_POLR3B', 'ENSG00000013523_ANGEL1', 'ENSG00000013561_RNF14', 'ENSG00000013563_DNASE1L1', 'ENSG00000013573_DDX11', 'ENSG00000013583_HEBP1', 'ENSG00000013619_MAMLD1', 'ENSG00000013725_CD6', 'ENSG00000013810_TACC3', 'ENSG00000014123_UFL1', 'ENSG00000014138_POLA2', 'ENSG00000014164_ZC3H3', 'ENSG00000014216_CAPN1', 'ENSG00000014257_ACPP', 'ENSG00000014641_MDH1', 'ENSG00000014824_SLC30A9', 'ENSG00000014914_MTMR11', 'ENSG00000014919_COX15', 'ENSG00000015133_CCDC88C', 'ENSG00000015153_YAF2', 'ENSG00000015171_ZMYND11', 'ENSG00000015285_WAS', 'ENSG00000015475_BID', 'ENSG00000015479_MATR3', 'ENSG00000015532_XYLT2', 'ENSG00000015568_RGPD5', 'ENSG00000015676_NUDCD3', 'ENSG00000016391_CHDH', 'ENSG00000016864_GLT8D1', 'ENSG00000017260_ATP2C1', 'ENSG00000017483_SLC38A5', 'ENSG00000017797_RALBP1', 'ENSG00000018189_RUFY3', 'ENSG00000018280_SLC11A1', 'ENSG00000018408_WWTR1', 'ENSG00000018510_AGPS', 'ENSG00000018610_CXorf56', 'ENSG00000018699_TTC27', 'ENSG00000018869_ZNF582', 'ENSG00000019144_PHLDB1', 'ENSG00000019485_PRDM11', 'ENSG00000019582_CD74', 'ENSG00000019991_HGF', 'ENSG00000019995_ZRANB1', 'ENSG00000020129_NCDN', 'ENSG00000020181_ADGRA2', 'ENSG00000020256_ZFP64', 'ENSG00000020426_MNAT1', 'ENSG00000020577_SAMD4A', 'ENSG00000020633_RUNX3', 'ENSG00000020922_MRE11', 'ENSG00000021300_PLEKHB1', 'ENSG00000021355_SERPINB1', 'ENSG00000021574_SPAST', 'ENSG00000021762_OSBPL5', 'ENSG00000021776_AQR', 'ENSG00000021826_CPS1', 'ENSG00000022267_FHL1', 'ENSG00000022277_RTF2', 'ENSG00000022556_NLRP2', 'ENSG00000022567_SLC45A4', 'ENSG00000022840_RNF10', 'ENSG00000022976_ZNF839', 'ENSG00000023041_ZDHHC6', 'ENSG00000023171_GRAMD1B', 'ENSG00000023191_RNH1', 'ENSG00000023228_NDUFS1', 'ENSG00000023287_RB1CC1', 'ENSG00000023318_ERP44', 'ENSG00000023330_ALAS1', 'ENSG00000023445_BIRC3', 'ENSG00000023516_AKAP11', 'ENSG00000023572_GLRX2', 'ENSG00000023608_SNAPC1', 'ENSG00000023697_DERA', 'ENSG00000023734_STRAP', 'ENSG00000023839_ABCC2', 'ENSG00000023892_DEF6', 'ENSG00000023902_PLEKHO1', 'ENSG00000023909_GCLM', 'ENSG00000024048_UBR2', 'ENSG00000024422_EHD2', 'ENSG00000024526_DEPDC1', 'ENSG00000024862_CCDC28A', 'ENSG00000025039_RRAGD', 'ENSG00000025156_HSF2', 'ENSG00000025293_PHF20', 'ENSG00000025423_HSD17B6', 'ENSG00000025434_NR1H3', 'ENSG00000025708_TYMP', 'ENSG00000025770_NCAPH2', 'ENSG00000025772_TOMM34', 'ENSG00000025796_SEC63', 'ENSG00000025800_KPNA6', 'ENSG00000026025_VIM', 'ENSG00000026036_RTEL1-TNFRSF6B', 'ENSG00000026103_FAS', 'ENSG00000026297_RNASET2', 'ENSG00000026508_CD44', 'ENSG00000026652_AGPAT4', 'ENSG00000026751_SLAMF7', 'ENSG00000026950_BTN3A1', 'ENSG00000027001_MIPEP', 'ENSG00000027075_PRKCH', 'ENSG00000027697_IFNGR1', 'ENSG00000027847_B4GALT7', 'ENSG00000027869_SH2D2A', 'ENSG00000028116_VRK2', 'ENSG00000028137_TNFRSF1B', 'ENSG00000028203_VEZT', 'ENSG00000028277_POU2F2', 'ENSG00000028310_BRD9', 'ENSG00000028528_SNX1', 'ENSG00000028839_TBPL1', 'ENSG00000029153_ARNTL2', 'ENSG00000029363_BCLAF1', 'ENSG00000029364_SLC39A9', 'ENSG00000029534_ANK1', 'ENSG00000029639_TFB1M', 'ENSG00000029725_RABEP1', 'ENSG00000029993_HMGB3', 'ENSG00000030066_NUP160', 'ENSG00000030110_BAK1', 'ENSG00000030419_IKZF2', 'ENSG00000030582_GRN', 'ENSG00000031003_FAM13B', 'ENSG00000031081_ARHGAP31', 'ENSG00000031691_CENPQ', 'ENSG00000031698_SARS', 'ENSG00000031823_RANBP3', 'ENSG00000032219_ARID4A', 'ENSG00000032389_EIPR1', 'ENSG00000032444_PNPLA6', 'ENSG00000032742_IFT88', 'ENSG00000033011_ALG1', 'ENSG00000033030_ZCCHC8', 'ENSG00000033050_ABCF2', 'ENSG00000033100_CHPF2', 'ENSG00000033122_LRRC7', 'ENSG00000033170_FUT8', 'ENSG00000033178_UBA6', 'ENSG00000033327_GAB2', 'ENSG00000033627_ATP6V0A1', 'ENSG00000033800_PIAS1', 'ENSG00000033867_SLC4A7', 'ENSG00000034053_APBA2', 'ENSG00000034152_MAP2K3', 'ENSG00000034510_TMSB10', 'ENSG00000034533_ASTE1', 'ENSG00000034677_RNF19A', 'ENSG00000034693_PEX3', 'ENSG00000034713_GABARAPL2', 'ENSG00000035115_SH3YL1', 'ENSG00000035141_FAM136A', 'ENSG00000035403_VCL', 'ENSG00000035499_DEPDC1B', 'ENSG00000035664_DAPK2', 'ENSG00000035681_NSMAF', 'ENSG00000035687_ADSS', 'ENSG00000035720_STAP1', 'ENSG00000035862_TIMP2', 'ENSG00000035928_RFC1', 'ENSG00000036054_TBC1D23', 'ENSG00000036257_CUL3', 'ENSG00000036448_MYOM2', 'ENSG00000036530_CYP46A1', 'ENSG00000036549_AC118549.1', 'ENSG00000036672_USP2', 'ENSG00000037042_TUBG2', 'ENSG00000037241_RPL26L1', 'ENSG00000037280_FLT4', 'ENSG00000037474_NSUN2', 'ENSG00000037637_FBXO42', 'ENSG00000037749_MFAP3', 'ENSG00000037757_MRI1', 'ENSG00000037897_METTL1', 'ENSG00000038002_AGA', 'ENSG00000038210_PI4K2B', 'ENSG00000038219_BOD1L1', 'ENSG00000038274_MAT2B', 'ENSG00000038358_EDC4', 'ENSG00000038382_TRIO', 'ENSG00000038427_VCAN', 'ENSG00000038532_CLEC16A', 'ENSG00000039068_CDH1', 'ENSG00000039123_MTREX', 'ENSG00000039139_DNAH5', 'ENSG00000039319_ZFYVE16', 'ENSG00000039523_RIPOR1', 'ENSG00000039560_RAI14', 'ENSG00000039650_PNKP', 'ENSG00000040199_PHLPP2', 'ENSG00000040275_SPDL1', 'ENSG00000040341_STAU2', 'ENSG00000040487_PQLC2', 'ENSG00000040531_CTNS', 'ENSG00000040608_RTN4R', 'ENSG00000040633_PHF23', 'ENSG00000040933_INPP4A', 'ENSG00000041353_RAB27B', 'ENSG00000041357_PSMA4', 'ENSG00000041515_MYO16', 'ENSG00000041802_LSG1', 'ENSG00000041880_PARP3', 'ENSG00000041988_THAP3', 'ENSG00000042062_RIPOR3', 'ENSG00000042088_TDP1', 'ENSG00000042286_AIFM2', 'ENSG00000042317_SPATA7', 'ENSG00000042429_MED17', 'ENSG00000042445_RETSAT', 'ENSG00000042493_CAPG', 'ENSG00000042753_AP2S1', 'ENSG00000042813_ZPBP', 'ENSG00000042980_ADAM28', 'ENSG00000043093_DCUN1D1', 'ENSG00000043143_JADE2', 'ENSG00000043462_LCP2', 'ENSG00000043514_TRIT1', 'ENSG00000043591_ADRB1', 'ENSG00000044090_CUL7', 'ENSG00000044115_CTNNA1', 'ENSG00000044446_PHKA2', 'ENSG00000044459_CNTLN', 'ENSG00000044574_HSPA5', 'ENSG00000046604_DSG2', 'ENSG00000046647_GEMIN8', 'ENSG00000046651_OFD1', 'ENSG00000046653_GPM6B', 'ENSG00000046889_PREX2', 'ENSG00000047056_WDR37', 'ENSG00000047188_YTHDC2', 'ENSG00000047230_CTPS2', 'ENSG00000047249_ATP6V1H', 'ENSG00000047315_POLR2B', 'ENSG00000047346_FAM214A', 'ENSG00000047365_ARAP2', 'ENSG00000047410_TPR', 'ENSG00000047457_CP', 'ENSG00000047578_KIAA0556', 'ENSG00000047579_DTNBP1', 'ENSG00000047597_XK', 'ENSG00000047621_C12orf4', 'ENSG00000047634_SCML1', 'ENSG00000047644_WWC3', 'ENSG00000047648_ARHGAP6', 'ENSG00000047662_FAM184B', 'ENSG00000047849_MAP4', 'ENSG00000047932_GOPC', 'ENSG00000048028_USP28', 'ENSG00000048052_HDAC9', 'ENSG00000048140_TSPAN17', 'ENSG00000048162_NOP16', 'ENSG00000048342_CC2D2A', 'ENSG00000048392_RRM2B', 'ENSG00000048405_ZNF800', 'ENSG00000048471_SNX29', 'ENSG00000048544_MRPS10', 'ENSG00000048649_RSF1', 'ENSG00000048707_VPS13D', 'ENSG00000048740_CELF2', 'ENSG00000048828_FAM120A', 'ENSG00000048991_R3HDM1', 'ENSG00000049089_COL9A2', 'ENSG00000049130_KITLG', 'ENSG00000049167_ERCC8', 'ENSG00000049192_ADAMTS6', 'ENSG00000049239_H6PD', 'ENSG00000049245_VAMP3', 'ENSG00000049246_PER3', 'ENSG00000049247_UTS2', 'ENSG00000049249_TNFRSF9', 'ENSG00000049323_LTBP1', 'ENSG00000049449_RCN1', 'ENSG00000049540_ELN', 'ENSG00000049541_RFC2', 'ENSG00000049618_ARID1B', 'ENSG00000049656_CLPTM1L', 'ENSG00000049759_NEDD4L', 'ENSG00000049768_FOXP3', 'ENSG00000049769_PPP1R3F', 'ENSG00000049860_HEXB', 'ENSG00000049883_PTCD2', 'ENSG00000050030_NEXMIF', 'ENSG00000050130_JKAMP', 'ENSG00000050327_ARHGEF5', 'ENSG00000050344_NFE2L3', 'ENSG00000050393_MCUR1', 'ENSG00000050405_LIMA1', 'ENSG00000050426_LETMD1', 'ENSG00000050438_SLC4A8', 'ENSG00000050555_LAMC3', 'ENSG00000050628_PTGER3', 'ENSG00000050730_TNIP3', 'ENSG00000050748_MAPK9', 'ENSG00000050767_COL23A1', 'ENSG00000050820_BCAR1', 'ENSG00000051009_FAM160A2', 'ENSG00000051108_HERPUD1', 'ENSG00000051128_HOMER3', 'ENSG00000051180_RAD51', 'ENSG00000051341_POLQ', 'ENSG00000051382_PIK3CB', 'ENSG00000051523_CYBA', 'ENSG00000051596_THOC3', 'ENSG00000051620_HEBP2', 'ENSG00000051825_MPHOSPH9', 'ENSG00000052126_PLEKHA5', 'ENSG00000052723_SIKE1', 'ENSG00000052749_RRP12', 'ENSG00000052795_FNIP2', 'ENSG00000052802_MSMO1', 'ENSG00000052841_TTC17', 'ENSG00000053108_FSTL4', 'ENSG00000053254_FOXN3', 'ENSG00000053328_METTL24', 'ENSG00000053371_AKR7A2', 'ENSG00000053372_MRTO4', 'ENSG00000053438_NNAT', 'ENSG00000053501_USE1', 'ENSG00000053524_MCF2L2', 'ENSG00000053702_NRIP2', 'ENSG00000053747_LAMA3', 'ENSG00000053770_AP5M1', 'ENSG00000053900_ANAPC4', 'ENSG00000053918_KCNQ1', 'ENSG00000054116_TRAPPC3', 'ENSG00000054118_THRAP3', 'ENSG00000054148_PHPT1', 'ENSG00000054219_LY75', 'ENSG00000054267_ARID4B', 'ENSG00000054277_OPN3', 'ENSG00000054282_SDCCAG8', 'ENSG00000054356_PTPRN', 'ENSG00000054392_HHAT', 'ENSG00000054523_KIF1B', 'ENSG00000054598_FOXC1', 'ENSG00000054611_TBC1D22A', 'ENSG00000054654_SYNE2', 'ENSG00000054690_PLEKHH1', 'ENSG00000054793_ATP9A', 'ENSG00000054965_FAM168A', 'ENSG00000054967_RELT', 'ENSG00000054983_GALC', 'ENSG00000055044_NOP58', 'ENSG00000055070_SZRD1', 'ENSG00000055118_KCNH2', 'ENSG00000055130_CUL1', 'ENSG00000055147_FAM114A2', 'ENSG00000055163_CYFIP2', 'ENSG00000055208_TAB2', 'ENSG00000055211_GINM1', 'ENSG00000055332_EIF2AK2', 'ENSG00000055483_USP36', 'ENSG00000055609_KMT2C', 'ENSG00000055732_MCOLN3', 'ENSG00000055917_PUM2', 'ENSG00000055950_MRPL43', 'ENSG00000055955_ITIH4', 'ENSG00000056050_HPF1', 'ENSG00000056097_ZFR', 'ENSG00000056277_ZNF280C', 'ENSG00000056558_TRAF1', 'ENSG00000056586_RC3H2', 'ENSG00000056736_IL17RB', 'ENSG00000056972_TRAF3IP2', 'ENSG00000056998_GYG2', 'ENSG00000057019_DCBLD2', 'ENSG00000057252_SOAT1', 'ENSG00000057294_PKP2', 'ENSG00000057593_F7', 'ENSG00000057608_GDI2', 'ENSG00000057657_PRDM1', 'ENSG00000057663_ATG5', 'ENSG00000057704_TMCC3', 'ENSG00000057757_PITHD1', 'ENSG00000057935_MTA3', 'ENSG00000058056_USP13', 'ENSG00000058063_ATP11B', 'ENSG00000058091_CDK14', 'ENSG00000058262_SEC61A1', 'ENSG00000058272_PPP1R12A', 'ENSG00000058335_RASGRF1', 'ENSG00000058404_CAMK2B', 'ENSG00000058453_CROCC', 'ENSG00000058600_POLR3E', 'ENSG00000058668_ATP2B4', 'ENSG00000058673_ZC3H11A', 'ENSG00000058729_RIOK2', 'ENSG00000058799_YIPF1', 'ENSG00000058804_NDC1', 'ENSG00000058866_DGKG', 'ENSG00000059122_FLYWCH1', 'ENSG00000059145_UNKL', 'ENSG00000059377_TBXAS1', 'ENSG00000059378_PARP12', 'ENSG00000059573_ALDH18A1', 'ENSG00000059588_TARBP1', 'ENSG00000059691_GATB', 'ENSG00000059728_MXD1', 'ENSG00000059758_CDK17', 'ENSG00000059769_DNAJC25', 'ENSG00000059804_SLC2A3', 'ENSG00000059915_PSD', 'ENSG00000060069_CTDP1', 'ENSG00000060138_YBX3', 'ENSG00000060140_STYK1', 'ENSG00000060237_WNK1', 'ENSG00000060339_CCAR1', 'ENSG00000060491_OGFR', 'ENSG00000060558_GNA15', 'ENSG00000060642_PIGV', 'ENSG00000060656_PTPRU', 'ENSG00000060688_SNRNP40', 'ENSG00000060749_QSER1', 'ENSG00000060762_MPC1', 'ENSG00000060971_ACAA1', 'ENSG00000060982_BCAT1', 'ENSG00000061273_HDAC7', 'ENSG00000061656_SPAG4', 'ENSG00000061676_NCKAP1', 'ENSG00000061794_MRPS35', 'ENSG00000061918_GUCY1B1', 'ENSG00000061936_SFSWAP', 'ENSG00000061938_TNK2', 'ENSG00000061987_MON2', 'ENSG00000062194_GPBP1', 'ENSG00000062282_DGAT2', 'ENSG00000062370_ZNF112', 'ENSG00000062485_CS', 'ENSG00000062524_LTK', 'ENSG00000062582_MRPS24', 'ENSG00000062598_ELMO2', 'ENSG00000062650_WAPL', 'ENSG00000062716_VMP1', 'ENSG00000062725_APPBP2', 'ENSG00000062822_POLD1', 'ENSG00000063015_SEZ6', 'ENSG00000063046_EIF4B', 'ENSG00000063127_SLC6A16', 'ENSG00000063169_BICRA', 'ENSG00000063176_SPHK2', 'ENSG00000063177_RPL18', 'ENSG00000063180_CA11', 'ENSG00000063241_ISOC2', 'ENSG00000063244_U2AF2', 'ENSG00000063245_EPN1', 'ENSG00000063322_MED29', 'ENSG00000063438_AHRR', 'ENSG00000063587_ZNF275', 'ENSG00000063601_MTMR1', 'ENSG00000063660_GPC1', 'ENSG00000063761_ADCK1', 'ENSG00000063854_HAGH', 'ENSG00000063978_RNF4', 'ENSG00000064012_CASP8', 'ENSG00000064042_LIMCH1', 'ENSG00000064102_INTS13', 'ENSG00000064115_TM7SF3', 'ENSG00000064199_SPA17', 'ENSG00000064201_TSPAN32', 'ENSG00000064205_WISP2', 'ENSG00000064225_ST3GAL6', 'ENSG00000064309_CDON', 'ENSG00000064313_TAF2', 'ENSG00000064393_HIPK2', 'ENSG00000064419_TNPO3', 'ENSG00000064490_RFXANK', 'ENSG00000064545_TMEM161A', 'ENSG00000064547_LPAR2', 'ENSG00000064601_CTSA', 'ENSG00000064607_SUGP2', 'ENSG00000064651_SLC12A2', 'ENSG00000064652_SNX24', 'ENSG00000064666_CNN2', 'ENSG00000064687_ABCA7', 'ENSG00000064703_DDX20', 'ENSG00000064726_BTBD1', 'ENSG00000064763_FAR2', 'ENSG00000064932_SBNO2', 'ENSG00000064933_PMS1', 'ENSG00000064961_HMG20B', 'ENSG00000064989_CALCRL', 'ENSG00000064995_TAF11', 'ENSG00000064999_ANKS1A', 'ENSG00000065000_AP3D1', 'ENSG00000065029_ZNF76', 'ENSG00000065054_SLC9A3R2', 'ENSG00000065057_NTHL1', 'ENSG00000065060_UHRF1BP1', 'ENSG00000065135_GNAI3', 'ENSG00000065150_IPO5', 'ENSG00000065154_OAT', 'ENSG00000065183_WDR3', 'ENSG00000065243_PKN2', 'ENSG00000065268_WDR18', 'ENSG00000065308_TRAM2', 'ENSG00000065320_NTN1', 'ENSG00000065328_MCM10', 'ENSG00000065357_DGKA', 'ENSG00000065361_ERBB3', 'ENSG00000065413_ANKRD44', 'ENSG00000065427_KARS', 'ENSG00000065457_ADAT1', 'ENSG00000065485_PDIA5', 'ENSG00000065491_TBC1D22B', 'ENSG00000065518_NDUFB4', 'ENSG00000065526_SPEN', 'ENSG00000065534_MYLK', 'ENSG00000065548_ZC3H15', 'ENSG00000065559_MAP2K4', 'ENSG00000065600_TMEM206', 'ENSG00000065613_SLK', 'ENSG00000065615_CYB5R4', 'ENSG00000065618_COL17A1', 'ENSG00000065621_GSTO2', 'ENSG00000065665_SEC61A2', 'ENSG00000065675_PRKCQ', 'ENSG00000065802_ASB1', 'ENSG00000065809_FAM107B', 'ENSG00000065833_ME1', 'ENSG00000065882_TBC1D1', 'ENSG00000065883_CDK13', 'ENSG00000065911_MTHFD2', 'ENSG00000065923_SLC9A7', 'ENSG00000065970_FOXJ2', 'ENSG00000065978_YBX1', 'ENSG00000065989_PDE4A', 'ENSG00000066027_PPP2R5A', 'ENSG00000066044_ELAVL1', 'ENSG00000066056_TIE1', 'ENSG00000066084_DIP2B', 'ENSG00000066117_SMARCD1', 'ENSG00000066135_KDM4A', 'ENSG00000066136_NFYC', 'ENSG00000066185_ZMYND12', 'ENSG00000066230_SLC9A3', 'ENSG00000066279_ASPM', 'ENSG00000066294_CD84', 'ENSG00000066322_ELOVL1', 'ENSG00000066336_SPI1', 'ENSG00000066379_ZNRD1', 'ENSG00000066382_MPPED2', 'ENSG00000066422_ZBTB11', 'ENSG00000066427_ATXN3', 'ENSG00000066455_GOLGA5', 'ENSG00000066468_FGFR2', 'ENSG00000066557_LRRC40', 'ENSG00000066583_ISOC1', 'ENSG00000066651_TRMT11', 'ENSG00000066654_THUMPD1', 'ENSG00000066697_MSANTD3', 'ENSG00000066735_KIF26A', 'ENSG00000066739_ATG2B', 'ENSG00000066777_ARFGEF1', 'ENSG00000066827_ZFAT', 'ENSG00000066855_MTFR1', 'ENSG00000066923_STAG3', 'ENSG00000066926_FECH', 'ENSG00000066933_MYO9A', 'ENSG00000067048_DDX3Y', 'ENSG00000067057_PFKP', 'ENSG00000067064_IDI1', 'ENSG00000067066_SP100', 'ENSG00000067082_KLF6', 'ENSG00000067113_PLPP1', 'ENSG00000067141_NEO1', 'ENSG00000067167_TRAM1', 'ENSG00000067177_PHKA1', 'ENSG00000067182_TNFRSF1A', 'ENSG00000067191_CACNB1', 'ENSG00000067208_EVI5', 'ENSG00000067221_STOML1', 'ENSG00000067225_PKM', 'ENSG00000067248_DHX29', 'ENSG00000067334_DNTTIP2', 'ENSG00000067365_METTL22', 'ENSG00000067369_TP53BP1', 'ENSG00000067445_TRO', 'ENSG00000067533_RRP15', 'ENSG00000067560_RHOA', 'ENSG00000067596_DHX8', 'ENSG00000067601_PMS2P4', 'ENSG00000067606_PRKCZ', 'ENSG00000067646_ZFY', 'ENSG00000067704_IARS2', 'ENSG00000067715_SYT1', 'ENSG00000067829_IDH3G', 'ENSG00000067836_ROGDI', 'ENSG00000067900_ROCK1', 'ENSG00000067955_CBFB', 'ENSG00000067992_PDK3', 'ENSG00000068001_HYAL2', 'ENSG00000068024_HDAC4', 'ENSG00000068028_RASSF1', 'ENSG00000068078_FGFR3', 'ENSG00000068079_IFI35', 'ENSG00000068097_HEATR6', 'ENSG00000068120_COASY', 'ENSG00000068137_PLEKHH3', 'ENSG00000068305_MEF2A', 'ENSG00000068308_OTUD5', 'ENSG00000068323_TFE3', 'ENSG00000068354_TBC1D25', 'ENSG00000068366_ACSL4', 'ENSG00000068383_INPP5A', 'ENSG00000068394_GPKOW', 'ENSG00000068400_GRIPAP1', 'ENSG00000068438_FTSJ1', 'ENSG00000068489_PRR11', 'ENSG00000068615_REEP1', 'ENSG00000068650_ATP11A', 'ENSG00000068654_POLR1A', 'ENSG00000068697_LAPTM4A', 'ENSG00000068724_TTC7A', 'ENSG00000068745_IP6K2', 'ENSG00000068784_SRBD1', 'ENSG00000068796_KIF2A', 'ENSG00000068831_RASGRP2', 'ENSG00000068878_PSME4', 'ENSG00000068885_IFT80', 'ENSG00000068903_SIRT2', 'ENSG00000068912_ERLEC1', 'ENSG00000068971_PPP2R5B', 'ENSG00000068976_PYGM', 'ENSG00000069020_MAST4', 'ENSG00000069188_SDK2', 'ENSG00000069248_NUP133', 'ENSG00000069275_NUCKS1', 'ENSG00000069329_VPS35', 'ENSG00000069345_DNAJA2', 'ENSG00000069399_BCL3', 'ENSG00000069424_KCNAB2', 'ENSG00000069482_GAL', 'ENSG00000069493_CLEC2D', 'ENSG00000069509_FUNDC1', 'ENSG00000069667_RORA', 'ENSG00000069696_DRD4', 'ENSG00000069702_TGFBR3', 'ENSG00000069812_HES2', 'ENSG00000069849_ATP1B3', 'ENSG00000069869_NEDD4', 'ENSG00000069943_PIGB', 'ENSG00000069956_MAPK6', 'ENSG00000069966_GNB5', 'ENSG00000069974_RAB27A', 'ENSG00000069998_HDHD5', 'ENSG00000070010_UFD1', 'ENSG00000070018_LRP6', 'ENSG00000070047_PHRF1', 'ENSG00000070061_ELP1', 'ENSG00000070081_NUCB2', 'ENSG00000070087_PFN2', 'ENSG00000070182_SPTB', 'ENSG00000070190_DAPP1', 'ENSG00000070214_SLC44A1', 'ENSG00000070269_TMEM260', 'ENSG00000070366_SMG6', 'ENSG00000070367_EXOC5', 'ENSG00000070371_CLTCL1', 'ENSG00000070388_FGF22', 'ENSG00000070404_FSTL3', 'ENSG00000070413_DGCR2', 'ENSG00000070423_RNF126', 'ENSG00000070444_MNT', 'ENSG00000070476_ZXDC', 'ENSG00000070495_JMJD6', 'ENSG00000070501_POLB', 'ENSG00000070526_ST6GALNAC1', 'ENSG00000070540_WIPI1', 'ENSG00000070601_FRMPD1', 'ENSG00000070610_GBA2', 'ENSG00000070614_NDST1', 'ENSG00000070669_ASNS', 'ENSG00000070718_AP3M2', 'ENSG00000070756_PABPC1', 'ENSG00000070759_TESK2', 'ENSG00000070761_CFAP20', 'ENSG00000070770_CSNK2A2', 'ENSG00000070785_EIF2B3', 'ENSG00000070814_TCOF1', 'ENSG00000070831_CDC42', 'ENSG00000070882_OSBPL3', 'ENSG00000070950_RAD18', 'ENSG00000070961_ATP2B1', 'ENSG00000071051_NCK2', 'ENSG00000071054_MAP4K4', 'ENSG00000071073_MGAT4A', 'ENSG00000071082_RPL31', 'ENSG00000071127_WDR1', 'ENSG00000071189_SNX13', 'ENSG00000071205_ARHGAP10', 'ENSG00000071242_RPS6KA2', 'ENSG00000071243_ING3', 'ENSG00000071246_VASH1', 'ENSG00000071282_LMCD1', 'ENSG00000071462_BUD23', 'ENSG00000071537_SEL1L', 'ENSG00000071539_TRIP13', 'ENSG00000071553_ATP6AP1', 'ENSG00000071564_TCF3', 'ENSG00000071575_TRIB2', 'ENSG00000071626_DAZAP1', 'ENSG00000071655_MBD3', 'ENSG00000071794_HLTF', 'ENSG00000071859_FAM50A', 'ENSG00000071889_FAM3A', 'ENSG00000071894_CPSF1', 'ENSG00000071967_CYBRD1', 'ENSG00000071994_PDCD2', 'ENSG00000072042_RDH11', 'ENSG00000072062_PRKACA', 'ENSG00000072071_ADGRL1', 'ENSG00000072110_ACTN1', 'ENSG00000072121_ZFYVE26', 'ENSG00000072133_RPS6KA6', 'ENSG00000072134_EPN2', 'ENSG00000072135_PTPN18', 'ENSG00000072163_LIMS2', 'ENSG00000072182_ASIC4', 'ENSG00000072195_SPEG', 'ENSG00000072201_LNX1', 'ENSG00000072210_ALDH3A2', 'ENSG00000072274_TFRC', 'ENSG00000072310_SREBF1', 'ENSG00000072364_AFF4', 'ENSG00000072401_UBE2D1', 'ENSG00000072415_MPP5', 'ENSG00000072422_RHOBTB1', 'ENSG00000072501_SMC1A', 'ENSG00000072506_HSD17B10', 'ENSG00000072518_MARK2', 'ENSG00000072571_HMMR', 'ENSG00000072609_CHFR', 'ENSG00000072657_TRHDE', 'ENSG00000072682_P4HA2', 'ENSG00000072694_FCGR2B', 'ENSG00000072736_NFATC3', 'ENSG00000072756_TRNT1', 'ENSG00000072778_ACADVL', 'ENSG00000072786_STK10', 'ENSG00000072803_FBXW11', 'ENSG00000072818_ACAP1', 'ENSG00000072840_EVC', 'ENSG00000072849_DERL2', 'ENSG00000072858_SIDT1', 'ENSG00000072864_NDE1', 'ENSG00000072952_MRVI1', 'ENSG00000072954_TMEM38A', 'ENSG00000072958_AP1M1', 'ENSG00000073008_PVR', 'ENSG00000073050_XRCC1', 'ENSG00000073060_SCARB1', 'ENSG00000073111_MCM2', 'ENSG00000073146_MOV10L1', 'ENSG00000073150_PANX2', 'ENSG00000073169_SELENOO', 'ENSG00000073331_ALPK1', 'ENSG00000073350_LLGL2', 'ENSG00000073417_PDE8A', 'ENSG00000073464_CLCN4', 'ENSG00000073536_NLE1', 'ENSG00000073578_SDHA', 'ENSG00000073584_SMARCE1', 'ENSG00000073605_GSDMB', 'ENSG00000073614_KDM5A', 'ENSG00000073670_ADAM11', 'ENSG00000073711_PPP2R3A', 'ENSG00000073712_FERMT2', 'ENSG00000073734_ABCB11', 'ENSG00000073737_DHRS9', 'ENSG00000073756_PTGS2', 'ENSG00000073792_IGF2BP2', 'ENSG00000073803_MAP3K13', 'ENSG00000073849_ST6GAL1', 'ENSG00000073905_VDAC1P1', 'ENSG00000073910_FRY', 'ENSG00000073921_PICALM', 'ENSG00000073969_NSF', 'ENSG00000074047_GLI2', 'ENSG00000074054_CLASP1', 'ENSG00000074071_MRPS34', 'ENSG00000074181_NOTCH3', 'ENSG00000074201_CLNS1A', 'ENSG00000074266_EED', 'ENSG00000074319_TSG101', 'ENSG00000074356_NCBP3', 'ENSG00000074370_ATP2A3', 'ENSG00000074416_MGLL', 'ENSG00000074582_BCS1L', 'ENSG00000074603_DPP8', 'ENSG00000074621_SLC24A1', 'ENSG00000074657_ZNF532', 'ENSG00000074660_SCARF1', 'ENSG00000074695_LMAN1', 'ENSG00000074696_HACD3', 'ENSG00000074706_IPCEF1', 'ENSG00000074755_ZZEF1', 'ENSG00000074800_ENO1', 'ENSG00000074842_MYDGF', 'ENSG00000074855_ANO8', 'ENSG00000074935_TUBE1', 'ENSG00000074966_TXK', 'ENSG00000075035_WSCD2', 'ENSG00000075043_KCNQ2', 'ENSG00000075089_ACTR6', 'ENSG00000075131_TIPIN', 'ENSG00000075142_SRI', 'ENSG00000075151_EIF4G3', 'ENSG00000075188_NUP37', 'ENSG00000075213_SEMA3A', 'ENSG00000075218_GTSE1', 'ENSG00000075223_SEMA3C', 'ENSG00000075234_TTC38', 'ENSG00000075239_ACAT1', 'ENSG00000075240_GRAMD4', 'ENSG00000075275_CELSR1', 'ENSG00000075292_ZNF638', 'ENSG00000075303_SLC25A40', 'ENSG00000075336_TIMM21', 'ENSG00000075340_ADD2', 'ENSG00000075391_RASAL2', 'ENSG00000075399_VPS9D1', 'ENSG00000075407_ZNF37A', 'ENSG00000075413_MARK3', 'ENSG00000075415_SLC25A3', 'ENSG00000075420_FNDC3B', 'ENSG00000075426_FOSL2', 'ENSG00000075539_FRYL', 'ENSG00000075568_TMEM131', 'ENSG00000075618_FSCN1', 'ENSG00000075624_ACTB', 'ENSG00000075643_MOCOS', 'ENSG00000075651_PLD1', 'ENSG00000075673_ATP12A', 'ENSG00000075702_WDR62', 'ENSG00000075711_DLG1', 'ENSG00000075785_RAB7A', 'ENSG00000075790_BCAP29', 'ENSG00000075826_SEC31B', 'ENSG00000075856_SART3', 'ENSG00000075884_ARHGAP15', 'ENSG00000075914_EXOSC7', 'ENSG00000075945_KIFAP3', 'ENSG00000075975_MKRN2', 'ENSG00000076003_MCM6', 'ENSG00000076043_REXO2', 'ENSG00000076053_RBM7', 'ENSG00000076067_RBMS2', 'ENSG00000076108_BAZ2A', 'ENSG00000076201_PTPN23', 'ENSG00000076242_MLH1', 'ENSG00000076248_UNG', 'ENSG00000076258_FMO4', 'ENSG00000076321_KLHL20', 'ENSG00000076351_SLC46A1', 'ENSG00000076356_PLXNA2', 'ENSG00000076382_SPAG5', 'ENSG00000076513_ANKRD13A', 'ENSG00000076554_TPD52', 'ENSG00000076555_ACACB', 'ENSG00000076604_TRAF4', 'ENSG00000076641_PAG1', 'ENSG00000076650_GPATCH1', 'ENSG00000076662_ICAM3', 'ENSG00000076685_NT5C2', 'ENSG00000076706_MCAM', 'ENSG00000076770_MBNL3', 'ENSG00000076826_CAMSAP3', 'ENSG00000076864_RAP1GAP', 'ENSG00000076924_XAB2', 'ENSG00000076928_ARHGEF1', 'ENSG00000076944_STXBP2', 'ENSG00000076984_MAP2K7', 'ENSG00000077044_DGKD', 'ENSG00000077063_CTTNBP2', 'ENSG00000077092_RARB', 'ENSG00000077097_TOP2B', 'ENSG00000077147_TM9SF3', 'ENSG00000077150_NFKB2', 'ENSG00000077152_UBE2T', 'ENSG00000077157_PPP1R12B', 'ENSG00000077232_DNAJC10', 'ENSG00000077235_GTF3C1', 'ENSG00000077238_IL4R', 'ENSG00000077254_USP33', 'ENSG00000077312_SNRPA', 'ENSG00000077327_SPAG6', 'ENSG00000077348_EXOSC5', 'ENSG00000077380_DYNC1I2', 'ENSG00000077420_APBB1IP', 'ENSG00000077454_LRCH4', 'ENSG00000077458_FAM76B', 'ENSG00000077463_SIRT6', 'ENSG00000077514_POLD3', 'ENSG00000077522_ACTN2', 'ENSG00000077549_CAPZB', 'ENSG00000077585_GPR137B', 'ENSG00000077616_NAALAD2', 'ENSG00000077684_JADE1', 'ENSG00000077713_SLC25A43', 'ENSG00000077721_UBE2A', 'ENSG00000077782_FGFR1', 'ENSG00000077935_SMC1B', 'ENSG00000077942_FBLN1', 'ENSG00000077943_ITGA8', 'ENSG00000077984_CST7', 'ENSG00000078018_MAP2', 'ENSG00000078043_PIAS2', 'ENSG00000078061_ARAF', 'ENSG00000078070_MCCC1', 'ENSG00000078081_LAMP3', 'ENSG00000078124_ACER3', 'ENSG00000078140_UBE2K', 'ENSG00000078142_PIK3C3', 'ENSG00000078177_N4BP2', 'ENSG00000078237_TIGAR', 'ENSG00000078246_TULP3', 'ENSG00000078269_SYNJ2', 'ENSG00000078295_ADCY2', 'ENSG00000078304_PPP2R5C', 'ENSG00000078319_PMS2P1', 'ENSG00000078369_GNB1', 'ENSG00000078399_HOXA9', 'ENSG00000078403_MLLT10', 'ENSG00000078487_ZCWPW1', 'ENSG00000078579_FGF20', 'ENSG00000078589_P2RY10', 'ENSG00000078596_ITM2A', 'ENSG00000078618_NRDC', 'ENSG00000078668_VDAC3', 'ENSG00000078674_PCM1', 'ENSG00000078687_TNRC6C', 'ENSG00000078699_CBFA2T2', 'ENSG00000078747_ITCH', 'ENSG00000078795_PKD2L2', 'ENSG00000078804_TP53INP2', 'ENSG00000078808_SDF4', 'ENSG00000078814_MYH7B', 'ENSG00000078900_TP73', 'ENSG00000078902_TOLLIP', 'ENSG00000078967_UBE2D4', 'ENSG00000079102_RUNX1T1', 'ENSG00000079134_THOC1', 'ENSG00000079150_FKBP7', 'ENSG00000079156_OSBPL6', 'ENSG00000079215_SLC1A3', 'ENSG00000079246_XRCC5', 'ENSG00000079257_LXN', 'ENSG00000079263_SP140', 'ENSG00000079277_MKNK1', 'ENSG00000079308_TNS1', 'ENSG00000079313_REXO1', 'ENSG00000079332_SAR1A', 'ENSG00000079335_CDC14A', 'ENSG00000079337_RAPGEF3', 'ENSG00000079385_CEACAM1', 'ENSG00000079387_SENP1', 'ENSG00000079432_CIC', 'ENSG00000079435_LIPE', 'ENSG00000079459_FDFT1', 'ENSG00000079462_PAFAH1B3', 'ENSG00000079482_OPHN1', 'ENSG00000079616_KIF22', 'ENSG00000079691_CARMIL1', 'ENSG00000079739_PGM1', 'ENSG00000079785_DDX1', 'ENSG00000079805_DNM2', 'ENSG00000079819_EPB41L2', 'ENSG00000079950_STX7', 'ENSG00000079974_RABL2B', 'ENSG00000079999_KEAP1', 'ENSG00000080007_DDX43', 'ENSG00000080189_SLC35C2', 'ENSG00000080200_CRYBG3', 'ENSG00000080298_RFX3', 'ENSG00000080345_RIF1', 'ENSG00000080371_RAB21', 'ENSG00000080493_SLC4A4', 'ENSG00000080503_SMARCA2', 'ENSG00000080546_SESN1', 'ENSG00000080603_SRCAP', 'ENSG00000080608_PUM3', 'ENSG00000080802_CNOT4', 'ENSG00000080815_PSEN1', 'ENSG00000080819_CPOX', 'ENSG00000080822_CLDND1', 'ENSG00000080823_MOK', 'ENSG00000080824_HSP90AA1', 'ENSG00000080839_RBL1', 'ENSG00000080845_DLGAP4', 'ENSG00000080910_CFHR2', 'ENSG00000080947_CROCCP3', 'ENSG00000080986_NDC80', 'ENSG00000081014_AP4E1', 'ENSG00000081019_RSBN1', 'ENSG00000081026_MAGI3', 'ENSG00000081041_CXCL2', 'ENSG00000081051_AFP', 'ENSG00000081052_COL4A4', 'ENSG00000081059_TCF7', 'ENSG00000081087_OSTM1', 'ENSG00000081138_CDH7', 'ENSG00000081148_IMPG2', 'ENSG00000081154_PCNP', 'ENSG00000081177_EXD2', 'ENSG00000081181_ARG2', 'ENSG00000081189_MEF2C', 'ENSG00000081237_PTPRC', 'ENSG00000081307_UBA5', 'ENSG00000081320_STK17B', 'ENSG00000081377_CDC14B', 'ENSG00000081386_ZNF510', 'ENSG00000081479_LRP2', 'ENSG00000081665_ZNF506', 'ENSG00000081692_JMJD4', 'ENSG00000081721_DUSP12', 'ENSG00000081760_AACS', 'ENSG00000081791_DELE1', 'ENSG00000081803_CADPS2', 'ENSG00000081870_HSPB11', 'ENSG00000081913_PHLPP1', 'ENSG00000081923_ATP8B1', 'ENSG00000081985_IL12RB2', 'ENSG00000082014_SMARCD3', 'ENSG00000082068_WDR70', 'ENSG00000082074_FYB1', 'ENSG00000082126_MPP4', 'ENSG00000082146_STRADB', 'ENSG00000082153_BZW1', 'ENSG00000082196_C1QTNF3', 'ENSG00000082212_ME2', 'ENSG00000082213_C5orf22', 'ENSG00000082258_CCNT2', 'ENSG00000082269_FAM135A', 'ENSG00000082397_EPB41L3', 'ENSG00000082438_COBLL1', 'ENSG00000082458_DLG3', 'ENSG00000082512_TRAF5', 'ENSG00000082515_MRPL22', 'ENSG00000082516_GEMIN5', 'ENSG00000082641_NFE2L1', 'ENSG00000082701_GSK3B', 'ENSG00000082781_ITGB5', 'ENSG00000082805_ERC1', 'ENSG00000082898_XPO1', 'ENSG00000082996_RNF13', 'ENSG00000083067_TRPM3', 'ENSG00000083093_PALB2', 'ENSG00000083097_DOP1A', 'ENSG00000083099_LYRM2', 'ENSG00000083123_BCKDHB', 'ENSG00000083168_KAT6A', 'ENSG00000083223_TUT7', 'ENSG00000083290_ULK2', 'ENSG00000083312_TNPO1', 'ENSG00000083444_PLOD1', 'ENSG00000083454_P2RX5', 'ENSG00000083457_ITGAE', 'ENSG00000083520_DIS3', 'ENSG00000083535_PIBF1', 'ENSG00000083544_TDRD3', 'ENSG00000083635_NUFIP1', 'ENSG00000083642_PDS5B', 'ENSG00000083720_OXCT1', 'ENSG00000083750_RRAGB', 'ENSG00000083799_CYLD', 'ENSG00000083807_SLC27A5', 'ENSG00000083812_ZNF324', 'ENSG00000083814_ZNF671', 'ENSG00000083817_ZNF416', 'ENSG00000083828_ZNF586', 'ENSG00000083838_ZNF446', 'ENSG00000083844_ZNF264', 'ENSG00000083845_RPS5', 'ENSG00000083896_YTHDC1', 'ENSG00000083937_CHMP2B', 'ENSG00000084070_SMAP2', 'ENSG00000084072_PPIE', 'ENSG00000084073_ZMPSTE24', 'ENSG00000084090_STARD7', 'ENSG00000084092_NOA1', 'ENSG00000084093_REST', 'ENSG00000084110_HAL', 'ENSG00000084112_SSH1', 'ENSG00000084207_GSTP1', 'ENSG00000084234_APLP2', 'ENSG00000084444_FAM234B', 'ENSG00000084463_WBP11', 'ENSG00000084623_EIF3I', 'ENSG00000084636_COL16A1', 'ENSG00000084652_TXLNA', 'ENSG00000084676_NCOA1', 'ENSG00000084693_AGBL5', 'ENSG00000084710_EFR3B', 'ENSG00000084731_KIF3C', 'ENSG00000084733_RAB10', 'ENSG00000084754_HADHA', 'ENSG00000084764_MAPRE3', 'ENSG00000084774_CAD', 'ENSG00000085063_CD59', 'ENSG00000085117_CD82', 'ENSG00000085185_BCORL1', 'ENSG00000085224_ATRX', 'ENSG00000085231_AK6', 'ENSG00000085265_FCN1', 'ENSG00000085274_MYNN', 'ENSG00000085276_MECOM', 'ENSG00000085365_SCAMP1', 'ENSG00000085377_PREP', 'ENSG00000085382_HACE1', 'ENSG00000085415_SEH1L', 'ENSG00000085433_WDR47', 'ENSG00000085449_WDFY1', 'ENSG00000085465_OVGP1', 'ENSG00000085491_SLC25A24', 'ENSG00000085511_MAP3K4', 'ENSG00000085514_PILRA', 'ENSG00000085552_IGSF9', 'ENSG00000085563_ABCB1', 'ENSG00000085644_ZNF213', 'ENSG00000085662_AKR1B1', 'ENSG00000085719_CPNE3', 'ENSG00000085721_RRN3', 'ENSG00000085733_CTTN', 'ENSG00000085741_WNT11', 'ENSG00000085760_MTIF2', 'ENSG00000085788_DDHD2', 'ENSG00000085831_TTC39A', 'ENSG00000085832_EPS15', 'ENSG00000085840_ORC1', 'ENSG00000085871_MGST2', 'ENSG00000085872_CHERP', 'ENSG00000085978_ATG16L1', 'ENSG00000085982_USP40', 'ENSG00000085998_POMGNT1', 'ENSG00000085999_RAD54L', 'ENSG00000086015_MAST2', 'ENSG00000086061_DNAJA1', 'ENSG00000086062_B4GALT1', 'ENSG00000086065_CHMP5', 'ENSG00000086102_NFX1', 'ENSG00000086189_DIMT1', 'ENSG00000086200_IPO11', 'ENSG00000086205_FOLH1', 'ENSG00000086232_EIF2AK1', 'ENSG00000086288_NME8', 'ENSG00000086289_EPDR1', 'ENSG00000086300_SNX10', 'ENSG00000086475_SEPHS1', 'ENSG00000086504_MRPL28', 'ENSG00000086506_HBQ1', 'ENSG00000086544_ITPKC', 'ENSG00000086548_CEACAM6', 'ENSG00000086589_RBM22', 'ENSG00000086598_TMED2', 'ENSG00000086619_ERO1B', 'ENSG00000086666_ZFAND6', 'ENSG00000086712_TXLNG', 'ENSG00000086730_LAT2', 'ENSG00000086758_HUWE1', 'ENSG00000086827_ZW10', 'ENSG00000086848_ALG9', 'ENSG00000087008_ACOX3', 'ENSG00000087053_MTMR2', 'ENSG00000087074_PPP1R15A', 'ENSG00000087076_HSD17B14', 'ENSG00000087077_TRIP6', 'ENSG00000087085_ACHE', 'ENSG00000087086_FTL', 'ENSG00000087087_SRRT', 'ENSG00000087088_BAX', 'ENSG00000087095_NLK', 'ENSG00000087111_PIGS', 'ENSG00000087152_ATXN7L3', 'ENSG00000087157_PGS1', 'ENSG00000087191_PSMC5', 'ENSG00000087206_UIMC1', 'ENSG00000087237_CETP', 'ENSG00000087245_MMP2', 'ENSG00000087250_MT3', 'ENSG00000087253_LPCAT2', 'ENSG00000087263_OGFOD1', 'ENSG00000087266_SH3BP2', 'ENSG00000087269_NOP14', 'ENSG00000087274_ADD1', 'ENSG00000087299_L2HGDH', 'ENSG00000087301_TXNDC16', 'ENSG00000087302_RTRAF', 'ENSG00000087303_NID2', 'ENSG00000087338_GMCL1', 'ENSG00000087365_SF3B2', 'ENSG00000087448_KLHL42', 'ENSG00000087460_GNAS', 'ENSG00000087470_DNM1L', 'ENSG00000087495_PHACTR3', 'ENSG00000087502_ERGIC2', 'ENSG00000087586_AURKA', 'ENSG00000087589_CASS4', 'ENSG00000087842_PIR', 'ENSG00000087884_AAMDC', 'ENSG00000087903_RFX2', 'ENSG00000087995_METTL2A', 'ENSG00000088002_SULT2B1', 'ENSG00000088035_ALG6', 'ENSG00000088038_CNOT3', 'ENSG00000088053_GP6', 'ENSG00000088179_PTPN4', 'ENSG00000088205_DDX18', 'ENSG00000088247_KHSRP', 'ENSG00000088256_GNA11', 'ENSG00000088280_ASAP3', 'ENSG00000088298_EDEM2', 'ENSG00000088305_DNMT3B', 'ENSG00000088325_TPX2', 'ENSG00000088356_PDRG1', 'ENSG00000088367_EPB41L1', 'ENSG00000088387_DOCK9', 'ENSG00000088448_ANKRD10', 'ENSG00000088451_TGDS', 'ENSG00000088538_DOCK3', 'ENSG00000088543_C3orf18', 'ENSG00000088682_COQ9', 'ENSG00000088726_TMEM40', 'ENSG00000088727_KIF9', 'ENSG00000088756_ARHGAP28', 'ENSG00000088766_CRLS1', 'ENSG00000088808_PPP1R13B', 'ENSG00000088812_ATRN', 'ENSG00000088826_SMOX', 'ENSG00000088832_FKBP1A', 'ENSG00000088833_NSFL1C', 'ENSG00000088836_SLC4A11', 'ENSG00000088854_C20orf194', 'ENSG00000088876_ZNF343', 'ENSG00000088881_EBF4', 'ENSG00000088882_CPXM1', 'ENSG00000088888_MAVS', 'ENSG00000088899_LZTS3', 'ENSG00000088930_XRN2', 'ENSG00000088970_KIZ', 'ENSG00000088986_DYNLL1', 'ENSG00000088992_TESC', 'ENSG00000089006_SNX5', 'ENSG00000089009_RPL6', 'ENSG00000089022_MAPKAPK5', 'ENSG00000089041_P2RX7', 'ENSG00000089048_ESF1', 'ENSG00000089050_RBBP9', 'ENSG00000089053_ANAPC5', 'ENSG00000089057_SLC23A2', 'ENSG00000089060_SLC8B1', 'ENSG00000089063_TMEM230', 'ENSG00000089091_DZANK1', 'ENSG00000089094_KDM2B', 'ENSG00000089123_TASP1', 'ENSG00000089127_OAS1', 'ENSG00000089154_GCN1', 'ENSG00000089157_RPLP0', 'ENSG00000089159_PXN', 'ENSG00000089163_SIRT4', 'ENSG00000089177_KIF16B', 'ENSG00000089195_TRMT6', 'ENSG00000089199_CHGB', 'ENSG00000089220_PEBP1', 'ENSG00000089234_BRAP', 'ENSG00000089248_ERP29', 'ENSG00000089280_FUS', 'ENSG00000089289_IGBP1', 'ENSG00000089327_FXYD5', 'ENSG00000089335_ZNF302', 'ENSG00000089351_GRAMD1A', 'ENSG00000089356_FXYD3', 'ENSG00000089472_HEPH', 'ENSG00000089486_CDIP1', 'ENSG00000089558_KCNH4', 'ENSG00000089597_GANAB', 'ENSG00000089639_GMIP', 'ENSG00000089682_RBM41', 'ENSG00000089685_BIRC5', 'ENSG00000089692_LAG3', 'ENSG00000089693_MLF2', 'ENSG00000089723_OTUB2', 'ENSG00000089737_DDX24', 'ENSG00000089775_ZBTB25', 'ENSG00000089818_NECAP1', 'ENSG00000089820_ARHGAP4', 'ENSG00000089847_ANKRD24', 'ENSG00000089876_DHX32', 'ENSG00000089902_RCOR1', 'ENSG00000089916_GPATCH2L', 'ENSG00000090006_LTBP4', 'ENSG00000090013_BLVRB', 'ENSG00000090020_SLC9A1', 'ENSG00000090054_SPTLC1', 'ENSG00000090060_PAPOLA', 'ENSG00000090061_CCNK', 'ENSG00000090097_PCBP4', 'ENSG00000090104_RGS1', 'ENSG00000090238_YPEL3', 'ENSG00000090263_MRPS33', 'ENSG00000090266_NDUFB2', 'ENSG00000090273_NUDC', 'ENSG00000090316_MAEA', 'ENSG00000090339_ICAM1', 'ENSG00000090372_STRN4', 'ENSG00000090376_IRAK3', 'ENSG00000090382_LYZ', 'ENSG00000090432_MUL1', 'ENSG00000090447_TFAP4', 'ENSG00000090470_PDCD7', 'ENSG00000090487_SPG21', 'ENSG00000090520_DNAJB11', 'ENSG00000090554_FLT3LG', 'ENSG00000090565_RAB11FIP3', 'ENSG00000090581_GNPTG', 'ENSG00000090612_ZNF268', 'ENSG00000090615_GOLGA3', 'ENSG00000090621_PABPC4', 'ENSG00000090661_CERS4', 'ENSG00000090674_MCOLN1', 'ENSG00000090686_USP48', 'ENSG00000090776_EFNB1', 'ENSG00000090857_PDPR', 'ENSG00000090861_AARS', 'ENSG00000090863_GLG1', 'ENSG00000090889_KIF4A', 'ENSG00000090905_TNRC6A', 'ENSG00000090924_PLEKHG2', 'ENSG00000090932_DLL3', 'ENSG00000090971_NAT14', 'ENSG00000090975_PITPNM2', 'ENSG00000090989_EXOC1', 'ENSG00000091009_RBM27', 'ENSG00000091039_OSBPL8', 'ENSG00000091073_DTX2', 'ENSG00000091106_NLRC4', 'ENSG00000091127_PUS7', 'ENSG00000091128_LAMB4', 'ENSG00000091136_LAMB1', 'ENSG00000091137_SLC26A4', 'ENSG00000091140_DLD', 'ENSG00000091157_WDR7', 'ENSG00000091164_TXNL1', 'ENSG00000091181_IL5RA', 'ENSG00000091262_ABCC6', 'ENSG00000091317_CMTM6', 'ENSG00000091409_ITGA6', 'ENSG00000091436_MAP3K20', 'ENSG00000091483_FH', 'ENSG00000091490_SEL1L3', 'ENSG00000091513_TF', 'ENSG00000091527_CDV3', 'ENSG00000091536_MYO15A', 'ENSG00000091542_ALKBH5', 'ENSG00000091592_NLRP1', 'ENSG00000091622_PITPNM3', 'ENSG00000091640_SPAG7', 'ENSG00000091651_ORC6', 'ENSG00000091732_ZC3HC1', 'ENSG00000091831_ESR1', 'ENSG00000091844_RGS17', 'ENSG00000091879_ANGPT2', 'ENSG00000091947_TMEM101', 'ENSG00000091972_CD200', 'ENSG00000092009_CMA1', 'ENSG00000092010_PSME1', 'ENSG00000092020_PPP2R3C', 'ENSG00000092036_HAUS4', 'ENSG00000092051_JPH4', 'ENSG00000092067_CEBPE', 'ENSG00000092068_SLC7A8', 'ENSG00000092094_OSGEP', 'ENSG00000092096_SLC22A17', 'ENSG00000092098_RNF31', 'ENSG00000092108_SCFD1', 'ENSG00000092140_G2E3', 'ENSG00000092148_HECTD1', 'ENSG00000092199_HNRNPC', 'ENSG00000092201_SUPT16H', 'ENSG00000092203_TOX4', 'ENSG00000092208_GEMIN2', 'ENSG00000092295_TGM1', 'ENSG00000092330_TINF2', 'ENSG00000092421_SEMA6A', 'ENSG00000092439_TRPM7', 'ENSG00000092445_TYRO3', 'ENSG00000092470_WDR76', 'ENSG00000092529_CAPN3', 'ENSG00000092531_SNAP23', 'ENSG00000092621_PHGDH', 'ENSG00000092758_COL9A3', 'ENSG00000092820_EZR', 'ENSG00000092841_MYL6', 'ENSG00000092847_AGO1', 'ENSG00000092853_CLSPN', 'ENSG00000092871_RFFL', 'ENSG00000092929_UNC13D', 'ENSG00000092931_MFSD11', 'ENSG00000092964_DPYSL2', 'ENSG00000092978_GPATCH2', 'ENSG00000093000_NUP50', 'ENSG00000093009_CDC45', 'ENSG00000093010_COMT', 'ENSG00000093072_ADA2', 'ENSG00000093144_ECHDC1', 'ENSG00000093167_LRRFIP2', 'ENSG00000093183_SEC22C', 'ENSG00000093217_XYLB', 'ENSG00000094631_HDAC6', 'ENSG00000094804_CDC6', 'ENSG00000094841_UPRT', 'ENSG00000094880_CDC23', 'ENSG00000094914_AAAS', 'ENSG00000094916_CBX5', 'ENSG00000094975_SUCO', 'ENSG00000095002_MSH2', 'ENSG00000095015_MAP3K1', 'ENSG00000095059_DHPS', 'ENSG00000095066_HOOK2', 'ENSG00000095139_ARCN1', 'ENSG00000095209_TMEM38B', 'ENSG00000095261_PSMD5', 'ENSG00000095303_PTGS1', 'ENSG00000095319_NUP188', 'ENSG00000095321_CRAT', 'ENSG00000095370_SH2D3C', 'ENSG00000095380_NANS', 'ENSG00000095383_TBC1D2', 'ENSG00000095485_CWF19L1', 'ENSG00000095539_SEMA4G', 'ENSG00000095564_BTAF1', 'ENSG00000095574_IKZF5', 'ENSG00000095585_BLNK', 'ENSG00000095627_TDRD1', 'ENSG00000095637_SORBS1', 'ENSG00000095739_BAMBI', 'ENSG00000095787_WAC', 'ENSG00000095794_CREM', 'ENSG00000095906_NUBP2', 'ENSG00000095917_TPSD1', 'ENSG00000095932_SMIM24', 'ENSG00000095951_HIVEP1', 'ENSG00000096006_CRISP3', 'ENSG00000096060_FKBP5', 'ENSG00000096063_SRPK1', 'ENSG00000096070_BRPF3', 'ENSG00000096080_MRPS18A', 'ENSG00000096092_TMEM14A', 'ENSG00000096093_EFHC1', 'ENSG00000096384_HSP90AB1', 'ENSG00000096401_CDC5L', 'ENSG00000096433_ITPR3', 'ENSG00000096654_ZNF184', 'ENSG00000096717_SIRT1', 'ENSG00000096746_HNRNPH3', 'ENSG00000096872_IFT74', 'ENSG00000096968_JAK2', 'ENSG00000096996_IL12RB1', 'ENSG00000097007_ABL1', 'ENSG00000097021_ACOT7', 'ENSG00000097033_SH3GLB1', 'ENSG00000097046_CDC7', 'ENSG00000097096_SYDE2', 'ENSG00000099139_PCSK5', 'ENSG00000099194_SCD', 'ENSG00000099203_TMED1', 'ENSG00000099204_ABLIM1', 'ENSG00000099219_ERMP1', 'ENSG00000099246_RAB18', 'ENSG00000099250_NRP1', 'ENSG00000099251_HSD17B7P2', 'ENSG00000099256_PRTFDC1', 'ENSG00000099282_TSPAN15', 'ENSG00000099284_H2AFY2', 'ENSG00000099290_WASHC2A', 'ENSG00000099308_MAST3', 'ENSG00000099326_MZF1', 'ENSG00000099330_OCEL1', 'ENSG00000099331_MYO9B', 'ENSG00000099337_KCNK6', 'ENSG00000099338_CATSPERG', 'ENSG00000099341_PSMD8', 'ENSG00000099364_FBXL19', 'ENSG00000099365_STX1B', 'ENSG00000099377_HSD3B7', 'ENSG00000099381_SETD1A', 'ENSG00000099385_BCL7C', 'ENSG00000099622_CIRBP', 'ENSG00000099624_ATP5F1D', 'ENSG00000099625_CBARP', 'ENSG00000099725_PRKY', 'ENSG00000099783_HNRNPM', 'ENSG00000099785_MARCH2', 'ENSG00000099795_NDUFB7', 'ENSG00000099797_TECR', 'ENSG00000099800_TIMM13', 'ENSG00000099804_CDC34', 'ENSG00000099810_MTAP', 'ENSG00000099814_CEP170B', 'ENSG00000099817_POLR2E', 'ENSG00000099821_POLRMT', 'ENSG00000099822_HCN2', 'ENSG00000099840_IZUMO4', 'ENSG00000099849_RASSF7', 'ENSG00000099860_GADD45B', 'ENSG00000099864_PALM', 'ENSG00000099866_MADCAM1', 'ENSG00000099875_MKNK2', 'ENSG00000099889_ARVCF', 'ENSG00000099899_TRMT2A', 'ENSG00000099901_RANBP1', 'ENSG00000099904_ZDHHC8', 'ENSG00000099910_KLHL22', 'ENSG00000099917_MED15', 'ENSG00000099940_SNAP29', 'ENSG00000099942_CRKL', 'ENSG00000099949_LZTR1', 'ENSG00000099953_MMP11', 'ENSG00000099956_SMARCB1', 'ENSG00000099957_P2RX6', 'ENSG00000099958_DERL3', 'ENSG00000099968_BCL2L13', 'ENSG00000099974_DDTL', 'ENSG00000099977_DDT', 'ENSG00000099985_OSM', 'ENSG00000099991_CABIN1', 'ENSG00000099992_TBC1D10A', 'ENSG00000099994_SUSD2', 'ENSG00000099995_SF3A1', 'ENSG00000099998_GGT5', 'ENSG00000099999_RNF215', 'ENSG00000100003_SEC14L2', 'ENSG00000100014_SPECC1L', 'ENSG00000100023_PPIL2', 'ENSG00000100024_UPB1', 'ENSG00000100027_YPEL1', 'ENSG00000100028_SNRPD3', 'ENSG00000100029_PES1', 'ENSG00000100030_MAPK1', 'ENSG00000100031_GGT1', 'ENSG00000100033_PRODH', 'ENSG00000100034_PPM1F', 'ENSG00000100036_SLC35E4', 'ENSG00000100038_TOP3B', 'ENSG00000100055_CYTH4', 'ENSG00000100056_ESS2', 'ENSG00000100058_CRYBB2P1', 'ENSG00000100060_MFNG', 'ENSG00000100065_CARD10', 'ENSG00000100068_LRP5L', 'ENSG00000100075_SLC25A1', 'ENSG00000100077_GRK3', 'ENSG00000100078_PLA2G3', 'ENSG00000100079_LGALS2', 'ENSG00000100083_GGA1', 'ENSG00000100084_HIRA', 'ENSG00000100092_SH3BP1', 'ENSG00000100095_SEZ6L', 'ENSG00000100097_LGALS1', 'ENSG00000100099_HPS4', 'ENSG00000100100_PIK3IP1', 'ENSG00000100101_Z83844.1', 'ENSG00000100104_SRRD', 'ENSG00000100105_PATZ1', 'ENSG00000100106_TRIOBP', 'ENSG00000100109_TFIP11', 'ENSG00000100116_GCAT', 'ENSG00000100122_CRYBB1', 'ENSG00000100124_ANKRD54', 'ENSG00000100129_EIF3L', 'ENSG00000100138_SNU13', 'ENSG00000100139_MICALL1', 'ENSG00000100142_POLR2F', 'ENSG00000100147_CCDC134', 'ENSG00000100150_DEPDC5', 'ENSG00000100151_PICK1', 'ENSG00000100154_TTC28', 'ENSG00000100156_SLC16A8', 'ENSG00000100162_CENPM', 'ENSG00000100167_SEPT3', 'ENSG00000100181_TPTEP1', 'ENSG00000100196_KDELR3', 'ENSG00000100201_DDX17', 'ENSG00000100206_DMC1', 'ENSG00000100207_TCF20', 'ENSG00000100209_HSCB', 'ENSG00000100211_CBY1', 'ENSG00000100216_TOMM22', 'ENSG00000100219_XBP1', 'ENSG00000100220_RTCB', 'ENSG00000100221_JOSD1', 'ENSG00000100225_FBXO7', 'ENSG00000100226_GTPBP1', 'ENSG00000100227_POLDIP3', 'ENSG00000100228_RAB36', 'ENSG00000100234_TIMP3', 'ENSG00000100239_PPP6R2', 'ENSG00000100241_SBF1', 'ENSG00000100242_SUN2', 'ENSG00000100243_CYB5R3', 'ENSG00000100246_DNAL4', 'ENSG00000100253_MIOX', 'ENSG00000100258_LMF2', 'ENSG00000100263_RHBDD3', 'ENSG00000100266_PACSIN2', 'ENSG00000100271_TTLL1', 'ENSG00000100276_RASL10A', 'ENSG00000100280_AP1B1', 'ENSG00000100281_HMGXB4', 'ENSG00000100284_TOM1', 'ENSG00000100285_NEFH', 'ENSG00000100288_CHKB', 'ENSG00000100290_BIK', 'ENSG00000100292_HMOX1', 'ENSG00000100294_MCAT', 'ENSG00000100296_THOC5', 'ENSG00000100297_MCM5', 'ENSG00000100298_APOBEC3H', 'ENSG00000100299_ARSA', 'ENSG00000100300_TSPO', 'ENSG00000100304_TTLL12', 'ENSG00000100307_CBX7', 'ENSG00000100314_CABP7', 'ENSG00000100316_RPL3', 'ENSG00000100319_ZMAT5', 'ENSG00000100320_RBFOX2', 'ENSG00000100321_SYNGR1', 'ENSG00000100324_TAB1', 'ENSG00000100325_ASCC2', 'ENSG00000100330_MTMR3', 'ENSG00000100335_MIEF1', 'ENSG00000100336_APOL4', 'ENSG00000100342_APOL1', 'ENSG00000100344_PNPLA3', 'ENSG00000100345_MYH9', 'ENSG00000100347_SAMM50', 'ENSG00000100348_TXN2', 'ENSG00000100350_FOXRED2', 'ENSG00000100351_GRAP2', 'ENSG00000100353_EIF3D', 'ENSG00000100354_TNRC6B', 'ENSG00000100359_SGSM3', 'ENSG00000100360_IFT27', 'ENSG00000100364_KIAA0930', 'ENSG00000100365_NCF4', 'ENSG00000100368_CSF2RB', 'ENSG00000100372_SLC25A17', 'ENSG00000100376_FAM118A', 'ENSG00000100379_KCTD17', 'ENSG00000100380_ST13', 'ENSG00000100385_IL2RB', 'ENSG00000100387_RBX1', 'ENSG00000100393_EP300', 'ENSG00000100395_L3MBTL2', 'ENSG00000100399_CHADL', 'ENSG00000100401_RANGAP1', 'ENSG00000100403_ZC3H7B', 'ENSG00000100410_PHF5A', 'ENSG00000100412_ACO2', 'ENSG00000100413_POLR3H', 'ENSG00000100416_TRMU', 'ENSG00000100417_PMM1', 'ENSG00000100418_DESI1', 'ENSG00000100422_CERK', 'ENSG00000100425_BRD1', 'ENSG00000100426_ZBED4', 'ENSG00000100427_MLC1', 'ENSG00000100429_HDAC10', 'ENSG00000100439_ABHD4', 'ENSG00000100441_KHNYN', 'ENSG00000100442_FKBP3', 'ENSG00000100445_SDR39U1', 'ENSG00000100448_CTSG', 'ENSG00000100453_GZMB', 'ENSG00000100461_RBM23', 'ENSG00000100462_PRMT5', 'ENSG00000100473_COCH', 'ENSG00000100478_AP4S1', 'ENSG00000100479_POLE2', 'ENSG00000100483_VCPKMT', 'ENSG00000100485_SOS2', 'ENSG00000100490_CDKL1', 'ENSG00000100503_NIN', 'ENSG00000100504_PYGL', 'ENSG00000100505_TRIM9', 'ENSG00000100519_PSMC6', 'ENSG00000100522_GNPNAT1', 'ENSG00000100523_DDHD1', 'ENSG00000100526_CDKN3', 'ENSG00000100528_CNIH1', 'ENSG00000100532_CGRRF1', 'ENSG00000100554_ATP6V1D', 'ENSG00000100558_PLEK2', 'ENSG00000100564_PIGH', 'ENSG00000100567_PSMA3', 'ENSG00000100568_VTI1B', 'ENSG00000100575_TIMM9', 'ENSG00000100577_GSTZ1', 'ENSG00000100578_KIAA0586', 'ENSG00000100580_TMED8', 'ENSG00000100583_SAMD15', 'ENSG00000100591_AHSA1', 'ENSG00000100592_DAAM1', 'ENSG00000100596_SPTLC2', 'ENSG00000100599_RIN3', 'ENSG00000100600_LGMN', 'ENSG00000100601_ALKBH1', 'ENSG00000100603_SNW1', 'ENSG00000100605_ITPK1', 'ENSG00000100612_DHRS7', 'ENSG00000100614_PPM1A', 'ENSG00000100628_ASB2', 'ENSG00000100629_CEP128', 'ENSG00000100632_ERH', 'ENSG00000100644_HIF1A', 'ENSG00000100647_SUSD6', 'ENSG00000100650_SRSF5', 'ENSG00000100664_EIF5', 'ENSG00000100678_SLC8A3', 'ENSG00000100697_DICER1', 'ENSG00000100711_ZFYVE21', 'ENSG00000100714_MTHFD1', 'ENSG00000100722_ZC3H14', 'ENSG00000100726_TELO2', 'ENSG00000100731_PCNX1', 'ENSG00000100744_GSKIP', 'ENSG00000100749_VRK1', 'ENSG00000100764_PSMC1', 'ENSG00000100767_PAPLN', 'ENSG00000100784_RPS6KA5', 'ENSG00000100796_PPP4R3A', 'ENSG00000100802_C14orf93', 'ENSG00000100804_PSMB5', 'ENSG00000100811_YY1', 'ENSG00000100813_ACIN1', 'ENSG00000100814_CCNB1IP1', 'ENSG00000100815_TRIP11', 'ENSG00000100823_APEX1', 'ENSG00000100836_PABPN1', 'ENSG00000100852_ARHGAP5', 'ENSG00000100865_CINP', 'ENSG00000100883_SRP54', 'ENSG00000100888_CHD8', 'ENSG00000100889_PCK2', 'ENSG00000100890_KIAA0391', 'ENSG00000100897_DCAF11', 'ENSG00000100902_PSMA6', 'ENSG00000100906_NFKBIA', 'ENSG00000100908_EMC9', 'ENSG00000100911_PSME2', 'ENSG00000100916_BRMS1L', 'ENSG00000100918_REC8', 'ENSG00000100926_TM9SF1', 'ENSG00000100934_SEC23A', 'ENSG00000100938_GMPR2', 'ENSG00000100941_PNN', 'ENSG00000100949_RABGGTA', 'ENSG00000100968_NFATC4', 'ENSG00000100979_PLTP', 'ENSG00000100982_PCIF1', 'ENSG00000100983_GSS', 'ENSG00000100991_TRPC4AP', 'ENSG00000100994_PYGB', 'ENSG00000100997_ABHD12', 'ENSG00000101000_PROCR', 'ENSG00000101003_GINS1', 'ENSG00000101004_NINL', 'ENSG00000101017_CD40', 'ENSG00000101019_UQCC1', 'ENSG00000101040_ZMYND8', 'ENSG00000101049_SGK2', 'ENSG00000101052_IFT52', 'ENSG00000101057_MYBL2', 'ENSG00000101079_NDRG3', 'ENSG00000101082_SLA2', 'ENSG00000101084_RAB5IF', 'ENSG00000101096_NFATC2', 'ENSG00000101104_PABPC1L', 'ENSG00000101109_STK4', 'ENSG00000101115_SALL4', 'ENSG00000101126_ADNP', 'ENSG00000101132_PFDN4', 'ENSG00000101138_CSTF1', 'ENSG00000101146_RAE1', 'ENSG00000101150_TPD52L2', 'ENSG00000101152_DNAJC5', 'ENSG00000101158_NELFCD', 'ENSG00000101160_CTSZ', 'ENSG00000101161_PRPF6', 'ENSG00000101162_TUBB1', 'ENSG00000101166_PRELID3B', 'ENSG00000101181_MTG2', 'ENSG00000101182_PSMA7', 'ENSG00000101187_SLCO4A1', 'ENSG00000101188_NTSR1', 'ENSG00000101189_MRGBP', 'ENSG00000101190_TCFL5', 'ENSG00000101191_DIDO1', 'ENSG00000101193_GID8', 'ENSG00000101194_SLC17A9', 'ENSG00000101199_ARFGAP1', 'ENSG00000101200_AVP', 'ENSG00000101210_EEF1A2', 'ENSG00000101213_PTK6', 'ENSG00000101216_GMEB2', 'ENSG00000101220_C20orf27', 'ENSG00000101224_CDC25B', 'ENSG00000101236_RNF24', 'ENSG00000101246_ARFRP1', 'ENSG00000101247_NDUFAF5', 'ENSG00000101255_TRIB3', 'ENSG00000101265_RASSF2', 'ENSG00000101266_CSNK2A1', 'ENSG00000101276_SLC52A3', 'ENSG00000101290_CDS2', 'ENSG00000101294_HM13', 'ENSG00000101306_MYLK2', 'ENSG00000101307_SIRPB1', 'ENSG00000101310_SEC23B', 'ENSG00000101311_FERMT1', 'ENSG00000101333_PLCB4', 'ENSG00000101335_MYL9', 'ENSG00000101336_HCK', 'ENSG00000101337_TM9SF4', 'ENSG00000101342_TLDC2', 'ENSG00000101343_CRNKL1', 'ENSG00000101346_POFUT1', 'ENSG00000101347_SAMHD1', 'ENSG00000101350_KIF3B', 'ENSG00000101353_MROH8', 'ENSG00000101361_NOP56', 'ENSG00000101363_MANBAL', 'ENSG00000101365_IDH3B', 'ENSG00000101367_MAPRE1', 'ENSG00000101384_JAG1', 'ENSG00000101391_CDK5RAP1', 'ENSG00000101400_SNTA1', 'ENSG00000101405_OXT', 'ENSG00000101407_TTI1', 'ENSG00000101412_E2F1', 'ENSG00000101413_RPRD1B', 'ENSG00000101417_PXMP4', 'ENSG00000101421_CHMP4B', 'ENSG00000101425_BPI', 'ENSG00000101439_CST3', 'ENSG00000101442_ACTR5', 'ENSG00000101443_WFDC2', 'ENSG00000101444_AHCY', 'ENSG00000101445_PPP1R16B', 'ENSG00000101447_FAM83D', 'ENSG00000101452_DHX35', 'ENSG00000101457_DNTTIP1', 'ENSG00000101460_MAP1LC3A', 'ENSG00000101464_PIGU', 'ENSG00000101470_TNNC2', 'ENSG00000101473_ACOT8', 'ENSG00000101474_APMAP', 'ENSG00000101489_CELF4', 'ENSG00000101493_ZNF516', 'ENSG00000101544_ADNP2', 'ENSG00000101546_RBFA', 'ENSG00000101557_USP14', 'ENSG00000101558_VAPA', 'ENSG00000101574_METTL4', 'ENSG00000101577_LPIN2', 'ENSG00000101596_SMCHD1', 'ENSG00000101605_MYOM1', 'ENSG00000101608_MYL12A', 'ENSG00000101624_CEP76', 'ENSG00000101639_CEP192', 'ENSG00000101654_RNMT', 'ENSG00000101665_SMAD7', 'ENSG00000101670_LIPG', 'ENSG00000101695_RNF125', 'ENSG00000101745_ANKRD12', 'ENSG00000101751_POLI', 'ENSG00000101752_MIB1', 'ENSG00000101773_RBBP8', 'ENSG00000101782_RIOK3', 'ENSG00000101811_CSTF2', 'ENSG00000101843_PSMD10', 'ENSG00000101844_ATG4A', 'ENSG00000101846_STS', 'ENSG00000101849_TBL1X', 'ENSG00000101856_PGRMC1', 'ENSG00000101868_POLA1', 'ENSG00000101871_MID1', 'ENSG00000101882_NKAP', 'ENSG00000101888_NXT2', 'ENSG00000101901_ALG13', 'ENSG00000101911_PRPS2', 'ENSG00000101916_TLR8', 'ENSG00000101928_MOSPD1', 'ENSG00000101935_AMMECR1', 'ENSG00000101938_CHRDL1', 'ENSG00000101940_WDR13', 'ENSG00000101945_SUV39H1', 'ENSG00000101955_SRPX', 'ENSG00000101966_XIAP', 'ENSG00000101972_STAG2', 'ENSG00000101974_ATP11C', 'ENSG00000101986_ABCD1', 'ENSG00000101997_CCDC22', 'ENSG00000102001_CACNA1F', 'ENSG00000102003_SYP', 'ENSG00000102007_PLP2', 'ENSG00000102010_BMX', 'ENSG00000102024_PLS3', 'ENSG00000102030_NAA10', 'ENSG00000102032_RENBP', 'ENSG00000102034_ELF4', 'ENSG00000102038_SMARCA1', 'ENSG00000102043_MTMR8', 'ENSG00000102048_ASB9', 'ENSG00000102053_ZC3H12B', 'ENSG00000102054_RBBP7', 'ENSG00000102057_KCND1', 'ENSG00000102078_SLC25A14', 'ENSG00000102081_FMR1', 'ENSG00000102096_PIM2', 'ENSG00000102098_SCML2', 'ENSG00000102100_SLC35A2', 'ENSG00000102103_PQBP1', 'ENSG00000102109_PCSK1N', 'ENSG00000102119_EMD', 'ENSG00000102125_TAZ', 'ENSG00000102144_PGK1', 'ENSG00000102145_GATA1', 'ENSG00000102158_MAGT1', 'ENSG00000102172_SMS', 'ENSG00000102178_UBL4A', 'ENSG00000102181_CD99L2', 'ENSG00000102189_EEA1', 'ENSG00000102218_RP2', 'ENSG00000102221_JADE3', 'ENSG00000102225_CDK16', 'ENSG00000102226_USP11', 'ENSG00000102230_PCYT1B', 'ENSG00000102241_HTATSF1', 'ENSG00000102245_CD40LG', 'ENSG00000102265_TIMP1', 'ENSG00000102290_PCDH11X', 'ENSG00000102302_FGD1', 'ENSG00000102309_PIN4', 'ENSG00000102312_PORCN', 'ENSG00000102316_MAGED2', 'ENSG00000102317_RBM3', 'ENSG00000102349_KLF8', 'ENSG00000102359_SRPX2', 'ENSG00000102362_SYTL4', 'ENSG00000102383_ZDHHC15', 'ENSG00000102384_CENPI', 'ENSG00000102385_DRP2', 'ENSG00000102390_PBDC1', 'ENSG00000102393_GLA', 'ENSG00000102401_ARMCX3', 'ENSG00000102409_BEX4', 'ENSG00000102445_RUBCNL', 'ENSG00000102468_HTR2A', 'ENSG00000102471_NDFIP2', 'ENSG00000102524_TNFSF13B', 'ENSG00000102531_FNDC3A', 'ENSG00000102543_CDADC1', 'ENSG00000102547_CAB39L', 'ENSG00000102554_KLF5', 'ENSG00000102572_STK24', 'ENSG00000102575_ACP5', 'ENSG00000102580_DNAJC3', 'ENSG00000102595_UGGT2', 'ENSG00000102606_ARHGEF7', 'ENSG00000102699_PARP4', 'ENSG00000102710_SUPT20H', 'ENSG00000102738_MRPS31', 'ENSG00000102743_SLC25A15', 'ENSG00000102753_KPNA3', 'ENSG00000102755_FLT1', 'ENSG00000102760_RGCC', 'ENSG00000102763_VWA8', 'ENSG00000102780_DGKH', 'ENSG00000102781_KATNAL1', 'ENSG00000102786_INTS6', 'ENSG00000102796_DHRS12', 'ENSG00000102804_TSC22D1', 'ENSG00000102805_CLN5', 'ENSG00000102858_MGRN1', 'ENSG00000102870_ZNF629', 'ENSG00000102871_TRADD', 'ENSG00000102878_HSF4', 'ENSG00000102879_CORO1A', 'ENSG00000102882_MAPK3', 'ENSG00000102886_GDPD3', 'ENSG00000102890_ELMO3', 'ENSG00000102893_PHKB', 'ENSG00000102897_LYRM1', 'ENSG00000102898_NUTF2', 'ENSG00000102900_NUP93', 'ENSG00000102901_CENPT', 'ENSG00000102904_TSNAXIP1', 'ENSG00000102908_NFAT5', 'ENSG00000102910_LONP2', 'ENSG00000102921_N4BP1', 'ENSG00000102931_ARL2BP', 'ENSG00000102934_PLLP', 'ENSG00000102935_ZNF423', 'ENSG00000102967_DHODH', 'ENSG00000102974_CTCF', 'ENSG00000102977_ACD', 'ENSG00000102978_POLR2C', 'ENSG00000102981_PARD6A', 'ENSG00000102984_ZNF821', 'ENSG00000102996_MMP15', 'ENSG00000103005_USB1', 'ENSG00000103018_CYB5B', 'ENSG00000103021_CCDC113', 'ENSG00000103024_NME3', 'ENSG00000103035_PSMD7', 'ENSG00000103037_SETD6', 'ENSG00000103042_SLC38A7', 'ENSG00000103043_VAC14', 'ENSG00000103044_HAS3', 'ENSG00000103047_TANGO6', 'ENSG00000103051_COG4', 'ENSG00000103056_SMPD3', 'ENSG00000103061_SLC7A6OS', 'ENSG00000103064_SLC7A6', 'ENSG00000103066_PLA2G15', 'ENSG00000103067_ESRP2', 'ENSG00000103091_WDR59', 'ENSG00000103111_MON1B', 'ENSG00000103121_CMC2', 'ENSG00000103126_AXIN1', 'ENSG00000103145_HCFC1R1', 'ENSG00000103148_NPRL3', 'ENSG00000103150_MLYCD', 'ENSG00000103152_MPG', 'ENSG00000103160_HSDL1', 'ENSG00000103168_TAF1C', 'ENSG00000103174_NAGPA', 'ENSG00000103175_WFDC1', 'ENSG00000103184_SEC14L5', 'ENSG00000103187_COTL1', 'ENSG00000103194_USP10', 'ENSG00000103196_CRISPLD2', 'ENSG00000103197_TSC2', 'ENSG00000103199_ZNF500', 'ENSG00000103202_NME4', 'ENSG00000103222_ABCC1', 'ENSG00000103226_NOMO3', 'ENSG00000103227_LMF1', 'ENSG00000103245_CIAO3', 'ENSG00000103248_MTHFSD', 'ENSG00000103249_CLCN7', 'ENSG00000103253_HAGHL', 'ENSG00000103254_FAM173A', 'ENSG00000103257_SLC7A5', 'ENSG00000103260_METRN', 'ENSG00000103264_FBXO31', 'ENSG00000103266_STUB1', 'ENSG00000103269_RHBDL1', 'ENSG00000103274_NUBP1', 'ENSG00000103275_UBE2I', 'ENSG00000103316_CRYM', 'ENSG00000103319_EEF2K', 'ENSG00000103326_CAPN15', 'ENSG00000103335_PIEZO1', 'ENSG00000103342_GSPT1', 'ENSG00000103343_ZNF174', 'ENSG00000103351_CLUAP1', 'ENSG00000103353_UBFD1', 'ENSG00000103356_EARS2', 'ENSG00000103363_ELOB', 'ENSG00000103365_GGA2', 'ENSG00000103381_CPPED1', 'ENSG00000103404_USP31', 'ENSG00000103415_HMOX2', 'ENSG00000103423_DNAJA3', 'ENSG00000103429_BFAR', 'ENSG00000103472_RRN3P2', 'ENSG00000103479_RBL2', 'ENSG00000103485_QPRT', 'ENSG00000103489_XYLT1', 'ENSG00000103490_PYCARD', 'ENSG00000103494_RPGRIP1L', 'ENSG00000103495_MAZ', 'ENSG00000103496_STX4', 'ENSG00000103502_CDIPT', 'ENSG00000103507_BCKDK', 'ENSG00000103510_KAT8', 'ENSG00000103512_NOMO1', 'ENSG00000103522_IL21R', 'ENSG00000103534_TMC5', 'ENSG00000103540_CCP110', 'ENSG00000103544_VPS35L', 'ENSG00000103546_SLC6A2', 'ENSG00000103549_RNF40', 'ENSG00000103550_KNOP1', 'ENSG00000103591_AAGAB', 'ENSG00000103599_IQCH', 'ENSG00000103642_LACTB', 'ENSG00000103647_CORO2B', 'ENSG00000103653_CSK', 'ENSG00000103657_HERC1', 'ENSG00000103671_TRIP4', 'ENSG00000103707_MTFMT', 'ENSG00000103723_AP3B2', 'ENSG00000103740_ACSBG1', 'ENSG00000103769_RAB11A', 'ENSG00000103811_CTSH', 'ENSG00000103852_TTC23', 'ENSG00000103855_CD276', 'ENSG00000103876_FAH', 'ENSG00000103888_CEMIP', 'ENSG00000103932_RPAP1', 'ENSG00000103942_HOMER2', 'ENSG00000103966_EHD4', 'ENSG00000103978_TMEM87A', 'ENSG00000103994_ZNF106', 'ENSG00000103995_CEP152', 'ENSG00000104043_ATP8B4', 'ENSG00000104047_DTWD1', 'ENSG00000104055_TGM5', 'ENSG00000104064_GABPB1', 'ENSG00000104067_TJP1', 'ENSG00000104081_BMF', 'ENSG00000104093_DMXL2', 'ENSG00000104129_DNAJC17', 'ENSG00000104131_EIF3J', 'ENSG00000104133_SPG11', 'ENSG00000104140_RHOV', 'ENSG00000104142_VPS18', 'ENSG00000104147_OIP5', 'ENSG00000104154_SLC30A4', 'ENSG00000104164_BLOC1S6', 'ENSG00000104177_MYEF2', 'ENSG00000104205_SGK3', 'ENSG00000104218_CSPP1', 'ENSG00000104219_ZDHHC2', 'ENSG00000104221_BRF2', 'ENSG00000104228_TRIM35', 'ENSG00000104231_ZFAND1', 'ENSG00000104267_CA2', 'ENSG00000104290_FZD3', 'ENSG00000104299_INTS9', 'ENSG00000104312_RIPK2', 'ENSG00000104320_NBN', 'ENSG00000104324_CPQ', 'ENSG00000104325_DECR1', 'ENSG00000104331_IMPAD1', 'ENSG00000104341_LAPTM4B', 'ENSG00000104343_UBE2W', 'ENSG00000104356_POP1', 'ENSG00000104361_NIPAL2', 'ENSG00000104365_IKBKB', 'ENSG00000104368_PLAT', 'ENSG00000104369_JPH1', 'ENSG00000104375_STK3', 'ENSG00000104381_GDAP1', 'ENSG00000104388_RAB2A', 'ENSG00000104408_EIF3E', 'ENSG00000104412_EMC2', 'ENSG00000104419_NDRG1', 'ENSG00000104427_ZC2HC1A', 'ENSG00000104432_IL7', 'ENSG00000104442_ARMC1', 'ENSG00000104447_TRPS1', 'ENSG00000104450_SPAG1', 'ENSG00000104472_CHRAC1', 'ENSG00000104490_NCALD', 'ENSG00000104497_SNX16', 'ENSG00000104517_UBR5', 'ENSG00000104518_GSDMD', 'ENSG00000104522_TSTA3', 'ENSG00000104524_PYCR3', 'ENSG00000104529_EEF1D', 'ENSG00000104549_SQLE', 'ENSG00000104611_SH2D4A', 'ENSG00000104613_INTS10', 'ENSG00000104626_ERI1', 'ENSG00000104635_SLC39A14', 'ENSG00000104643_MTMR9', 'ENSG00000104660_LEPROTL1', 'ENSG00000104671_DCTN6', 'ENSG00000104679_R3HCC1', 'ENSG00000104687_GSR', 'ENSG00000104689_TNFRSF10A', 'ENSG00000104691_UBXN8', 'ENSG00000104695_PPP2CB', 'ENSG00000104714_ERICH1', 'ENSG00000104723_TUSC3', 'ENSG00000104728_ARHGEF10', 'ENSG00000104731_KLHDC4', 'ENSG00000104738_MCM4', 'ENSG00000104756_KCTD9', 'ENSG00000104763_ASAH1', 'ENSG00000104765_BNIP3L', 'ENSG00000104774_MAN2B1', 'ENSG00000104783_KCNN4', 'ENSG00000104805_NUCB1', 'ENSG00000104808_DHDH', 'ENSG00000104812_GYS1', 'ENSG00000104814_MAP4K1', 'ENSG00000104823_ECH1', 'ENSG00000104824_HNRNPL', 'ENSG00000104825_NFKBIB', 'ENSG00000104833_TUBB4A', 'ENSG00000104835_SARS2', 'ENSG00000104848_KCNA7', 'ENSG00000104852_SNRNP70', 'ENSG00000104853_CLPTM1', 'ENSG00000104856_RELB', 'ENSG00000104859_CLASRP', 'ENSG00000104863_LIN7B', 'ENSG00000104866_PPP1R37', 'ENSG00000104870_FCGRT', 'ENSG00000104872_PIH1D1', 'ENSG00000104879_CKM', 'ENSG00000104880_ARHGEF18', 'ENSG00000104881_PPP1R13L', 'ENSG00000104883_PEX11G', 'ENSG00000104884_ERCC2', 'ENSG00000104885_DOT1L', 'ENSG00000104886_PLEKHJ1', 'ENSG00000104889_RNASEH2A', 'ENSG00000104894_CD37', 'ENSG00000104897_SF3A2', 'ENSG00000104899_AMH', 'ENSG00000104903_LYL1', 'ENSG00000104904_OAZ1', 'ENSG00000104907_TRMT1', 'ENSG00000104915_STX10', 'ENSG00000104918_RETN', 'ENSG00000104921_FCER2', 'ENSG00000104936_DMPK', 'ENSG00000104946_TBC1D17', 'ENSG00000104951_IL4I1', 'ENSG00000104953_TLE6', 'ENSG00000104957_CCDC130', 'ENSG00000104960_PTOV1', 'ENSG00000104964_AES', 'ENSG00000104969_SGTA', 'ENSG00000104972_LILRB1', 'ENSG00000104973_MED25', 'ENSG00000104974_LILRA1', 'ENSG00000104976_SNAPC2', 'ENSG00000104979_C19orf53', 'ENSG00000104980_TIMM44', 'ENSG00000104983_CCDC61', 'ENSG00000104998_IL27RA', 'ENSG00000105011_ASF1B', 'ENSG00000105048_TNNT1', 'ENSG00000105053_VRK3', 'ENSG00000105058_FAM32A', 'ENSG00000105063_PPP6R1', 'ENSG00000105072_C19orf44', 'ENSG00000105085_MED26', 'ENSG00000105122_RASAL3', 'ENSG00000105127_AKAP8', 'ENSG00000105135_ILVBL', 'ENSG00000105136_ZNF419', 'ENSG00000105137_SYDE1', 'ENSG00000105143_SLC1A6', 'ENSG00000105146_AURKC', 'ENSG00000105171_POP4', 'ENSG00000105173_CCNE1', 'ENSG00000105176_URI1', 'ENSG00000105185_PDCD5', 'ENSG00000105186_ANKRD27', 'ENSG00000105193_RPS16', 'ENSG00000105197_TIMM50', 'ENSG00000105202_FBL', 'ENSG00000105204_DYRK1B', 'ENSG00000105205_CLC', 'ENSG00000105220_GPI', 'ENSG00000105221_AKT2', 'ENSG00000105223_PLD3', 'ENSG00000105227_PRX', 'ENSG00000105229_PIAS4', 'ENSG00000105245_NUMBL', 'ENSG00000105248_YJU2', 'ENSG00000105251_SHD', 'ENSG00000105254_TBCB', 'ENSG00000105255_FSD1', 'ENSG00000105258_POLR2I', 'ENSG00000105270_CLIP3', 'ENSG00000105278_ZFR2', 'ENSG00000105281_SLC1A5', 'ENSG00000105287_PRKD2', 'ENSG00000105298_CACTIN', 'ENSG00000105321_CCDC9', 'ENSG00000105323_HNRNPUL1', 'ENSG00000105325_FZR1', 'ENSG00000105327_BBC3', 'ENSG00000105329_TGFB1', 'ENSG00000105339_DENND3', 'ENSG00000105341_DMAC2', 'ENSG00000105352_CEACAM4', 'ENSG00000105355_PLIN3', 'ENSG00000105364_MRPL4', 'ENSG00000105366_SIGLEC8', 'ENSG00000105369_CD79A', 'ENSG00000105371_ICAM4', 'ENSG00000105372_RPS19', 'ENSG00000105373_NOP53', 'ENSG00000105374_NKG7', 'ENSG00000105376_ICAM5', 'ENSG00000105379_ETFB', 'ENSG00000105383_CD33', 'ENSG00000105392_CRX', 'ENSG00000105393_BABAM1', 'ENSG00000105397_TYK2', 'ENSG00000105401_CDC37', 'ENSG00000105402_NAPA', 'ENSG00000105404_RABAC1', 'ENSG00000105409_ATP1A3', 'ENSG00000105426_PTPRS', 'ENSG00000105427_CNFN', 'ENSG00000105429_MEGF8', 'ENSG00000105438_KDELR1', 'ENSG00000105443_CYTH2', 'ENSG00000105447_GRWD1', 'ENSG00000105464_GRIN2D', 'ENSG00000105467_SYNGR4', 'ENSG00000105472_CLEC11A', 'ENSG00000105479_CCDC114', 'ENSG00000105483_CARD8', 'ENSG00000105486_LIG1', 'ENSG00000105492_SIGLEC6', 'ENSG00000105497_ZNF175', 'ENSG00000105499_PLA2G4C', 'ENSG00000105501_SIGLEC5', 'ENSG00000105514_RAB3D', 'ENSG00000105516_DBP', 'ENSG00000105518_TMEM205', 'ENSG00000105519_CAPS', 'ENSG00000105520_PLPPR2', 'ENSG00000105523_FAM83E', 'ENSG00000105538_RASIP1', 'ENSG00000105552_BCAT2', 'ENSG00000105556_MIER2', 'ENSG00000105559_PLEKHA4', 'ENSG00000105568_PPP2R1A', 'ENSG00000105576_TNPO2', 'ENSG00000105583_WDR83OS', 'ENSG00000105607_GCDH', 'ENSG00000105610_KLF1', 'ENSG00000105612_DNASE2', 'ENSG00000105613_MAST1', 'ENSG00000105617_LENG1', 'ENSG00000105618_PRPF31', 'ENSG00000105619_TFPT', 'ENSG00000105639_JAK3', 'ENSG00000105640_RPL18A', 'ENSG00000105641_SLC5A5', 'ENSG00000105643_ARRDC2', 'ENSG00000105647_PIK3R2', 'ENSG00000105649_RAB3A', 'ENSG00000105655_ISYNA1', 'ENSG00000105656_ELL', 'ENSG00000105662_CRTC1', 'ENSG00000105668_UPK1A', 'ENSG00000105669_COPE', 'ENSG00000105671_DDX49', 'ENSG00000105672_ETV2', 'ENSG00000105676_ARMC6', 'ENSG00000105677_TMEM147', 'ENSG00000105679_GAPDHS', 'ENSG00000105697_HAMP', 'ENSG00000105698_USF2', 'ENSG00000105699_LSR', 'ENSG00000105700_KXD1', 'ENSG00000105701_FKBP8', 'ENSG00000105705_SUGP1', 'ENSG00000105708_ZNF14', 'ENSG00000105711_SCN1B', 'ENSG00000105717_PBX4', 'ENSG00000105722_ERF', 'ENSG00000105723_GSK3A', 'ENSG00000105726_ATP13A1', 'ENSG00000105732_ZNF574', 'ENSG00000105737_GRIK5', 'ENSG00000105738_SIPA1L3', 'ENSG00000105750_ZNF85', 'ENSG00000105755_ETHE1', 'ENSG00000105767_CADM4', 'ENSG00000105771_SMG9', 'ENSG00000105778_AVL9', 'ENSG00000105784_RUNDC3B', 'ENSG00000105792_CFAP69', 'ENSG00000105793_GTPBP10', 'ENSG00000105808_RASA4', 'ENSG00000105810_CDK6', 'ENSG00000105819_PMPCB', 'ENSG00000105821_DNAJC2', 'ENSG00000105829_BET1', 'ENSG00000105835_NAMPT', 'ENSG00000105849_TWISTNB', 'ENSG00000105851_PIK3CG', 'ENSG00000105854_PON2', 'ENSG00000105855_ITGB8', 'ENSG00000105856_HBP1', 'ENSG00000105865_DUS4L', 'ENSG00000105866_SP4', 'ENSG00000105875_WDR91', 'ENSG00000105877_DNAH11', 'ENSG00000105879_CBLL1', 'ENSG00000105887_MTPN', 'ENSG00000105889_STEAP1B', 'ENSG00000105926_MPP6', 'ENSG00000105928_GSDME', 'ENSG00000105939_ZC3HAV1', 'ENSG00000105948_TTC26', 'ENSG00000105953_OGDH', 'ENSG00000105963_ADAP1', 'ENSG00000105967_TFEC', 'ENSG00000105968_H2AFV', 'ENSG00000105971_CAV2', 'ENSG00000105974_CAV1', 'ENSG00000105976_MET', 'ENSG00000105982_RNF32', 'ENSG00000105983_LMBR1', 'ENSG00000105988_NHP2P1', 'ENSG00000105991_HOXA1', 'ENSG00000105993_DNAJB6', 'ENSG00000105996_HOXA2', 'ENSG00000105997_HOXA3', 'ENSG00000106003_LFNG', 'ENSG00000106004_HOXA5', 'ENSG00000106006_HOXA6', 'ENSG00000106009_BRAT1', 'ENSG00000106012_IQCE', 'ENSG00000106013_ANKRD7', 'ENSG00000106025_TSPAN12', 'ENSG00000106028_SSBP1', 'ENSG00000106034_CPED1', 'ENSG00000106049_HIBADH', 'ENSG00000106052_TAX1BP1', 'ENSG00000106066_CPVL', 'ENSG00000106069_CHN2', 'ENSG00000106070_GRB10', 'ENSG00000106077_ABHD11', 'ENSG00000106080_FKBP14', 'ENSG00000106086_PLEKHA8', 'ENSG00000106089_STX1A', 'ENSG00000106100_NOD1', 'ENSG00000106105_GARS', 'ENSG00000106123_EPHB6', 'ENSG00000106125_MINDY4', 'ENSG00000106133_NSUN5P2', 'ENSG00000106144_CASP2', 'ENSG00000106153_CHCHD2', 'ENSG00000106211_HSPB1', 'ENSG00000106236_NPTX2', 'ENSG00000106244_PDAP1', 'ENSG00000106245_BUD31', 'ENSG00000106246_PTCD1', 'ENSG00000106258_CYP3A5', 'ENSG00000106261_ZKSCAN1', 'ENSG00000106263_EIF3B', 'ENSG00000106266_SNX8', 'ENSG00000106268_NUDT1', 'ENSG00000106290_TAF6', 'ENSG00000106299_WASL', 'ENSG00000106305_AIMP2', 'ENSG00000106327_TFR2', 'ENSG00000106330_MOSPD3', 'ENSG00000106333_PCOLCE', 'ENSG00000106336_FBXO24', 'ENSG00000106344_RBM28', 'ENSG00000106346_USP42', 'ENSG00000106348_IMPDH1', 'ENSG00000106351_AGFG2', 'ENSG00000106355_LSM5', 'ENSG00000106366_SERPINE1', 'ENSG00000106367_AP1S1', 'ENSG00000106392_C1GALT1', 'ENSG00000106397_PLOD3', 'ENSG00000106399_RPA3', 'ENSG00000106400_ZNHIT1', 'ENSG00000106404_CLDN15', 'ENSG00000106415_GLCCI1', 'ENSG00000106443_PHF14', 'ENSG00000106459_NRF1', 'ENSG00000106460_TMEM106B', 'ENSG00000106462_EZH2', 'ENSG00000106477_CEP41', 'ENSG00000106479_ZNF862', 'ENSG00000106484_MEST', 'ENSG00000106524_ANKMY2', 'ENSG00000106526_ACTR3C', 'ENSG00000106537_TSPAN13', 'ENSG00000106538_RARRES2', 'ENSG00000106546_AHR', 'ENSG00000106554_CHCHD3', 'ENSG00000106560_GIMAP2', 'ENSG00000106565_TMEM176B', 'ENSG00000106588_PSMA2', 'ENSG00000106591_MRPL32', 'ENSG00000106603_COA1', 'ENSG00000106605_BLVRA', 'ENSG00000106608_URGCP', 'ENSG00000106609_TMEM248', 'ENSG00000106610_STAG3L4', 'ENSG00000106615_RHEB', 'ENSG00000106617_PRKAG2', 'ENSG00000106624_AEBP1', 'ENSG00000106628_POLD2', 'ENSG00000106633_GCK', 'ENSG00000106635_BCL7B', 'ENSG00000106636_YKT6', 'ENSG00000106638_TBL2', 'ENSG00000106665_CLIP2', 'ENSG00000106682_EIF4H', 'ENSG00000106683_LIMK1', 'ENSG00000106686_SPATA6L', 'ENSG00000106692_FKTN', 'ENSG00000106701_FSD1L', 'ENSG00000106723_SPIN1', 'ENSG00000106733_NMRK1', 'ENSG00000106771_TMEM245', 'ENSG00000106772_PRUNE2', 'ENSG00000106780_MEGF9', 'ENSG00000106785_TRIM14', 'ENSG00000106789_CORO2A', 'ENSG00000106799_TGFBR1', 'ENSG00000106803_SEC61B', 'ENSG00000106804_C5', 'ENSG00000106809_OGN', 'ENSG00000106819_ASPN', 'ENSG00000106823_ECM2', 'ENSG00000106829_TLE4', 'ENSG00000106852_LHX6', 'ENSG00000106853_PTGR1', 'ENSG00000106868_SUSD1', 'ENSG00000106948_AKNA', 'ENSG00000106952_TNFSF8', 'ENSG00000106976_DNM1', 'ENSG00000106991_ENG', 'ENSG00000106992_AK1', 'ENSG00000106993_CDC37L1', 'ENSG00000107014_RLN2', 'ENSG00000107018_RLN1', 'ENSG00000107020_PLGRKT', 'ENSG00000107021_TBC1D13', 'ENSG00000107036_RIC1', 'ENSG00000107077_KDM4C', 'ENSG00000107099_DOCK8', 'ENSG00000107104_KANK1', 'ENSG00000107130_NCS1', 'ENSG00000107140_TESK1', 'ENSG00000107159_CA9', 'ENSG00000107164_FUBP3', 'ENSG00000107175_CREB3', 'ENSG00000107185_RGP1', 'ENSG00000107186_MPDZ', 'ENSG00000107201_DDX58', 'ENSG00000107223_EDF1', 'ENSG00000107242_PIP5K1B', 'ENSG00000107249_GLIS3', 'ENSG00000107262_BAG1', 'ENSG00000107263_RAPGEF1', 'ENSG00000107281_NPDC1', 'ENSG00000107290_SETX', 'ENSG00000107331_ABCA2', 'ENSG00000107338_SHB', 'ENSG00000107341_UBE2R2', 'ENSG00000107362_ABHD17B', 'ENSG00000107371_EXOSC3', 'ENSG00000107372_ZFAND5', 'ENSG00000107404_DVL1', 'ENSG00000107438_PDLIM1', 'ENSG00000107443_CCNJ', 'ENSG00000107447_DNTT', 'ENSG00000107485_GATA3', 'ENSG00000107521_HPS1', 'ENSG00000107537_PHYH', 'ENSG00000107551_RASSF4', 'ENSG00000107554_DNMBP', 'ENSG00000107560_RAB11FIP2', 'ENSG00000107562_CXCL12', 'ENSG00000107566_ERLIN1', 'ENSG00000107581_EIF3A', 'ENSG00000107611_CUBN', 'ENSG00000107614_TRDMT1', 'ENSG00000107625_DDX50', 'ENSG00000107643_MAPK8', 'ENSG00000107651_SEC23IP', 'ENSG00000107669_ATE1', 'ENSG00000107672_NSMCE4A', 'ENSG00000107679_PLEKHA1', 'ENSG00000107719_PALD1', 'ENSG00000107731_UNC5B', 'ENSG00000107736_CDH23', 'ENSG00000107738_VSIR', 'ENSG00000107742_SPOCK2', 'ENSG00000107745_MICU1', 'ENSG00000107758_PPP3CB', 'ENSG00000107771_CCSER2', 'ENSG00000107779_BMPR1A', 'ENSG00000107789_MINPP1', 'ENSG00000107796_ACTA2', 'ENSG00000107798_LIPA', 'ENSG00000107815_TWNK', 'ENSG00000107816_LZTS2', 'ENSG00000107819_SFXN3', 'ENSG00000107821_KAZALD1', 'ENSG00000107829_FBXW4', 'ENSG00000107833_NPM3', 'ENSG00000107854_TNKS2', 'ENSG00000107862_GBF1', 'ENSG00000107863_ARHGAP21', 'ENSG00000107864_CPEB3', 'ENSG00000107872_FBXL15', 'ENSG00000107874_CUEDC2', 'ENSG00000107882_SUFU', 'ENSG00000107890_ANKRD26', 'ENSG00000107897_ACBD5', 'ENSG00000107902_LHPP', 'ENSG00000107929_LARP4B', 'ENSG00000107937_GTPBP4', 'ENSG00000107938_EDRF1', 'ENSG00000107949_BCCIP', 'ENSG00000107951_MTPAP', 'ENSG00000107954_NEURL1', 'ENSG00000107957_SH3PXD2A', 'ENSG00000107959_PITRM1', 'ENSG00000107960_STN1', 'ENSG00000107968_MAP3K8', 'ENSG00000107984_DKK1', 'ENSG00000108001_EBF3', 'ENSG00000108010_GLRX3', 'ENSG00000108021_FAM208B', 'ENSG00000108039_XPNPEP1', 'ENSG00000108055_SMC3', 'ENSG00000108061_SHOC2', 'ENSG00000108064_TFAM', 'ENSG00000108091_CCDC6', 'ENSG00000108094_CUL2', 'ENSG00000108100_CCNY', 'ENSG00000108106_UBE2S', 'ENSG00000108107_RPL28', 'ENSG00000108175_ZMIZ1', 'ENSG00000108176_DNAJC12', 'ENSG00000108179_PPIF', 'ENSG00000108187_PBLD', 'ENSG00000108219_TSPAN14', 'ENSG00000108239_TBC1D12', 'ENSG00000108256_NUFIP2', 'ENSG00000108262_GIT1', 'ENSG00000108298_RPL19', 'ENSG00000108306_FBXL20', 'ENSG00000108309_RUNDC3A', 'ENSG00000108312_UBTF', 'ENSG00000108344_PSMD3', 'ENSG00000108349_CASC3', 'ENSG00000108352_RAPGEFL1', 'ENSG00000108370_RGS9', 'ENSG00000108375_RNF43', 'ENSG00000108379_WNT3', 'ENSG00000108384_RAD51C', 'ENSG00000108387_SEPT4', 'ENSG00000108389_MTMR4', 'ENSG00000108395_TRIM37', 'ENSG00000108405_P2RX1', 'ENSG00000108406_DHX40', 'ENSG00000108423_TUBD1', 'ENSG00000108424_KPNB1', 'ENSG00000108433_GOSR2', 'ENSG00000108439_PNPO', 'ENSG00000108443_RPS6KB1', 'ENSG00000108448_TRIM16L', 'ENSG00000108465_CDK5RAP3', 'ENSG00000108468_CBX1', 'ENSG00000108469_RECQL5', 'ENSG00000108474_PIGL', 'ENSG00000108479_GALK1', 'ENSG00000108506_INTS2', 'ENSG00000108509_CAMTA2', 'ENSG00000108510_MED13', 'ENSG00000108511_HOXB6', 'ENSG00000108515_ENO3', 'ENSG00000108518_PFN1', 'ENSG00000108523_RNF167', 'ENSG00000108528_SLC25A11', 'ENSG00000108551_RASD1', 'ENSG00000108556_CHRNE', 'ENSG00000108557_RAI1', 'ENSG00000108559_NUP88', 'ENSG00000108561_C1QBP', 'ENSG00000108576_SLC6A4', 'ENSG00000108578_BLMH', 'ENSG00000108582_CPD', 'ENSG00000108587_GOSR1', 'ENSG00000108588_CCDC47', 'ENSG00000108590_MED31', 'ENSG00000108591_DRG2', 'ENSG00000108592_FTSJ3', 'ENSG00000108599_AKAP10', 'ENSG00000108602_ALDH3A1', 'ENSG00000108604_SMARCD2', 'ENSG00000108622_ICAM2', 'ENSG00000108639_SYNGR2', 'ENSG00000108641_B9D1', 'ENSG00000108651_UTP6', 'ENSG00000108654_DDX5', 'ENSG00000108666_C17orf75', 'ENSG00000108669_CYTH1', 'ENSG00000108671_PSMD11', 'ENSG00000108679_LGALS3BP', 'ENSG00000108691_CCL2', 'ENSG00000108733_PEX12', 'ENSG00000108771_DHX58', 'ENSG00000108773_KAT2A', 'ENSG00000108774_RAB5C', 'ENSG00000108784_NAGLU', 'ENSG00000108786_HSD17B1', 'ENSG00000108788_MLX', 'ENSG00000108798_ABI3', 'ENSG00000108799_EZH1', 'ENSG00000108813_DLX4', 'ENSG00000108819_PPP1R9B', 'ENSG00000108826_MRPL27', 'ENSG00000108828_VAT1', 'ENSG00000108829_LRRC59', 'ENSG00000108830_RND2', 'ENSG00000108839_ALOX12', 'ENSG00000108840_HDAC5', 'ENSG00000108846_ABCC3', 'ENSG00000108848_LUC7L3', 'ENSG00000108852_MPP2', 'ENSG00000108854_SMURF2', 'ENSG00000108861_DUSP3', 'ENSG00000108883_EFTUD2', 'ENSG00000108924_HLF', 'ENSG00000108932_SLC16A6', 'ENSG00000108946_PRKAR1A', 'ENSG00000108950_FAM20A', 'ENSG00000108953_YWHAE', 'ENSG00000108960_MMD', 'ENSG00000108961_RANGRF', 'ENSG00000108963_DPH1', 'ENSG00000108984_MAP2K6', 'ENSG00000109016_DHRS7B', 'ENSG00000109046_WSB1', 'ENSG00000109062_SLC9A3R1', 'ENSG00000109063_MYH3', 'ENSG00000109065_NAT9', 'ENSG00000109066_TMEM104', 'ENSG00000109079_TNFAIP1', 'ENSG00000109083_IFT20', 'ENSG00000109084_TMEM97', 'ENSG00000109089_CDR2L', 'ENSG00000109099_PMP22', 'ENSG00000109103_UNC119', 'ENSG00000109107_ALDOC', 'ENSG00000109111_SUPT6H', 'ENSG00000109113_RAB34', 'ENSG00000109118_PHF12', 'ENSG00000109133_TMEM33', 'ENSG00000109171_SLAIN2', 'ENSG00000109180_OCIAD1', 'ENSG00000109184_DCUN1D4', 'ENSG00000109189_USP46', 'ENSG00000109220_CHIC2', 'ENSG00000109255_NMU', 'ENSG00000109265_KIAA1211', 'ENSG00000109270_LAMTOR3', 'ENSG00000109272_PF4V1', 'ENSG00000109320_NFKB1', 'ENSG00000109321_AREG', 'ENSG00000109323_MANBA', 'ENSG00000109332_UBE2D3', 'ENSG00000109339_MAPK10', 'ENSG00000109381_ELF2', 'ENSG00000109390_NDUFC1', 'ENSG00000109436_TBC1D9', 'ENSG00000109445_ZNF330', 'ENSG00000109452_INPP4B', 'ENSG00000109458_GAB1', 'ENSG00000109466_KLHL2', 'ENSG00000109472_CPE', 'ENSG00000109475_RPL34', 'ENSG00000109501_WFS1', 'ENSG00000109519_GRPEL1', 'ENSG00000109534_GAR1', 'ENSG00000109536_FRG1', 'ENSG00000109572_CLCN3', 'ENSG00000109576_AADAT', 'ENSG00000109586_GALNT7', 'ENSG00000109606_DHX15', 'ENSG00000109618_SEPSECS', 'ENSG00000109654_TRIM2', 'ENSG00000109667_SLC2A9', 'ENSG00000109670_FBXW7', 'ENSG00000109674_NEIL3', 'ENSG00000109680_TBC1D19', 'ENSG00000109684_CLNK', 'ENSG00000109685_NSD2', 'ENSG00000109686_SH3D19', 'ENSG00000109689_STIM2', 'ENSG00000109736_MFSD10', 'ENSG00000109738_GLRB', 'ENSG00000109743_BST1', 'ENSG00000109756_RAPGEF2', 'ENSG00000109762_SNX25', 'ENSG00000109771_LRP2BP', 'ENSG00000109775_UFSP2', 'ENSG00000109787_KLF3', 'ENSG00000109790_KLHL5', 'ENSG00000109805_NCAPG', 'ENSG00000109814_UGDH', 'ENSG00000109854_HTATIP2', 'ENSG00000109861_CTSC', 'ENSG00000109881_CCDC34', 'ENSG00000109906_ZBTB16', 'ENSG00000109911_ELP4', 'ENSG00000109917_ZPR1', 'ENSG00000109919_MTCH2', 'ENSG00000109920_FNBP4', 'ENSG00000109927_TECTA', 'ENSG00000109929_SC5D', 'ENSG00000109943_CRTAM', 'ENSG00000109944_JHY', 'ENSG00000109971_HSPA8', 'ENSG00000110002_VWA5A', 'ENSG00000110011_DNAJC4', 'ENSG00000110013_SIAE', 'ENSG00000110025_SNX15', 'ENSG00000110031_LPXN', 'ENSG00000110042_DTX4', 'ENSG00000110046_ATG2A', 'ENSG00000110047_EHD1', 'ENSG00000110048_OSBP', 'ENSG00000110057_UNC93B1', 'ENSG00000110060_PUS3', 'ENSG00000110063_DCPS', 'ENSG00000110066_KMT5B', 'ENSG00000110074_FOXRED1', 'ENSG00000110075_PPP6R3', 'ENSG00000110076_NRXN2', 'ENSG00000110077_MS4A6A', 'ENSG00000110079_MS4A4A', 'ENSG00000110080_ST3GAL4', 'ENSG00000110090_CPT1A', 'ENSG00000110092_CCND1', 'ENSG00000110104_CCDC86', 'ENSG00000110107_PRPF19', 'ENSG00000110108_TMEM109', 'ENSG00000110171_TRIM3', 'ENSG00000110172_CHORDC1', 'ENSG00000110200_ANAPC15', 'ENSG00000110218_PANX1', 'ENSG00000110237_ARHGEF17', 'ENSG00000110274_CEP164', 'ENSG00000110315_RNF141', 'ENSG00000110318_CEP126', 'ENSG00000110321_EIF4G2', 'ENSG00000110324_IL10RA', 'ENSG00000110328_GALNT18', 'ENSG00000110330_BIRC2', 'ENSG00000110344_UBE4A', 'ENSG00000110367_DDX6', 'ENSG00000110395_CBL', 'ENSG00000110400_NECTIN1', 'ENSG00000110422_HIPK3', 'ENSG00000110429_FBXO3', 'ENSG00000110435_PDHX', 'ENSG00000110442_COMMD9', 'ENSG00000110446_SLC15A3', 'ENSG00000110455_ACCS', 'ENSG00000110492_MDK', 'ENSG00000110497_AMBRA1', 'ENSG00000110514_MADD', 'ENSG00000110536_PTPMT1', 'ENSG00000110583_NAA40', 'ENSG00000110619_CARS', 'ENSG00000110628_SLC22A18', 'ENSG00000110651_CD81', 'ENSG00000110660_SLC35F2', 'ENSG00000110665_C11orf21', 'ENSG00000110693_SOX6', 'ENSG00000110696_C11orf58', 'ENSG00000110697_PITPNM1', 'ENSG00000110700_RPS13', 'ENSG00000110711_AIP', 'ENSG00000110713_NUP98', 'ENSG00000110717_NDUFS8', 'ENSG00000110719_TCIRG1', 'ENSG00000110721_CHKA', 'ENSG00000110723_EXPH5', 'ENSG00000110756_HPS5', 'ENSG00000110768_GTF2H1', 'ENSG00000110799_VWF', 'ENSG00000110801_PSMD9', 'ENSG00000110811_P3H3', 'ENSG00000110841_PPFIBP1', 'ENSG00000110844_PRPF40B', 'ENSG00000110848_CD69', 'ENSG00000110851_PRDM4', 'ENSG00000110852_CLEC2B', 'ENSG00000110871_COQ5', 'ENSG00000110876_SELPLG', 'ENSG00000110880_CORO1C', 'ENSG00000110881_ASIC1', 'ENSG00000110888_CAPRIN2', 'ENSG00000110906_KCTD10', 'ENSG00000110911_SLC11A2', 'ENSG00000110917_MLEC', 'ENSG00000110921_MVK', 'ENSG00000110925_CSRNP2', 'ENSG00000110931_CAMKK2', 'ENSG00000110934_BIN2', 'ENSG00000110944_IL23A', 'ENSG00000110955_ATP5F1B', 'ENSG00000110958_PTGES3', 'ENSG00000110987_BCL7A', 'ENSG00000111011_RSRC2', 'ENSG00000111012_CYP27B1', 'ENSG00000111052_LIN7A', 'ENSG00000111057_KRT18', 'ENSG00000111058_ACSS3', 'ENSG00000111077_TNS2', 'ENSG00000111087_GLI1', 'ENSG00000111110_PPM1H', 'ENSG00000111142_METAP2', 'ENSG00000111144_LTA4H', 'ENSG00000111145_ELK3', 'ENSG00000111186_WNT5B', 'ENSG00000111196_MAGOHB', 'ENSG00000111203_ITFG2', 'ENSG00000111206_FOXM1', 'ENSG00000111215_PRR4', 'ENSG00000111224_PARP11', 'ENSG00000111229_ARPC3', 'ENSG00000111231_GPN3', 'ENSG00000111237_VPS29', 'ENSG00000111247_RAD51AP1', 'ENSG00000111252_SH2B3', 'ENSG00000111254_AKAP3', 'ENSG00000111261_MANSC1', 'ENSG00000111266_DUSP16', 'ENSG00000111269_CREBL2', 'ENSG00000111271_ACAD10', 'ENSG00000111275_ALDH2', 'ENSG00000111276_CDKN1B', 'ENSG00000111300_NAA25', 'ENSG00000111305_GSG1', 'ENSG00000111319_SCNN1A', 'ENSG00000111321_LTBR', 'ENSG00000111325_OGFOD2', 'ENSG00000111328_CDK2AP1', 'ENSG00000111331_OAS3', 'ENSG00000111335_OAS2', 'ENSG00000111344_RASAL1', 'ENSG00000111348_ARHGDIB', 'ENSG00000111358_GTF2H3', 'ENSG00000111361_EIF2B1', 'ENSG00000111364_DDX55', 'ENSG00000111371_SLC38A1', 'ENSG00000111412_C12orf49', 'ENSG00000111424_VDR', 'ENSG00000111445_RFC5', 'ENSG00000111450_STX2', 'ENSG00000111481_COPZ1', 'ENSG00000111490_TBC1D30', 'ENSG00000111530_CAND1', 'ENSG00000111540_RAB5B', 'ENSG00000111554_MDM1', 'ENSG00000111581_NUP107', 'ENSG00000111596_CNOT2', 'ENSG00000111602_TIMELESS', 'ENSG00000111605_CPSF6', 'ENSG00000111615_KRR1', 'ENSG00000111639_MRPL51', 'ENSG00000111640_GAPDH', 'ENSG00000111641_NOP2', 'ENSG00000111642_CHD4', 'ENSG00000111644_ACRBP', 'ENSG00000111647_UHRF1BP1L', 'ENSG00000111652_COPS7A', 'ENSG00000111653_ING4', 'ENSG00000111664_GNB3', 'ENSG00000111665_CDCA3', 'ENSG00000111666_CHPT1', 'ENSG00000111667_USP5', 'ENSG00000111669_TPI1', 'ENSG00000111670_GNPTAB', 'ENSG00000111671_SPSB2', 'ENSG00000111674_ENO2', 'ENSG00000111676_ATN1', 'ENSG00000111678_C12orf57', 'ENSG00000111679_PTPN6', 'ENSG00000111684_LPCAT3', 'ENSG00000111696_NT5DC3', 'ENSG00000111704_NANOG', 'ENSG00000111707_SUDS3', 'ENSG00000111711_GOLT1B', 'ENSG00000111716_LDHB', 'ENSG00000111725_PRKAB1', 'ENSG00000111726_CMAS', 'ENSG00000111727_HCFC2', 'ENSG00000111728_ST8SIA1', 'ENSG00000111729_CLEC4A', 'ENSG00000111731_C2CD5', 'ENSG00000111737_RAB35', 'ENSG00000111752_PHC1', 'ENSG00000111775_COX6A1', 'ENSG00000111785_RIC8B', 'ENSG00000111786_SRSF9', 'ENSG00000111788_AC009533.1', 'ENSG00000111790_FGFR1OP2', 'ENSG00000111796_KLRB1', 'ENSG00000111801_BTN3A3', 'ENSG00000111802_TDP2', 'ENSG00000111816_FRK', 'ENSG00000111817_DSE', 'ENSG00000111832_RWDD1', 'ENSG00000111834_RSPH4A', 'ENSG00000111837_MAK', 'ENSG00000111843_TMEM14C', 'ENSG00000111845_PAK1IP1', 'ENSG00000111846_GCNT2', 'ENSG00000111850_SMIM8', 'ENSG00000111859_NEDD9', 'ENSG00000111860_CEP85L', 'ENSG00000111875_ASF1A', 'ENSG00000111877_MCM9', 'ENSG00000111879_FAM184A', 'ENSG00000111880_RNGTT', 'ENSG00000111885_MAN1A1', 'ENSG00000111897_SERINC1', 'ENSG00000111906_HDDC2', 'ENSG00000111907_TPD52L1', 'ENSG00000111911_HINT3', 'ENSG00000111912_NCOA7', 'ENSG00000111913_RIPOR2', 'ENSG00000111962_UST', 'ENSG00000111981_ULBP1', 'ENSG00000112029_FBXO5', 'ENSG00000112031_MTRF1L', 'ENSG00000112033_PPARD', 'ENSG00000112039_FANCE', 'ENSG00000112053_SLC26A8', 'ENSG00000112062_MAPK14', 'ENSG00000112077_RHAG', 'ENSG00000112078_KCTD20', 'ENSG00000112079_STK38', 'ENSG00000112081_SRSF3', 'ENSG00000112096_SOD2', 'ENSG00000112110_MRPL18', 'ENSG00000112118_MCM3', 'ENSG00000112130_RNF8', 'ENSG00000112137_PHACTR1', 'ENSG00000112139_MDGA1', 'ENSG00000112144_ICK', 'ENSG00000112146_FBXO9', 'ENSG00000112149_CD83', 'ENSG00000112159_MDN1', 'ENSG00000112167_SAYSD1', 'ENSG00000112182_BACH2', 'ENSG00000112195_TREML2', 'ENSG00000112200_ZNF451', 'ENSG00000112208_BAG2', 'ENSG00000112210_RAB23', 'ENSG00000112212_TSPO2', 'ENSG00000112218_GPR63', 'ENSG00000112232_KHDRBS2', 'ENSG00000112234_FBXL4', 'ENSG00000112237_CCNC', 'ENSG00000112242_E2F3', 'ENSG00000112249_ASCC3', 'ENSG00000112282_MED23', 'ENSG00000112290_WASF1', 'ENSG00000112293_GPLD1', 'ENSG00000112294_ALDH5A1', 'ENSG00000112297_CRYBG1', 'ENSG00000112299_VNN1', 'ENSG00000112303_VNN2', 'ENSG00000112304_ACOT13', 'ENSG00000112305_SMAP1', 'ENSG00000112306_RPS12', 'ENSG00000112308_C6orf62', 'ENSG00000112309_B3GAT2', 'ENSG00000112312_GMNN', 'ENSG00000112320_SOBP', 'ENSG00000112335_SNX3', 'ENSG00000112339_HBS1L', 'ENSG00000112343_TRIM38', 'ENSG00000112357_PEX7', 'ENSG00000112365_ZBTB24', 'ENSG00000112367_FIG4', 'ENSG00000112378_PERP', 'ENSG00000112379_ARFGEF3', 'ENSG00000112394_SLC16A10', 'ENSG00000112406_HECA', 'ENSG00000112414_ADGRG6', 'ENSG00000112419_PHACTR2', 'ENSG00000112425_EPM2A', 'ENSG00000112473_SLC39A7', 'ENSG00000112511_PHF1', 'ENSG00000112514_CUTA', 'ENSG00000112530_PACRG', 'ENSG00000112531_QKI', 'ENSG00000112559_MDFI', 'ENSG00000112561_TFEB', 'ENSG00000112576_CCND3', 'ENSG00000112578_BYSL', 'ENSG00000112584_FAM120B', 'ENSG00000112592_TBP', 'ENSG00000112624_BICRAL', 'ENSG00000112640_PPP2R5D', 'ENSG00000112651_MRPL2', 'ENSG00000112655_PTK7', 'ENSG00000112658_SRF', 'ENSG00000112659_CUL9', 'ENSG00000112667_DNPH1', 'ENSG00000112679_DUSP22', 'ENSG00000112685_EXOC2', 'ENSG00000112695_COX7A2', 'ENSG00000112697_TMEM30A', 'ENSG00000112699_GMDS', 'ENSG00000112701_SENP6', 'ENSG00000112715_VEGFA', 'ENSG00000112739_PRPF4B', 'ENSG00000112742_TTK', 'ENSG00000112759_SLC29A1', 'ENSG00000112763_BTN2A1', 'ENSG00000112773_TENT5A', 'ENSG00000112787_FBRSL1', 'ENSG00000112796_ENPP5', 'ENSG00000112799_LY86', 'ENSG00000112851_ERBIN', 'ENSG00000112855_HARS2', 'ENSG00000112874_NUDT12', 'ENSG00000112877_CEP72', 'ENSG00000112893_MAN2A1', 'ENSG00000112941_TENT4A', 'ENSG00000112964_GHR', 'ENSG00000112972_HMGCS1', 'ENSG00000112977_DAP', 'ENSG00000112981_NME5', 'ENSG00000112983_BRD8', 'ENSG00000112984_KIF20A', 'ENSG00000112992_NNT', 'ENSG00000112996_MRPS30', 'ENSG00000113013_HSPA9', 'ENSG00000113048_MRPS27', 'ENSG00000113068_PFDN1', 'ENSG00000113070_HBEGF', 'ENSG00000113083_LOX', 'ENSG00000113100_CDH9', 'ENSG00000113108_APBB3', 'ENSG00000113119_TMCO6', 'ENSG00000113140_SPARC', 'ENSG00000113141_IK', 'ENSG00000113161_HMGCR', 'ENSG00000113163_COL4A3BP', 'ENSG00000113194_FAF2', 'ENSG00000113231_PDE8B', 'ENSG00000113240_CLK4', 'ENSG00000113263_ITK', 'ENSG00000113269_RNF130', 'ENSG00000113272_THG1L', 'ENSG00000113273_ARSB', 'ENSG00000113282_CLINT1', 'ENSG00000113296_THBS4', 'ENSG00000113300_CNOT6', 'ENSG00000113312_TTC1', 'ENSG00000113318_MSH3', 'ENSG00000113328_CCNG1', 'ENSG00000113356_POLR3G', 'ENSG00000113360_DROSHA', 'ENSG00000113368_LMNB1', 'ENSG00000113369_ARRDC3', 'ENSG00000113384_GOLPH3', 'ENSG00000113387_SUB1', 'ENSG00000113389_NPR3', 'ENSG00000113391_FAM172A', 'ENSG00000113396_SLC27A6', 'ENSG00000113407_TARS', 'ENSG00000113441_LNPEP', 'ENSG00000113448_PDE4D', 'ENSG00000113456_RAD1', 'ENSG00000113460_BRIX1', 'ENSG00000113504_SLC12A7', 'ENSG00000113520_IL4', 'ENSG00000113522_RAD50', 'ENSG00000113532_ST8SIA4', 'ENSG00000113552_GNPDA1', 'ENSG00000113555_PCDH12', 'ENSG00000113558_SKP1', 'ENSG00000113569_NUP155', 'ENSG00000113575_PPP2CA', 'ENSG00000113580_NR3C1', 'ENSG00000113583_C5orf15', 'ENSG00000113593_PPWD1', 'ENSG00000113595_TRIM23', 'ENSG00000113597_TRAPPC13', 'ENSG00000113615_SEC24A', 'ENSG00000113621_TXNDC15', 'ENSG00000113638_TTC33', 'ENSG00000113643_RARS', 'ENSG00000113645_WWC1', 'ENSG00000113648_H2AFY', 'ENSG00000113649_TCERG1', 'ENSG00000113657_DPYSL3', 'ENSG00000113658_SMAD5', 'ENSG00000113712_CSNK1A1', 'ENSG00000113716_HMGXB3', 'ENSG00000113719_ERGIC1', 'ENSG00000113721_PDGFRB', 'ENSG00000113732_ATP6V0E1', 'ENSG00000113734_BNIP1', 'ENSG00000113739_STC2', 'ENSG00000113742_CPEB4', 'ENSG00000113749_HRH2', 'ENSG00000113758_DBN1', 'ENSG00000113761_ZNF346', 'ENSG00000113790_EHHADH', 'ENSG00000113810_SMC4', 'ENSG00000113811_SELENOK', 'ENSG00000113812_ACTR8', 'ENSG00000113838_TBCCD1', 'ENSG00000113845_TIMMDC1', 'ENSG00000113851_CRBN', 'ENSG00000113916_BCL6', 'ENSG00000113924_HGD', 'ENSG00000113946_CLDN16', 'ENSG00000113966_ARL6', 'ENSG00000113971_NPHP3', 'ENSG00000114013_CD86', 'ENSG00000114019_AMOTL2', 'ENSG00000114021_NIT2', 'ENSG00000114023_FAM162A', 'ENSG00000114026_OGG1', 'ENSG00000114030_KPNA1', 'ENSG00000114054_PCCB', 'ENSG00000114062_UBE3A', 'ENSG00000114098_ARMC8', 'ENSG00000114107_CEP70', 'ENSG00000114120_SLC25A36', 'ENSG00000114125_RNF7', 'ENSG00000114126_TFDP2', 'ENSG00000114127_XRN1', 'ENSG00000114166_KAT2B', 'ENSG00000114209_PDCD10', 'ENSG00000114268_PFKFB4', 'ENSG00000114270_COL7A1', 'ENSG00000114302_PRKAR2A', 'ENSG00000114315_HES1', 'ENSG00000114316_USP4', 'ENSG00000114331_ACAP2', 'ENSG00000114346_ECT2', 'ENSG00000114353_GNAI2', 'ENSG00000114354_TFG', 'ENSG00000114374_USP9Y', 'ENSG00000114383_TUSC2', 'ENSG00000114388_NPRL2', 'ENSG00000114391_RPL24', 'ENSG00000114395_CYB561D2', 'ENSG00000114405_C3orf14', 'ENSG00000114416_FXR1', 'ENSG00000114423_CBLB', 'ENSG00000114439_BBX', 'ENSG00000114446_IFT57', 'ENSG00000114450_GNB4', 'ENSG00000114473_IQCG', 'ENSG00000114480_GBE1', 'ENSG00000114487_MORC1', 'ENSG00000114491_UMPS', 'ENSG00000114503_NCBP2', 'ENSG00000114520_SNX4', 'ENSG00000114529_C3orf52', 'ENSG00000114541_FRMD4B', 'ENSG00000114544_SLC41A3', 'ENSG00000114554_PLXNA1', 'ENSG00000114573_ATP6V1A', 'ENSG00000114626_ABTB1', 'ENSG00000114631_PODXL2', 'ENSG00000114646_CSPG5', 'ENSG00000114648_KLHL18', 'ENSG00000114650_SCAP', 'ENSG00000114654_EFCC1', 'ENSG00000114656_KIAA1257', 'ENSG00000114670_NEK11', 'ENSG00000114686_MRPL3', 'ENSG00000114698_PLSCR4', 'ENSG00000114735_HEMK1', 'ENSG00000114737_CISH', 'ENSG00000114738_MAPKAPK3', 'ENSG00000114739_ACVR2B', 'ENSG00000114742_WDR48', 'ENSG00000114744_COMMD2', 'ENSG00000114745_GORASP1', 'ENSG00000114767_RRP9', 'ENSG00000114770_ABCC5', 'ENSG00000114779_ABHD14B', 'ENSG00000114784_EIF1B', 'ENSG00000114786_ABHD14A-ACY1', 'ENSG00000114790_ARHGEF26', 'ENSG00000114796_KLHL24', 'ENSG00000114805_PLCH1', 'ENSG00000114812_VIPR1', 'ENSG00000114841_DNAH1', 'ENSG00000114850_SSR3', 'ENSG00000114853_ZBTB47', 'ENSG00000114854_TNNC1', 'ENSG00000114857_NKTR', 'ENSG00000114859_CLCN2', 'ENSG00000114861_FOXP1', 'ENSG00000114867_EIF4G1', 'ENSG00000114902_SPCS1', 'ENSG00000114904_NEK4', 'ENSG00000114933_INO80D', 'ENSG00000114942_EEF1B2', 'ENSG00000114948_ADAM23', 'ENSG00000114956_DGUOK', 'ENSG00000114978_MOB1A', 'ENSG00000114982_KANSL3', 'ENSG00000114988_LMAN2L', 'ENSG00000114993_RTKN', 'ENSG00000114999_TTL', 'ENSG00000115008_IL1A', 'ENSG00000115020_PIKFYVE', 'ENSG00000115041_KCNIP3', 'ENSG00000115042_FAHD2A', 'ENSG00000115053_NCL', 'ENSG00000115073_ACTR1B', 'ENSG00000115084_SLC35F5', 'ENSG00000115085_ZAP70', 'ENSG00000115091_ACTR3', 'ENSG00000115107_STEAP3', 'ENSG00000115109_EPB41L5', 'ENSG00000115112_TFCP2L1', 'ENSG00000115128_SF3B6', 'ENSG00000115129_TP53I3', 'ENSG00000115137_DNAJC27', 'ENSG00000115138_POMC', 'ENSG00000115145_STAM2', 'ENSG00000115159_GPD2', 'ENSG00000115163_CENPA', 'ENSG00000115165_CYTIP', 'ENSG00000115170_ACVR1', 'ENSG00000115183_TANC1', 'ENSG00000115194_SLC30A3', 'ENSG00000115204_MPV17', 'ENSG00000115207_GTF3C2', 'ENSG00000115211_EIF2B4', 'ENSG00000115216_NRBP1', 'ENSG00000115226_FNDC4', 'ENSG00000115232_ITGA4', 'ENSG00000115233_PSMD14', 'ENSG00000115234_SNX17', 'ENSG00000115239_ASB3', 'ENSG00000115241_PPM1G', 'ENSG00000115252_PDE1A', 'ENSG00000115255_REEP6', 'ENSG00000115257_PCSK4', 'ENSG00000115267_IFIH1', 'ENSG00000115268_RPS15', 'ENSG00000115271_GCA', 'ENSG00000115274_INO80B', 'ENSG00000115275_MOGS', 'ENSG00000115282_TTC31', 'ENSG00000115286_NDUFS7', 'ENSG00000115289_PCGF1', 'ENSG00000115290_GRB14', 'ENSG00000115295_CLIP4', 'ENSG00000115306_SPTBN1', 'ENSG00000115307_AUP1', 'ENSG00000115310_RTN4', 'ENSG00000115317_HTRA2', 'ENSG00000115318_LOXL3', 'ENSG00000115325_DOK1', 'ENSG00000115339_GALNT3', 'ENSG00000115350_POLE4', 'ENSG00000115355_CCDC88A', 'ENSG00000115364_MRPL19', 'ENSG00000115365_LANCL1', 'ENSG00000115368_WDR75', 'ENSG00000115392_FANCL', 'ENSG00000115414_FN1', 'ENSG00000115415_STAT1', 'ENSG00000115419_GLS', 'ENSG00000115421_PAPOLG', 'ENSG00000115423_DNAH6', 'ENSG00000115425_PECR', 'ENSG00000115446_UNC50', 'ENSG00000115457_IGFBP2', 'ENSG00000115459_ELMOD3', 'ENSG00000115461_IGFBP5', 'ENSG00000115464_USP34', 'ENSG00000115484_CCT4', 'ENSG00000115486_GGCX', 'ENSG00000115504_EHBP1', 'ENSG00000115514_TXNDC9', 'ENSG00000115520_COQ10B', 'ENSG00000115523_GNLY', 'ENSG00000115524_SF3B1', 'ENSG00000115525_ST3GAL5', 'ENSG00000115526_CHST10', 'ENSG00000115539_PDCL3', 'ENSG00000115540_MOB4', 'ENSG00000115541_HSPE1', 'ENSG00000115548_KDM3A', 'ENSG00000115556_PLCD4', 'ENSG00000115561_CHMP3', 'ENSG00000115568_ZNF142', 'ENSG00000115594_IL1R1', 'ENSG00000115596_WNT6', 'ENSG00000115602_IL1RL1', 'ENSG00000115604_IL18R1', 'ENSG00000115607_IL18RAP', 'ENSG00000115641_FHL2', 'ENSG00000115649_CNPPD1', 'ENSG00000115652_UXS1', 'ENSG00000115657_ABCB6', 'ENSG00000115661_STK16', 'ENSG00000115677_HDLBP', 'ENSG00000115685_PPP1R7', 'ENSG00000115687_PASK', 'ENSG00000115694_STK25', 'ENSG00000115718_PROC', 'ENSG00000115738_ID2', 'ENSG00000115750_TAF1B', 'ENSG00000115756_HPCAL1', 'ENSG00000115758_ODC1', 'ENSG00000115760_BIRC6', 'ENSG00000115761_NOL10', 'ENSG00000115762_PLEKHB2', 'ENSG00000115806_GORASP2', 'ENSG00000115808_STRN', 'ENSG00000115816_CEBPZ', 'ENSG00000115825_PRKD3', 'ENSG00000115827_DCAF17', 'ENSG00000115828_QPCT', 'ENSG00000115839_RAB3GAP1', 'ENSG00000115840_SLC25A12', 'ENSG00000115841_RMDN2', 'ENSG00000115850_LCT', 'ENSG00000115866_DARS', 'ENSG00000115875_SRSF7', 'ENSG00000115884_SDC1', 'ENSG00000115896_PLCL1', 'ENSG00000115902_SLC1A4', 'ENSG00000115904_SOS1', 'ENSG00000115919_KYNU', 'ENSG00000115935_WIPF1', 'ENSG00000115942_ORC2', 'ENSG00000115944_COX7A2L', 'ENSG00000115946_PNO1', 'ENSG00000115947_ORC4', 'ENSG00000115956_PLEK', 'ENSG00000115966_ATF2', 'ENSG00000115970_THADA', 'ENSG00000115977_AAK1', 'ENSG00000115993_TRAK2', 'ENSG00000115998_C2orf42', 'ENSG00000116001_TIA1', 'ENSG00000116005_PCYOX1', 'ENSG00000116014_KISS1R', 'ENSG00000116016_EPAS1', 'ENSG00000116017_ARID3A', 'ENSG00000116030_SUMO1', 'ENSG00000116044_NFE2L2', 'ENSG00000116062_MSH6', 'ENSG00000116095_PLEKHA3', 'ENSG00000116096_SPR', 'ENSG00000116106_EPHA4', 'ENSG00000116117_PARD3B', 'ENSG00000116120_FARSB', 'ENSG00000116127_ALMS1', 'ENSG00000116128_BCL9', 'ENSG00000116133_DHCR24', 'ENSG00000116138_DNAJC16', 'ENSG00000116151_MORN1', 'ENSG00000116157_GPX7', 'ENSG00000116161_CACYBP', 'ENSG00000116171_SCP2', 'ENSG00000116176_TPSG1', 'ENSG00000116191_RALGPS2', 'ENSG00000116198_CEP104', 'ENSG00000116199_FAM20B', 'ENSG00000116205_TCEANC2', 'ENSG00000116209_TMEM59', 'ENSG00000116212_LRRC42', 'ENSG00000116213_WRAP73', 'ENSG00000116221_MRPL37', 'ENSG00000116237_ICMT', 'ENSG00000116251_RPL22', 'ENSG00000116254_CHD5', 'ENSG00000116260_QSOX1', 'ENSG00000116266_STXBP3', 'ENSG00000116273_PHF13', 'ENSG00000116288_PARK7', 'ENSG00000116299_KIAA1324', 'ENSG00000116337_AMPD2', 'ENSG00000116350_SRSF4', 'ENSG00000116353_MECR', 'ENSG00000116396_KCNC4', 'ENSG00000116406_EDEM3', 'ENSG00000116455_WDR77', 'ENSG00000116459_ATP5PB', 'ENSG00000116473_RAP1A', 'ENSG00000116478_HDAC1', 'ENSG00000116489_CAPZA1', 'ENSG00000116497_S100PBP', 'ENSG00000116514_RNF19B', 'ENSG00000116521_SCAMP3', 'ENSG00000116525_TRIM62', 'ENSG00000116539_ASH1L', 'ENSG00000116544_DLGAP3', 'ENSG00000116560_SFPQ', 'ENSG00000116574_RHOU', 'ENSG00000116580_GON4L', 'ENSG00000116584_ARHGEF2', 'ENSG00000116586_LAMTOR2', 'ENSG00000116604_MEF2D', 'ENSG00000116641_DOCK7', 'ENSG00000116649_SRM', 'ENSG00000116652_DLEU2L', 'ENSG00000116663_FBXO6', 'ENSG00000116667_C1orf21', 'ENSG00000116668_SWT1', 'ENSG00000116670_MAD2L2', 'ENSG00000116675_DNAJC6', 'ENSG00000116678_LEPR', 'ENSG00000116679_IVNS1ABP', 'ENSG00000116685_KIAA2013', 'ENSG00000116688_MFN2', 'ENSG00000116690_PRG4', 'ENSG00000116691_MIIP', 'ENSG00000116698_SMG7', 'ENSG00000116701_NCF2', 'ENSG00000116704_SLC35D1', 'ENSG00000116711_PLA2G4A', 'ENSG00000116717_GADD45A', 'ENSG00000116731_PRDM2', 'ENSG00000116741_RGS2', 'ENSG00000116747_TROVE2', 'ENSG00000116750_UCHL5', 'ENSG00000116752_BCAS2', 'ENSG00000116754_SRSF11', 'ENSG00000116761_CTH', 'ENSG00000116771_AGMAT', 'ENSG00000116786_PLEKHM2', 'ENSG00000116791_CRYZ', 'ENSG00000116793_PHTF1', 'ENSG00000116809_ZBTB17', 'ENSG00000116815_CD58', 'ENSG00000116819_TFAP2E', 'ENSG00000116824_CD2', 'ENSG00000116830_TTF2', 'ENSG00000116833_NR5A2', 'ENSG00000116852_KIF21B', 'ENSG00000116857_TMEM9', 'ENSG00000116863_ADPRHL2', 'ENSG00000116871_MAP7D1', 'ENSG00000116874_WARS2', 'ENSG00000116883_AL591845.1', 'ENSG00000116885_OSCP1', 'ENSG00000116898_MRPS15', 'ENSG00000116903_EXOC8', 'ENSG00000116906_GNPAT', 'ENSG00000116918_TSNAX', 'ENSG00000116922_C1orf109', 'ENSG00000116954_RRAGC', 'ENSG00000116962_NID1', 'ENSG00000116977_LGALS8', 'ENSG00000116984_MTR', 'ENSG00000116985_BMP8B', 'ENSG00000116990_MYCL', 'ENSG00000116991_SIPA1L2', 'ENSG00000117000_RLF', 'ENSG00000117009_KMO', 'ENSG00000117010_ZNF684', 'ENSG00000117013_KCNQ4', 'ENSG00000117016_RIMS3', 'ENSG00000117020_AKT3', 'ENSG00000117036_ETV3', 'ENSG00000117054_ACADM', 'ENSG00000117090_SLAMF1', 'ENSG00000117091_CD48', 'ENSG00000117115_PADI2', 'ENSG00000117118_SDHB', 'ENSG00000117122_MFAP2', 'ENSG00000117133_RPF1', 'ENSG00000117139_KDM5B', 'ENSG00000117143_UAP1', 'ENSG00000117148_ACTL8', 'ENSG00000117151_CTBS', 'ENSG00000117153_KLHL12', 'ENSG00000117155_SSX2IP', 'ENSG00000117174_ZNHIT6', 'ENSG00000117222_RBBP5', 'ENSG00000117226_GBP3', 'ENSG00000117228_GBP1', 'ENSG00000117242_PINK1-AS', 'ENSG00000117245_KIF17', 'ENSG00000117262_GPR89A', 'ENSG00000117266_CDK18', 'ENSG00000117280_RAB29', 'ENSG00000117281_CD160', 'ENSG00000117298_ECE1', 'ENSG00000117305_HMGCL', 'ENSG00000117308_GALE', 'ENSG00000117318_ID3', 'ENSG00000117335_CD46', 'ENSG00000117360_PRPF3', 'ENSG00000117362_APH1A', 'ENSG00000117385_P3H1', 'ENSG00000117394_SLC2A1', 'ENSG00000117395_EBNA1BP2', 'ENSG00000117399_CDC20', 'ENSG00000117400_MPL', 'ENSG00000117407_ARTN', 'ENSG00000117408_IPO13', 'ENSG00000117410_ATP6V0B', 'ENSG00000117411_B4GALT2', 'ENSG00000117419_ERI3', 'ENSG00000117425_PTCH2', 'ENSG00000117448_AKR1A1', 'ENSG00000117450_PRDX1', 'ENSG00000117461_PIK3R3', 'ENSG00000117475_BLZF1', 'ENSG00000117477_CCDC181', 'ENSG00000117479_SLC19A2', 'ENSG00000117480_FAAH', 'ENSG00000117481_NSUN4', 'ENSG00000117500_TMED5', 'ENSG00000117505_DR1', 'ENSG00000117519_CNN3', 'ENSG00000117523_PRRC2C', 'ENSG00000117525_F3', 'ENSG00000117528_ABCD3', 'ENSG00000117533_VAMP4', 'ENSG00000117543_DPH5', 'ENSG00000117569_PTBP2', 'ENSG00000117586_TNFSF4', 'ENSG00000117592_PRDX6', 'ENSG00000117593_DARS2', 'ENSG00000117595_IRF6', 'ENSG00000117597_UTP25', 'ENSG00000117602_RCAN3', 'ENSG00000117614_SYF2', 'ENSG00000117616_RSRP1', 'ENSG00000117620_SLC35A3', 'ENSG00000117625_RCOR3', 'ENSG00000117632_STMN1', 'ENSG00000117640_MTFR1L', 'ENSG00000117643_MAN1C1', 'ENSG00000117650_NEK2', 'ENSG00000117676_RPS6KA1', 'ENSG00000117682_DHDDS', 'ENSG00000117691_NENF', 'ENSG00000117697_NSL1', 'ENSG00000117713_ARID1A', 'ENSG00000117724_CENPF', 'ENSG00000117748_RPA2', 'ENSG00000117751_PPP1R8', 'ENSG00000117758_STX12', 'ENSG00000117791_MARC2', 'ENSG00000117859_OSBPL9', 'ENSG00000117862_TXNDC12', 'ENSG00000117868_ESYT2', 'ENSG00000117877_CD3EAP', 'ENSG00000117899_MESD', 'ENSG00000117906_RCN2', 'ENSG00000117984_CTSD', 'ENSG00000118007_STAG1', 'ENSG00000118046_STK11', 'ENSG00000118058_KMT2A', 'ENSG00000118096_IFT46', 'ENSG00000118137_APOA1', 'ENSG00000118156_ZNF541', 'ENSG00000118162_KPTN', 'ENSG00000118181_RPS25', 'ENSG00000118193_KIF14', 'ENSG00000118197_DDX59', 'ENSG00000118200_CAMSAP2', 'ENSG00000118217_ATF6', 'ENSG00000118231_CRYGD', 'ENSG00000118242_MREG', 'ENSG00000118246_FASTKD2', 'ENSG00000118257_NRP2', 'ENSG00000118260_CREB1', 'ENSG00000118263_KLF7', 'ENSG00000118276_B4GALT6', 'ENSG00000118292_C1orf54', 'ENSG00000118307_CASC1', 'ENSG00000118308_LRMP', 'ENSG00000118363_SPCS2', 'ENSG00000118369_USP35', 'ENSG00000118407_FILIP1', 'ENSG00000118412_CASP8AP2', 'ENSG00000118418_HMGN3', 'ENSG00000118420_UBE3D', 'ENSG00000118454_ANKRD13C', 'ENSG00000118473_SGIP1', 'ENSG00000118482_PHF3', 'ENSG00000118495_PLAGL1', 'ENSG00000118496_FBXO30', 'ENSG00000118503_TNFAIP3', 'ENSG00000118507_AKAP7', 'ENSG00000118508_RAB32', 'ENSG00000118513_MYB', 'ENSG00000118514_ALDH8A1', 'ENSG00000118515_SGK1', 'ENSG00000118518_RNF146', 'ENSG00000118557_PMFBP1', 'ENSG00000118564_FBXL5', 'ENSG00000118579_MED28', 'ENSG00000118596_SLC16A7', 'ENSG00000118600_RXYLT1', 'ENSG00000118620_ZNF430', 'ENSG00000118640_VAMP8', 'ENSG00000118655_DCLRE1B', 'ENSG00000118680_MYL12B', 'ENSG00000118689_FOXO3', 'ENSG00000118690_ARMC2', 'ENSG00000118705_RPN2', 'ENSG00000118707_TGIF2', 'ENSG00000118762_PKD2', 'ENSG00000118777_ABCG2', 'ENSG00000118785_SPP1', 'ENSG00000118804_STBD1', 'ENSG00000118816_CCNI', 'ENSG00000118849_RARRES1', 'ENSG00000118855_MFSD1', 'ENSG00000118873_RAB3GAP2', 'ENSG00000118894_EEF2KMT', 'ENSG00000118898_PPL', 'ENSG00000118900_UBN1', 'ENSG00000118922_KLF12', 'ENSG00000118939_UCHL3', 'ENSG00000118960_HS1BP3', 'ENSG00000118961_LDAH', 'ENSG00000118965_WDR35', 'ENSG00000118971_CCND2', 'ENSG00000118976_OTUD4P1', 'ENSG00000118985_ELL2', 'ENSG00000118997_DNAH7', 'ENSG00000119004_CYP20A1', 'ENSG00000119013_NDUFB3', 'ENSG00000119041_GTF3C3', 'ENSG00000119042_SATB2', 'ENSG00000119048_UBE2B', 'ENSG00000119121_TRPM6', 'ENSG00000119138_KLF9', 'ENSG00000119139_TJP2', 'ENSG00000119147_C2orf40', 'ENSG00000119185_ITGB1BP1', 'ENSG00000119203_CPSF3', 'ENSG00000119227_PIGZ', 'ENSG00000119231_SENP5', 'ENSG00000119242_CCDC92', 'ENSG00000119280_C1orf198', 'ENSG00000119285_HEATR1', 'ENSG00000119314_PTBP3', 'ENSG00000119318_RAD23B', 'ENSG00000119321_FKBP15', 'ENSG00000119326_CTNNAL1', 'ENSG00000119328_FAM206A', 'ENSG00000119333_WDR34', 'ENSG00000119335_SET', 'ENSG00000119383_PTPA', 'ENSG00000119392_GLE1', 'ENSG00000119396_RAB14', 'ENSG00000119397_CNTRL', 'ENSG00000119401_TRIM32', 'ENSG00000119402_FBXW2', 'ENSG00000119403_PHF19', 'ENSG00000119408_NEK6', 'ENSG00000119411_BSPRY', 'ENSG00000119414_PPP6C', 'ENSG00000119421_NDUFA8', 'ENSG00000119431_HDHD3', 'ENSG00000119446_RBM18', 'ENSG00000119471_HSDL2', 'ENSG00000119487_MAPKAP1', 'ENSG00000119508_NR4A3', 'ENSG00000119509_INVS', 'ENSG00000119514_GALNT12', 'ENSG00000119522_DENND1A', 'ENSG00000119523_ALG2', 'ENSG00000119535_CSF3R', 'ENSG00000119537_KDSR', 'ENSG00000119541_VPS4B', 'ENSG00000119559_C19orf25', 'ENSG00000119574_ZBTB45', 'ENSG00000119596_YLPM1', 'ENSG00000119599_DCAF4', 'ENSG00000119608_PROX2', 'ENSG00000119616_FCF1', 'ENSG00000119630_PGF', 'ENSG00000119632_IFI27L2', 'ENSG00000119636_BBOF1', 'ENSG00000119638_NEK9', 'ENSG00000119640_ACYP1', 'ENSG00000119650_IFT43', 'ENSG00000119655_NPC2', 'ENSG00000119661_DNAL1', 'ENSG00000119669_IRF2BPL', 'ENSG00000119673_ACOT2', 'ENSG00000119681_LTBP2', 'ENSG00000119682_AREL1', 'ENSG00000119684_MLH3', 'ENSG00000119685_TTLL5', 'ENSG00000119686_FLVCR2', 'ENSG00000119688_ABCD4', 'ENSG00000119689_DLST', 'ENSG00000119698_PPP4R4', 'ENSG00000119699_TGFB3', 'ENSG00000119703_ZC2HC1C', 'ENSG00000119705_SLIRP', 'ENSG00000119707_RBM25', 'ENSG00000119711_ALDH6A1', 'ENSG00000119714_GPR68', 'ENSG00000119718_EIF2B2', 'ENSG00000119720_NRDE2', 'ENSG00000119723_COQ6', 'ENSG00000119725_ZNF410', 'ENSG00000119729_RHOQ', 'ENSG00000119737_GPR75', 'ENSG00000119760_SUPT7L', 'ENSG00000119772_DNMT3A', 'ENSG00000119777_TMEM214', 'ENSG00000119778_ATAD2B', 'ENSG00000119782_FKBP1B', 'ENSG00000119787_ATL2', 'ENSG00000119801_YPEL5', 'ENSG00000119812_FAM98A', 'ENSG00000119820_YIPF4', 'ENSG00000119844_AFTPH', 'ENSG00000119862_LGALSL', 'ENSG00000119865_CNRIP1', 'ENSG00000119866_BCL11A', 'ENSG00000119878_CRIPT', 'ENSG00000119888_EPCAM', 'ENSG00000119899_SLC17A5', 'ENSG00000119900_OGFRL1', 'ENSG00000119906_SLF2', 'ENSG00000119912_IDE', 'ENSG00000119915_ELOVL3', 'ENSG00000119917_IFIT3', 'ENSG00000119919_NKX2-3', 'ENSG00000119922_IFIT2', 'ENSG00000119927_GPAM', 'ENSG00000119929_CUTC', 'ENSG00000119943_PYROXD2', 'ENSG00000119946_CNNM1', 'ENSG00000119950_MXI1', 'ENSG00000119953_SMNDC1', 'ENSG00000119965_C10orf88', 'ENSG00000119969_HELLS', 'ENSG00000119977_TCTN3', 'ENSG00000119979_FAM45A', 'ENSG00000119986_AVPI1', 'ENSG00000120008_WDR11', 'ENSG00000120029_ARMH3', 'ENSG00000120049_KCNIP2', 'ENSG00000120051_CFAP58', 'ENSG00000120053_GOT1', 'ENSG00000120055_C10orf95', 'ENSG00000120057_SFRP5', 'ENSG00000120063_GNA13', 'ENSG00000120071_KANSL1', 'ENSG00000120075_HOXB5', 'ENSG00000120093_HOXB3', 'ENSG00000120129_DUSP1', 'ENSG00000120137_PANK3', 'ENSG00000120156_TEK', 'ENSG00000120158_RCL1', 'ENSG00000120159_CAAP1', 'ENSG00000120162_MOB3B', 'ENSG00000120210_INSL6', 'ENSG00000120217_CD274', 'ENSG00000120253_NUP43', 'ENSG00000120254_MTHFD1L', 'ENSG00000120256_LRP11', 'ENSG00000120262_CCDC170', 'ENSG00000120265_PCMT1', 'ENSG00000120278_PLEKHG1', 'ENSG00000120279_MYCT1', 'ENSG00000120280_CXorf21', 'ENSG00000120306_CYSTM1', 'ENSG00000120314_WDR55', 'ENSG00000120318_ARAP3', 'ENSG00000120333_MRPS14', 'ENSG00000120334_CENPL', 'ENSG00000120370_GORAB', 'ENSG00000120437_ACAT2', 'ENSG00000120438_TCP1', 'ENSG00000120451_SNX19', 'ENSG00000120457_KCNJ5', 'ENSG00000120458_MSANTD2', 'ENSG00000120509_PDZD11', 'ENSG00000120519_SLC10A7', 'ENSG00000120526_NUDCD1', 'ENSG00000120533_ENY2', 'ENSG00000120539_MASTL', 'ENSG00000120555_SEPT7P9', 'ENSG00000120594_PLXDC2', 'ENSG00000120616_EPC1', 'ENSG00000120645_IQSEC3', 'ENSG00000120647_CCDC77', 'ENSG00000120656_TAF12', 'ENSG00000120658_ENOX1', 'ENSG00000120662_MTRF1', 'ENSG00000120664_SPART-AS1', 'ENSG00000120669_SOHLH2', 'ENSG00000120675_DNAJC15', 'ENSG00000120685_PROSER1', 'ENSG00000120686_UFM1', 'ENSG00000120688_WBP4', 'ENSG00000120690_ELF1', 'ENSG00000120693_SMAD9', 'ENSG00000120694_HSPH1', 'ENSG00000120696_KBTBD7', 'ENSG00000120697_ALG5', 'ENSG00000120699_EXOSC8', 'ENSG00000120705_ETF1', 'ENSG00000120708_TGFBI', 'ENSG00000120709_FAM53C', 'ENSG00000120725_SIL1', 'ENSG00000120727_PAIP2', 'ENSG00000120733_KDM3B', 'ENSG00000120738_EGR1', 'ENSG00000120742_SERP1', 'ENSG00000120756_PLS1', 'ENSG00000120784_ZFP30', 'ENSG00000120798_NR2C1', 'ENSG00000120800_UTP20', 'ENSG00000120802_TMPO', 'ENSG00000120805_ARL1', 'ENSG00000120820_GLT8D2', 'ENSG00000120832_MTERF2', 'ENSG00000120833_SOCS2', 'ENSG00000120837_NFYB', 'ENSG00000120860_WASHC3', 'ENSG00000120868_APAF1', 'ENSG00000120875_DUSP4', 'ENSG00000120885_CLU', 'ENSG00000120889_TNFRSF10B', 'ENSG00000120896_SORBS3', 'ENSG00000120899_PTK2B', 'ENSG00000120910_PPP3CC', 'ENSG00000120913_PDLIM2', 'ENSG00000120915_EPHX2', 'ENSG00000120925_RNF170', 'ENSG00000120942_UBIAD1', 'ENSG00000120948_TARDBP', 'ENSG00000120949_TNFRSF8', 'ENSG00000120963_ZNF706', 'ENSG00000120992_LYPLA1', 'ENSG00000121005_CRISPLD1', 'ENSG00000121022_COPS5', 'ENSG00000121039_RDH10', 'ENSG00000121053_EPX', 'ENSG00000121057_AKAP1', 'ENSG00000121058_COIL', 'ENSG00000121060_TRIM25', 'ENSG00000121064_SCPEP1', 'ENSG00000121067_SPOP', 'ENSG00000121073_SLC35B1', 'ENSG00000121089_NACA3P', 'ENSG00000121101_TEX14', 'ENSG00000121104_FAM117A', 'ENSG00000121152_NCAPH', 'ENSG00000121207_LRAT', 'ENSG00000121210_TMEM131L', 'ENSG00000121211_MND1', 'ENSG00000121236_TRIM6', 'ENSG00000121274_TENT4B', 'ENSG00000121281_ADCY7', 'ENSG00000121289_CEP89', 'ENSG00000121297_TSHZ3', 'ENSG00000121310_ECHDC2', 'ENSG00000121350_PYROXD1', 'ENSG00000121351_IAPP', 'ENSG00000121361_KCNJ8', 'ENSG00000121390_PSPC1', 'ENSG00000121406_ZNF549', 'ENSG00000121410_A1BG', 'ENSG00000121413_ZSCAN18', 'ENSG00000121417_ZNF211', 'ENSG00000121454_LHX4', 'ENSG00000121481_RNF2', 'ENSG00000121486_TRMT1L', 'ENSG00000121542_SEC22A', 'ENSG00000121552_CSTA', 'ENSG00000121570_DPPA4', 'ENSG00000121577_POPDC2', 'ENSG00000121578_B4GALT4', 'ENSG00000121579_NAA50', 'ENSG00000121594_CD80', 'ENSG00000121621_KIF18A', 'ENSG00000121644_DESI2', 'ENSG00000121653_MAPK8IP1', 'ENSG00000121671_CRY2', 'ENSG00000121680_PEX16', 'ENSG00000121690_DEPDC7', 'ENSG00000121691_CAT', 'ENSG00000121716_PILRB', 'ENSG00000121741_ZMYM2', 'ENSG00000121749_TBC1D15', 'ENSG00000121753_ADGRB2', 'ENSG00000121766_ZCCHC17', 'ENSG00000121769_FABP3', 'ENSG00000121774_KHDRBS1', 'ENSG00000121775_TMEM39B', 'ENSG00000121797_CCRL2', 'ENSG00000121807_CCR2', 'ENSG00000121851_POLR3GL', 'ENSG00000121858_TNFSF10', 'ENSG00000121864_ZNF639', 'ENSG00000121879_PIK3CA', 'ENSG00000121892_PDS5A', 'ENSG00000121895_TMEM156', 'ENSG00000121897_LIAS', 'ENSG00000121900_TMEM54', 'ENSG00000121903_ZSCAN20', 'ENSG00000121931_LRIF1', 'ENSG00000121933_TMIGD3', 'ENSG00000121940_CLCC1', 'ENSG00000121957_GPSM2', 'ENSG00000121964_GTDC1', 'ENSG00000121966_CXCR4', 'ENSG00000121988_ZRANB3', 'ENSG00000121989_ACVR2A', 'ENSG00000122008_POLK', 'ENSG00000122012_SV2C', 'ENSG00000122025_FLT3', 'ENSG00000122026_RPL21', 'ENSG00000122033_MTIF3', 'ENSG00000122034_GTF3A', 'ENSG00000122035_RASL11A', 'ENSG00000122042_UBL3', 'ENSG00000122068_FYTTD1', 'ENSG00000122085_MTERF4', 'ENSG00000122121_XPNPEP2', 'ENSG00000122122_SASH3', 'ENSG00000122126_OCRL', 'ENSG00000122140_MRPS2', 'ENSG00000122188_LAX1', 'ENSG00000122203_KIAA1191', 'ENSG00000122218_COPA', 'ENSG00000122223_CD244', 'ENSG00000122257_RBBP6', 'ENSG00000122299_ZC3H7A', 'ENSG00000122335_SERAC1', 'ENSG00000122359_ANXA11', 'ENSG00000122376_SHLD2', 'ENSG00000122378_PRXL2A', 'ENSG00000122386_ZNF205', 'ENSG00000122390_NAA60', 'ENSG00000122406_RPL5', 'ENSG00000122417_ODF2L', 'ENSG00000122432_SPATA1', 'ENSG00000122435_TRMT13', 'ENSG00000122477_LRRC39', 'ENSG00000122481_RWDD3', 'ENSG00000122482_ZNF644', 'ENSG00000122483_CCDC18', 'ENSG00000122484_RPAP2', 'ENSG00000122490_PQLC1', 'ENSG00000122507_BBS9', 'ENSG00000122512_PMS2', 'ENSG00000122515_ZMIZ2', 'ENSG00000122545_SEPT7', 'ENSG00000122547_EEPD1', 'ENSG00000122548_KIAA0087', 'ENSG00000122550_KLHL7', 'ENSG00000122557_HERPUD2', 'ENSG00000122565_CBX3', 'ENSG00000122566_HNRNPA2B1', 'ENSG00000122574_WIPF3', 'ENSG00000122591_FAM126A', 'ENSG00000122592_HOXA7', 'ENSG00000122641_INHBA', 'ENSG00000122642_FKBP9', 'ENSG00000122643_NT5C3A', 'ENSG00000122644_ARL4A', 'ENSG00000122674_CCZ1', 'ENSG00000122678_POLM', 'ENSG00000122687_MRM2', 'ENSG00000122691_TWIST1', 'ENSG00000122692_SMU1', 'ENSG00000122694_GLIPR2', 'ENSG00000122696_SLC25A51', 'ENSG00000122705_CLTA', 'ENSG00000122707_RECK', 'ENSG00000122711_SPINK4', 'ENSG00000122729_ACO1', 'ENSG00000122741_DCAF10', 'ENSG00000122778_KIAA1549', 'ENSG00000122779_TRIM24', 'ENSG00000122783_CYREN', 'ENSG00000122786_CALD1', 'ENSG00000122824_NUDT10', 'ENSG00000122861_PLAU', 'ENSG00000122862_SRGN', 'ENSG00000122870_BICC1', 'ENSG00000122873_CISD1', 'ENSG00000122877_EGR2', 'ENSG00000122882_ECD', 'ENSG00000122884_P4HA1', 'ENSG00000122912_SLC25A16', 'ENSG00000122952_ZWINT', 'ENSG00000122958_VPS26A', 'ENSG00000122965_RBM19', 'ENSG00000122966_CIT', 'ENSG00000122970_IFT81', 'ENSG00000122971_ACADS', 'ENSG00000122986_HVCN1', 'ENSG00000123009_NME2P1', 'ENSG00000123064_DDX54', 'ENSG00000123066_MED13L', 'ENSG00000123080_CDKN2C', 'ENSG00000123091_RNF11', 'ENSG00000123094_RASSF8', 'ENSG00000123095_BHLHE41', 'ENSG00000123096_SSPN', 'ENSG00000123104_ITPR2', 'ENSG00000123106_CCDC91', 'ENSG00000123119_NECAB1', 'ENSG00000123124_WWP1', 'ENSG00000123130_ACOT9', 'ENSG00000123131_PRDX4', 'ENSG00000123136_DDX39A', 'ENSG00000123143_PKN1', 'ENSG00000123144_TRIR', 'ENSG00000123146_ADGRE5', 'ENSG00000123154_WDR83', 'ENSG00000123159_GIPC1', 'ENSG00000123178_SPRYD7', 'ENSG00000123179_EBPL', 'ENSG00000123191_ATP7B', 'ENSG00000123200_ZC3H13', 'ENSG00000123213_NLN', 'ENSG00000123219_CENPK', 'ENSG00000123240_OPTN', 'ENSG00000123268_ATF1', 'ENSG00000123297_TSFM', 'ENSG00000123329_ARHGAP9', 'ENSG00000123338_NCKAP1L', 'ENSG00000123342_MMP19', 'ENSG00000123349_PFDN5', 'ENSG00000123352_SPATS2', 'ENSG00000123353_ORMDL2', 'ENSG00000123358_NR4A1', 'ENSG00000123360_PDE1B', 'ENSG00000123374_CDK2', 'ENSG00000123384_LRP1', 'ENSG00000123395_ATG101', 'ENSG00000123405_NFE2', 'ENSG00000123411_IKZF4', 'ENSG00000123415_SMUG1', 'ENSG00000123416_TUBA1B', 'ENSG00000123427_EEF1AKMT3', 'ENSG00000123444_KBTBD4', 'ENSG00000123453_SARDH', 'ENSG00000123472_ATPAF1', 'ENSG00000123473_STIL', 'ENSG00000123485_HJURP', 'ENSG00000123500_COL10A1', 'ENSG00000123505_AMD1', 'ENSG00000123545_NDUFAF4', 'ENSG00000123552_USP45', 'ENSG00000123562_MORF4L2', 'ENSG00000123570_RAB9B', 'ENSG00000123575_FAM199X', 'ENSG00000123595_RAB9A', 'ENSG00000123600_METTL8', 'ENSG00000123607_TTC21B', 'ENSG00000123609_NMI', 'ENSG00000123610_TNFAIP6', 'ENSG00000123636_BAZ2B', 'ENSG00000123643_SLC36A1', 'ENSG00000123684_LPGAT1', 'ENSG00000123685_BATF3', 'ENSG00000123689_G0S2', 'ENSG00000123700_KCNJ2', 'ENSG00000123728_RAP2C', 'ENSG00000123737_EXOSC9', 'ENSG00000123739_PLA2G12A', 'ENSG00000123810_B9D2', 'ENSG00000123815_COQ8B', 'ENSG00000123836_PFKFB2', 'ENSG00000123870_ZNF137P', 'ENSG00000123892_RAB38', 'ENSG00000123908_AGO2', 'ENSG00000123933_MXD4', 'ENSG00000123965_PMS2P5', 'ENSG00000123975_CKS2', 'ENSG00000123983_ACSL3', 'ENSG00000123989_CHPF', 'ENSG00000123992_DNPEP', 'ENSG00000123999_INHA', 'ENSG00000124006_OBSL1', 'ENSG00000124019_FAM124B', 'ENSG00000124067_SLC12A4', 'ENSG00000124074_ENKD1', 'ENSG00000124097_HMGB1P1', 'ENSG00000124098_FAM210B', 'ENSG00000124103_FAM209A', 'ENSG00000124104_SNX21', 'ENSG00000124107_SLPI', 'ENSG00000124116_WFDC3', 'ENSG00000124120_TTPAL', 'ENSG00000124126_PREX1', 'ENSG00000124134_KCNS1', 'ENSG00000124140_SLC12A5', 'ENSG00000124145_SDC4', 'ENSG00000124151_NCOA3', 'ENSG00000124155_PIGT', 'ENSG00000124159_MATN4', 'ENSG00000124160_NCOA5', 'ENSG00000124164_VAPB', 'ENSG00000124171_PARD6B', 'ENSG00000124172_ATP5F1E', 'ENSG00000124177_CHD6', 'ENSG00000124181_PLCG1', 'ENSG00000124193_SRSF6', 'ENSG00000124198_ARFGEF2', 'ENSG00000124201_ZNFX1', 'ENSG00000124207_CSE1L', 'ENSG00000124209_RAB22A', 'ENSG00000124214_STAU1', 'ENSG00000124215_CDH26', 'ENSG00000124216_SNAI1', 'ENSG00000124217_MOCS3', 'ENSG00000124222_STX16', 'ENSG00000124224_PPP4R1L', 'ENSG00000124225_PMEPA1', 'ENSG00000124226_RNF114', 'ENSG00000124228_DDX27', 'ENSG00000124243_BCAS4', 'ENSG00000124249_KCNK15', 'ENSG00000124257_NEURL2', 'ENSG00000124275_MTRR', 'ENSG00000124279_FASTKD3', 'ENSG00000124299_PEPD', 'ENSG00000124313_IQSEC2', 'ENSG00000124333_VAMP7', 'ENSG00000124334_IL9R', 'ENSG00000124356_STAMBP', 'ENSG00000124357_NAGK', 'ENSG00000124370_MCEE', 'ENSG00000124374_PAIP2B', 'ENSG00000124380_SNRNP27', 'ENSG00000124383_MPHOSPH10', 'ENSG00000124399_NDUFB4P12', 'ENSG00000124406_ATP8A1', 'ENSG00000124422_USP22', 'ENSG00000124444_ZNF576', 'ENSG00000124459_ZNF45', 'ENSG00000124466_LYPD3', 'ENSG00000124486_USP9X', 'ENSG00000124490_CRISP2', 'ENSG00000124491_F13A1', 'ENSG00000124496_TRERF1', 'ENSG00000124507_PACSIN1', 'ENSG00000124508_BTN2A2', 'ENSG00000124523_SIRT5', 'ENSG00000124532_MRS2', 'ENSG00000124535_WRNIP1', 'ENSG00000124541_RRP36', 'ENSG00000124549_BTN2A3P', 'ENSG00000124562_SNRPC', 'ENSG00000124570_SERPINB6', 'ENSG00000124571_XPO5', 'ENSG00000124574_ABCC10', 'ENSG00000124575_HIST1H1D', 'ENSG00000124587_PEX6', 'ENSG00000124588_NQO2', 'ENSG00000124593_AL365205.1', 'ENSG00000124596_OARD1', 'ENSG00000124602_UNC5CL', 'ENSG00000124608_AARS2', 'ENSG00000124610_HIST1H1A', 'ENSG00000124613_ZNF391', 'ENSG00000124614_RPS10', 'ENSG00000124615_MOCS1', 'ENSG00000124635_HIST1H2BJ', 'ENSG00000124641_MED20', 'ENSG00000124659_TBCC', 'ENSG00000124678_TCP11', 'ENSG00000124688_MAD2L1BP', 'ENSG00000124702_KLHDC3', 'ENSG00000124713_GNMT', 'ENSG00000124731_TREM1', 'ENSG00000124733_MEA1', 'ENSG00000124743_KLHL31', 'ENSG00000124762_CDKN1A', 'ENSG00000124766_SOX4', 'ENSG00000124767_GLO1', 'ENSG00000124772_CPNE5', 'ENSG00000124780_KCNK17', 'ENSG00000124782_RREB1', 'ENSG00000124783_SSR1', 'ENSG00000124784_RIOK1', 'ENSG00000124786_SLC35B3', 'ENSG00000124787_RPP40', 'ENSG00000124788_ATXN1', 'ENSG00000124789_NUP153', 'ENSG00000124795_DEK', 'ENSG00000124802_EEF1E1', 'ENSG00000124813_RUNX2', 'ENSG00000124831_LRRFIP1', 'ENSG00000124835_AC105760.1', 'ENSG00000124875_CXCL6', 'ENSG00000124882_EREG', 'ENSG00000124920_MYRF', 'ENSG00000124942_AHNAK', 'ENSG00000125037_EMC3', 'ENSG00000125089_SH3TC1', 'ENSG00000125107_CNOT1', 'ENSG00000125122_LRRC29', 'ENSG00000125124_BBS2', 'ENSG00000125144_MT1G', 'ENSG00000125148_MT2A', 'ENSG00000125149_C16orf70', 'ENSG00000125166_GOT2', 'ENSG00000125170_DOK4', 'ENSG00000125207_PIWIL1', 'ENSG00000125245_GPR18', 'ENSG00000125246_CLYBL', 'ENSG00000125247_TMTC4', 'ENSG00000125249_RAP2A', 'ENSG00000125257_ABCC4', 'ENSG00000125266_EFNB2', 'ENSG00000125304_TM9SF2', 'ENSG00000125319_C17orf53', 'ENSG00000125347_IRF1', 'ENSG00000125351_UPF3B', 'ENSG00000125352_RNF113A', 'ENSG00000125354_SEPT6', 'ENSG00000125356_NDUFA1', 'ENSG00000125375_ATP5S', 'ENSG00000125378_BMP4', 'ENSG00000125384_PTGER2', 'ENSG00000125386_FAM193A', 'ENSG00000125388_GRK4', 'ENSG00000125430_HS3ST3B1', 'ENSG00000125434_SLC25A35', 'ENSG00000125445_MRPS7', 'ENSG00000125447_GGA3', 'ENSG00000125449_ARMC7', 'ENSG00000125450_NUP85', 'ENSG00000125454_SLC25A19', 'ENSG00000125457_MIF4GD', 'ENSG00000125458_NT5C', 'ENSG00000125459_MSTO1', 'ENSG00000125482_TTF1', 'ENSG00000125484_GTF3C4', 'ENSG00000125485_DDX31', 'ENSG00000125503_PPP1R12C', 'ENSG00000125505_MBOAT7', 'ENSG00000125510_OPRL1', 'ENSG00000125520_SLC2A4RG', 'ENSG00000125531_FNDC11', 'ENSG00000125534_PPDPF', 'ENSG00000125538_IL1B', 'ENSG00000125611_CHCHD5', 'ENSG00000125618_PAX8', 'ENSG00000125629_INSIG2', 'ENSG00000125630_POLR1B', 'ENSG00000125633_CCDC93', 'ENSG00000125637_PSD4', 'ENSG00000125648_SLC25A23', 'ENSG00000125650_PSPN', 'ENSG00000125651_GTF2F1', 'ENSG00000125652_ALKBH7', 'ENSG00000125656_CLPP', 'ENSG00000125657_TNFSF9', 'ENSG00000125676_THOC2', 'ENSG00000125686_MED1', 'ENSG00000125691_RPL23', 'ENSG00000125703_ATG4C', 'ENSG00000125726_CD70', 'ENSG00000125730_C3', 'ENSG00000125731_SH2D3A', 'ENSG00000125733_TRIP10', 'ENSG00000125734_GPR108', 'ENSG00000125735_TNFSF14', 'ENSG00000125740_FOSB', 'ENSG00000125741_OPA3', 'ENSG00000125743_SNRPD2', 'ENSG00000125744_RTN2', 'ENSG00000125746_EML2', 'ENSG00000125753_VASP', 'ENSG00000125755_SYMPK', 'ENSG00000125772_GPCPD1', 'ENSG00000125775_SDCBP2', 'ENSG00000125779_PANK2', 'ENSG00000125804_FAM182A', 'ENSG00000125810_CD93', 'ENSG00000125812_GZF1', 'ENSG00000125814_NAPB', 'ENSG00000125817_CENPB', 'ENSG00000125818_PSMF1', 'ENSG00000125821_DTD1', 'ENSG00000125826_RBCK1', 'ENSG00000125827_TMX4', 'ENSG00000125834_STK35', 'ENSG00000125835_SNRPB', 'ENSG00000125841_NRSN2', 'ENSG00000125843_AP5S1', 'ENSG00000125844_RRBP1', 'ENSG00000125846_ZNF133', 'ENSG00000125863_MKKS', 'ENSG00000125864_BFSP1', 'ENSG00000125868_DSTN', 'ENSG00000125869_LAMP5', 'ENSG00000125870_SNRPB2', 'ENSG00000125871_MGME1', 'ENSG00000125875_TBC1D20', 'ENSG00000125877_ITPA', 'ENSG00000125878_TCF15', 'ENSG00000125885_MCM8', 'ENSG00000125898_FAM110A', 'ENSG00000125901_MRPS26', 'ENSG00000125910_S1PR4', 'ENSG00000125912_NCLN', 'ENSG00000125944_HNRNPR', 'ENSG00000125945_ZNF436', 'ENSG00000125952_MAX', 'ENSG00000125962_ARMCX5', 'ENSG00000125967_NECAB3', 'ENSG00000125968_ID1', 'ENSG00000125970_RALY', 'ENSG00000125971_DYNLRB1', 'ENSG00000125977_EIF2S2', 'ENSG00000125991_ERGIC3', 'ENSG00000125995_ROMO1', 'ENSG00000126001_CEP250', 'ENSG00000126003_PLAGL2', 'ENSG00000126005_MMP24OS', 'ENSG00000126012_KDM5C', 'ENSG00000126016_AMOT', 'ENSG00000126062_TMEM115', 'ENSG00000126067_PSMB2', 'ENSG00000126070_AGO3', 'ENSG00000126088_UROD', 'ENSG00000126091_ST3GAL3', 'ENSG00000126106_TMEM53', 'ENSG00000126107_HECTD3', 'ENSG00000126214_KLC1', 'ENSG00000126215_XRCC3', 'ENSG00000126216_TUBGCP3', 'ENSG00000126217_MCF2L', 'ENSG00000126226_PCID2', 'ENSG00000126231_PROZ', 'ENSG00000126243_LRFN3', 'ENSG00000126246_IGFLR1', 'ENSG00000126247_CAPNS1', 'ENSG00000126249_PDCD2L', 'ENSG00000126254_RBM42', 'ENSG00000126261_UBA2', 'ENSG00000126264_HCST', 'ENSG00000126267_COX6B1', 'ENSG00000126351_THRA', 'ENSG00000126353_CCR7', 'ENSG00000126368_NR1D1', 'ENSG00000126391_FRMD8', 'ENSG00000126432_PRDX5', 'ENSG00000126453_BCL2L12', 'ENSG00000126456_IRF3', 'ENSG00000126457_PRMT1', 'ENSG00000126458_RRAS', 'ENSG00000126460_PRRG2', 'ENSG00000126461_SCAF1', 'ENSG00000126464_PRR12', 'ENSG00000126467_TSKS', 'ENSG00000126500_FLRT1', 'ENSG00000126522_ASL', 'ENSG00000126524_SBDS', 'ENSG00000126561_STAT5A', 'ENSG00000126581_BECN1', 'ENSG00000126602_TRAP1', 'ENSG00000126603_GLIS2', 'ENSG00000126653_NSRP1', 'ENSG00000126698_DNAJC8', 'ENSG00000126705_AHDC1', 'ENSG00000126709_IFI6', 'ENSG00000126746_ZNF384', 'ENSG00000126749_EMG1', 'ENSG00000126756_UXT', 'ENSG00000126759_CFP', 'ENSG00000126767_ELK1', 'ENSG00000126768_TIMM17B', 'ENSG00000126773_PCNX4', 'ENSG00000126775_ATG14', 'ENSG00000126777_KTN1', 'ENSG00000126787_DLGAP5', 'ENSG00000126790_L3HYPDH', 'ENSG00000126803_HSPA2', 'ENSG00000126804_ZBTB1', 'ENSG00000126814_TRMT5', 'ENSG00000126821_SGPP1', 'ENSG00000126822_PLEKHG3', 'ENSG00000126858_RHOT1', 'ENSG00000126860_EVI2A', 'ENSG00000126861_OMG', 'ENSG00000126870_WDR60', 'ENSG00000126878_AIF1L', 'ENSG00000126882_FAM78A', 'ENSG00000126883_NUP214', 'ENSG00000126903_SLC10A3', 'ENSG00000126934_MAP2K2', 'ENSG00000126945_HNRNPH2', 'ENSG00000126947_ARMCX1', 'ENSG00000126953_TIMM8A', 'ENSG00000126970_ZC4H2', 'ENSG00000127022_CANX', 'ENSG00000127054_INTS11', 'ENSG00000127074_RGS13', 'ENSG00000127080_IPPK', 'ENSG00000127081_ZNF484', 'ENSG00000127083_OMD', 'ENSG00000127084_FGD3', 'ENSG00000127124_HIVEP3', 'ENSG00000127125_PPCS', 'ENSG00000127184_COX7C', 'ENSG00000127191_TRAF2', 'ENSG00000127220_ABHD8', 'ENSG00000127249_ATP13A4', 'ENSG00000127252_HRASLS', 'ENSG00000127311_HELB', 'ENSG00000127314_RAP1B', 'ENSG00000127325_BEST3', 'ENSG00000127328_RAB3IP', 'ENSG00000127329_PTPRB', 'ENSG00000127334_DYRK2', 'ENSG00000127337_YEATS4', 'ENSG00000127364_TAS2R4', 'ENSG00000127399_LRRC61', 'ENSG00000127415_IDUA', 'ENSG00000127418_FGFRL1', 'ENSG00000127419_TMEM175', 'ENSG00000127423_AUNIP', 'ENSG00000127445_PIN1', 'ENSG00000127452_FBXL12', 'ENSG00000127463_EMC1', 'ENSG00000127481_UBR4', 'ENSG00000127483_HP1BP3', 'ENSG00000127507_ADGRE2', 'ENSG00000127511_SIN3B', 'ENSG00000127526_SLC35E1', 'ENSG00000127527_EPS15L1', 'ENSG00000127528_KLF2', 'ENSG00000127533_F2RL3', 'ENSG00000127540_UQCR11', 'ENSG00000127554_GFER', 'ENSG00000127561_SYNGR3', 'ENSG00000127564_PKMYT1', 'ENSG00000127578_WFIKKN1', 'ENSG00000127580_WDR24', 'ENSG00000127585_FBXL16', 'ENSG00000127586_CHTF18', 'ENSG00000127589_TUBBP1', 'ENSG00000127603_MACF1', 'ENSG00000127616_SMARCA4', 'ENSG00000127663_KDM4B', 'ENSG00000127666_TICAM1', 'ENSG00000127720_METTL25', 'ENSG00000127774_EMC6', 'ENSG00000127804_METTL16', 'ENSG00000127824_TUBA4A', 'ENSG00000127831_VIL1', 'ENSG00000127837_AAMP', 'ENSG00000127838_PNKD', 'ENSG00000127863_TNFRSF19', 'ENSG00000127870_RNF6', 'ENSG00000127884_ECHS1', 'ENSG00000127903_ZNF835', 'ENSG00000127914_AKAP9', 'ENSG00000127920_GNG11', 'ENSG00000127922_SEM1', 'ENSG00000127946_HIP1', 'ENSG00000127947_PTPN12', 'ENSG00000127948_POR', 'ENSG00000127951_FGL2', 'ENSG00000127952_STYXL1', 'ENSG00000127954_STEAP4', 'ENSG00000127955_GNAI1', 'ENSG00000127957_PMS2P3', 'ENSG00000127980_PEX1', 'ENSG00000127989_MTERF1', 'ENSG00000127990_SGCE', 'ENSG00000127993_RBM48', 'ENSG00000127995_CASD1', 'ENSG00000128000_ZNF780B', 'ENSG00000128011_LRFN1', 'ENSG00000128016_ZFP36', 'ENSG00000128039_SRD5A3', 'ENSG00000128040_SPINK2', 'ENSG00000128045_RASL11B', 'ENSG00000128050_PAICS', 'ENSG00000128059_PPAT', 'ENSG00000128159_TUBGCP6', 'ENSG00000128165_ADM2', 'ENSG00000128185_DGCR6L', 'ENSG00000128191_DGCR8', 'ENSG00000128203_ASPHD2', 'ENSG00000128218_VPREB3', 'ENSG00000128228_SDF2L1', 'ENSG00000128245_YWHAH', 'ENSG00000128266_GNAZ', 'ENSG00000128268_MGAT3', 'ENSG00000128272_ATF4', 'ENSG00000128274_A4GALT', 'ENSG00000128283_CDC42EP1', 'ENSG00000128284_APOL3', 'ENSG00000128294_TPST2', 'ENSG00000128298_BAIAP2L2', 'ENSG00000128309_MPST', 'ENSG00000128310_GALR3', 'ENSG00000128311_TST', 'ENSG00000128322_IGLL1', 'ENSG00000128335_APOL2', 'ENSG00000128340_RAC2', 'ENSG00000128342_LIF', 'ENSG00000128346_C22orf23', 'ENSG00000128383_APOBEC3A', 'ENSG00000128394_APOBEC3F', 'ENSG00000128408_RIBC2', 'ENSG00000128463_EMC4', 'ENSG00000128487_SPECC1', 'ENSG00000128512_DOCK4', 'ENSG00000128513_POT1', 'ENSG00000128524_ATP6V1F', 'ENSG00000128534_LSM8', 'ENSG00000128536_CDHR3', 'ENSG00000128563_PRKRIP1', 'ENSG00000128564_VGF', 'ENSG00000128567_PODXL', 'ENSG00000128578_STRIP2', 'ENSG00000128581_IFT22', 'ENSG00000128585_MKLN1', 'ENSG00000128590_DNAJB9', 'ENSG00000128594_LRRC4', 'ENSG00000128595_CALU', 'ENSG00000128596_CCDC136', 'ENSG00000128602_SMO', 'ENSG00000128604_IRF5', 'ENSG00000128606_LRRC17', 'ENSG00000128607_KLHDC10', 'ENSG00000128609_NDUFA5', 'ENSG00000128626_MRPS12', 'ENSG00000128641_MYO1B', 'ENSG00000128654_MTX2', 'ENSG00000128655_PDE11A', 'ENSG00000128656_CHN1', 'ENSG00000128683_GAD1', 'ENSG00000128692_EIF2S2P4', 'ENSG00000128694_OSGEPL1', 'ENSG00000128699_ORMDL1', 'ENSG00000128708_HAT1', 'ENSG00000128731_HERC2', 'ENSG00000128739_SNRPN', 'ENSG00000128789_PSMG2', 'ENSG00000128791_TWSG1', 'ENSG00000128805_ARHGAP22', 'ENSG00000128815_WDFY4', 'ENSG00000128829_EIF2AK4', 'ENSG00000128833_MYO5C', 'ENSG00000128872_TMOD2', 'ENSG00000128881_TTBK2', 'ENSG00000128891_CCDC32', 'ENSG00000128908_INO80', 'ENSG00000128915_ICE2', 'ENSG00000128923_MINDY2', 'ENSG00000128928_IVD', 'ENSG00000128944_KNSTRN', 'ENSG00000128951_DUT', 'ENSG00000128965_CHAC1', 'ENSG00000128973_CLN6', 'ENSG00000128989_ARPP19', 'ENSG00000129003_VPS13C', 'ENSG00000129007_CALML4', 'ENSG00000129028_THAP10', 'ENSG00000129038_LOXL1', 'ENSG00000129048_ACKR4', 'ENSG00000129055_ANAPC13', 'ENSG00000129071_MBD4', 'ENSG00000129083_COPB1', 'ENSG00000129084_PSMA1', 'ENSG00000129103_SUMF2', 'ENSG00000129116_PALLD', 'ENSG00000129128_SPCS3', 'ENSG00000129158_SERGEF', 'ENSG00000129159_KCNC1', 'ENSG00000129167_TPH1', 'ENSG00000129173_E2F8', 'ENSG00000129187_DCTD', 'ENSG00000129194_SOX15', 'ENSG00000129195_PIMREG', 'ENSG00000129197_RPAIN', 'ENSG00000129204_USP6', 'ENSG00000129219_PLD2', 'ENSG00000129226_CD68', 'ENSG00000129235_TXNDC17', 'ENSG00000129244_ATP1B2', 'ENSG00000129245_FXR2', 'ENSG00000129250_KIF1C', 'ENSG00000129255_MPDU1', 'ENSG00000129292_PHF20L1', 'ENSG00000129295_LRRC6', 'ENSG00000129315_CCNT1', 'ENSG00000129317_PUS7L', 'ENSG00000129347_KRI1', 'ENSG00000129351_ILF3', 'ENSG00000129353_SLC44A2', 'ENSG00000129355_CDKN2D', 'ENSG00000129422_MTUS1', 'ENSG00000129450_SIGLEC9', 'ENSG00000129460_NGDN', 'ENSG00000129465_RIPK3', 'ENSG00000129467_ADCY4', 'ENSG00000129472_RAB2B', 'ENSG00000129473_BCL2L2', 'ENSG00000129474_AJUBA', 'ENSG00000129480_DTD2', 'ENSG00000129484_PARP2', 'ENSG00000129493_HEATR5A', 'ENSG00000129515_SNX6', 'ENSG00000129518_EAPP', 'ENSG00000129521_EGLN3', 'ENSG00000129534_MIS18BP1', 'ENSG00000129535_NRL', 'ENSG00000129538_RNASE1', 'ENSG00000129559_NEDD8', 'ENSG00000129562_DAD1', 'ENSG00000129566_TEP1', 'ENSG00000129595_EPB41L4A', 'ENSG00000129625_REEP5', 'ENSG00000129636_ITFG1', 'ENSG00000129646_QRICH2', 'ENSG00000129654_FOXJ1', 'ENSG00000129657_SEC14L1', 'ENSG00000129667_RHBDF2', 'ENSG00000129675_ARHGEF6', 'ENSG00000129680_MAP7D3', 'ENSG00000129682_FGF13', 'ENSG00000129691_ASH2L', 'ENSG00000129696_TTI2', 'ENSG00000129749_CHRNA10', 'ENSG00000129757_CDKN1C', 'ENSG00000129810_SGO1', 'ENSG00000129824_RPS4Y1', 'ENSG00000129911_KLF16', 'ENSG00000129925_TMEM8A', 'ENSG00000129932_DOHH', 'ENSG00000129933_MAU2', 'ENSG00000129951_PLPPR3', 'ENSG00000129968_ABHD17A', 'ENSG00000129990_SYT5', 'ENSG00000129991_TNNI3', 'ENSG00000129993_CBFA2T3', 'ENSG00000130005_GAMT', 'ENSG00000130021_PUDP', 'ENSG00000130023_ERMARD', 'ENSG00000130024_PHF10', 'ENSG00000130032_PRRG3', 'ENSG00000130035_GALNT8', 'ENSG00000130038_CRACR2A', 'ENSG00000130052_STARD8', 'ENSG00000130054_FAM155B', 'ENSG00000130066_SAT1', 'ENSG00000130119_GNL3L', 'ENSG00000130147_SH3BP4', 'ENSG00000130150_MOSPD2', 'ENSG00000130158_DOCK6', 'ENSG00000130159_ECSIT', 'ENSG00000130164_LDLR', 'ENSG00000130165_ELOF1', 'ENSG00000130175_PRKCSH', 'ENSG00000130177_CDC16', 'ENSG00000130193_THEM6', 'ENSG00000130202_NECTIN2', 'ENSG00000130203_APOE', 'ENSG00000130204_TOMM40', 'ENSG00000130208_APOC1', 'ENSG00000130222_GADD45G', 'ENSG00000130224_LRCH2', 'ENSG00000130227_XPO7', 'ENSG00000130244_FAM98C', 'ENSG00000130254_SAFB2', 'ENSG00000130255_RPL36', 'ENSG00000130270_ATP8B3', 'ENSG00000130299_GTPBP3', 'ENSG00000130300_PLVAP', 'ENSG00000130303_BST2', 'ENSG00000130304_SLC27A1', 'ENSG00000130305_NSUN5', 'ENSG00000130309_COLGALT1', 'ENSG00000130311_DDA1', 'ENSG00000130312_MRPL34', 'ENSG00000130313_PGLS', 'ENSG00000130332_LSM7', 'ENSG00000130338_TULP4', 'ENSG00000130340_SNX9', 'ENSG00000130347_RTN4IP1', 'ENSG00000130348_QRSL1', 'ENSG00000130349_C6orf203', 'ENSG00000130363_RSPH3', 'ENSG00000130368_MAS1', 'ENSG00000130382_MLLT1', 'ENSG00000130396_AFDN', 'ENSG00000130402_ACTN4', 'ENSG00000130413_STK33', 'ENSG00000130414_NDUFA10', 'ENSG00000130429_ARPC1B', 'ENSG00000130449_ZSWIM6', 'ENSG00000130475_FCHO1', 'ENSG00000130477_UNC13A', 'ENSG00000130479_MAP1S', 'ENSG00000130487_KLHDC7B', 'ENSG00000130489_SCO2', 'ENSG00000130508_PXDN', 'ENSG00000130511_SSBP4', 'ENSG00000130513_GDF15', 'ENSG00000130517_PGPEP1', 'ENSG00000130518_IQCN', 'ENSG00000130520_LSM4', 'ENSG00000130522_JUND', 'ENSG00000130529_TRPM4', 'ENSG00000130544_ZNF557', 'ENSG00000130558_OLFM1', 'ENSG00000130559_CAMSAP1', 'ENSG00000130560_UBAC1', 'ENSG00000130561_SAG', 'ENSG00000130584_ZBTB46', 'ENSG00000130589_HELZ2', 'ENSG00000130590_SAMD10', 'ENSG00000130592_LSP1', 'ENSG00000130595_TNNT3', 'ENSG00000130598_TNNI2', 'ENSG00000130600_H19', 'ENSG00000130635_COL5A1', 'ENSG00000130638_ATXN10', 'ENSG00000130640_TUBGCP2', 'ENSG00000130649_CYP2E1', 'ENSG00000130653_PNPLA7', 'ENSG00000130656_HBZ', 'ENSG00000130669_PAK4', 'ENSG00000130684_ZNF337', 'ENSG00000130695_CEP85', 'ENSG00000130699_TAF4', 'ENSG00000130700_GATA5', 'ENSG00000130702_LAMA5', 'ENSG00000130703_OSBPL2', 'ENSG00000130706_ADRM1', 'ENSG00000130707_ASS1', 'ENSG00000130711_PRDM12', 'ENSG00000130713_EXOSC2', 'ENSG00000130714_POMT1', 'ENSG00000130717_UCK1', 'ENSG00000130720_FIBCD1', 'ENSG00000130723_PRRC2B', 'ENSG00000130724_CHMP2A', 'ENSG00000130725_UBE2M', 'ENSG00000130726_TRIM28', 'ENSG00000130731_METTL26', 'ENSG00000130733_YIPF2', 'ENSG00000130734_ATG4D', 'ENSG00000130741_EIF2S3', 'ENSG00000130748_TMEM160', 'ENSG00000130749_ZC3H4', 'ENSG00000130751_NPAS1', 'ENSG00000130755_GMFG', 'ENSG00000130758_MAP3K10', 'ENSG00000130762_ARHGEF16', 'ENSG00000130764_LRRC47', 'ENSG00000130766_SESN2', 'ENSG00000130768_SMPDL3B', 'ENSG00000130770_ATP5IF1', 'ENSG00000130772_MED18', 'ENSG00000130775_THEMIS2', 'ENSG00000130779_CLIP1', 'ENSG00000130783_CCDC62', 'ENSG00000130787_HIP1R', 'ENSG00000130803_ZNF317', 'ENSG00000130810_PPAN', 'ENSG00000130811_EIF3G', 'ENSG00000130812_ANGPTL6', 'ENSG00000130813_C19orf66', 'ENSG00000130816_DNMT1', 'ENSG00000130818_ZNF426', 'ENSG00000130821_SLC6A8', 'ENSG00000130822_PNCK', 'ENSG00000130826_DKC1', 'ENSG00000130827_PLXNA3', 'ENSG00000130829_DUSP9', 'ENSG00000130830_MPP1', 'ENSG00000130844_ZNF331', 'ENSG00000130856_ZNF236', 'ENSG00000130881_LRP3', 'ENSG00000130921_C12orf65', 'ENSG00000130935_NOL11', 'ENSG00000130939_UBE4B', 'ENSG00000130940_CASZ1', 'ENSG00000130956_HABP4', 'ENSG00000130958_SLC35D2', 'ENSG00000130962_PRRG1', 'ENSG00000130985_UBA1', 'ENSG00000130988_RGN', 'ENSG00000130997_POLN', 'ENSG00000131002_TXLNGY', 'ENSG00000131013_PPIL4', 'ENSG00000131015_ULBP2', 'ENSG00000131016_AKAP12', 'ENSG00000131018_SYNE1', 'ENSG00000131023_LATS1', 'ENSG00000131037_EPS8L1', 'ENSG00000131042_LILRB2', 'ENSG00000131043_AAR2', 'ENSG00000131051_RBM39', 'ENSG00000131061_ZNF341', 'ENSG00000131067_GGT7', 'ENSG00000131069_ACSS2', 'ENSG00000131080_EDA2R', 'ENSG00000131089_ARHGEF9', 'ENSG00000131100_ATP6V1E1', 'ENSG00000131115_ZNF227', 'ENSG00000131116_ZNF428', 'ENSG00000131127_ZNF141', 'ENSG00000131142_CCL25', 'ENSG00000131143_COX4I1', 'ENSG00000131148_EMC8', 'ENSG00000131149_GSE1', 'ENSG00000131153_GINS2', 'ENSG00000131165_CHMP1A', 'ENSG00000131171_SH3BGRL', 'ENSG00000131174_COX7B', 'ENSG00000131187_F12', 'ENSG00000131188_PRR7', 'ENSG00000131196_NFATC1', 'ENSG00000131236_CAP1', 'ENSG00000131238_PPT1', 'ENSG00000131242_RAB11FIP4', 'ENSG00000131263_RLIM', 'ENSG00000131269_ABCB7', 'ENSG00000131323_TRAF3', 'ENSG00000131351_HAUS8', 'ENSG00000131355_ADGRE3', 'ENSG00000131368_MRPS25', 'ENSG00000131370_SH3BP5', 'ENSG00000131373_HACL1', 'ENSG00000131374_TBC1D5', 'ENSG00000131375_CAPN7', 'ENSG00000131378_RFTN1', 'ENSG00000131381_RBSN', 'ENSG00000131389_SLC6A6', 'ENSG00000131398_KCNC3', 'ENSG00000131401_NAPSB', 'ENSG00000131408_NR1H2', 'ENSG00000131435_PDLIM4', 'ENSG00000131437_KIF3A', 'ENSG00000131446_MGAT1', 'ENSG00000131459_GFPT2', 'ENSG00000131462_TUBG1', 'ENSG00000131467_PSME3', 'ENSG00000131469_RPL27', 'ENSG00000131470_PSMC3IP', 'ENSG00000131471_AOC3', 'ENSG00000131473_ACLY', 'ENSG00000131475_VPS25', 'ENSG00000131477_RAMP2', 'ENSG00000131480_AOC2', 'ENSG00000131484_AC091132.1', 'ENSG00000131495_NDUFA2', 'ENSG00000131503_ANKHD1', 'ENSG00000131504_DIAPH1', 'ENSG00000131507_NDFIP1', 'ENSG00000131508_UBE2D2', 'ENSG00000131558_EXOC4', 'ENSG00000131584_ACAP3', 'ENSG00000131591_C1orf159', 'ENSG00000131626_PPFIA1', 'ENSG00000131634_TMEM204', 'ENSG00000131650_KREMEN2', 'ENSG00000131652_THOC6', 'ENSG00000131653_TRAF7', 'ENSG00000131668_BARX1', 'ENSG00000131669_NINJ1', 'ENSG00000131697_NPHP4', 'ENSG00000131711_MAP1B', 'ENSG00000131724_IL13RA1', 'ENSG00000131725_WDR44', 'ENSG00000131732_ZCCHC9', 'ENSG00000131747_TOP2A', 'ENSG00000131748_STARD3', 'ENSG00000131759_RARA', 'ENSG00000131773_KHDRBS3', 'ENSG00000131778_CHD1L', 'ENSG00000131779_PEX11B', 'ENSG00000131781_FMO5', 'ENSG00000131788_PIAS3', 'ENSG00000131791_PRKAB2', 'ENSG00000131797_CLUHP3', 'ENSG00000131828_PDHA1', 'ENSG00000131844_MCCC2', 'ENSG00000131845_ZNF304', 'ENSG00000131848_ZSCAN5A', 'ENSG00000131849_ZNF132', 'ENSG00000131871_SELENOS', 'ENSG00000131873_CHSY1', 'ENSG00000131876_SNRPA1', 'ENSG00000131899_LLGL1', 'ENSG00000131931_THAP1', 'ENSG00000131943_C19orf12', 'ENSG00000131944_FAAP24', 'ENSG00000131966_ACTR10', 'ENSG00000131969_ABHD12B', 'ENSG00000131979_GCH1', 'ENSG00000131981_LGALS3', 'ENSG00000132000_PODNL1', 'ENSG00000132002_DNAJB1', 'ENSG00000132003_ZSWIM4', 'ENSG00000132004_FBXW9', 'ENSG00000132005_RFX1', 'ENSG00000132010_ZNF20', 'ENSG00000132016_C19orf57', 'ENSG00000132017_DCAF15', 'ENSG00000132024_CC2D1A', 'ENSG00000132109_TRIM21', 'ENSG00000132122_SPATA6', 'ENSG00000132128_LRRC41', 'ENSG00000132141_CCT6B', 'ENSG00000132153_DHX30', 'ENSG00000132155_RAF1', 'ENSG00000132164_SLC6A11', 'ENSG00000132182_NUP210', 'ENSG00000132185_FCRLA', 'ENSG00000132196_HSD17B7', 'ENSG00000132199_ENOSF1', 'ENSG00000132204_LINC00470', 'ENSG00000132205_EMILIN2', 'ENSG00000132207_SLX1A', 'ENSG00000132254_ARFIP2', 'ENSG00000132256_TRIM5', 'ENSG00000132274_TRIM22', 'ENSG00000132275_RRP8', 'ENSG00000132286_TIMM10B', 'ENSG00000132294_EFR3A', 'ENSG00000132300_PTCD3', 'ENSG00000132305_IMMT', 'ENSG00000132313_MRPL35', 'ENSG00000132321_IQCA1', 'ENSG00000132323_ILKAP', 'ENSG00000132326_PER2', 'ENSG00000132329_RAMP1', 'ENSG00000132330_SCLY', 'ENSG00000132334_PTPRE', 'ENSG00000132341_RAN', 'ENSG00000132356_PRKAA1', 'ENSG00000132357_CARD6', 'ENSG00000132359_RAP1GAP2', 'ENSG00000132361_CLUH', 'ENSG00000132376_INPP5K', 'ENSG00000132382_MYBBP1A', 'ENSG00000132383_RPA1', 'ENSG00000132386_SERPINF1', 'ENSG00000132388_UBE2G1', 'ENSG00000132394_EEFSEC', 'ENSG00000132405_TBC1D14', 'ENSG00000132406_TMEM128', 'ENSG00000132423_COQ3', 'ENSG00000132424_PNISR', 'ENSG00000132432_SEC61G', 'ENSG00000132434_LANCL2', 'ENSG00000132436_FIGNL1', 'ENSG00000132437_DDC', 'ENSG00000132463_GRSF1', 'ENSG00000132465_JCHAIN', 'ENSG00000132466_ANKRD17', 'ENSG00000132467_UTP3', 'ENSG00000132470_ITGB4', 'ENSG00000132471_WBP2', 'ENSG00000132475_H3F3B', 'ENSG00000132478_UNK', 'ENSG00000132481_TRIM47', 'ENSG00000132485_ZRANB2', 'ENSG00000132507_EIF5A', 'ENSG00000132510_KDM6B', 'ENSG00000132514_CLEC10A', 'ENSG00000132518_GUCY2D', 'ENSG00000132522_GPS2', 'ENSG00000132530_XAF1', 'ENSG00000132535_DLG4', 'ENSG00000132541_RIDA', 'ENSG00000132549_VPS13B', 'ENSG00000132561_MATN2', 'ENSG00000132563_REEP2', 'ENSG00000132570_PCBD2', 'ENSG00000132581_SDF2', 'ENSG00000132589_FLOT2', 'ENSG00000132591_ERAL1', 'ENSG00000132600_PRMT7', 'ENSG00000132603_NIP7', 'ENSG00000132604_TERF2', 'ENSG00000132612_VPS4A', 'ENSG00000132613_MTSS1L', 'ENSG00000132623_ANKEF1', 'ENSG00000132635_PCED1A', 'ENSG00000132639_SNAP25', 'ENSG00000132640_BTBD3', 'ENSG00000132646_PCNA', 'ENSG00000132661_NXT1', 'ENSG00000132664_POLR3F', 'ENSG00000132669_RIN2', 'ENSG00000132670_PTPRA', 'ENSG00000132676_DAP3', 'ENSG00000132680_KHDC4', 'ENSG00000132688_NES', 'ENSG00000132692_BCAN', 'ENSG00000132694_ARHGEF11', 'ENSG00000132704_FCRL2', 'ENSG00000132716_DCAF8', 'ENSG00000132718_SYT11', 'ENSG00000132740_IGHMBP2', 'ENSG00000132744_ACY3', 'ENSG00000132749_TESMIN', 'ENSG00000132763_MMACHC', 'ENSG00000132768_DPH2', 'ENSG00000132773_TOE1', 'ENSG00000132780_NASP', 'ENSG00000132781_MUTYH', 'ENSG00000132792_CTNNBL1', 'ENSG00000132793_LPIN3', 'ENSG00000132801_ZSWIM3', 'ENSG00000132819_RBM38', 'ENSG00000132823_OSER1', 'ENSG00000132824_SERINC3', 'ENSG00000132825_PPP1R3D', 'ENSG00000132837_DMGDH', 'ENSG00000132842_AP3B1', 'ENSG00000132846_ZBED3', 'ENSG00000132849_PATJ', 'ENSG00000132854_KANK4', 'ENSG00000132874_SLC14A2', 'ENSG00000132879_FBXO44', 'ENSG00000132881_CPLANE2', 'ENSG00000132906_CASP9', 'ENSG00000132912_DCTN4', 'ENSG00000132915_PDE6A', 'ENSG00000132932_ATP8A2', 'ENSG00000132938_MTUS2', 'ENSG00000132950_ZMYM5', 'ENSG00000132952_USPL1', 'ENSG00000132953_XPO4', 'ENSG00000132963_POMP', 'ENSG00000132964_CDK8', 'ENSG00000132965_ALOX5AP', 'ENSG00000132967_HMGB1P5', 'ENSG00000132970_WASF3', 'ENSG00000132972_RNF17', 'ENSG00000132975_GPR12', 'ENSG00000133019_CHRM3', 'ENSG00000133026_MYH10', 'ENSG00000133027_PEMT', 'ENSG00000133028_SCO1', 'ENSG00000133030_MPRIP', 'ENSG00000133048_CHI3L1', 'ENSG00000133056_PIK3C2B', 'ENSG00000133059_DSTYK', 'ENSG00000133063_CHIT1', 'ENSG00000133065_SLC41A1', 'ENSG00000133067_LGR6', 'ENSG00000133069_TMCC2', 'ENSG00000133101_CCNA1', 'ENSG00000133103_COG6', 'ENSG00000133104_SPART', 'ENSG00000133106_EPSTI1', 'ENSG00000133111_RFXAP', 'ENSG00000133112_TPT1', 'ENSG00000133114_GPALPP1', 'ENSG00000133116_KL', 'ENSG00000133119_RFC3', 'ENSG00000133121_STARD13', 'ENSG00000133131_MORC4', 'ENSG00000133134_BEX2', 'ENSG00000133138_TBC1D8B', 'ENSG00000133142_TCEAL4', 'ENSG00000133169_BEX1', 'ENSG00000133193_FAM104A', 'ENSG00000133195_SLC39A11', 'ENSG00000133216_EPHB2', 'ENSG00000133226_SRRM1', 'ENSG00000133243_BTBD2', 'ENSG00000133246_PRAM1', 'ENSG00000133247_KMT5C', 'ENSG00000133250_ZNF414', 'ENSG00000133256_PDE6B', 'ENSG00000133265_HSPBP1', 'ENSG00000133275_CSNK1G2', 'ENSG00000133302_SLF1', 'ENSG00000133313_CNDP2', 'ENSG00000133315_MACROD1', 'ENSG00000133316_WDR74', 'ENSG00000133317_LGALS12', 'ENSG00000133318_RTN3', 'ENSG00000133321_RARRES3', 'ENSG00000133328_HRASLS2', 'ENSG00000133392_MYH11', 'ENSG00000133393_FOPNL', 'ENSG00000133398_MED10', 'ENSG00000133401_PDZD2', 'ENSG00000133422_MORC2', 'ENSG00000133424_LARGE1', 'ENSG00000133460_SLC2A11', 'ENSG00000133466_C1QTNF6', 'ENSG00000133477_FAM83F', 'ENSG00000133488_SEC14L4', 'ENSG00000133519_ZDHHC8P1', 'ENSG00000133561_GIMAP6', 'ENSG00000133574_GIMAP4', 'ENSG00000133597_ADCK2', 'ENSG00000133606_MKRN1', 'ENSG00000133612_AGAP3', 'ENSG00000133619_KRBA1', 'ENSG00000133624_ZNF767P', 'ENSG00000133627_ACTR3B', 'ENSG00000133636_NTS', 'ENSG00000133639_BTG1', 'ENSG00000133641_C12orf29', 'ENSG00000133657_ATP13A3', 'ENSG00000133661_SFTPD', 'ENSG00000133665_DYDC2', 'ENSG00000133678_TMEM254', 'ENSG00000133687_TMTC1', 'ENSG00000133703_KRAS', 'ENSG00000133704_IPO8', 'ENSG00000133706_LARS', 'ENSG00000133731_IMPA1', 'ENSG00000133739_LRRCC1', 'ENSG00000133740_E2F5', 'ENSG00000133742_CA1', 'ENSG00000133773_CCDC59', 'ENSG00000133789_SWAP70', 'ENSG00000133794_ARNTL', 'ENSG00000133805_AMPD3', 'ENSG00000133812_SBF2', 'ENSG00000133816_MICAL2', 'ENSG00000133818_RRAS2', 'ENSG00000133835_HSD17B4', 'ENSG00000133858_ZFC3H1', 'ENSG00000133863_TEX15', 'ENSG00000133872_SARAF', 'ENSG00000133874_RNF122', 'ENSG00000133884_DPF2', 'ENSG00000133895_MEN1', 'ENSG00000133935_ERG28', 'ENSG00000133943_DGLUCY', 'ENSG00000133958_UNC79', 'ENSG00000133961_NUMB', 'ENSG00000133983_COX16', 'ENSG00000133985_TTC9', 'ENSG00000133997_MED6', 'ENSG00000134001_EIF2S1', 'ENSG00000134007_ADAM20', 'ENSG00000134013_LOXL2', 'ENSG00000134014_ELP3', 'ENSG00000134020_PEBP4', 'ENSG00000134028_ADAMDEC1', 'ENSG00000134030_CTIF', 'ENSG00000134046_MBD2', 'ENSG00000134049_IER3IP1', 'ENSG00000134056_MRPS36', 'ENSG00000134057_CCNB1', 'ENSG00000134058_CDK7', 'ENSG00000134061_CD180', 'ENSG00000134070_IRAK2', 'ENSG00000134072_CAMK1', 'ENSG00000134077_THUMPD3', 'ENSG00000134086_VHL', 'ENSG00000134107_BHLHE40', 'ENSG00000134108_ARL8B', 'ENSG00000134109_EDEM1', 'ENSG00000134138_MEIS2', 'ENSG00000134146_DPH6', 'ENSG00000134152_KATNBL1', 'ENSG00000134153_EMC7', 'ENSG00000134184_GSTM1', 'ENSG00000134186_PRPF38B', 'ENSG00000134193_REG4', 'ENSG00000134198_TSPAN2', 'ENSG00000134201_GSTM5', 'ENSG00000134202_GSTM3', 'ENSG00000134215_VAV3', 'ENSG00000134222_PSRC1', 'ENSG00000134242_PTPN22', 'ENSG00000134243_SORT1', 'ENSG00000134245_WNT2B', 'ENSG00000134247_PTGFRN', 'ENSG00000134248_LAMTOR5', 'ENSG00000134250_NOTCH2', 'ENSG00000134253_TRIM45', 'ENSG00000134255_CEPT1', 'ENSG00000134256_CD101', 'ENSG00000134262_AP4B1', 'ENSG00000134265_NAPG', 'ENSG00000134278_SPIRE1', 'ENSG00000134283_PPHLN1', 'ENSG00000134285_FKBP11', 'ENSG00000134287_ARF3', 'ENSG00000134291_TMEM106C', 'ENSG00000134294_SLC38A2', 'ENSG00000134297_PLEKHA8P1', 'ENSG00000134308_YWHAQ', 'ENSG00000134313_KIDINS220', 'ENSG00000134317_GRHL1', 'ENSG00000134318_ROCK2', 'ENSG00000134321_RSAD2', 'ENSG00000134323_MYCN', 'ENSG00000134324_LPIN1', 'ENSG00000134326_CMPK2', 'ENSG00000134330_IAH1', 'ENSG00000134333_LDHA', 'ENSG00000134352_IL6ST', 'ENSG00000134363_FST', 'ENSG00000134369_NAV1', 'ENSG00000134371_CDC73', 'ENSG00000134375_TIMM17A', 'ENSG00000134376_CRB1', 'ENSG00000134419_RPS15A', 'ENSG00000134440_NARS', 'ENSG00000134444_RELCH', 'ENSG00000134452_FBH1', 'ENSG00000134453_RBM17', 'ENSG00000134460_IL2RA', 'ENSG00000134461_ANKRD16', 'ENSG00000134463_ECHDC3', 'ENSG00000134470_IL15RA', 'ENSG00000134480_CCNH', 'ENSG00000134489_HRH4', 'ENSG00000134490_TMEM241', 'ENSG00000134504_KCTD1', 'ENSG00000134508_CABLES1', 'ENSG00000134516_DOCK2', 'ENSG00000134531_EMP1', 'ENSG00000134539_KLRD1', 'ENSG00000134548_SPX', 'ENSG00000134569_LRP4', 'ENSG00000134571_MYBPC3', 'ENSG00000134574_DDB2', 'ENSG00000134575_ACP2', 'ENSG00000134590_RTL8C', 'ENSG00000134594_RAB33A', 'ENSG00000134597_RBMX2', 'ENSG00000134602_STK26', 'ENSG00000134627_PIWIL4', 'ENSG00000134644_PUM1', 'ENSG00000134684_YARS', 'ENSG00000134686_PHC2', 'ENSG00000134690_CDCA8', 'ENSG00000134697_GNL2', 'ENSG00000134698_AGO4', 'ENSG00000134709_HOOK1', 'ENSG00000134716_CYP2J2', 'ENSG00000134717_BTF3L4', 'ENSG00000134744_TUT4', 'ENSG00000134748_PRPF38A', 'ENSG00000134755_DSC2', 'ENSG00000134758_RNF138', 'ENSG00000134759_ELP2', 'ENSG00000134769_DTNA', 'ENSG00000134779_TPGS2', 'ENSG00000134780_DAGLA', 'ENSG00000134802_SLC43A3', 'ENSG00000134809_TIMM10', 'ENSG00000134815_DHX34', 'ENSG00000134824_FADS2', 'ENSG00000134825_TMEM258', 'ENSG00000134827_TCN1', 'ENSG00000134830_C5AR2', 'ENSG00000134851_TMEM165', 'ENSG00000134852_CLOCK', 'ENSG00000134864_GGACT', 'ENSG00000134873_CLDN10', 'ENSG00000134874_DZIP1', 'ENSG00000134882_UBAC2', 'ENSG00000134884_ARGLU1', 'ENSG00000134897_BIVM', 'ENSG00000134899_ERCC5', 'ENSG00000134900_TPP2', 'ENSG00000134901_KDELC1', 'ENSG00000134905_CARS2', 'ENSG00000134909_ARHGAP32', 'ENSG00000134910_STT3A', 'ENSG00000134940_ACRV1', 'ENSG00000134954_ETS1', 'ENSG00000134955_SLC37A2', 'ENSG00000134962_KLB', 'ENSG00000134970_TMED7', 'ENSG00000134982_APC', 'ENSG00000134986_NREP', 'ENSG00000134987_WDR36', 'ENSG00000134996_OSTF1', 'ENSG00000135002_RFK', 'ENSG00000135018_UBQLN1', 'ENSG00000135040_NAA35', 'ENSG00000135045_C9orf40', 'ENSG00000135046_ANXA1', 'ENSG00000135047_CTSL', 'ENSG00000135048_CEMIP2', 'ENSG00000135049_AGTPBP1', 'ENSG00000135052_GOLM1', 'ENSG00000135069_PSAT1', 'ENSG00000135070_ISCA1', 'ENSG00000135074_ADAM19', 'ENSG00000135077_HAVCR2', 'ENSG00000135090_TAOK3', 'ENSG00000135093_USP30', 'ENSG00000135094_SDS', 'ENSG00000135108_FBXO21', 'ENSG00000135114_OASL', 'ENSG00000135119_RNFT2', 'ENSG00000135124_P2RX4', 'ENSG00000135127_BICDL1', 'ENSG00000135144_DTX1', 'ENSG00000135148_TRAFD1', 'ENSG00000135164_DMTF1', 'ENSG00000135185_TMEM243', 'ENSG00000135205_CCDC146', 'ENSG00000135211_TMEM60', 'ENSG00000135218_CD36', 'ENSG00000135241_PNPLA8', 'ENSG00000135245_HILPDA', 'ENSG00000135249_RINT1', 'ENSG00000135250_SRPK2', 'ENSG00000135253_KCP', 'ENSG00000135269_TES', 'ENSG00000135272_MDFIC', 'ENSG00000135297_MTO1', 'ENSG00000135299_ANKRD6', 'ENSG00000135314_KHDC1', 'ENSG00000135315_CEP162', 'ENSG00000135316_SYNCRIP', 'ENSG00000135317_SNX14', 'ENSG00000135334_AKIRIN2', 'ENSG00000135336_ORC3', 'ENSG00000135338_LCA5', 'ENSG00000135341_MAP3K7', 'ENSG00000135362_PRR5L', 'ENSG00000135363_LMO2', 'ENSG00000135365_PHF21A', 'ENSG00000135372_NAT10', 'ENSG00000135378_PRRG4', 'ENSG00000135387_CAPRIN1', 'ENSG00000135390_ATP5MC2', 'ENSG00000135392_DNAJC14', 'ENSG00000135404_CD63', 'ENSG00000135407_AVIL', 'ENSG00000135409_AMHR2', 'ENSG00000135414_GDF11', 'ENSG00000135423_GLS2', 'ENSG00000135424_ITGA7', 'ENSG00000135426_TESPA1', 'ENSG00000135437_RDH5', 'ENSG00000135439_AGAP2', 'ENSG00000135441_BLOC1S1', 'ENSG00000135446_CDK4', 'ENSG00000135451_TROAP', 'ENSG00000135452_TSPAN31', 'ENSG00000135454_B4GALNT1', 'ENSG00000135457_TFCP2', 'ENSG00000135469_COQ10A', 'ENSG00000135473_PAN2', 'ENSG00000135476_ESPL1', 'ENSG00000135480_KRT7', 'ENSG00000135482_ZC3H10', 'ENSG00000135486_HNRNPA1', 'ENSG00000135503_ACVR1B', 'ENSG00000135506_OS9', 'ENSG00000135519_KCNH3', 'ENSG00000135521_LTV1', 'ENSG00000135525_MAP7', 'ENSG00000135535_CD164', 'ENSG00000135537_AFG1L', 'ENSG00000135540_NHSL1', 'ENSG00000135541_AHI1', 'ENSG00000135549_PKIB', 'ENSG00000135587_SMPD2', 'ENSG00000135596_MICAL1', 'ENSG00000135597_REPS1', 'ENSG00000135604_STX11', 'ENSG00000135605_TEC', 'ENSG00000135617_PRADC1', 'ENSG00000135622_SEMA4F', 'ENSG00000135624_CCT7', 'ENSG00000135631_RAB11FIP5', 'ENSG00000135632_SMYD5', 'ENSG00000135636_DYSF', 'ENSG00000135637_CCDC142', 'ENSG00000135643_KCNMB4', 'ENSG00000135655_USP15', 'ENSG00000135677_GNS', 'ENSG00000135678_CPM', 'ENSG00000135679_MDM2', 'ENSG00000135686_KLHL36', 'ENSG00000135698_MPHOSPH6', 'ENSG00000135709_KIAA0513', 'ENSG00000135720_DYNC1LI2', 'ENSG00000135722_FBXL8', 'ENSG00000135723_FHOD1', 'ENSG00000135736_CCDC102A', 'ENSG00000135740_SLC9A5', 'ENSG00000135747_ZNF670-ZNF695', 'ENSG00000135749_PCNX2', 'ENSG00000135763_URB2', 'ENSG00000135766_EGLN1', 'ENSG00000135775_COG2', 'ENSG00000135776_ABCB10', 'ENSG00000135778_NTPCR', 'ENSG00000135801_TAF5L', 'ENSG00000135821_GLUL', 'ENSG00000135823_STX6', 'ENSG00000135828_RNASEL', 'ENSG00000135829_DHX9', 'ENSG00000135835_KIAA1614', 'ENSG00000135837_CEP350', 'ENSG00000135838_NPL', 'ENSG00000135842_FAM129A', 'ENSG00000135845_PIGC', 'ENSG00000135862_LAMC1', 'ENSG00000135870_RC3H1', 'ENSG00000135899_SP110', 'ENSG00000135900_MRPL44', 'ENSG00000135905_DOCK10', 'ENSG00000135912_TTLL4', 'ENSG00000135913_USP37', 'ENSG00000135914_HTR2B', 'ENSG00000135916_ITM2C', 'ENSG00000135919_SERPINE2', 'ENSG00000135924_DNAJB2', 'ENSG00000135926_TMBIM1', 'ENSG00000135930_EIF4E2', 'ENSG00000135931_ARMC9', 'ENSG00000135932_CAB39', 'ENSG00000135940_COX5B', 'ENSG00000135945_REV1', 'ENSG00000135951_TSGA10', 'ENSG00000135953_MFSD9', 'ENSG00000135956_TMEM127', 'ENSG00000135966_TGFBRAP1', 'ENSG00000135968_GCC2', 'ENSG00000135972_MRPS9', 'ENSG00000135974_C2orf49', 'ENSG00000135976_ANKRD36', 'ENSG00000135999_EPC2', 'ENSG00000136003_ISCU', 'ENSG00000136010_ALDH1L2', 'ENSG00000136014_USP44', 'ENSG00000136021_SCYL2', 'ENSG00000136026_CKAP4', 'ENSG00000136040_PLXNC1', 'ENSG00000136044_APPL2', 'ENSG00000136045_PWP1', 'ENSG00000136048_DRAM1', 'ENSG00000136051_WASHC4', 'ENSG00000136052_SLC41A2', 'ENSG00000136059_VILL', 'ENSG00000136068_FLNB', 'ENSG00000136098_NEK3', 'ENSG00000136100_VPS36', 'ENSG00000136104_RNASEH2B', 'ENSG00000136108_CKAP2', 'ENSG00000136110_CNMD', 'ENSG00000136111_TBC1D4', 'ENSG00000136114_THSD1', 'ENSG00000136122_BORA', 'ENSG00000136141_LRCH1', 'ENSG00000136143_SUCLA2', 'ENSG00000136144_RCBTB1', 'ENSG00000136146_MED4', 'ENSG00000136147_PHF11', 'ENSG00000136149_RPL13AP25', 'ENSG00000136152_COG3', 'ENSG00000136153_LMO7', 'ENSG00000136156_ITM2B', 'ENSG00000136158_SPRY2', 'ENSG00000136159_NUDT15', 'ENSG00000136160_EDNRB', 'ENSG00000136161_RCBTB2', 'ENSG00000136167_LCP1', 'ENSG00000136169_SETDB2', 'ENSG00000136193_SCRN1', 'ENSG00000136197_C7orf25', 'ENSG00000136205_TNS3', 'ENSG00000136206_SPDYE1', 'ENSG00000136213_CHST12', 'ENSG00000136231_IGF2BP3', 'ENSG00000136235_GPNMB', 'ENSG00000136237_RAPGEF5', 'ENSG00000136238_RAC1', 'ENSG00000136240_KDELR2', 'ENSG00000136243_NUPL2', 'ENSG00000136247_ZDHHC4', 'ENSG00000136250_AOAH', 'ENSG00000136261_BZW2', 'ENSG00000136270_TBRG4', 'ENSG00000136271_DDX56', 'ENSG00000136273_HUS1', 'ENSG00000136279_DBNL', 'ENSG00000136280_CCM2', 'ENSG00000136286_MYO1G', 'ENSG00000136295_TTYH3', 'ENSG00000136305_CIDEB', 'ENSG00000136315_AL355922.1', 'ENSG00000136319_TTC5', 'ENSG00000136367_ZFHX2', 'ENSG00000136371_MTHFS', 'ENSG00000136378_ADAMTS7', 'ENSG00000136379_ABHD17C', 'ENSG00000136381_IREB2', 'ENSG00000136383_ALPK3', 'ENSG00000136404_TM6SF1', 'ENSG00000136425_CIB2', 'ENSG00000136436_CALCOCO2', 'ENSG00000136444_RSAD1', 'ENSG00000136448_NMT1', 'ENSG00000136449_MYCBPAP', 'ENSG00000136450_SRSF1', 'ENSG00000136451_VEZF1', 'ENSG00000136463_TACO1', 'ENSG00000136478_TEX2', 'ENSG00000136485_DCAF7', 'ENSG00000136490_LIMD2', 'ENSG00000136492_BRIP1', 'ENSG00000136504_KAT7', 'ENSG00000136514_RTP4', 'ENSG00000136518_ACTL6A', 'ENSG00000136521_NDUFB5', 'ENSG00000136522_MRPL47', 'ENSG00000136527_TRA2B', 'ENSG00000136531_SCN2A', 'ENSG00000136536_MARCH7', 'ENSG00000136541_ERMN', 'ENSG00000136542_GALNT5', 'ENSG00000136560_TANK', 'ENSG00000136603_SKIL', 'ENSG00000136628_EPRS', 'ENSG00000136630_HLX', 'ENSG00000136631_VPS45', 'ENSG00000136634_IL10', 'ENSG00000136636_KCTD3', 'ENSG00000136643_RPS6KC1', 'ENSG00000136682_CBWD2', 'ENSG00000136689_IL1RN', 'ENSG00000136699_SMPD4', 'ENSG00000136709_WDR33', 'ENSG00000136710_CCDC115', 'ENSG00000136715_SAP130', 'ENSG00000136717_BIN1', 'ENSG00000136718_IMP4', 'ENSG00000136720_HS6ST1', 'ENSG00000136731_UGGT1', 'ENSG00000136732_GYPC', 'ENSG00000136738_STAM', 'ENSG00000136754_ABI1', 'ENSG00000136758_YME1L1', 'ENSG00000136770_DNAJC1', 'ENSG00000136783_NIPSNAP3A', 'ENSG00000136802_LRRC8A', 'ENSG00000136807_CDK9', 'ENSG00000136810_TXN', 'ENSG00000136811_ODF2', 'ENSG00000136813_ECPAS', 'ENSG00000136816_TOR1B', 'ENSG00000136819_C9orf78', 'ENSG00000136824_SMC2', 'ENSG00000136826_KLF4', 'ENSG00000136827_TOR1A', 'ENSG00000136828_RALGPS1', 'ENSG00000136830_FAM129B', 'ENSG00000136840_ST6GALNAC4', 'ENSG00000136842_TMOD1', 'ENSG00000136848_DAB2IP', 'ENSG00000136854_STXBP1', 'ENSG00000136856_SLC2A8', 'ENSG00000136861_CDK5RAP2', 'ENSG00000136866_ZFP37', 'ENSG00000136867_SLC31A2', 'ENSG00000136868_SLC31A1', 'ENSG00000136869_TLR4', 'ENSG00000136870_ZNF189', 'ENSG00000136874_STX17', 'ENSG00000136875_PRPF4', 'ENSG00000136877_FPGS', 'ENSG00000136878_USP20', 'ENSG00000136888_ATP6V1G1', 'ENSG00000136891_TEX10', 'ENSG00000136895_GARNL3', 'ENSG00000136897_MRPL50', 'ENSG00000136908_DPM2', 'ENSG00000136925_TSTD2', 'ENSG00000136929_HEMGN', 'ENSG00000136930_PSMB7', 'ENSG00000136932_TRMO', 'ENSG00000136933_RABEPK', 'ENSG00000136935_GOLGA1', 'ENSG00000136936_XPA', 'ENSG00000136937_NCBP1', 'ENSG00000136938_ANP32B', 'ENSG00000136940_PDCL', 'ENSG00000136942_RPL35', 'ENSG00000136943_CTSV', 'ENSG00000136950_ARPC5L', 'ENSG00000136960_ENPP2', 'ENSG00000136982_DSCC1', 'ENSG00000136986_DERL1', 'ENSG00000136997_MYC', 'ENSG00000136999_NOV', 'ENSG00000137038_DMAC1', 'ENSG00000137040_RANBP6', 'ENSG00000137054_POLR1E', 'ENSG00000137055_PLAA', 'ENSG00000137070_IL11RA', 'ENSG00000137073_UBAP2', 'ENSG00000137074_APTX', 'ENSG00000137075_RNF38', 'ENSG00000137076_TLN1', 'ENSG00000137094_DNAJB5', 'ENSG00000137098_SPAG8', 'ENSG00000137100_DCTN3', 'ENSG00000137101_CD72', 'ENSG00000137103_TMEM8B', 'ENSG00000137106_GRHPR', 'ENSG00000137124_ALDH1B1', 'ENSG00000137133_HINT2', 'ENSG00000137135_ARHGEF39', 'ENSG00000137145_DENND4C', 'ENSG00000137154_RPS6', 'ENSG00000137161_CNPY3', 'ENSG00000137166_FOXP4', 'ENSG00000137168_PPIL1', 'ENSG00000137171_KLC4', 'ENSG00000137177_KIF13A', 'ENSG00000137185_ZSCAN9', 'ENSG00000137193_PIM1', 'ENSG00000137198_GMPR', 'ENSG00000137200_CMTR1', 'ENSG00000137203_TFAP2A', 'ENSG00000137207_YIPF3', 'ENSG00000137210_TMEM14B', 'ENSG00000137216_TMEM63B', 'ENSG00000137218_FRS3', 'ENSG00000137221_TJAP1', 'ENSG00000137225_CAPN11', 'ENSG00000137265_IRF4', 'ENSG00000137266_SLC22A23', 'ENSG00000137267_TUBB2A', 'ENSG00000137269_LRRC1', 'ENSG00000137274_BPHL', 'ENSG00000137275_RIPK1', 'ENSG00000137285_TUBB2B', 'ENSG00000137288_UQCC2', 'ENSG00000137309_HMGA1', 'ENSG00000137310_TCF19', 'ENSG00000137312_FLOT1', 'ENSG00000137331_IER3', 'ENSG00000137337_MDC1', 'ENSG00000137338_PGBD1', 'ENSG00000137343_ATAT1', 'ENSG00000137364_TPMT', 'ENSG00000137393_RNF144B', 'ENSG00000137404_NRM', 'ENSG00000137409_MTCH1', 'ENSG00000137411_VARS2', 'ENSG00000137413_TAF8', 'ENSG00000137414_FAM8A1', 'ENSG00000137434_C6orf52', 'ENSG00000137441_FGFBP2', 'ENSG00000137449_CPEB2', 'ENSG00000137460_FHDC1', 'ENSG00000137462_TLR2', 'ENSG00000137463_MGARP', 'ENSG00000137478_FCHSD2', 'ENSG00000137486_ARRB1', 'ENSG00000137492_THAP12', 'ENSG00000137494_ANKRD42', 'ENSG00000137496_IL18BP', 'ENSG00000137497_NUMA1', 'ENSG00000137500_CCDC90B', 'ENSG00000137501_SYTL2', 'ENSG00000137502_RAB30', 'ENSG00000137504_CREBZF', 'ENSG00000137507_LRRC32', 'ENSG00000137509_PRCP', 'ENSG00000137513_NARS2', 'ENSG00000137522_RNF121', 'ENSG00000137547_MRPL15', 'ENSG00000137563_GGH', 'ENSG00000137571_SLCO5A1', 'ENSG00000137574_TGS1', 'ENSG00000137575_SDCBP', 'ENSG00000137601_NEK1', 'ENSG00000137628_DDX60', 'ENSG00000137642_SORL1', 'ENSG00000137656_BUD13', 'ENSG00000137672_TRPC6', 'ENSG00000137673_MMP7', 'ENSG00000137692_DCUN1D5', 'ENSG00000137693_YAP1', 'ENSG00000137699_TRIM29', 'ENSG00000137700_SLC37A4', 'ENSG00000137710_RDX', 'ENSG00000137713_PPP2R1B', 'ENSG00000137714_FDX1', 'ENSG00000137720_C11orf1', 'ENSG00000137726_FXYD6', 'ENSG00000137727_ARHGAP20', 'ENSG00000137752_CASP1', 'ENSG00000137760_ALKBH8', 'ENSG00000137764_MAP2K5', 'ENSG00000137767_SQOR', 'ENSG00000137770_CTDSPL2', 'ENSG00000137776_SLTM', 'ENSG00000137801_THBS1', 'ENSG00000137802_MAPKBP1', 'ENSG00000137804_NUSAP1', 'ENSG00000137806_NDUFAF1', 'ENSG00000137807_KIF23', 'ENSG00000137809_ITGA11', 'ENSG00000137812_KNL1', 'ENSG00000137814_HAUS2', 'ENSG00000137815_RTF1', 'ENSG00000137817_PARP6', 'ENSG00000137818_RPLP1', 'ENSG00000137819_PAQR5', 'ENSG00000137821_LRRC49', 'ENSG00000137822_TUBGCP4', 'ENSG00000137824_RMDN3', 'ENSG00000137825_ITPKA', 'ENSG00000137831_UACA', 'ENSG00000137834_SMAD6', 'ENSG00000137841_PLCB2', 'ENSG00000137842_TMEM62', 'ENSG00000137843_PAK6', 'ENSG00000137845_ADAM10', 'ENSG00000137857_DUOX1', 'ENSG00000137860_SLC28A2', 'ENSG00000137871_ZNF280D', 'ENSG00000137875_BCL2L10', 'ENSG00000137876_RSL24D1', 'ENSG00000137880_GCHFR', 'ENSG00000137936_BCAR3', 'ENSG00000137941_TTLL7', 'ENSG00000137942_FNBP1L', 'ENSG00000137944_KYAT3', 'ENSG00000137947_GTF2B', 'ENSG00000137955_RABGGTB', 'ENSG00000137959_IFI44L', 'ENSG00000137960_GIPC2', 'ENSG00000137962_ARHGAP29', 'ENSG00000137965_IFI44', 'ENSG00000137968_SLC44A5', 'ENSG00000137970_RPL7P9', 'ENSG00000137992_DBT', 'ENSG00000137996_RTCA', 'ENSG00000138002_IFT172', 'ENSG00000138018_SELENOI', 'ENSG00000138028_CGREF1', 'ENSG00000138029_HADHB', 'ENSG00000138030_KHK', 'ENSG00000138031_ADCY3', 'ENSG00000138032_PPM1B', 'ENSG00000138035_PNPT1', 'ENSG00000138036_DYNC2LI1', 'ENSG00000138050_THUMPD2', 'ENSG00000138061_CYP1B1', 'ENSG00000138069_RAB1A', 'ENSG00000138071_ACTR2', 'ENSG00000138073_PREB', 'ENSG00000138074_SLC5A6', 'ENSG00000138078_PREPL', 'ENSG00000138079_SLC3A1', 'ENSG00000138080_EMILIN1', 'ENSG00000138081_FBXO11', 'ENSG00000138085_ATRAID', 'ENSG00000138092_CENPO', 'ENSG00000138095_LRPPRC', 'ENSG00000138100_TRIM54', 'ENSG00000138101_DTNB', 'ENSG00000138107_ACTR1A', 'ENSG00000138111_MFSD13A', 'ENSG00000138115_CYP2C8', 'ENSG00000138119_MYOF', 'ENSG00000138131_LOXL4', 'ENSG00000138134_STAMBPL1', 'ENSG00000138135_CH25H', 'ENSG00000138138_ATAD1', 'ENSG00000138160_KIF11', 'ENSG00000138161_CUZD1', 'ENSG00000138162_TACC2', 'ENSG00000138166_DUSP5', 'ENSG00000138172_CALHM2', 'ENSG00000138175_ARL3', 'ENSG00000138180_CEP55', 'ENSG00000138182_KIF20B', 'ENSG00000138185_ENTPD1', 'ENSG00000138190_EXOC6', 'ENSG00000138193_PLCE1', 'ENSG00000138207_RBP4', 'ENSG00000138231_DBR1', 'ENSG00000138246_DNAJC13', 'ENSG00000138279_ANXA7', 'ENSG00000138286_FAM149B1', 'ENSG00000138303_ASCC1', 'ENSG00000138316_ADAMTS14', 'ENSG00000138326_RPS24', 'ENSG00000138336_TET1', 'ENSG00000138346_DNA2', 'ENSG00000138356_AOX1', 'ENSG00000138363_ATIC', 'ENSG00000138375_SMARCAL1', 'ENSG00000138376_BARD1', 'ENSG00000138378_STAT4', 'ENSG00000138380_CARF', 'ENSG00000138381_ASNSD1', 'ENSG00000138382_METTL5', 'ENSG00000138385_SSB', 'ENSG00000138386_NAB1', 'ENSG00000138395_CDK15', 'ENSG00000138398_PPIG', 'ENSG00000138399_FASTKD1', 'ENSG00000138400_MDH1B', 'ENSG00000138413_IDH1', 'ENSG00000138430_OLA1', 'ENSG00000138433_CIR1', 'ENSG00000138434_ITPRID2', 'ENSG00000138439_FAM117B', 'ENSG00000138442_WDR12', 'ENSG00000138443_ABI2', 'ENSG00000138448_ITGAV', 'ENSG00000138449_SLC40A1', 'ENSG00000138459_SLC35A5', 'ENSG00000138463_DIRC2', 'ENSG00000138468_SENP7', 'ENSG00000138495_COX17', 'ENSG00000138496_PARP9', 'ENSG00000138587_MNS1', 'ENSG00000138592_USP8', 'ENSG00000138593_SECISBP2L', 'ENSG00000138594_TMOD3', 'ENSG00000138600_SPPL2A', 'ENSG00000138604_GLCE', 'ENSG00000138606_SHF', 'ENSG00000138613_APH1B', 'ENSG00000138614_INTS14', 'ENSG00000138617_PARP16', 'ENSG00000138621_PPCDC', 'ENSG00000138623_SEMA7A', 'ENSG00000138629_UBL7', 'ENSG00000138639_ARHGAP24', 'ENSG00000138640_FAM13A', 'ENSG00000138641_HERC3', 'ENSG00000138642_HERC6', 'ENSG00000138646_HERC5', 'ENSG00000138658_ZGRF1', 'ENSG00000138660_AP1AR', 'ENSG00000138663_COPS4', 'ENSG00000138668_HNRNPD', 'ENSG00000138669_PRKG2', 'ENSG00000138670_RASGEF1B', 'ENSG00000138674_SEC31A', 'ENSG00000138678_GPAT3', 'ENSG00000138685_FGF2', 'ENSG00000138686_BBS7', 'ENSG00000138688_KIAA1109', 'ENSG00000138696_BMPR1B', 'ENSG00000138698_RAP1GDS1', 'ENSG00000138709_LARP1B', 'ENSG00000138722_MMRN1', 'ENSG00000138735_PDE5A', 'ENSG00000138738_PRDM5', 'ENSG00000138744_NAAA', 'ENSG00000138750_NUP54', 'ENSG00000138756_BMP2K', 'ENSG00000138757_G3BP2', 'ENSG00000138758_SEPT11', 'ENSG00000138760_SCARB2', 'ENSG00000138764_CCNG2', 'ENSG00000138767_CNOT6L', 'ENSG00000138768_USO1', 'ENSG00000138772_ANXA3', 'ENSG00000138777_PPA2', 'ENSG00000138778_CENPE', 'ENSG00000138780_GSTCD', 'ENSG00000138785_INTS12', 'ENSG00000138792_ENPEP', 'ENSG00000138794_CASP6', 'ENSG00000138795_LEF1', 'ENSG00000138796_HADH', 'ENSG00000138798_EGF', 'ENSG00000138801_PAPSS1', 'ENSG00000138802_SEC24B', 'ENSG00000138814_PPP3CA', 'ENSG00000138821_SLC39A8', 'ENSG00000138829_FBN2', 'ENSG00000138834_MAPK8IP3', 'ENSG00000138835_RGS3', 'ENSG00000138867_GUCD1', 'ENSG00000138942_RNF185', 'ENSG00000138964_PARVG', 'ENSG00000139044_B4GALNT3', 'ENSG00000139053_PDE6H', 'ENSG00000139055_ERP27', 'ENSG00000139083_ETV6', 'ENSG00000139112_GABARAPL1', 'ENSG00000139116_KIF21A', 'ENSG00000139117_CPNE8', 'ENSG00000139131_YARS2', 'ENSG00000139132_FGD4', 'ENSG00000139133_ALG10', 'ENSG00000139146_SINHCAF', 'ENSG00000139154_AEBP2', 'ENSG00000139160_ETFBKMT', 'ENSG00000139163_ETNK1', 'ENSG00000139168_ZCRB1', 'ENSG00000139173_TMEM117', 'ENSG00000139174_PRICKLE1', 'ENSG00000139178_C1RL', 'ENSG00000139180_NDUFA9', 'ENSG00000139182_CLSTN3', 'ENSG00000139187_KLRG1', 'ENSG00000139190_VAMP1', 'ENSG00000139192_TAPBPL', 'ENSG00000139193_CD27', 'ENSG00000139194_RBP5', 'ENSG00000139197_PEX5', 'ENSG00000139211_AMIGO2', 'ENSG00000139218_SCAF11', 'ENSG00000139219_COL2A1', 'ENSG00000139233_LLPH', 'ENSG00000139239_RPL14P1', 'ENSG00000139266_MARCH9', 'ENSG00000139269_INHBE', 'ENSG00000139278_GLIPR1', 'ENSG00000139289_PHLDA1', 'ENSG00000139291_TMEM19', 'ENSG00000139318_DUSP6', 'ENSG00000139323_POC1B', 'ENSG00000139324_TMTC3', 'ENSG00000139329_LUM', 'ENSG00000139343_SNRPF', 'ENSG00000139344_AMDHD1', 'ENSG00000139350_NEDD1', 'ENSG00000139351_SYCP3', 'ENSG00000139354_GAS2L3', 'ENSG00000139370_SLC15A4', 'ENSG00000139372_TDG', 'ENSG00000139405_RITA1', 'ENSG00000139410_SDSL', 'ENSG00000139428_MMAB', 'ENSG00000139433_GLTP', 'ENSG00000139436_GIT2', 'ENSG00000139437_TCHP', 'ENSG00000139438_FAM222A', 'ENSG00000139496_NUP58', 'ENSG00000139505_MTMR6', 'ENSG00000139508_SLC46A3', 'ENSG00000139514_SLC7A1', 'ENSG00000139517_LNX2', 'ENSG00000139531_SUOX', 'ENSG00000139537_CCDC65', 'ENSG00000139546_TARBP2', 'ENSG00000139567_ACVRL1', 'ENSG00000139572_GPR84', 'ENSG00000139579_NABP2', 'ENSG00000139597_N4BP2L1', 'ENSG00000139610_CELA1', 'ENSG00000139613_SMARCC2', 'ENSG00000139618_BRCA2', 'ENSG00000139620_KANSL2', 'ENSG00000139624_CERS5', 'ENSG00000139625_MAP3K12', 'ENSG00000139626_ITGB7', 'ENSG00000139629_GALNT6', 'ENSG00000139631_CSAD', 'ENSG00000139636_LMBR1L', 'ENSG00000139637_C12orf10', 'ENSG00000139641_ESYT1', 'ENSG00000139644_TMBIM6', 'ENSG00000139645_ANKRD52', 'ENSG00000139651_ZNF740', 'ENSG00000139668_WDFY2', 'ENSG00000139675_HNRNPA1L2', 'ENSG00000139679_LPAR6', 'ENSG00000139684_ESD', 'ENSG00000139687_RB1', 'ENSG00000139697_SBNO1', 'ENSG00000139714_MORN3', 'ENSG00000139718_SETD1B', 'ENSG00000139719_VPS33A', 'ENSG00000139722_VPS37B', 'ENSG00000139725_RHOF', 'ENSG00000139726_DENR', 'ENSG00000139734_DIAPH3', 'ENSG00000139737_SLAIN1', 'ENSG00000139746_RBM26', 'ENSG00000139793_MBNL2', 'ENSG00000139826_ABHD13', 'ENSG00000139832_RAB20', 'ENSG00000139835_GRTP1', 'ENSG00000139842_CUL4A', 'ENSG00000139880_CDH24', 'ENSG00000139890_REM2', 'ENSG00000139899_CBLN3', 'ENSG00000139910_NOVA1', 'ENSG00000139914_FITM1', 'ENSG00000139915_MDGA2', 'ENSG00000139921_TMX1', 'ENSG00000139926_FRMD6', 'ENSG00000139946_PELI2', 'ENSG00000139970_RTN1', 'ENSG00000139971_ARMH4', 'ENSG00000139974_SLC38A6', 'ENSG00000139977_NAA30', 'ENSG00000139985_ADAM21', 'ENSG00000139990_DCAF5', 'ENSG00000139998_RAB15', 'ENSG00000140006_WDR89', 'ENSG00000140009_ESR2', 'ENSG00000140022_STON2', 'ENSG00000140025_EFCAB11', 'ENSG00000140030_GPR65', 'ENSG00000140043_PTGR2', 'ENSG00000140044_JDP2', 'ENSG00000140057_AK7', 'ENSG00000140090_SLC24A4', 'ENSG00000140092_FBLN5', 'ENSG00000140104_CLBA1', 'ENSG00000140105_WARS', 'ENSG00000140153_WDR20', 'ENSG00000140157_NIPA2', 'ENSG00000140199_SLC12A6', 'ENSG00000140259_MFAP1', 'ENSG00000140262_TCF12', 'ENSG00000140263_SORD', 'ENSG00000140264_SERF2', 'ENSG00000140265_ZSCAN29', 'ENSG00000140280_LYSMD2', 'ENSG00000140284_SLC27A2', 'ENSG00000140285_FGF7', 'ENSG00000140287_HDC', 'ENSG00000140299_BNIP2', 'ENSG00000140307_GTF2A2', 'ENSG00000140319_SRP14', 'ENSG00000140320_BAHD1', 'ENSG00000140323_DISP2', 'ENSG00000140326_CDAN1', 'ENSG00000140332_TLE3', 'ENSG00000140350_ANP32A', 'ENSG00000140365_COMMD4', 'ENSG00000140367_UBE2Q2', 'ENSG00000140368_PSTPIP1', 'ENSG00000140374_ETFA', 'ENSG00000140379_BCL2A1', 'ENSG00000140382_HMG20A', 'ENSG00000140386_SCAPER', 'ENSG00000140391_TSPAN3', 'ENSG00000140395_WDR61', 'ENSG00000140396_NCOA2', 'ENSG00000140398_NEIL1', 'ENSG00000140400_MAN2C1', 'ENSG00000140403_DNAJA4', 'ENSG00000140406_TLNRD1', 'ENSG00000140416_TPM1', 'ENSG00000140443_IGF1R', 'ENSG00000140450_ARRDC4', 'ENSG00000140451_PIF1', 'ENSG00000140455_USP3', 'ENSG00000140459_CYP11A1', 'ENSG00000140463_BBS4', 'ENSG00000140464_PML', 'ENSG00000140465_CYP1A1', 'ENSG00000140471_LINS1', 'ENSG00000140474_ULK3', 'ENSG00000140479_PCSK6', 'ENSG00000140497_SCAMP2', 'ENSG00000140511_HAPLN3', 'ENSG00000140521_POLG', 'ENSG00000140525_FANCI', 'ENSG00000140526_ABHD2', 'ENSG00000140527_WDR93', 'ENSG00000140534_TICRR', 'ENSG00000140543_DET1', 'ENSG00000140545_MFGE8', 'ENSG00000140548_ZNF710', 'ENSG00000140553_UNC45A', 'ENSG00000140563_MCTP2', 'ENSG00000140564_FURIN', 'ENSG00000140575_IQGAP1', 'ENSG00000140577_CRTC3', 'ENSG00000140598_EFL1', 'ENSG00000140600_SH3GL3', 'ENSG00000140612_SEC11A', 'ENSG00000140632_GLYR1', 'ENSG00000140650_PMM2', 'ENSG00000140675_SLC5A2', 'ENSG00000140678_ITGAX', 'ENSG00000140682_TGFB1I1', 'ENSG00000140688_C16orf58', 'ENSG00000140691_ARMC5', 'ENSG00000140694_PARN', 'ENSG00000140718_FTO', 'ENSG00000140740_UQCRC2', 'ENSG00000140743_CDR2', 'ENSG00000140749_IGSF6', 'ENSG00000140750_ARHGAP17', 'ENSG00000140795_MYLK3', 'ENSG00000140807_NKD1', 'ENSG00000140829_DHX38', 'ENSG00000140830_TXNL4B', 'ENSG00000140835_CHST4', 'ENSG00000140836_ZFHX3', 'ENSG00000140848_CPNE2', 'ENSG00000140853_NLRC5', 'ENSG00000140854_KATNB1', 'ENSG00000140859_KIFC3', 'ENSG00000140876_NUDT7', 'ENSG00000140905_GCSH', 'ENSG00000140931_CMTM3', 'ENSG00000140932_CMTM2', 'ENSG00000140939_NOL3', 'ENSG00000140941_MAP1LC3B', 'ENSG00000140943_MBTPS1', 'ENSG00000140948_ZCCHC14', 'ENSG00000140950_TLDC1', 'ENSG00000140961_OSGIN1', 'ENSG00000140968_IRF8', 'ENSG00000140983_RHOT2', 'ENSG00000140987_ZSCAN32', 'ENSG00000140988_RPS2', 'ENSG00000140990_NDUFB10', 'ENSG00000140992_PDPK1', 'ENSG00000140993_TIGD7', 'ENSG00000140995_DEF8', 'ENSG00000141002_TCF25', 'ENSG00000141012_GALNS', 'ENSG00000141013_GAS8', 'ENSG00000141026_MED9', 'ENSG00000141027_NCOR1', 'ENSG00000141030_COPS3', 'ENSG00000141034_GID4', 'ENSG00000141040_ZNF287', 'ENSG00000141068_KSR1', 'ENSG00000141076_UTP4', 'ENSG00000141084_RANBP10', 'ENSG00000141086_CTRL', 'ENSG00000141096_DPEP3', 'ENSG00000141098_GFOD2', 'ENSG00000141101_NOB1', 'ENSG00000141127_PRPSAP2', 'ENSG00000141179_PCTP', 'ENSG00000141194_OR4D1', 'ENSG00000141198_TOM1L1', 'ENSG00000141219_C17orf80', 'ENSG00000141232_TOB1', 'ENSG00000141252_VPS53', 'ENSG00000141258_SGSM2', 'ENSG00000141279_NPEPPS', 'ENSG00000141293_SKAP1', 'ENSG00000141294_LRRC46', 'ENSG00000141295_SCRN2', 'ENSG00000141298_SSH2', 'ENSG00000141337_ARSG', 'ENSG00000141349_G6PC3', 'ENSG00000141367_CLTC', 'ENSG00000141371_C17orf64', 'ENSG00000141376_BCAS3', 'ENSG00000141378_PTRH2', 'ENSG00000141380_SS18', 'ENSG00000141384_TAF4B', 'ENSG00000141385_AFG3L2', 'ENSG00000141391_PRELID3A', 'ENSG00000141401_IMPA2', 'ENSG00000141404_GNAL', 'ENSG00000141424_SLC39A6', 'ENSG00000141425_RPRD1A', 'ENSG00000141428_C18orf21', 'ENSG00000141429_GALNT1', 'ENSG00000141431_ASXL3', 'ENSG00000141433_ADCYAP1', 'ENSG00000141441_GAREM1', 'ENSG00000141446_ESCO1', 'ENSG00000141447_OSBPL1A', 'ENSG00000141449_GREB1L', 'ENSG00000141452_RMC1', 'ENSG00000141456_PELP1', 'ENSG00000141458_NPC1', 'ENSG00000141469_SLC14A1', 'ENSG00000141480_ARRB2', 'ENSG00000141499_WRAP53', 'ENSG00000141503_MINK1', 'ENSG00000141504_SAT2', 'ENSG00000141505_ASGR1', 'ENSG00000141506_PIK3R5', 'ENSG00000141510_TP53', 'ENSG00000141519_CCDC40', 'ENSG00000141522_ARHGDIA', 'ENSG00000141524_TMC6', 'ENSG00000141526_SLC16A3', 'ENSG00000141540_TTYH2', 'ENSG00000141542_RAB40B', 'ENSG00000141543_EIF4A3', 'ENSG00000141551_CSNK1D', 'ENSG00000141552_ANAPC11', 'ENSG00000141556_TBCD', 'ENSG00000141560_FN3KRP', 'ENSG00000141562_NARF', 'ENSG00000141564_RPTOR', 'ENSG00000141568_FOXK2', 'ENSG00000141569_TRIM65', 'ENSG00000141570_CBX8', 'ENSG00000141574_SECTM1', 'ENSG00000141576_RNF157', 'ENSG00000141577_CEP131', 'ENSG00000141580_WDR45B', 'ENSG00000141582_CBX4', 'ENSG00000141622_RNF165', 'ENSG00000141627_DYM', 'ENSG00000141642_ELAC1', 'ENSG00000141644_MBD1', 'ENSG00000141646_SMAD4', 'ENSG00000141655_TNFRSF11A', 'ENSG00000141664_ZCCHC2', 'ENSG00000141665_FBXO15', 'ENSG00000141682_PMAIP1', 'ENSG00000141696_P3H4', 'ENSG00000141698_NT5C3B', 'ENSG00000141699_RETREG3', 'ENSG00000141736_ERBB2', 'ENSG00000141741_MIEN1', 'ENSG00000141744_PNMT', 'ENSG00000141750_STAC2', 'ENSG00000141753_IGFBP4', 'ENSG00000141756_FKBP10', 'ENSG00000141759_TXNL4A', 'ENSG00000141837_CACNA1A', 'ENSG00000141854_MISP3', 'ENSG00000141858_SAMD1', 'ENSG00000141867_BRD4', 'ENSG00000141873_SLC39A3', 'ENSG00000141905_NFIC', 'ENSG00000141933_TPGS1', 'ENSG00000141934_PLPP2', 'ENSG00000141956_PRDM15', 'ENSG00000141959_PFKL', 'ENSG00000141965_FEM1A', 'ENSG00000141968_VAV1', 'ENSG00000141971_MVB12A', 'ENSG00000141977_CIB3', 'ENSG00000141985_SH3GL1', 'ENSG00000141994_DUS3L', 'ENSG00000142002_DPP9', 'ENSG00000142039_CCDC97', 'ENSG00000142046_TMEM91', 'ENSG00000142065_ZFP14', 'ENSG00000142082_SIRT3', 'ENSG00000142089_IFITM3', 'ENSG00000142102_PGGHG', 'ENSG00000142156_COL6A1', 'ENSG00000142166_IFNAR1', 'ENSG00000142168_SOD1', 'ENSG00000142173_COL6A2', 'ENSG00000142185_TRPM2', 'ENSG00000142186_SCYL1', 'ENSG00000142188_TMEM50B', 'ENSG00000142192_APP', 'ENSG00000142197_DOP1B', 'ENSG00000142207_URB1', 'ENSG00000142208_AKT1', 'ENSG00000142227_EMP3', 'ENSG00000142230_SAE1', 'ENSG00000142233_NTN5', 'ENSG00000142235_LMTK3', 'ENSG00000142252_GEMIN7', 'ENSG00000142279_WTIP', 'ENSG00000142303_ADAMTS10', 'ENSG00000142327_RNPEPL1', 'ENSG00000142330_CAPN10', 'ENSG00000142347_MYO1F', 'ENSG00000142396_ERVK3-1', 'ENSG00000142405_NLRP12', 'ENSG00000142408_CACNG8', 'ENSG00000142409_ZNF787', 'ENSG00000142444_TIMM29', 'ENSG00000142453_CARM1', 'ENSG00000142459_EVI5L', 'ENSG00000142494_SLC47A1', 'ENSG00000142507_PSMB6', 'ENSG00000142512_SIGLEC10', 'ENSG00000142528_ZNF473', 'ENSG00000142530_FAM71E1', 'ENSG00000142534_RPS11', 'ENSG00000142541_RPL13A', 'ENSG00000142544_CTU1', 'ENSG00000142546_NOSIP', 'ENSG00000142549_IGLON5', 'ENSG00000142552_RCN3', 'ENSG00000142556_ZNF614', 'ENSG00000142583_SLC2A5', 'ENSG00000142599_RERE', 'ENSG00000142606_MMEL1', 'ENSG00000142609_CFAP74', 'ENSG00000142611_PRDM16', 'ENSG00000142619_PADI3', 'ENSG00000142632_ARHGEF19', 'ENSG00000142634_EFHD2', 'ENSG00000142655_PEX14', 'ENSG00000142657_PGD', 'ENSG00000142669_SH3BGRL3', 'ENSG00000142675_CNKSR1', 'ENSG00000142676_RPL11', 'ENSG00000142684_ZNF593', 'ENSG00000142686_C1orf216', 'ENSG00000142687_KIAA0319L', 'ENSG00000142694_EVA1B', 'ENSG00000142731_PLK4', 'ENSG00000142733_MAP3K6', 'ENSG00000142751_GPN2', 'ENSG00000142765_SYTL1', 'ENSG00000142784_WDTC1', 'ENSG00000142794_NBPF3', 'ENSG00000142798_HSPG2', 'ENSG00000142856_ITGB3BP', 'ENSG00000142864_SERBP1', 'ENSG00000142867_BCL10', 'ENSG00000142871_CYR61', 'ENSG00000142875_PRKACB', 'ENSG00000142892_PIGK', 'ENSG00000142920_AZIN2', 'ENSG00000142937_RPS8', 'ENSG00000142945_KIF2C', 'ENSG00000142949_PTPRF', 'ENSG00000142959_BEST4', 'ENSG00000142961_MOB3C', 'ENSG00000143013_LMO4', 'ENSG00000143033_MTF2', 'ENSG00000143061_IGSF3', 'ENSG00000143067_ZNF697', 'ENSG00000143079_CTTNBP2NL', 'ENSG00000143093_STRIP1', 'ENSG00000143106_PSMA5', 'ENSG00000143110_C1orf162', 'ENSG00000143119_CD53', 'ENSG00000143126_CELSR2', 'ENSG00000143127_ITGA10', 'ENSG00000143147_GPR161', 'ENSG00000143149_ALDH9A1', 'ENSG00000143153_ATP1B1', 'ENSG00000143155_TIPRL', 'ENSG00000143156_NME7', 'ENSG00000143157_POGK', 'ENSG00000143158_MPC2', 'ENSG00000143162_CREG1', 'ENSG00000143164_DCAF6', 'ENSG00000143167_GPA33', 'ENSG00000143178_TBX19', 'ENSG00000143179_UCK2', 'ENSG00000143183_TMCO1', 'ENSG00000143190_POU2F1', 'ENSG00000143195_ILDR2', 'ENSG00000143198_MGST3', 'ENSG00000143207_COP1', 'ENSG00000143222_UFC1', 'ENSG00000143224_PPOX', 'ENSG00000143226_FCGR2A', 'ENSG00000143228_NUF2', 'ENSG00000143248_RGS5', 'ENSG00000143252_SDHC', 'ENSG00000143256_PFDN2', 'ENSG00000143257_NR1I3', 'ENSG00000143258_USP21', 'ENSG00000143294_PRCC', 'ENSG00000143303_RRNAD1', 'ENSG00000143314_MRPL24', 'ENSG00000143315_PIGM', 'ENSG00000143319_ISG20L2', 'ENSG00000143321_HDGF', 'ENSG00000143322_ABL2', 'ENSG00000143324_XPR1', 'ENSG00000143333_RGS16', 'ENSG00000143337_TOR1AIP1', 'ENSG00000143344_RGL1', 'ENSG00000143353_LYPLAL1', 'ENSG00000143363_PRUNE1', 'ENSG00000143367_TUFT1', 'ENSG00000143368_SF3B4', 'ENSG00000143369_ECM1', 'ENSG00000143373_ZNF687', 'ENSG00000143374_TARS2', 'ENSG00000143375_CGN', 'ENSG00000143376_SNX27', 'ENSG00000143379_SETDB1', 'ENSG00000143382_ADAMTSL4', 'ENSG00000143384_MCL1', 'ENSG00000143387_CTSK', 'ENSG00000143390_RFX5', 'ENSG00000143393_PI4KB', 'ENSG00000143398_PIP5K1A', 'ENSG00000143401_ANP32E', 'ENSG00000143409_MINDY1', 'ENSG00000143412_ANXA9', 'ENSG00000143416_SELENBP1', 'ENSG00000143418_CERS2', 'ENSG00000143420_ENSA', 'ENSG00000143429_LSP1P4', 'ENSG00000143434_SEMA6C', 'ENSG00000143436_MRPL9', 'ENSG00000143437_ARNT', 'ENSG00000143442_POGZ', 'ENSG00000143443_C1orf56', 'ENSG00000143450_OAZ3', 'ENSG00000143457_GOLPH3L', 'ENSG00000143458_GABPB2', 'ENSG00000143473_KCNH1', 'ENSG00000143476_DTL', 'ENSG00000143479_DYRK3', 'ENSG00000143486_EIF2D', 'ENSG00000143493_INTS7', 'ENSG00000143494_VASH2', 'ENSG00000143498_TAF1A', 'ENSG00000143499_SMYD2', 'ENSG00000143507_DUSP10', 'ENSG00000143514_TP53BP2', 'ENSG00000143515_ATP8B2', 'ENSG00000143537_ADAM15', 'ENSG00000143543_JTB', 'ENSG00000143545_RAB13', 'ENSG00000143546_S100A8', 'ENSG00000143549_TPM3', 'ENSG00000143553_SNAPIN', 'ENSG00000143554_SLC27A3', 'ENSG00000143569_UBAP2L', 'ENSG00000143570_SLC39A1', 'ENSG00000143575_HAX1', 'ENSG00000143578_CREB3L4', 'ENSG00000143590_EFNA3', 'ENSG00000143595_AQP10', 'ENSG00000143603_KCNN3', 'ENSG00000143612_C1orf43', 'ENSG00000143614_GATAD2B', 'ENSG00000143621_ILF2', 'ENSG00000143622_RIT1', 'ENSG00000143624_INTS3', 'ENSG00000143627_PKLR', 'ENSG00000143630_HCN3', 'ENSG00000143633_C1orf131', 'ENSG00000143641_GALNT2', 'ENSG00000143643_TTC13', 'ENSG00000143653_SCCPDH', 'ENSG00000143669_LYST', 'ENSG00000143674_MAP3K21', 'ENSG00000143702_CEP170', 'ENSG00000143727_ACP1', 'ENSG00000143740_SNAP47', 'ENSG00000143742_SRP9', 'ENSG00000143748_NVL', 'ENSG00000143751_SDE2', 'ENSG00000143753_DEGS1', 'ENSG00000143756_FBXO28', 'ENSG00000143761_ARF1', 'ENSG00000143771_CNIH4', 'ENSG00000143772_ITPKB', 'ENSG00000143774_GUK1', 'ENSG00000143776_CDC42BPA', 'ENSG00000143786_CNIH3', 'ENSG00000143793_C1orf35', 'ENSG00000143797_MBOAT2', 'ENSG00000143799_PARP1', 'ENSG00000143801_PSEN2', 'ENSG00000143811_PYCR2', 'ENSG00000143815_LBR', 'ENSG00000143819_EPHX1', 'ENSG00000143842_SOX13', 'ENSG00000143845_ETNK2', 'ENSG00000143847_PPFIA4', 'ENSG00000143850_PLEKHA6', 'ENSG00000143851_PTPN7', 'ENSG00000143862_ARL8A', 'ENSG00000143869_GDF7', 'ENSG00000143870_PDIA6', 'ENSG00000143878_RHOB', 'ENSG00000143882_ATP6V1C2', 'ENSG00000143889_HNRNPLL', 'ENSG00000143891_GALM', 'ENSG00000143919_CAMKMT', 'ENSG00000143921_ABCG8', 'ENSG00000143924_EML4', 'ENSG00000143933_CALM2', 'ENSG00000143942_CHAC2', 'ENSG00000143947_RPS27A', 'ENSG00000143951_WDPCP', 'ENSG00000143952_VPS54', 'ENSG00000143970_ASXL2', 'ENSG00000143971_ETAA1', 'ENSG00000143977_SNRPG', 'ENSG00000143994_ABHD1', 'ENSG00000143995_MEIS1', 'ENSG00000144021_CIAO1', 'ENSG00000144026_ZNF514', 'ENSG00000144028_SNRNP200', 'ENSG00000144029_MRPS5', 'ENSG00000144034_TPRKB', 'ENSG00000144036_EXOC6B', 'ENSG00000144040_SFXN5', 'ENSG00000144043_TEX261', 'ENSG00000144045_DQX1', 'ENSG00000144048_DUSP11', 'ENSG00000144057_ST6GAL2', 'ENSG00000144061_NPHP1', 'ENSG00000144063_MALL', 'ENSG00000144115_THNSL2', 'ENSG00000144118_RALB', 'ENSG00000144120_TMEM177', 'ENSG00000144134_RABL2A', 'ENSG00000144136_SLC20A1', 'ENSG00000144161_ZC3H8', 'ENSG00000144182_LIPT1', 'ENSG00000144199_FAHD2B', 'ENSG00000144218_AFF3', 'ENSG00000144224_UBXN4', 'ENSG00000144228_SPOPL', 'ENSG00000144231_POLR2D', 'ENSG00000144233_AMMECR1L', 'ENSG00000144278_GALNT13', 'ENSG00000144283_PKP4', 'ENSG00000144306_SCRN3', 'ENSG00000144320_LNPK', 'ENSG00000144354_CDCA7', 'ENSG00000144355_DLX1', 'ENSG00000144357_UBR3', 'ENSG00000144362_PHOSPHO2', 'ENSG00000144369_FAM171B', 'ENSG00000144381_HSPD1', 'ENSG00000144395_CCDC150', 'ENSG00000144401_METTL21A', 'ENSG00000144406_UNC80', 'ENSG00000144407_PTH2R', 'ENSG00000144426_NBEAL1', 'ENSG00000144445_KANSL1L', 'ENSG00000144451_SPAG16', 'ENSG00000144455_SUMF1', 'ENSG00000144468_RHBDD1', 'ENSG00000144476_ACKR3', 'ENSG00000144485_HES6', 'ENSG00000144488_ESPNL', 'ENSG00000144504_ANKMY1', 'ENSG00000144524_COPS7B', 'ENSG00000144535_DIS3L2', 'ENSG00000144550_CPNE9', 'ENSG00000144554_FANCD2', 'ENSG00000144559_TAMM41', 'ENSG00000144560_VGLL4', 'ENSG00000144566_RAB5A', 'ENSG00000144567_RETREG2', 'ENSG00000144579_CTDSP1', 'ENSG00000144580_CNOT9', 'ENSG00000144589_STK11IP', 'ENSG00000144591_GMPPA', 'ENSG00000144596_GRIP2', 'ENSG00000144597_EAF1', 'ENSG00000144635_DYNC1LI1', 'ENSG00000144645_OSBPL10', 'ENSG00000144647_POMGNT2', 'ENSG00000144648_ACKR2', 'ENSG00000144649_FAM198A', 'ENSG00000144655_CSRNP1', 'ENSG00000144659_SLC25A38', 'ENSG00000144668_ITGA9', 'ENSG00000144674_GOLGA4', 'ENSG00000144677_CTDSPL', 'ENSG00000144681_STAC', 'ENSG00000144711_IQSEC1', 'ENSG00000144712_CAND2', 'ENSG00000144713_RPL32', 'ENSG00000144724_PTPRG', 'ENSG00000144736_SHQ1', 'ENSG00000144741_SLC25A26', 'ENSG00000144744_UBA3', 'ENSG00000144746_ARL6IP5', 'ENSG00000144747_TMF1', 'ENSG00000144749_LRIG1', 'ENSG00000144791_LIMD1', 'ENSG00000144792_ZNF660', 'ENSG00000144802_NFKBIZ', 'ENSG00000144810_COL8A1', 'ENSG00000144815_NXPE3', 'ENSG00000144821_MYH15', 'ENSG00000144824_PHLDB2', 'ENSG00000144827_ABHD10', 'ENSG00000144840_RABL3', 'ENSG00000144843_ADPRH', 'ENSG00000144848_ATG3', 'ENSG00000144852_NR1I2', 'ENSG00000144867_SRPRB', 'ENSG00000144868_TMEM108', 'ENSG00000144893_MED12L', 'ENSG00000144895_EIF2A', 'ENSG00000144909_OSBPL11', 'ENSG00000144935_TRPC1', 'ENSG00000144959_NCEH1', 'ENSG00000145012_LPP', 'ENSG00000145014_TMEM44', 'ENSG00000145016_RUBCN', 'ENSG00000145020_AMT', 'ENSG00000145022_TCTA', 'ENSG00000145041_DCAF1', 'ENSG00000145050_MANF', 'ENSG00000145088_EAF2', 'ENSG00000145191_EIF2B5', 'ENSG00000145214_DGKQ', 'ENSG00000145216_FIP1L1', 'ENSG00000145217_SLC26A1', 'ENSG00000145220_LYAR', 'ENSG00000145241_CENPC', 'ENSG00000145246_ATP10D', 'ENSG00000145247_OCIAD2', 'ENSG00000145248_SLC10A4', 'ENSG00000145284_SCD5', 'ENSG00000145287_PLAC8', 'ENSG00000145293_ENOPH1', 'ENSG00000145331_TRMT10A', 'ENSG00000145332_KLHL8', 'ENSG00000145335_SNCA', 'ENSG00000145337_PYURF', 'ENSG00000145348_TBCK', 'ENSG00000145349_CAMK2D', 'ENSG00000145354_CISD2', 'ENSG00000145362_ANK2', 'ENSG00000145365_TIFA', 'ENSG00000145375_SPATA5', 'ENSG00000145386_CCNA2', 'ENSG00000145388_METTL14', 'ENSG00000145390_USP53', 'ENSG00000145391_SETD7', 'ENSG00000145414_NAF1', 'ENSG00000145416_MARCH1', 'ENSG00000145425_RPS3A', 'ENSG00000145431_PDGFC', 'ENSG00000145439_CBR4', 'ENSG00000145476_CYP4V2', 'ENSG00000145491_ROPN1L', 'ENSG00000145494_NDUFS6', 'ENSG00000145495_MARCH6', 'ENSG00000145506_NKD2', 'ENSG00000145545_SRD5A1', 'ENSG00000145555_MYO10', 'ENSG00000145569_OTULINL', 'ENSG00000145592_RPL37', 'ENSG00000145604_SKP2', 'ENSG00000145632_PLK2', 'ENSG00000145649_GZMA', 'ENSG00000145675_PIK3R1', 'ENSG00000145685_LHFPL2', 'ENSG00000145687_SSBP2', 'ENSG00000145700_ANKRD31', 'ENSG00000145703_IQGAP2', 'ENSG00000145708_CRHBP', 'ENSG00000145715_RASA1', 'ENSG00000145723_GIN1', 'ENSG00000145725_PPIP5K2', 'ENSG00000145730_PAM', 'ENSG00000145734_BDP1', 'ENSG00000145736_GTF2H2', 'ENSG00000145740_SLC30A5', 'ENSG00000145741_BTF3', 'ENSG00000145743_FBXL17', 'ENSG00000145757_SPATA9', 'ENSG00000145777_TSLP', 'ENSG00000145779_TNFAIP8', 'ENSG00000145780_FEM1C', 'ENSG00000145781_COMMD10', 'ENSG00000145782_ATG12', 'ENSG00000145817_YIPF5', 'ENSG00000145819_ARHGAP26', 'ENSG00000145824_CXCL14', 'ENSG00000145832_SLC25A48', 'ENSG00000145833_DDX46', 'ENSG00000145860_RNF145', 'ENSG00000145868_FBXO38', 'ENSG00000145882_PCYOX1L', 'ENSG00000145901_TNIP1', 'ENSG00000145907_G3BP1', 'ENSG00000145908_ZNF300', 'ENSG00000145911_N4BP3', 'ENSG00000145912_NHP2', 'ENSG00000145916_RMND5B', 'ENSG00000145919_BOD1', 'ENSG00000145936_KCNMB1', 'ENSG00000145945_FAM50B', 'ENSG00000145949_MYLK4', 'ENSG00000145979_TBC1D7', 'ENSG00000145982_FARS2', 'ENSG00000145990_GFOD1', 'ENSG00000145996_CDKAL1', 'ENSG00000146006_LRRTM2', 'ENSG00000146007_ZMAT2', 'ENSG00000146013_GFRA3', 'ENSG00000146021_KLHL3', 'ENSG00000146054_TRIM7', 'ENSG00000146063_TRIM41', 'ENSG00000146066_HIGD2A', 'ENSG00000146067_FAM193B', 'ENSG00000146070_PLA2G7', 'ENSG00000146072_TNFRSF21', 'ENSG00000146083_RNF44', 'ENSG00000146085_MUT', 'ENSG00000146094_DOK3', 'ENSG00000146109_ABT1', 'ENSG00000146112_PPP1R18', 'ENSG00000146143_PRIM2', 'ENSG00000146192_FGD2', 'ENSG00000146205_ANO7', 'ENSG00000146215_CRIP3', 'ENSG00000146223_RPL7L1', 'ENSG00000146232_NFKBIE', 'ENSG00000146242_TPBG', 'ENSG00000146243_IRAK1BP1', 'ENSG00000146247_PHIP', 'ENSG00000146263_MMS22L', 'ENSG00000146267_FAXC', 'ENSG00000146278_PNRC1', 'ENSG00000146281_PM20D2', 'ENSG00000146282_RARS2', 'ENSG00000146285_SCML4', 'ENSG00000146350_TBC1D32', 'ENSG00000146373_RNF217', 'ENSG00000146376_ARHGAP18', 'ENSG00000146386_ABRACL', 'ENSG00000146409_SLC18B1', 'ENSG00000146410_MTFR2', 'ENSG00000146414_SHPRH', 'ENSG00000146416_AIG1', 'ENSG00000146425_DYNLT1', 'ENSG00000146426_TIAM2', 'ENSG00000146433_TMEM181', 'ENSG00000146453_PNLDC1', 'ENSG00000146457_WTAP', 'ENSG00000146463_ZMYM4', 'ENSG00000146476_ARMT1', 'ENSG00000146530_VWDE', 'ENSG00000146535_GNA12', 'ENSG00000146540_C7orf50', 'ENSG00000146555_SDK1', 'ENSG00000146556_WASH2P', 'ENSG00000146574_CCZ1B', 'ENSG00000146576_C7orf26', 'ENSG00000146587_RBAK', 'ENSG00000146592_CREB5', 'ENSG00000146670_CDCA5', 'ENSG00000146676_PURB', 'ENSG00000146677_AC004453.1', 'ENSG00000146700_SSC4D', 'ENSG00000146701_MDH2', 'ENSG00000146707_POMZP3', 'ENSG00000146722_AC211486.1', 'ENSG00000146729_NIPSNAP2', 'ENSG00000146731_CCT6A', 'ENSG00000146733_PSPH', 'ENSG00000146757_ZNF92', 'ENSG00000146776_ATXN7L1', 'ENSG00000146802_TMEM168', 'ENSG00000146826_C7orf43', 'ENSG00000146828_SLC12A9', 'ENSG00000146830_GIGYF1', 'ENSG00000146833_TRIM4', 'ENSG00000146834_MEPCE', 'ENSG00000146839_ZAN', 'ENSG00000146842_TMEM209', 'ENSG00000146856_AGBL3', 'ENSG00000146858_ZC3HAV1L', 'ENSG00000146859_TMEM140', 'ENSG00000146872_TLK2', 'ENSG00000146904_EPHA1', 'ENSG00000146909_NOM1', 'ENSG00000146918_NCAPG2', 'ENSG00000146963_LUC7L2', 'ENSG00000147003_CLTRN', 'ENSG00000147010_SH3KBP1', 'ENSG00000147036_LANCL3', 'ENSG00000147044_CASK', 'ENSG00000147050_KDM6A', 'ENSG00000147059_SPIN2A', 'ENSG00000147065_MSN', 'ENSG00000147099_HDAC8', 'ENSG00000147100_SLC16A2', 'ENSG00000147117_ZNF157', 'ENSG00000147118_ZNF182', 'ENSG00000147119_CHST7', 'ENSG00000147121_KRBOX4', 'ENSG00000147123_NDUFB11', 'ENSG00000147124_ZNF41', 'ENSG00000147130_ZMYM3', 'ENSG00000147133_TAF1', 'ENSG00000147138_GPR174', 'ENSG00000147140_NONO', 'ENSG00000147144_CCDC120', 'ENSG00000147145_LPAR4', 'ENSG00000147155_EBP', 'ENSG00000147162_OGT', 'ENSG00000147164_SNX12', 'ENSG00000147166_ITGB1BP2', 'ENSG00000147168_IL2RG', 'ENSG00000147174_GCNA', 'ENSG00000147180_ZNF711', 'ENSG00000147202_DIAPH2', 'ENSG00000147206_NXF3', 'ENSG00000147224_PRPS1', 'ENSG00000147231_CXorf57', 'ENSG00000147251_DOCK11', 'ENSG00000147257_GPC3', 'ENSG00000147274_RBMX', 'ENSG00000147316_MCPH1', 'ENSG00000147324_MFHAS1', 'ENSG00000147364_FBXO25', 'ENSG00000147383_NSDHL', 'ENSG00000147394_ZNF185', 'ENSG00000147400_CETN2', 'ENSG00000147403_RPL10', 'ENSG00000147408_CSGALNACT1', 'ENSG00000147416_ATP6V1B2', 'ENSG00000147419_CCDC25', 'ENSG00000147421_HMBOX1', 'ENSG00000147434_CHRNA6', 'ENSG00000147437_GNRH1', 'ENSG00000147439_BIN3', 'ENSG00000147443_DOK2', 'ENSG00000147454_SLC25A37', 'ENSG00000147457_CHMP7', 'ENSG00000147459_DOCK5', 'ENSG00000147465_STAR', 'ENSG00000147471_PLPBP', 'ENSG00000147475_ERLIN2', 'ENSG00000147488_ST18', 'ENSG00000147509_RGS20', 'ENSG00000147526_TACC1', 'ENSG00000147533_GOLGA7', 'ENSG00000147535_PLPP5', 'ENSG00000147536_GINS4', 'ENSG00000147548_NSD3', 'ENSG00000147570_DNAJC5B', 'ENSG00000147576_ADHFE1', 'ENSG00000147586_MRPS28', 'ENSG00000147592_LACTB2', 'ENSG00000147601_TERF1', 'ENSG00000147604_RPL7', 'ENSG00000147649_MTDH', 'ENSG00000147650_LRP12', 'ENSG00000147654_EBAG9', 'ENSG00000147669_POLR2K', 'ENSG00000147677_EIF3H', 'ENSG00000147679_UTP23', 'ENSG00000147684_NDUFB9', 'ENSG00000147687_TATDN1', 'ENSG00000147689_FAM83A', 'ENSG00000147789_ZNF7', 'ENSG00000147799_ARHGAP39', 'ENSG00000147804_SLC39A4', 'ENSG00000147813_NAPRT', 'ENSG00000147852_VLDLR', 'ENSG00000147853_AK3', 'ENSG00000147854_UHRF2', 'ENSG00000147862_NFIB', 'ENSG00000147872_PLIN2', 'ENSG00000147874_HAUS6', 'ENSG00000147883_CDKN2B', 'ENSG00000147889_CDKN2A', 'ENSG00000147894_C9orf72', 'ENSG00000147905_ZCCHC7', 'ENSG00000147912_FBXO10', 'ENSG00000147955_SIGMAR1', 'ENSG00000147996_CBWD5', 'ENSG00000148019_CEP78', 'ENSG00000148057_IDNK', 'ENSG00000148090_AUH', 'ENSG00000148110_MFSD14B', 'ENSG00000148120_C9orf3', 'ENSG00000148143_ZNF462', 'ENSG00000148153_INIP', 'ENSG00000148154_UGCG', 'ENSG00000148158_SNX30', 'ENSG00000148175_STOM', 'ENSG00000148180_GSN', 'ENSG00000148187_MRRF', 'ENSG00000148200_NR6A1', 'ENSG00000148204_CRB2', 'ENSG00000148218_ALAD', 'ENSG00000148219_ASTN2', 'ENSG00000148225_WDR31', 'ENSG00000148229_POLE3', 'ENSG00000148248_SURF4', 'ENSG00000148288_GBGT1', 'ENSG00000148290_SURF1', 'ENSG00000148291_SURF2', 'ENSG00000148296_SURF6', 'ENSG00000148297_MED22', 'ENSG00000148300_REXO4', 'ENSG00000148303_RPL7A', 'ENSG00000148308_GTF3C5', 'ENSG00000148331_ASB6', 'ENSG00000148334_PTGES2', 'ENSG00000148335_NTMT1', 'ENSG00000148337_CIZ1', 'ENSG00000148339_SLC25A25', 'ENSG00000148341_SH3GLB2', 'ENSG00000148343_MIGA2', 'ENSG00000148346_LCN2', 'ENSG00000148356_LRSAM1', 'ENSG00000148358_GPR107', 'ENSG00000148362_PAXX', 'ENSG00000148377_IDI2', 'ENSG00000148384_INPP5E', 'ENSG00000148396_SEC16A', 'ENSG00000148399_DPH7', 'ENSG00000148400_NOTCH1', 'ENSG00000148411_NACC2', 'ENSG00000148426_PROSER2', 'ENSG00000148429_USP6NL', 'ENSG00000148444_COMMD3', 'ENSG00000148450_MSRB2', 'ENSG00000148459_PDSS1', 'ENSG00000148468_FAM171A1', 'ENSG00000148481_MINDY3', 'ENSG00000148483_TMEM236', 'ENSG00000148484_RSU1', 'ENSG00000148488_ST8SIA6', 'ENSG00000148498_PARD3', 'ENSG00000148516_ZEB1', 'ENSG00000148572_NRBF2', 'ENSG00000148600_CDHR1', 'ENSG00000148606_POLR3A', 'ENSG00000148634_HERC4', 'ENSG00000148655_LRMDA', 'ENSG00000148660_CAMK2G', 'ENSG00000148672_GLUD1', 'ENSG00000148680_HTR7', 'ENSG00000148688_RPP30', 'ENSG00000148690_FRA10AC1', 'ENSG00000148700_ADD3', 'ENSG00000148719_DNAJB12', 'ENSG00000148730_EIF4EBP2', 'ENSG00000148737_TCF7L2', 'ENSG00000148773_MKI67', 'ENSG00000148798_INA', 'ENSG00000148803_FUOM', 'ENSG00000148814_LRRC27', 'ENSG00000148824_MTG1', 'ENSG00000148832_PAOX', 'ENSG00000148834_GSTO1', 'ENSG00000148835_TAF5', 'ENSG00000148840_PPRC1', 'ENSG00000148841_ITPRIP', 'ENSG00000148842_CNNM2', 'ENSG00000148843_PDCD11', 'ENSG00000148848_ADAM12', 'ENSG00000148908_RGS10', 'ENSG00000148925_BTBD10', 'ENSG00000148926_ADM', 'ENSG00000148935_GAS2', 'ENSG00000148943_LIN7C', 'ENSG00000148948_LRRC4C', 'ENSG00000148950_IMMP1L', 'ENSG00000148985_PGAP2', 'ENSG00000149016_TUT1', 'ENSG00000149050_ZNF214', 'ENSG00000149054_ZNF215', 'ENSG00000149084_HSD17B12', 'ENSG00000149089_APIP', 'ENSG00000149091_DGKZ', 'ENSG00000149100_EIF3M', 'ENSG00000149115_TNKS1BP1', 'ENSG00000149131_SERPING1', 'ENSG00000149136_SSRP1', 'ENSG00000149150_SLC43A1', 'ENSG00000149177_PTPRJ', 'ENSG00000149179_C11orf49', 'ENSG00000149182_ARFGAP2', 'ENSG00000149187_CELF1', 'ENSG00000149196_HIKESHI', 'ENSG00000149201_CCDC81', 'ENSG00000149212_SESN3', 'ENSG00000149218_ENDOD1', 'ENSG00000149231_CCDC82', 'ENSG00000149243_KLHL35', 'ENSG00000149257_SERPINH1', 'ENSG00000149260_CAPN5', 'ENSG00000149262_INTS4', 'ENSG00000149269_PAK1', 'ENSG00000149273_RPS3', 'ENSG00000149289_ZC3H12C', 'ENSG00000149292_TTC12', 'ENSG00000149294_NCAM1', 'ENSG00000149308_NPAT', 'ENSG00000149311_ATM', 'ENSG00000149313_AASDHPPT', 'ENSG00000149328_GLB1L2', 'ENSG00000149346_SLX4IP', 'ENSG00000149357_LAMTOR1', 'ENSG00000149380_P4HA3', 'ENSG00000149418_ST14', 'ENSG00000149428_HYOU1', 'ENSG00000149474_KAT14', 'ENSG00000149476_TKFC', 'ENSG00000149480_MTA2', 'ENSG00000149483_TMEM138', 'ENSG00000149485_FADS1', 'ENSG00000149489_ROM1', 'ENSG00000149499_EML3', 'ENSG00000149503_INCENP', 'ENSG00000149516_MS4A3', 'ENSG00000149531_FRG1BP', 'ENSG00000149532_CPSF7', 'ENSG00000149534_MS4A2', 'ENSG00000149541_B3GAT3', 'ENSG00000149547_EI24', 'ENSG00000149548_CCDC15', 'ENSG00000149554_CHEK1', 'ENSG00000149557_FEZ1', 'ENSG00000149564_ESAM', 'ENSG00000149571_KIRREL3', 'ENSG00000149573_MPZL2', 'ENSG00000149577_SIDT2', 'ENSG00000149582_TMEM25', 'ENSG00000149591_TAGLN', 'ENSG00000149599_DUSP15', 'ENSG00000149600_COMMD7', 'ENSG00000149609_C20orf144', 'ENSG00000149636_DSN1', 'ENSG00000149639_SOGA1', 'ENSG00000149646_CNBD2', 'ENSG00000149657_LSM14B', 'ENSG00000149658_YTHDF1', 'ENSG00000149679_CABLES2', 'ENSG00000149716_LTO1', 'ENSG00000149743_TRPT1', 'ENSG00000149761_NUDT22', 'ENSG00000149781_FERMT3', 'ENSG00000149782_PLCB3', 'ENSG00000149792_MRPL49', 'ENSG00000149806_FAU', 'ENSG00000149809_TM7SF2', 'ENSG00000149823_VPS51', 'ENSG00000149922_TBX6', 'ENSG00000149923_PPP4C', 'ENSG00000149925_ALDOA', 'ENSG00000149926_FAM57B', 'ENSG00000149927_DOC2A', 'ENSG00000149929_HIRIP3', 'ENSG00000149930_TAOK2', 'ENSG00000149932_TMEM219', 'ENSG00000149948_HMGA2', 'ENSG00000149970_CNKSR2', 'ENSG00000150045_KLRF1', 'ENSG00000150054_MPP7', 'ENSG00000150093_ITGB1', 'ENSG00000150281_CTF1', 'ENSG00000150316_CWC15', 'ENSG00000150337_FCGR1A', 'ENSG00000150347_ARID5B', 'ENSG00000150401_DCUN1D2', 'ENSG00000150403_TMCO3', 'ENSG00000150433_TMEM218', 'ENSG00000150455_TIRAP', 'ENSG00000150456_EEF1AKMT1', 'ENSG00000150457_LATS2', 'ENSG00000150459_SAP18', 'ENSG00000150471_ADGRL3', 'ENSG00000150477_KIAA1328', 'ENSG00000150510_FAM124A', 'ENSG00000150527_MIA2', 'ENSG00000150540_HNMT', 'ENSG00000150551_LYPD1', 'ENSG00000150556_LYPD6B', 'ENSG00000150593_PDCD4', 'ENSG00000150594_ADRA2A', 'ENSG00000150625_GPM6A', 'ENSG00000150627_WDR17', 'ENSG00000150636_CCDC102B', 'ENSG00000150637_CD226', 'ENSG00000150667_FSIP1', 'ENSG00000150672_DLG2', 'ENSG00000150681_RGS18', 'ENSG00000150687_PRSS23', 'ENSG00000150712_MTMR12', 'ENSG00000150753_CCT5', 'ENSG00000150756_FAM173B', 'ENSG00000150760_DOCK1', 'ENSG00000150764_DIXDC1', 'ENSG00000150768_DLAT', 'ENSG00000150773_PIH1D2', 'ENSG00000150776_NKAPD1', 'ENSG00000150779_TIMM8B', 'ENSG00000150782_IL18', 'ENSG00000150783_TEX12', 'ENSG00000150787_PTS', 'ENSG00000150867_PIP4K2A', 'ENSG00000150873_C2orf50', 'ENSG00000150907_FOXO1', 'ENSG00000150938_CRIM1', 'ENSG00000150961_SEC24D', 'ENSG00000150967_ABCB9', 'ENSG00000150977_RILPL2', 'ENSG00000150990_DHX37', 'ENSG00000150991_UBC', 'ENSG00000150995_ITPR1', 'ENSG00000151006_PRSS53', 'ENSG00000151012_SLC7A11', 'ENSG00000151014_NOCT', 'ENSG00000151023_ENKUR', 'ENSG00000151062_CACNA2D4', 'ENSG00000151065_DCP1B', 'ENSG00000151067_CACNA1C', 'ENSG00000151090_THRB', 'ENSG00000151092_NGLY1', 'ENSG00000151093_OXSM', 'ENSG00000151116_UEVLD', 'ENSG00000151117_TMEM86A', 'ENSG00000151131_C12orf45', 'ENSG00000151135_TMEM263', 'ENSG00000151136_BTBD11', 'ENSG00000151148_UBE3B', 'ENSG00000151150_ANK3', 'ENSG00000151151_IPMK', 'ENSG00000151164_RAD9B', 'ENSG00000151176_PLBD2', 'ENSG00000151208_DLG5', 'ENSG00000151229_SLC2A13', 'ENSG00000151233_GXYLT1', 'ENSG00000151239_TWF1', 'ENSG00000151240_DIP2C', 'ENSG00000151247_EIF4E', 'ENSG00000151276_MAGI1', 'ENSG00000151287_TEX30', 'ENSG00000151292_CSNK1G3', 'ENSG00000151303_AL136982.1', 'ENSG00000151304_SRFBP1', 'ENSG00000151320_AKAP6', 'ENSG00000151322_NPAS3', 'ENSG00000151327_FAM177A1', 'ENSG00000151332_MBIP', 'ENSG00000151338_MIPOL1', 'ENSG00000151348_EXT2', 'ENSG00000151353_TMEM18', 'ENSG00000151364_KCTD14', 'ENSG00000151366_NDUFC2', 'ENSG00000151376_ME3', 'ENSG00000151413_NUBPL', 'ENSG00000151414_NEK7', 'ENSG00000151422_FER', 'ENSG00000151445_VIPAS39', 'ENSG00000151458_ANKRD50', 'ENSG00000151461_UPF2', 'ENSG00000151465_CDC123', 'ENSG00000151466_SCLT1', 'ENSG00000151468_CCDC3', 'ENSG00000151470_C4orf33', 'ENSG00000151474_FRMD4A', 'ENSG00000151490_PTPRO', 'ENSG00000151491_EPS8', 'ENSG00000151498_ACAD8', 'ENSG00000151500_THYN1', 'ENSG00000151502_VPS26B', 'ENSG00000151503_NCAPD3', 'ENSG00000151532_VTI1A', 'ENSG00000151552_QDPR', 'ENSG00000151553_FAM160B1', 'ENSG00000151575_TEX9', 'ENSG00000151576_QTRT2', 'ENSG00000151611_MMAA', 'ENSG00000151612_ZNF827', 'ENSG00000151623_NR3C2', 'ENSG00000151632_AKR1C2', 'ENSG00000151640_DPYSL4', 'ENSG00000151651_ADAM8', 'ENSG00000151657_KIN', 'ENSG00000151665_PIGF', 'ENSG00000151687_ANKAR', 'ENSG00000151689_INPP1', 'ENSG00000151690_MFSD6', 'ENSG00000151692_RNF144A', 'ENSG00000151693_ASAP2', 'ENSG00000151694_ADAM17', 'ENSG00000151702_FLI1', 'ENSG00000151715_TMEM45B', 'ENSG00000151718_WWC2', 'ENSG00000151725_CENPU', 'ENSG00000151726_ACSL1', 'ENSG00000151729_SLC25A4', 'ENSG00000151743_AMN1', 'ENSG00000151746_BICD1', 'ENSG00000151748_SAV1', 'ENSG00000151773_CCDC122', 'ENSG00000151778_SERP2', 'ENSG00000151779_NBAS', 'ENSG00000151789_ZNF385D', 'ENSG00000151806_GUF1', 'ENSG00000151835_SACS', 'ENSG00000151838_CCDC175', 'ENSG00000151846_PABPC3', 'ENSG00000151849_CENPJ', 'ENSG00000151876_FBXO4', 'ENSG00000151881_TMEM267', 'ENSG00000151882_CCL28', 'ENSG00000151883_PARP8', 'ENSG00000151893_CACUL1', 'ENSG00000151914_DST', 'ENSG00000151917_BEND6', 'ENSG00000151923_TIAL1', 'ENSG00000151929_BAG3', 'ENSG00000151962_RBM46', 'ENSG00000151967_SCHIP1', 'ENSG00000152056_AP1S3', 'ENSG00000152061_RABGAP1L', 'ENSG00000152078_TMEM56', 'ENSG00000152082_MZT2B', 'ENSG00000152102_FAM168B', 'ENSG00000152104_PTPN14', 'ENSG00000152117_AC073869.1', 'ENSG00000152127_MGAT5', 'ENSG00000152128_TMEM163', 'ENSG00000152133_GPATCH11', 'ENSG00000152147_GEMIN6', 'ENSG00000152192_POU4F1', 'ENSG00000152193_RNF219', 'ENSG00000152207_CYSLTR2', 'ENSG00000152213_ARL11', 'ENSG00000152217_SETBP1', 'ENSG00000152219_ARL14EP', 'ENSG00000152223_EPG5', 'ENSG00000152229_PSTPIP2', 'ENSG00000152234_ATP5F1A', 'ENSG00000152240_HAUS1', 'ENSG00000152242_C18orf25', 'ENSG00000152253_SPC25', 'ENSG00000152256_PDK1', 'ENSG00000152270_PDE3B', 'ENSG00000152284_TCF7L1', 'ENSG00000152291_TGOLN2', 'ENSG00000152332_UHMK1', 'ENSG00000152348_ATG10', 'ENSG00000152359_POC5', 'ENSG00000152377_SPOCK1', 'ENSG00000152380_FAM151B', 'ENSG00000152382_TADA1', 'ENSG00000152404_CWF19L2', 'ENSG00000152409_JMY', 'ENSG00000152413_HOMER1', 'ENSG00000152422_XRCC4', 'ENSG00000152433_ZNF547', 'ENSG00000152439_ZNF773', 'ENSG00000152443_ZNF776', 'ENSG00000152454_ZNF256', 'ENSG00000152455_SUV39H2', 'ENSG00000152457_DCLRE1C', 'ENSG00000152464_RPP38', 'ENSG00000152465_NMT2', 'ENSG00000152467_ZSCAN1', 'ENSG00000152475_ZNF837', 'ENSG00000152484_USP12', 'ENSG00000152492_CCDC50', 'ENSG00000152518_ZFP36L2', 'ENSG00000152520_PAN3', 'ENSG00000152527_PLEKHH2', 'ENSG00000152556_PFKM', 'ENSG00000152558_TMEM123', 'ENSG00000152580_IGSF10', 'ENSG00000152582_SPEF2', 'ENSG00000152601_MBNL1', 'ENSG00000152620_NADK2', 'ENSG00000152642_GPD1L', 'ENSG00000152661_GJA1', 'ENSG00000152683_SLC30A6', 'ENSG00000152684_PELO', 'ENSG00000152689_RASGRP3', 'ENSG00000152700_SAR1B', 'ENSG00000152705_CATSPER3', 'ENSG00000152749_GPR180', 'ENSG00000152760_TCTEX1D1', 'ENSG00000152763_WDR78', 'ENSG00000152766_ANKRD22', 'ENSG00000152767_FARP1', 'ENSG00000152778_IFIT5', 'ENSG00000152782_PANK1', 'ENSG00000152784_PRDM8', 'ENSG00000152795_HNRNPDL', 'ENSG00000152804_HHEX', 'ENSG00000152818_UTRN', 'ENSG00000152904_GGPS1', 'ENSG00000152926_ZNF117', 'ENSG00000152931_PART1', 'ENSG00000152932_RAB3C', 'ENSG00000152939_MARVELD2', 'ENSG00000152942_RAD17', 'ENSG00000152944_MED21', 'ENSG00000152952_PLOD2', 'ENSG00000152953_STK32B', 'ENSG00000152990_ADGRA3', 'ENSG00000153002_CPB1', 'ENSG00000153006_SREK1IP1', 'ENSG00000153015_CWC27', 'ENSG00000153029_MR1', 'ENSG00000153037_SRP19', 'ENSG00000153044_CENPH', 'ENSG00000153046_CDYL', 'ENSG00000153048_CARHSP1', 'ENSG00000153064_BANK1', 'ENSG00000153066_TXNDC11', 'ENSG00000153071_DAB2', 'ENSG00000153094_BCL2L11', 'ENSG00000153107_ANAPC1', 'ENSG00000153113_CAST', 'ENSG00000153130_SCOC', 'ENSG00000153132_CLGN', 'ENSG00000153140_CETN3', 'ENSG00000153147_SMARCA5', 'ENSG00000153157_SYCP2L', 'ENSG00000153162_BMP6', 'ENSG00000153179_RASSF3', 'ENSG00000153187_HNRNPU', 'ENSG00000153201_RANBP2', 'ENSG00000153207_AHCTF1', 'ENSG00000153208_MERTK', 'ENSG00000153214_TMEM87B', 'ENSG00000153233_PTPRR', 'ENSG00000153234_NR4A2', 'ENSG00000153237_CCDC148', 'ENSG00000153246_PLA2R1', 'ENSG00000153250_RBMS1', 'ENSG00000153283_CD96', 'ENSG00000153291_SLC25A27', 'ENSG00000153310_FAM49B', 'ENSG00000153317_ASAP1', 'ENSG00000153339_TRAPPC8', 'ENSG00000153347_FAM81B', 'ENSG00000153363_LINC00467', 'ENSG00000153391_INO80C', 'ENSG00000153395_LPCAT1', 'ENSG00000153404_PLEKHG4B', 'ENSG00000153406_NMRAL1', 'ENSG00000153443_UBALD1', 'ENSG00000153485_TMEM251', 'ENSG00000153487_ING1', 'ENSG00000153531_ADPRHL1', 'ENSG00000153551_CMTM7', 'ENSG00000153558_FBXL2', 'ENSG00000153560_UBP1', 'ENSG00000153561_RMND5A', 'ENSG00000153574_RPIA', 'ENSG00000153707_PTPRD', 'ENSG00000153714_LURAP1L', 'ENSG00000153721_CNKSR3', 'ENSG00000153767_GTF2E1', 'ENSG00000153774_CFDP1', 'ENSG00000153786_ZDHHC7', 'ENSG00000153790_C7orf31', 'ENSG00000153814_JAZF1', 'ENSG00000153815_CMIP', 'ENSG00000153823_PID1', 'ENSG00000153827_TRIP12', 'ENSG00000153832_FBXO36', 'ENSG00000153879_CEBPG', 'ENSG00000153885_KCTD15', 'ENSG00000153896_ZNF599', 'ENSG00000153898_MCOLN2', 'ENSG00000153904_DDAH1', 'ENSG00000153914_SREK1', 'ENSG00000153922_CHD1', 'ENSG00000153933_DGKE', 'ENSG00000153936_HS2ST1', 'ENSG00000153944_MSI2', 'ENSG00000153956_CACNA2D1', 'ENSG00000153975_ZUP1', 'ENSG00000153976_HS3ST3A1', 'ENSG00000153982_GDPD1', 'ENSG00000153989_NUS1', 'ENSG00000154001_PPP2R5E', 'ENSG00000154016_GRAP', 'ENSG00000154025_SLC5A10', 'ENSG00000154027_AK5', 'ENSG00000154059_IMPACT', 'ENSG00000154065_ANKRD29', 'ENSG00000154079_SDHAF4', 'ENSG00000154096_THY1', 'ENSG00000154099_DNAAF1', 'ENSG00000154102_C16orf74', 'ENSG00000154114_TBCEL', 'ENSG00000154122_ANKH', 'ENSG00000154124_OTULIN', 'ENSG00000154127_UBASH3B', 'ENSG00000154133_ROBO4', 'ENSG00000154134_ROBO3', 'ENSG00000154144_TBRG1', 'ENSG00000154146_NRGN', 'ENSG00000154153_RETREG1', 'ENSG00000154174_TOMM70', 'ENSG00000154188_ANGPT1', 'ENSG00000154217_PITPNC1', 'ENSG00000154222_CC2D1B', 'ENSG00000154229_PRKCA', 'ENSG00000154237_LRRK1', 'ENSG00000154240_CEP112', 'ENSG00000154262_ABCA6', 'ENSG00000154263_ABCA10', 'ENSG00000154265_ABCA5', 'ENSG00000154269_ENPP3', 'ENSG00000154277_UCHL1', 'ENSG00000154305_MIA3', 'ENSG00000154309_DISP1', 'ENSG00000154310_TNIK', 'ENSG00000154328_NEIL2', 'ENSG00000154330_PGM5', 'ENSG00000154358_OBSCN', 'ENSG00000154359_LONRF1', 'ENSG00000154370_TRIM11', 'ENSG00000154380_ENAH', 'ENSG00000154429_CCSAP', 'ENSG00000154447_SH3RF1', 'ENSG00000154451_GBP5', 'ENSG00000154473_BUB3', 'ENSG00000154479_CCDC173', 'ENSG00000154511_FAM69A', 'ENSG00000154518_ATP5MC3', 'ENSG00000154537_FAM27C', 'ENSG00000154548_SRSF12', 'ENSG00000154582_ELOC', 'ENSG00000154589_LY96', 'ENSG00000154608_CEP170P1', 'ENSG00000154611_PSMA8', 'ENSG00000154620_TMSB4Y', 'ENSG00000154639_CXADR', 'ENSG00000154640_BTG3', 'ENSG00000154642_C21orf91', 'ENSG00000154654_NCAM2', 'ENSG00000154655_L3MBTL4', 'ENSG00000154678_PDE1C', 'ENSG00000154710_RABGEF1', 'ENSG00000154719_MRPL39', 'ENSG00000154721_JAM2', 'ENSG00000154723_ATP5PF', 'ENSG00000154727_GABPA', 'ENSG00000154734_ADAMTS1', 'ENSG00000154743_TSEN2', 'ENSG00000154760_SLFN13', 'ENSG00000154767_XPC', 'ENSG00000154781_CCDC174', 'ENSG00000154783_FGD5', 'ENSG00000154803_FLCN', 'ENSG00000154813_DPH3', 'ENSG00000154814_OXNAD1', 'ENSG00000154822_PLCL2', 'ENSG00000154832_CXXC1', 'ENSG00000154839_SKA1', 'ENSG00000154845_PPP4R1', 'ENSG00000154856_APCDD1', 'ENSG00000154864_PIEZO2', 'ENSG00000154874_CCDC144B', 'ENSG00000154889_MPPE1', 'ENSG00000154898_CCDC144CP', 'ENSG00000154914_USP43', 'ENSG00000154917_RAB6B', 'ENSG00000154920_EME1', 'ENSG00000154930_ACSS1', 'ENSG00000154945_ANKRD40', 'ENSG00000154957_ZNF18', 'ENSG00000154978_VOPP1', 'ENSG00000155008_APOOL', 'ENSG00000155016_CYP2U1', 'ENSG00000155034_FBXL18', 'ENSG00000155085_AK9', 'ENSG00000155090_KLF10', 'ENSG00000155093_PTPRN2', 'ENSG00000155096_AZIN1', 'ENSG00000155097_ATP6V1C1', 'ENSG00000155099_PIP4P2', 'ENSG00000155100_OTUD6B', 'ENSG00000155111_CDK19', 'ENSG00000155115_GTF3C6', 'ENSG00000155158_TTC39B', 'ENSG00000155189_AGPAT5', 'ENSG00000155229_MMS19', 'ENSG00000155252_PI4K2A', 'ENSG00000155254_MARVELD1', 'ENSG00000155256_ZFYVE27', 'ENSG00000155275_TRMT44', 'ENSG00000155287_SLC25A28', 'ENSG00000155304_HSPA13', 'ENSG00000155307_SAMSN1', 'ENSG00000155313_USP25', 'ENSG00000155324_GRAMD2B', 'ENSG00000155329_ZCCHC10', 'ENSG00000155330_C16orf87', 'ENSG00000155363_MOV10', 'ENSG00000155366_RHOC', 'ENSG00000155367_PPM1J', 'ENSG00000155368_DBI', 'ENSG00000155380_SLC16A1', 'ENSG00000155393_HEATR3', 'ENSG00000155438_NIFK', 'ENSG00000155463_OXA1L', 'ENSG00000155465_SLC7A7', 'ENSG00000155506_LARP1', 'ENSG00000155508_CNOT8', 'ENSG00000155542_SETD9', 'ENSG00000155545_MIER3', 'ENSG00000155561_NUP205', 'ENSG00000155592_ZKSCAN2', 'ENSG00000155621_C9orf85', 'ENSG00000155629_PIK3AP1', 'ENSG00000155636_RBM45', 'ENSG00000155657_TTN', 'ENSG00000155659_VSIG4', 'ENSG00000155660_PDIA4', 'ENSG00000155666_KDM8', 'ENSG00000155719_OTOA', 'ENSG00000155729_KCTD18', 'ENSG00000155744_FAM126B', 'ENSG00000155749_ALS2CR12', 'ENSG00000155754_C2CD6', 'ENSG00000155755_TMEM237', 'ENSG00000155760_FZD7', 'ENSG00000155761_SPAG17', 'ENSG00000155792_DEPTOR', 'ENSG00000155827_RNF20', 'ENSG00000155846_PPARGC1B', 'ENSG00000155849_ELMO1', 'ENSG00000155850_SLC26A2', 'ENSG00000155858_LSM11', 'ENSG00000155868_MED7', 'ENSG00000155875_SAXO1', 'ENSG00000155876_RRAGA', 'ENSG00000155893_PXYLP1', 'ENSG00000155903_RASA2', 'ENSG00000155906_RMND1', 'ENSG00000155926_SLA', 'ENSG00000155957_TMBIM4', 'ENSG00000155959_VBP1', 'ENSG00000155961_RAB39B', 'ENSG00000155962_CLIC2', 'ENSG00000155966_AFF2', 'ENSG00000155970_MICU3', 'ENSG00000155974_GRIP1', 'ENSG00000155975_VPS37A', 'ENSG00000155980_KIF5A', 'ENSG00000156011_PSD3', 'ENSG00000156017_CARNMT1', 'ENSG00000156026_MCU', 'ENSG00000156030_ELMSAN1', 'ENSG00000156042_CFAP70', 'ENSG00000156049_GNA14', 'ENSG00000156050_FAM161B', 'ENSG00000156052_GNAQ', 'ENSG00000156076_WIF1', 'ENSG00000156103_MMP16', 'ENSG00000156110_ADK', 'ENSG00000156127_BATF', 'ENSG00000156136_DCK', 'ENSG00000156140_ADAMTS3', 'ENSG00000156150_ALX3', 'ENSG00000156162_DPY19L4', 'ENSG00000156170_NDUFAF6', 'ENSG00000156171_DRAM2', 'ENSG00000156172_C8orf37', 'ENSG00000156206_CFAP161', 'ENSG00000156232_WHAMM', 'ENSG00000156239_N6AMT1', 'ENSG00000156253_RWDD2B', 'ENSG00000156256_USP16', 'ENSG00000156261_CCT8', 'ENSG00000156265_MAP3K7CL', 'ENSG00000156273_BACH1', 'ENSG00000156298_TSPAN7', 'ENSG00000156299_TIAM1', 'ENSG00000156304_SCAF4', 'ENSG00000156313_RPGR', 'ENSG00000156345_CDK20', 'ENSG00000156374_PCGF6', 'ENSG00000156381_ANKRD9', 'ENSG00000156384_SFR1', 'ENSG00000156398_SFXN2', 'ENSG00000156411_ATP5MPL', 'ENSG00000156413_FUT6', 'ENSG00000156414_TDRD9', 'ENSG00000156427_FGF18', 'ENSG00000156463_SH3RF2', 'ENSG00000156467_UQCRB', 'ENSG00000156469_MTERF3', 'ENSG00000156471_PTDSS1', 'ENSG00000156475_PPP2R2B', 'ENSG00000156482_RPL30', 'ENSG00000156500_FAM122C', 'ENSG00000156502_SUPV3L1', 'ENSG00000156504_FAM122B', 'ENSG00000156508_EEF1A1', 'ENSG00000156509_FBXO43', 'ENSG00000156515_HK1', 'ENSG00000156521_TYSND1', 'ENSG00000156531_PHF6', 'ENSG00000156535_CD109', 'ENSG00000156575_PRG3', 'ENSG00000156587_UBE2L6', 'ENSG00000156599_ZDHHC5', 'ENSG00000156603_MED19', 'ENSG00000156639_ZFAND3', 'ENSG00000156642_NPTN', 'ENSG00000156650_KAT6B', 'ENSG00000156671_SAMD8', 'ENSG00000156675_RAB11FIP1', 'ENSG00000156689_GLYATL2', 'ENSG00000156697_UTP14A', 'ENSG00000156709_AIFM1', 'ENSG00000156711_MAPK13', 'ENSG00000156735_BAG4', 'ENSG00000156738_MS4A1', 'ENSG00000156787_TBC1D31', 'ENSG00000156795_WDYHV1', 'ENSG00000156802_ATAD2', 'ENSG00000156804_FBXO32', 'ENSG00000156831_NSMCE2', 'ENSG00000156853_ZNF689', 'ENSG00000156858_PRR14', 'ENSG00000156860_FBRS', 'ENSG00000156869_FRRS1', 'ENSG00000156873_PHKG2', 'ENSG00000156875_MFSD14A', 'ENSG00000156876_SASS6', 'ENSG00000156928_MALSU1', 'ENSG00000156931_VPS8', 'ENSG00000156958_GALK2', 'ENSG00000156959_LHFPL4', 'ENSG00000156966_B3GNT7', 'ENSG00000156968_MPV17L', 'ENSG00000156970_BUB1B', 'ENSG00000156973_PDE6D', 'ENSG00000156976_EIF4A2', 'ENSG00000156983_BRPF1', 'ENSG00000156990_RPUSD3', 'ENSG00000157014_TATDN2', 'ENSG00000157017_GHRL', 'ENSG00000157020_SEC13', 'ENSG00000157021_FAM92A1P1', 'ENSG00000157036_EXOG', 'ENSG00000157045_NTAN1', 'ENSG00000157077_ZFYVE9', 'ENSG00000157106_SMG1', 'ENSG00000157107_FCHO2', 'ENSG00000157110_RBPMS', 'ENSG00000157181_ODR4', 'ENSG00000157184_CPT2', 'ENSG00000157191_NECAP2', 'ENSG00000157193_LRP8', 'ENSG00000157212_PAXIP1', 'ENSG00000157214_STEAP2', 'ENSG00000157216_SSBP3', 'ENSG00000157224_CLDN12', 'ENSG00000157227_MMP14', 'ENSG00000157240_FZD1', 'ENSG00000157259_GATAD1', 'ENSG00000157303_SUSD3', 'ENSG00000157306_ZFHX2-AS1', 'ENSG00000157315_TMED6', 'ENSG00000157322_CLEC18A', 'ENSG00000157326_DHRS4', 'ENSG00000157343_ARMC12', 'ENSG00000157349_DDX19B', 'ENSG00000157350_ST3GAL2', 'ENSG00000157353_FUK', 'ENSG00000157379_DHRS1', 'ENSG00000157388_CACNA1D', 'ENSG00000157399_ARSE', 'ENSG00000157404_KIT', 'ENSG00000157426_AASDH', 'ENSG00000157429_ZNF19', 'ENSG00000157445_CACNA2D3', 'ENSG00000157450_RNF111', 'ENSG00000157456_CCNB2', 'ENSG00000157470_FAM81A', 'ENSG00000157483_MYO1E', 'ENSG00000157500_APPL1', 'ENSG00000157510_AFAP1L1', 'ENSG00000157514_TSC22D3', 'ENSG00000157538_VPS26C', 'ENSG00000157540_DYRK1A', 'ENSG00000157554_ERG', 'ENSG00000157557_ETS2', 'ENSG00000157570_TSPAN18', 'ENSG00000157578_LCA5L', 'ENSG00000157593_SLC35B2', 'ENSG00000157600_TMEM164', 'ENSG00000157601_MX1', 'ENSG00000157613_CREB3L1', 'ENSG00000157617_C2CD2', 'ENSG00000157625_TAB3', 'ENSG00000157637_SLC38A10', 'ENSG00000157653_C9orf43', 'ENSG00000157657_ZNF618', 'ENSG00000157693_TMEM268', 'ENSG00000157703_SVOPL', 'ENSG00000157734_SNX22', 'ENSG00000157741_UBN2', 'ENSG00000157764_BRAF', 'ENSG00000157778_PSMG3', 'ENSG00000157782_CABP1', 'ENSG00000157796_WDR19', 'ENSG00000157800_SLC37A3', 'ENSG00000157823_AP3S2', 'ENSG00000157827_FMNL2', 'ENSG00000157833_GAREM2', 'ENSG00000157837_SPPL3', 'ENSG00000157869_RAB28', 'ENSG00000157870_PRXL2B', 'ENSG00000157873_TNFRSF14', 'ENSG00000157881_PANK4', 'ENSG00000157890_MEGF11', 'ENSG00000157895_C12orf43', 'ENSG00000157911_PEX10', 'ENSG00000157916_RER1', 'ENSG00000157927_RADIL', 'ENSG00000157933_SKI', 'ENSG00000157954_WIPI2', 'ENSG00000157978_LDLRAP1', 'ENSG00000157985_AGAP1', 'ENSG00000157992_KRTCAP3', 'ENSG00000158006_PAFAH2', 'ENSG00000158019_BABAM2', 'ENSG00000158023_WDR66', 'ENSG00000158042_MRPL17', 'ENSG00000158050_DUSP2', 'ENSG00000158062_UBXN11', 'ENSG00000158079_PTPDC1', 'ENSG00000158089_GALNT14', 'ENSG00000158092_NCK1', 'ENSG00000158106_RHPN1', 'ENSG00000158109_TPRG1L', 'ENSG00000158113_LRRC43', 'ENSG00000158122_PRXL2C', 'ENSG00000158156_XKR8', 'ENSG00000158158_CNNM4', 'ENSG00000158161_EYA3', 'ENSG00000158163_DZIP1L', 'ENSG00000158164_TMSB15A', 'ENSG00000158169_FANCC', 'ENSG00000158186_MRAS', 'ENSG00000158195_WASF2', 'ENSG00000158201_ABHD3', 'ENSG00000158220_ESYT3', 'ENSG00000158234_FAIM', 'ENSG00000158270_COLEC12', 'ENSG00000158286_RNF207', 'ENSG00000158290_CUL4B', 'ENSG00000158292_GPR153', 'ENSG00000158301_GPRASP2', 'ENSG00000158315_RHBDL2', 'ENSG00000158321_AUTS2', 'ENSG00000158352_SHROOM4', 'ENSG00000158373_HIST1H2BD', 'ENSG00000158402_CDC25C', 'ENSG00000158406_HIST1H4H', 'ENSG00000158411_MITD1', 'ENSG00000158417_EIF5B', 'ENSG00000158423_RIBC1', 'ENSG00000158427_TMSB15B', 'ENSG00000158428_CATIP', 'ENSG00000158435_CNOT11', 'ENSG00000158445_KCNB1', 'ENSG00000158457_TSPAN33', 'ENSG00000158458_NRG2', 'ENSG00000158467_AHCYL2', 'ENSG00000158470_B4GALT5', 'ENSG00000158473_CD1D', 'ENSG00000158480_SPATA2', 'ENSG00000158483_FAM86C1', 'ENSG00000158486_DNAH3', 'ENSG00000158517_NCF1', 'ENSG00000158526_TSR2', 'ENSG00000158528_PPP1R9A', 'ENSG00000158545_ZC3H18', 'ENSG00000158552_ZFAND2B', 'ENSG00000158555_GDPD5', 'ENSG00000158560_DYNC1I1', 'ENSG00000158578_ALAS2', 'ENSG00000158604_TMED4', 'ENSG00000158615_PPP1R15B', 'ENSG00000158623_COPG2', 'ENSG00000158636_EMSY', 'ENSG00000158669_GPAT4', 'ENSG00000158691_ZSCAN12', 'ENSG00000158710_TAGLN2', 'ENSG00000158711_ELK4', 'ENSG00000158714_SLAMF8', 'ENSG00000158715_SLC45A3', 'ENSG00000158716_DUSP23', 'ENSG00000158717_RNF166', 'ENSG00000158747_NBL1', 'ENSG00000158769_F11R', 'ENSG00000158773_USF1', 'ENSG00000158792_SPATA2L', 'ENSG00000158793_NIT1', 'ENSG00000158796_DEDD', 'ENSG00000158805_ZNF276', 'ENSG00000158806_NPM2', 'ENSG00000158813_EDA', 'ENSG00000158825_CDA', 'ENSG00000158828_PINK1', 'ENSG00000158850_B4GALT3', 'ENSG00000158856_DMTN', 'ENSG00000158863_FAM160B2', 'ENSG00000158864_NDUFS2', 'ENSG00000158869_FCER1G', 'ENSG00000158882_TOMM40L', 'ENSG00000158887_MPZ', 'ENSG00000158941_CCAR2', 'ENSG00000158966_CACHD1', 'ENSG00000158985_CDC42SE2', 'ENSG00000158987_RAPGEF6', 'ENSG00000159023_EPB41', 'ENSG00000159055_MIS18A', 'ENSG00000159063_ALG8', 'ENSG00000159069_FBXW5', 'ENSG00000159079_CFAP298', 'ENSG00000159082_SYNJ1', 'ENSG00000159086_PAXBP1', 'ENSG00000159110_IFNAR2', 'ENSG00000159111_MRPL10', 'ENSG00000159128_IFNGR2', 'ENSG00000159131_GART', 'ENSG00000159140_SON', 'ENSG00000159147_DONSON', 'ENSG00000159164_SV2A', 'ENSG00000159176_CSRP1', 'ENSG00000159199_ATP5MC1', 'ENSG00000159200_RCAN1', 'ENSG00000159202_UBE2Z', 'ENSG00000159208_CIART', 'ENSG00000159210_SNF8', 'ENSG00000159212_CLIC6', 'ENSG00000159214_CCDC24', 'ENSG00000159216_RUNX1', 'ENSG00000159228_CBR1', 'ENSG00000159231_CBR3', 'ENSG00000159256_MORC3', 'ENSG00000159259_CHAF1B', 'ENSG00000159267_HLCS', 'ENSG00000159307_SCUBE1', 'ENSG00000159314_ARHGAP27', 'ENSG00000159322_ADPGK', 'ENSG00000159335_PTMS', 'ENSG00000159339_PADI4', 'ENSG00000159346_ADIPOR1', 'ENSG00000159348_CYB5R1', 'ENSG00000159352_PSMD4', 'ENSG00000159363_ATP13A2', 'ENSG00000159374_M1AP', 'ENSG00000159377_PSMB4', 'ENSG00000159388_BTG2', 'ENSG00000159399_HK2', 'ENSG00000159403_C1R', 'ENSG00000159423_ALDH4A1', 'ENSG00000159433_STARD9', 'ENSG00000159445_THEM4', 'ENSG00000159450_TCHH', 'ENSG00000159459_UBR1', 'ENSG00000159461_AMFR', 'ENSG00000159479_MED8', 'ENSG00000159496_RGL4', 'ENSG00000159579_RSPRY1', 'ENSG00000159588_CCDC17', 'ENSG00000159592_GPBP1L1', 'ENSG00000159593_NAE1', 'ENSG00000159596_TMEM69', 'ENSG00000159618_ADGRG5', 'ENSG00000159625_DRC7', 'ENSG00000159640_ACE', 'ENSG00000159648_TEPP', 'ENSG00000159658_EFCAB14', 'ENSG00000159674_SPON2', 'ENSG00000159685_CHCHD6', 'ENSG00000159692_CTBP1', 'ENSG00000159708_LRRC36', 'ENSG00000159712_ANKRD18CP', 'ENSG00000159714_ZDHHC1', 'ENSG00000159720_ATP6V0D1', 'ENSG00000159723_AGRP', 'ENSG00000159733_ZFYVE28', 'ENSG00000159753_CARMIL2', 'ENSG00000159761_C16orf86', 'ENSG00000159784_FAM131B', 'ENSG00000159788_RGS12', 'ENSG00000159792_PSKH1', 'ENSG00000159840_ZYX', 'ENSG00000159842_ABR', 'ENSG00000159871_LYPD5', 'ENSG00000159873_CCDC117', 'ENSG00000159882_ZNF230', 'ENSG00000159884_CCDC107', 'ENSG00000159885_ZNF222', 'ENSG00000159899_NPR2', 'ENSG00000159905_ZNF221', 'ENSG00000159915_ZNF233', 'ENSG00000159917_ZNF235', 'ENSG00000159921_GNE', 'ENSG00000159958_TNFRSF13C', 'ENSG00000159961_OR3A3', 'ENSG00000160007_ARHGAP35', 'ENSG00000160013_PTGIR', 'ENSG00000160014_CALM3', 'ENSG00000160049_DFFA', 'ENSG00000160050_CCDC28B', 'ENSG00000160051_IQCC', 'ENSG00000160055_TMEM234', 'ENSG00000160058_BSDC1', 'ENSG00000160062_ZBTB8A', 'ENSG00000160072_ATAD3B', 'ENSG00000160075_SSU72', 'ENSG00000160087_UBE2J2', 'ENSG00000160094_ZNF362', 'ENSG00000160097_FNDC5', 'ENSG00000160111_CPAMD8', 'ENSG00000160113_NR2F6', 'ENSG00000160117_ANKLE1', 'ENSG00000160124_CCDC58', 'ENSG00000160131_VMA21', 'ENSG00000160145_KALRN', 'ENSG00000160161_CILP2', 'ENSG00000160172_FAM86C2P', 'ENSG00000160179_ABCG1', 'ENSG00000160185_UBASH3A', 'ENSG00000160188_RSPH1', 'ENSG00000160190_SLC37A1', 'ENSG00000160191_PDE9A', 'ENSG00000160193_WDR4', 'ENSG00000160194_NDUFV3', 'ENSG00000160199_PKNOX1', 'ENSG00000160200_CBS', 'ENSG00000160201_U2AF1', 'ENSG00000160207_HSF2BP', 'ENSG00000160208_RRP1B', 'ENSG00000160209_PDXK', 'ENSG00000160211_G6PD', 'ENSG00000160213_CSTB', 'ENSG00000160214_RRP1', 'ENSG00000160216_AGPAT3', 'ENSG00000160218_TRAPPC10', 'ENSG00000160219_GAB3', 'ENSG00000160221_GATD3A', 'ENSG00000160223_ICOSLG', 'ENSG00000160226_CFAP410', 'ENSG00000160229_ZNF66', 'ENSG00000160233_LRRC3', 'ENSG00000160255_ITGB2', 'ENSG00000160256_FAM207A', 'ENSG00000160271_RALGDS', 'ENSG00000160282_FTCD', 'ENSG00000160284_SPATC1L', 'ENSG00000160285_LSS', 'ENSG00000160293_VAV2', 'ENSG00000160294_MCM3AP', 'ENSG00000160298_C21orf58', 'ENSG00000160299_PCNT', 'ENSG00000160305_DIP2A', 'ENSG00000160307_S100B', 'ENSG00000160310_PRMT2', 'ENSG00000160318_CLDND2', 'ENSG00000160321_ZNF208', 'ENSG00000160323_ADAMTS13', 'ENSG00000160325_CACFD1', 'ENSG00000160326_SLC2A6', 'ENSG00000160336_ZNF761', 'ENSG00000160345_C9orf116', 'ENSG00000160352_ZNF714', 'ENSG00000160360_GPSM1', 'ENSG00000160392_C19orf47', 'ENSG00000160401_CFAP157', 'ENSG00000160404_TOR2A', 'ENSG00000160408_ST6GALNAC6', 'ENSG00000160410_SHKBP1', 'ENSG00000160439_RDH13', 'ENSG00000160445_ZER1', 'ENSG00000160446_ZDHHC12', 'ENSG00000160447_PKN3', 'ENSG00000160460_SPTBN4', 'ENSG00000160469_BRSK1', 'ENSG00000160471_COX6B2', 'ENSG00000160539_PLPP7', 'ENSG00000160551_TAOK1', 'ENSG00000160563_MED27', 'ENSG00000160570_DEDD2', 'ENSG00000160584_SIK3', 'ENSG00000160588_MPZL3', 'ENSG00000160593_JAML', 'ENSG00000160602_NEK8', 'ENSG00000160606_TLCD1', 'ENSG00000160613_PCSK7', 'ENSG00000160633_SAFB', 'ENSG00000160678_S100A1', 'ENSG00000160679_CHTOP', 'ENSG00000160685_ZBTB7B', 'ENSG00000160688_FLAD1', 'ENSG00000160691_SHC1', 'ENSG00000160695_VPS11', 'ENSG00000160703_NLRX1', 'ENSG00000160710_ADAR', 'ENSG00000160712_IL6R', 'ENSG00000160714_UBE2Q1', 'ENSG00000160741_CRTC2', 'ENSG00000160746_ANO10', 'ENSG00000160752_FDPS', 'ENSG00000160753_RUSC1', 'ENSG00000160766_GBAP1', 'ENSG00000160767_FAM189B', 'ENSG00000160781_PAQR6', 'ENSG00000160783_PMF1', 'ENSG00000160785_SLC25A44', 'ENSG00000160789_LMNA', 'ENSG00000160791_CCR5', 'ENSG00000160796_NBEAL2', 'ENSG00000160799_CCDC12', 'ENSG00000160803_UBQLN4', 'ENSG00000160813_PPP1R35', 'ENSG00000160818_GPATCH4', 'ENSG00000160867_FGFR4', 'ENSG00000160877_NACC1', 'ENSG00000160883_HK3', 'ENSG00000160886_LY6K', 'ENSG00000160888_IER2', 'ENSG00000160908_ZNF394', 'ENSG00000160917_CPSF4', 'ENSG00000160932_LY6E', 'ENSG00000160948_VPS28', 'ENSG00000160949_TONSL', 'ENSG00000160951_PTGER1', 'ENSG00000160953_MUM1', 'ENSG00000160957_RECQL4', 'ENSG00000160959_LRRC14', 'ENSG00000160961_ZNF333', 'ENSG00000160972_PPP1R16A', 'ENSG00000160991_ORAI2', 'ENSG00000160993_ALKBH4', 'ENSG00000160999_SH2B2', 'ENSG00000161010_MRNIP', 'ENSG00000161011_SQSTM1', 'ENSG00000161013_MGAT4B', 'ENSG00000161016_RPL8', 'ENSG00000161021_MAML1', 'ENSG00000161031_PGLYRP2', 'ENSG00000161036_LRWD1', 'ENSG00000161040_FBXL13', 'ENSG00000161048_NAPEPLD', 'ENSG00000161055_SCGB3A1', 'ENSG00000161057_PSMC2', 'ENSG00000161082_CELF5', 'ENSG00000161091_MFSD12', 'ENSG00000161133_USP41', 'ENSG00000161149_TUBA3FP', 'ENSG00000161179_YDJC', 'ENSG00000161202_DVL3', 'ENSG00000161203_AP2M1', 'ENSG00000161204_ABCF3', 'ENSG00000161217_PCYT1A', 'ENSG00000161243_FBXO27', 'ENSG00000161249_DMKN', 'ENSG00000161265_U2AF1L4', 'ENSG00000161267_BDH1', 'ENSG00000161277_THAP8', 'ENSG00000161281_COX7A1', 'ENSG00000161298_ZNF382', 'ENSG00000161328_LRRC56', 'ENSG00000161381_PLXDC1', 'ENSG00000161395_PGAP3', 'ENSG00000161405_IKZF3', 'ENSG00000161509_GRIN2C', 'ENSG00000161513_FDXR', 'ENSG00000161526_SAP30BP', 'ENSG00000161533_ACOX1', 'ENSG00000161542_PRPSAP1', 'ENSG00000161547_SRSF2', 'ENSG00000161551_ZNF577', 'ENSG00000161558_TMEM143', 'ENSG00000161610_HCRT', 'ENSG00000161618_ALDH16A1', 'ENSG00000161638_ITGA5', 'ENSG00000161642_ZNF385A', 'ENSG00000161647_MPP3', 'ENSG00000161654_LSM12', 'ENSG00000161664_ASB16', 'ENSG00000161671_EMC10', 'ENSG00000161677_JOSD2', 'ENSG00000161681_SHANK1', 'ENSG00000161682_FAM171A2', 'ENSG00000161692_DBF4B', 'ENSG00000161714_PLCD3', 'ENSG00000161791_FMNL3', 'ENSG00000161800_RACGAP1', 'ENSG00000161813_LARP4', 'ENSG00000161835_GRASP', 'ENSG00000161847_RAVER1', 'ENSG00000161860_SYCE2', 'ENSG00000161888_SPC24', 'ENSG00000161904_LEMD2', 'ENSG00000161911_TREML1', 'ENSG00000161912_ADCY10P1', 'ENSG00000161914_ZNF653', 'ENSG00000161920_MED11', 'ENSG00000161921_CXCL16', 'ENSG00000161929_SCIMP', 'ENSG00000161939_RNASEK-C17orf49', 'ENSG00000161940_BCL6B', 'ENSG00000161944_ASGR2', 'ENSG00000161955_TNFSF13', 'ENSG00000161956_SENP3', 'ENSG00000161958_FGF11', 'ENSG00000161960_EIF4A1', 'ENSG00000161970_RPL26', 'ENSG00000161973_CCDC42', 'ENSG00000161980_POLR3K', 'ENSG00000161981_SNRNP25', 'ENSG00000161996_WDR90', 'ENSG00000161999_JMJD8', 'ENSG00000162004_CCDC78', 'ENSG00000162032_SPSB3', 'ENSG00000162039_MEIOB', 'ENSG00000162062_TEDC2', 'ENSG00000162063_CCNF', 'ENSG00000162065_TBC1D24', 'ENSG00000162066_AMDHD2', 'ENSG00000162069_BICDL2', 'ENSG00000162073_PAQR4', 'ENSG00000162076_FLYWCH2', 'ENSG00000162078_ZG16B', 'ENSG00000162086_ZNF75A', 'ENSG00000162104_ADCY9', 'ENSG00000162105_SHANK2', 'ENSG00000162129_CLPB', 'ENSG00000162139_NEU3', 'ENSG00000162144_CYB561A3', 'ENSG00000162148_PPP1R32', 'ENSG00000162174_ASRGL1', 'ENSG00000162191_UBXN1', 'ENSG00000162194_LBHD1', 'ENSG00000162222_TTC9C', 'ENSG00000162227_TAF6L', 'ENSG00000162231_NXF1', 'ENSG00000162236_STX5', 'ENSG00000162241_SLC25A45', 'ENSG00000162244_RPL29', 'ENSG00000162298_SYVN1', 'ENSG00000162300_ZFPL1', 'ENSG00000162302_RPS6KA4', 'ENSG00000162337_LRP5', 'ENSG00000162341_TPCN2', 'ENSG00000162366_PDZK1IP1', 'ENSG00000162367_TAL1', 'ENSG00000162368_CMPK1', 'ENSG00000162373_BEND5', 'ENSG00000162374_ELAVL4', 'ENSG00000162377_COA7', 'ENSG00000162378_ZYG11B', 'ENSG00000162384_C1orf123', 'ENSG00000162385_MAGOH', 'ENSG00000162390_ACOT11', 'ENSG00000162396_PARS2', 'ENSG00000162402_USP24', 'ENSG00000162407_PLPP3', 'ENSG00000162408_NOL9', 'ENSG00000162413_KLHL21', 'ENSG00000162415_ZSWIM5', 'ENSG00000162419_GMEB1', 'ENSG00000162430_SELENON', 'ENSG00000162433_AK4', 'ENSG00000162434_JAK1', 'ENSG00000162437_RAVER2', 'ENSG00000162441_LZIC', 'ENSG00000162444_RBP7', 'ENSG00000162458_FBLIM1', 'ENSG00000162461_SLC25A34', 'ENSG00000162482_AKR7A3', 'ENSG00000162490_DRAXIN', 'ENSG00000162496_DHRS3', 'ENSG00000162510_MATN1', 'ENSG00000162511_LAPTM5', 'ENSG00000162512_SDC3', 'ENSG00000162517_PEF1', 'ENSG00000162520_SYNC', 'ENSG00000162521_RBBP4', 'ENSG00000162522_KIAA1522', 'ENSG00000162526_TSSK3', 'ENSG00000162542_TMCO4', 'ENSG00000162543_UBXN10', 'ENSG00000162571_TTLL10', 'ENSG00000162572_SCNN1D', 'ENSG00000162585_FAAP20', 'ENSG00000162591_MEGF6', 'ENSG00000162595_DIRAS3', 'ENSG00000162599_NFIA', 'ENSG00000162600_OMA1', 'ENSG00000162601_MYSM1', 'ENSG00000162604_TM2D1', 'ENSG00000162607_USP1', 'ENSG00000162613_FUBP1', 'ENSG00000162614_NEXN', 'ENSG00000162616_DNAJB4', 'ENSG00000162620_LRRIQ3', 'ENSG00000162623_TYW3', 'ENSG00000162627_SNX7', 'ENSG00000162630_B3GALT2', 'ENSG00000162636_FAM102B', 'ENSG00000162639_HENMT1', 'ENSG00000162641_AKNAD1', 'ENSG00000162642_C1orf52', 'ENSG00000162645_GBP2', 'ENSG00000162650_ATXN7L2', 'ENSG00000162654_GBP4', 'ENSG00000162664_ZNF326', 'ENSG00000162669_HFM1', 'ENSG00000162676_GFI1', 'ENSG00000162687_KCNT2', 'ENSG00000162688_AGL', 'ENSG00000162694_EXTL2', 'ENSG00000162695_SLC30A7', 'ENSG00000162702_ZNF281', 'ENSG00000162704_ARPC5', 'ENSG00000162711_NLRP3', 'ENSG00000162714_ZNF496', 'ENSG00000162722_TRIM58', 'ENSG00000162723_SLAMF9', 'ENSG00000162729_IGSF8', 'ENSG00000162733_DDR2', 'ENSG00000162734_PEA15', 'ENSG00000162735_PEX19', 'ENSG00000162736_NCSTN', 'ENSG00000162738_VANGL2', 'ENSG00000162739_SLAMF6', 'ENSG00000162745_OLFML2B', 'ENSG00000162746_FCRLB', 'ENSG00000162755_KLHDC9', 'ENSG00000162757_C1orf74', 'ENSG00000162769_FLVCR1', 'ENSG00000162772_ATF3', 'ENSG00000162775_RBM15', 'ENSG00000162777_DENND2D', 'ENSG00000162779_AXDND1', 'ENSG00000162783_IER5', 'ENSG00000162804_SNED1', 'ENSG00000162813_BPNT1', 'ENSG00000162814_SPATA17', 'ENSG00000162817_C1orf115', 'ENSG00000162819_BROX', 'ENSG00000162825_NBPF20', 'ENSG00000162836_ACP6', 'ENSG00000162849_KIF26B', 'ENSG00000162851_TFB2M', 'ENSG00000162852_CNST', 'ENSG00000162869_PPP1R21', 'ENSG00000162878_PKDCC', 'ENSG00000162881_OXER1', 'ENSG00000162882_HAAO', 'ENSG00000162885_B3GALNT2', 'ENSG00000162888_C1orf147', 'ENSG00000162889_MAPKAPK2', 'ENSG00000162894_FCMR', 'ENSG00000162909_CAPN2', 'ENSG00000162910_MRPL55', 'ENSG00000162913_OBSCN-AS1', 'ENSG00000162923_WDR26', 'ENSG00000162924_REL', 'ENSG00000162927_PUS10', 'ENSG00000162928_PEX13', 'ENSG00000162929_KIAA1841', 'ENSG00000162931_TRIM17', 'ENSG00000162946_DISC1', 'ENSG00000162959_MEMO1', 'ENSG00000162961_DPY30', 'ENSG00000162971_TYW5', 'ENSG00000162972_MAIP1', 'ENSG00000162976_PQLC3', 'ENSG00000162980_ARL5A', 'ENSG00000162989_KCNJ3', 'ENSG00000162992_NEUROD1', 'ENSG00000162994_CLHC1', 'ENSG00000162997_PRORSD1P', 'ENSG00000162999_DUSP19', 'ENSG00000163001_CFAP36', 'ENSG00000163002_NUP35', 'ENSG00000163006_CCDC138', 'ENSG00000163009_C2orf48', 'ENSG00000163013_FBXO41', 'ENSG00000163016_ALMS1P1', 'ENSG00000163026_WDCP', 'ENSG00000163029_SMC6', 'ENSG00000163040_CCDC74A', 'ENSG00000163041_H3F3A', 'ENSG00000163050_COQ8A', 'ENSG00000163053_SLC16A14', 'ENSG00000163060_TEKT4', 'ENSG00000163069_SGCB', 'ENSG00000163072_NOSTRIN', 'ENSG00000163082_SGPP2', 'ENSG00000163092_XIRP2', 'ENSG00000163093_BBS5', 'ENSG00000163104_SMARCAD1', 'ENSG00000163106_HPGDS', 'ENSG00000163110_PDLIM5', 'ENSG00000163125_RPRD2', 'ENSG00000163126_ANKRD23', 'ENSG00000163131_CTSS', 'ENSG00000163132_MSX1', 'ENSG00000163138_PACRGL', 'ENSG00000163141_BNIPL', 'ENSG00000163154_TNFAIP8L2', 'ENSG00000163155_LYSMD1', 'ENSG00000163156_SCNM1', 'ENSG00000163157_TMOD4', 'ENSG00000163159_VPS72', 'ENSG00000163161_ERCC3', 'ENSG00000163162_RNF149', 'ENSG00000163166_IWS1', 'ENSG00000163170_BOLA3', 'ENSG00000163171_CDC42EP3', 'ENSG00000163191_S100A11', 'ENSG00000163214_DHX57', 'ENSG00000163219_ARHGAP25', 'ENSG00000163220_S100A9', 'ENSG00000163221_S100A12', 'ENSG00000163235_TGFA', 'ENSG00000163249_CCNYL1', 'ENSG00000163251_FZD5', 'ENSG00000163257_DCAF16', 'ENSG00000163281_GNPDA2', 'ENSG00000163283_ALPP', 'ENSG00000163291_PAQR3', 'ENSG00000163293_NIPAL1', 'ENSG00000163297_ANTXR2', 'ENSG00000163312_HELQ', 'ENSG00000163319_MRPS18C', 'ENSG00000163320_CGGBP1', 'ENSG00000163322_ABRAXAS1', 'ENSG00000163328_GPR155', 'ENSG00000163331_DAPL1', 'ENSG00000163344_PMVK', 'ENSG00000163346_PBXIP1', 'ENSG00000163348_PYGO2', 'ENSG00000163349_HIPK1', 'ENSG00000163359_COL6A3', 'ENSG00000163362_INAVA', 'ENSG00000163364_LINC01116', 'ENSG00000163374_YY1AP1', 'ENSG00000163376_KBTBD8', 'ENSG00000163378_EOGT', 'ENSG00000163380_LMOD3', 'ENSG00000163382_NAXE', 'ENSG00000163389_POGLUT1', 'ENSG00000163393_SLC22A15', 'ENSG00000163399_ATP1A1', 'ENSG00000163406_SLC15A2', 'ENSG00000163412_EIF4E3', 'ENSG00000163421_PROK2', 'ENSG00000163428_LRRC58', 'ENSG00000163430_FSTL1', 'ENSG00000163435_ELF3', 'ENSG00000163444_TMEM183A', 'ENSG00000163449_TMEM169', 'ENSG00000163453_IGFBP7', 'ENSG00000163462_TRIM46', 'ENSG00000163463_KRTCAP2', 'ENSG00000163464_CXCR1', 'ENSG00000163466_ARPC2', 'ENSG00000163467_TSACC', 'ENSG00000163468_CCT3', 'ENSG00000163472_TMEM79', 'ENSG00000163479_SSR2', 'ENSG00000163481_RNF25', 'ENSG00000163482_STK36', 'ENSG00000163491_NEK10', 'ENSG00000163492_CCDC141', 'ENSG00000163507_CIP2A', 'ENSG00000163510_CWC22', 'ENSG00000163512_AZI2', 'ENSG00000163513_TGFBR2', 'ENSG00000163516_ANKZF1', 'ENSG00000163517_HDAC11', 'ENSG00000163519_TRAT1', 'ENSG00000163520_FBLN2', 'ENSG00000163521_GLB1L', 'ENSG00000163527_STT3B', 'ENSG00000163528_CHCHD4', 'ENSG00000163534_FCRL1', 'ENSG00000163535_SGO2', 'ENSG00000163536_SERPINI1', 'ENSG00000163539_CLASP2', 'ENSG00000163541_SUCLG1', 'ENSG00000163545_NUAK2', 'ENSG00000163554_SPTA1', 'ENSG00000163558_PRKCI', 'ENSG00000163563_MNDA', 'ENSG00000163564_PYHIN1', 'ENSG00000163565_IFI16', 'ENSG00000163568_AIM2', 'ENSG00000163576_EFHB', 'ENSG00000163577_EIF5A2', 'ENSG00000163584_RPL22L1', 'ENSG00000163590_PPM1L', 'ENSG00000163596_ICA1L', 'ENSG00000163597_SNHG16', 'ENSG00000163600_ICOS', 'ENSG00000163602_RYBP', 'ENSG00000163605_PPP4R2', 'ENSG00000163606_CD200R1', 'ENSG00000163607_GTPBP8', 'ENSG00000163608_NEPRO', 'ENSG00000163611_SPICE1', 'ENSG00000163617_CCDC191', 'ENSG00000163625_WDFY3', 'ENSG00000163626_COX18', 'ENSG00000163629_PTPN13', 'ENSG00000163631_ALB', 'ENSG00000163632_C3orf49', 'ENSG00000163634_THOC7', 'ENSG00000163635_ATXN7', 'ENSG00000163636_PSMD6', 'ENSG00000163638_ADAMTS9', 'ENSG00000163644_PPM1K', 'ENSG00000163655_GMPS', 'ENSG00000163659_TIPARP', 'ENSG00000163660_CCNL1', 'ENSG00000163661_PTX3', 'ENSG00000163666_HESX1', 'ENSG00000163681_SLMAP', 'ENSG00000163682_RPL9', 'ENSG00000163683_SMIM14', 'ENSG00000163684_RPP14', 'ENSG00000163686_ABHD6', 'ENSG00000163689_C3orf67', 'ENSG00000163694_RBM47', 'ENSG00000163697_APBB2', 'ENSG00000163701_IL17RE', 'ENSG00000163702_IL17RC', 'ENSG00000163703_CRELD1', 'ENSG00000163704_PRRT3', 'ENSG00000163710_PCOLCE2', 'ENSG00000163714_U2SURP', 'ENSG00000163719_MTMR14', 'ENSG00000163728_TTC14', 'ENSG00000163734_CXCL3', 'ENSG00000163736_PPBP', 'ENSG00000163737_PF4', 'ENSG00000163738_MTHFD2L', 'ENSG00000163739_CXCL1', 'ENSG00000163743_RCHY1', 'ENSG00000163751_CPA3', 'ENSG00000163754_GYG1', 'ENSG00000163755_HPS3', 'ENSG00000163762_TM4SF18', 'ENSG00000163781_TOPBP1', 'ENSG00000163785_RYK', 'ENSG00000163788_SNRK', 'ENSG00000163794_UCN', 'ENSG00000163795_ZNF513', 'ENSG00000163798_SLC4A1AP', 'ENSG00000163803_PLB1', 'ENSG00000163806_SPDYA', 'ENSG00000163807_KIAA1143', 'ENSG00000163808_KIF15', 'ENSG00000163811_WDR43', 'ENSG00000163812_ZDHHC3', 'ENSG00000163814_CDCP1', 'ENSG00000163815_CLEC3B', 'ENSG00000163818_LZTFL1', 'ENSG00000163820_FYCO1', 'ENSG00000163823_CCR1', 'ENSG00000163827_LRRC2', 'ENSG00000163832_ELP6', 'ENSG00000163833_FBXO40', 'ENSG00000163840_DTX3L', 'ENSG00000163848_ZNF148', 'ENSG00000163864_NMNAT3', 'ENSG00000163866_SMIM12', 'ENSG00000163867_ZMYM6', 'ENSG00000163870_TPRA1', 'ENSG00000163872_YEATS2', 'ENSG00000163874_ZC3H12A', 'ENSG00000163875_MEAF6', 'ENSG00000163877_SNIP1', 'ENSG00000163882_POLR2H', 'ENSG00000163884_KLF15', 'ENSG00000163888_CAMK2N2', 'ENSG00000163900_TMEM41A', 'ENSG00000163902_RPN1', 'ENSG00000163904_SENP2', 'ENSG00000163909_HEYL', 'ENSG00000163913_IFT122', 'ENSG00000163918_RFC4', 'ENSG00000163923_RPL39L', 'ENSG00000163930_BAP1', 'ENSG00000163931_TKT', 'ENSG00000163932_PRKCD', 'ENSG00000163933_RFT1', 'ENSG00000163935_SFMBT1', 'ENSG00000163938_GNL3', 'ENSG00000163939_PBRM1', 'ENSG00000163945_UVSSA', 'ENSG00000163946_FAM208A', 'ENSG00000163947_ARHGEF3', 'ENSG00000163950_SLBP', 'ENSG00000163956_LRPAP1', 'ENSG00000163958_ZDHHC19', 'ENSG00000163959_SLC51A', 'ENSG00000163960_UBXN7', 'ENSG00000163961_RNF168', 'ENSG00000163964_PIGX', 'ENSG00000163975_MELTF', 'ENSG00000163993_S100P', 'ENSG00000164002_EXO5', 'ENSG00000164008_C1orf50', 'ENSG00000164010_ERMAP', 'ENSG00000164011_ZNF691', 'ENSG00000164022_AIMP1', 'ENSG00000164023_SGMS2', 'ENSG00000164024_METAP1', 'ENSG00000164031_DNAJB14', 'ENSG00000164032_H2AFZ', 'ENSG00000164035_EMCN', 'ENSG00000164037_SLC9B1', 'ENSG00000164038_SLC9B2', 'ENSG00000164039_BDH2', 'ENSG00000164040_PGRMC2', 'ENSG00000164045_CDC25A', 'ENSG00000164047_CAMP', 'ENSG00000164048_ZNF589', 'ENSG00000164050_PLXNB1', 'ENSG00000164051_CCDC51', 'ENSG00000164053_ATRIP', 'ENSG00000164054_SHISA5', 'ENSG00000164056_SPRY1', 'ENSG00000164062_APEH', 'ENSG00000164066_INTU', 'ENSG00000164068_RNF123', 'ENSG00000164070_HSPA4L', 'ENSG00000164073_MFSD8', 'ENSG00000164074_ABHD18', 'ENSG00000164077_MON1A', 'ENSG00000164080_RAD54L2', 'ENSG00000164081_TEX264', 'ENSG00000164086_DUSP7', 'ENSG00000164087_POC1A', 'ENSG00000164088_PPM1M', 'ENSG00000164091_WDR82', 'ENSG00000164096_C4orf3', 'ENSG00000164100_NDST3', 'ENSG00000164104_HMGB2', 'ENSG00000164105_SAP30', 'ENSG00000164106_SCRG1', 'ENSG00000164109_MAD2L1', 'ENSG00000164111_ANXA5', 'ENSG00000164114_MAP9', 'ENSG00000164116_GUCY1A1', 'ENSG00000164117_FBXO8', 'ENSG00000164118_CEP44', 'ENSG00000164120_HPGD', 'ENSG00000164124_TMEM144', 'ENSG00000164125_FAM198B', 'ENSG00000164134_NAA15', 'ENSG00000164136_IL15', 'ENSG00000164144_ARFIP1', 'ENSG00000164151_ICE1', 'ENSG00000164161_HHIP', 'ENSG00000164162_ANAPC10', 'ENSG00000164163_ABCE1', 'ENSG00000164164_OTUD4', 'ENSG00000164167_LSM6', 'ENSG00000164168_TMEM184C', 'ENSG00000164169_PRMT9', 'ENSG00000164171_ITGA2', 'ENSG00000164172_MOCS2', 'ENSG00000164180_TMEM161B', 'ENSG00000164181_ELOVL7', 'ENSG00000164182_NDUFAF2', 'ENSG00000164187_LMBRD2', 'ENSG00000164190_NIPBL', 'ENSG00000164197_RNF180', 'ENSG00000164209_SLC25A46', 'ENSG00000164211_STARD4', 'ENSG00000164219_PGGT1B', 'ENSG00000164220_F2RL2', 'ENSG00000164221_CCDC112', 'ENSG00000164236_ANKRD33B', 'ENSG00000164237_CMBL', 'ENSG00000164241_C5orf63', 'ENSG00000164244_PRRC1', 'ENSG00000164251_F2RL1', 'ENSG00000164252_AGGF1', 'ENSG00000164253_WDR41', 'ENSG00000164258_NDUFS4', 'ENSG00000164284_GRPEL2', 'ENSG00000164291_ARSK', 'ENSG00000164292_RHOBTB3', 'ENSG00000164294_GPX8', 'ENSG00000164296_TIGD6', 'ENSG00000164300_SERINC5', 'ENSG00000164303_ENPP6', 'ENSG00000164304_CAGE1', 'ENSG00000164305_CASP3', 'ENSG00000164306_PRIMPOL', 'ENSG00000164307_ERAP1', 'ENSG00000164308_ERAP2', 'ENSG00000164309_CMYA5', 'ENSG00000164323_CFAP97', 'ENSG00000164327_RICTOR', 'ENSG00000164329_TENT2', 'ENSG00000164331_ANKRA2', 'ENSG00000164332_UBLCP1', 'ENSG00000164338_UTP15', 'ENSG00000164342_TLR3', 'ENSG00000164346_NSA2', 'ENSG00000164347_GFM2', 'ENSG00000164362_TERT', 'ENSG00000164366_CCDC127', 'ENSG00000164398_ACSL6', 'ENSG00000164402_SEPT8', 'ENSG00000164403_SHROOM1', 'ENSG00000164404_GDF9', 'ENSG00000164405_UQCRQ', 'ENSG00000164406_LEAP2', 'ENSG00000164414_SLC35A1', 'ENSG00000164430_CGAS', 'ENSG00000164440_TXLNB', 'ENSG00000164442_CITED2', 'ENSG00000164463_CREBRF', 'ENSG00000164465_DCBLD1', 'ENSG00000164466_SFXN1', 'ENSG00000164483_SAMD3', 'ENSG00000164484_TMEM200A', 'ENSG00000164494_PDSS2', 'ENSG00000164506_STXBP5', 'ENSG00000164512_ANKRD55', 'ENSG00000164520_RAET1E', 'ENSG00000164535_DAGLB', 'ENSG00000164542_KIAA0895', 'ENSG00000164543_STK17A', 'ENSG00000164548_TRA2A', 'ENSG00000164574_GALNT10', 'ENSG00000164576_SAP30L', 'ENSG00000164587_RPS14', 'ENSG00000164591_MYOZ3', 'ENSG00000164597_COG5', 'ENSG00000164603_BMT2', 'ENSG00000164604_GPR85', 'ENSG00000164609_SLU7', 'ENSG00000164610_RP9', 'ENSG00000164611_PTTG1', 'ENSG00000164615_CAMLG', 'ENSG00000164620_RELL2', 'ENSG00000164621_SMAD5-AS1', 'ENSG00000164626_KCNK5', 'ENSG00000164627_KIF6', 'ENSG00000164631_ZNF12', 'ENSG00000164638_SLC29A4', 'ENSG00000164647_STEAP1', 'ENSG00000164649_CDCA7L', 'ENSG00000164654_MIOS', 'ENSG00000164659_KIAA1324L', 'ENSG00000164663_USP49', 'ENSG00000164674_SYTL3', 'ENSG00000164675_IQUB', 'ENSG00000164684_ZNF704', 'ENSG00000164687_FABP5', 'ENSG00000164691_TAGAP', 'ENSG00000164695_CHMP4C', 'ENSG00000164707_SLC13A4', 'ENSG00000164713_BRI3', 'ENSG00000164715_LMTK2', 'ENSG00000164733_CTSB', 'ENSG00000164741_DLC1', 'ENSG00000164742_ADCY1', 'ENSG00000164743_C8orf48', 'ENSG00000164744_SUN3', 'ENSG00000164751_PEX2', 'ENSG00000164754_RAD21', 'ENSG00000164758_MED30', 'ENSG00000164764_SBSPON', 'ENSG00000164776_PHKG1', 'ENSG00000164808_SPIDR', 'ENSG00000164815_ORC5', 'ENSG00000164818_DNAAF5', 'ENSG00000164823_OSGIN2', 'ENSG00000164825_DEFB1', 'ENSG00000164828_SUN1', 'ENSG00000164830_OXR1', 'ENSG00000164841_TMEM74', 'ENSG00000164849_GPR146', 'ENSG00000164855_TMEM184A', 'ENSG00000164877_MICALL2', 'ENSG00000164879_CA3', 'ENSG00000164880_INTS1', 'ENSG00000164885_CDK5', 'ENSG00000164889_SLC4A2', 'ENSG00000164896_FASTK', 'ENSG00000164897_TMUB1', 'ENSG00000164898_FMC1', 'ENSG00000164902_PHAX', 'ENSG00000164904_ALDH7A1', 'ENSG00000164916_FOXK1', 'ENSG00000164919_COX6C', 'ENSG00000164924_YWHAZ', 'ENSG00000164929_BAALC', 'ENSG00000164930_FZD6', 'ENSG00000164932_CTHRC1', 'ENSG00000164933_SLC25A32', 'ENSG00000164934_DCAF13', 'ENSG00000164935_DCSTAMP', 'ENSG00000164938_TP53INP1', 'ENSG00000164941_INTS8', 'ENSG00000164944_VIRMA', 'ENSG00000164946_FREM1', 'ENSG00000164949_GEM', 'ENSG00000164951_PDP1', 'ENSG00000164953_TMEM67', 'ENSG00000164961_WASHC5', 'ENSG00000164967_RPP25L', 'ENSG00000164970_FAM219A', 'ENSG00000164972_C9orf24', 'ENSG00000164975_SNAPC3', 'ENSG00000164976_MYORG', 'ENSG00000164978_NUDT2', 'ENSG00000164983_TMEM65', 'ENSG00000164985_PSIP1', 'ENSG00000164989_CCDC171', 'ENSG00000165006_UBAP1', 'ENSG00000165025_SYK', 'ENSG00000165028_NIPSNAP3B', 'ENSG00000165029_ABCA1', 'ENSG00000165030_NFIL3', 'ENSG00000165046_LETM2', 'ENSG00000165055_METTL2B', 'ENSG00000165060_FXN', 'ENSG00000165071_TMEM71', 'ENSG00000165072_MAMDC2', 'ENSG00000165092_ALDH1A1', 'ENSG00000165097_KDM1B', 'ENSG00000165102_HGSNAT', 'ENSG00000165113_GKAP1', 'ENSG00000165115_KIF27', 'ENSG00000165118_C9orf64', 'ENSG00000165119_HNRNPK', 'ENSG00000165121_AL353743.1', 'ENSG00000165138_ANKS6', 'ENSG00000165140_FBP1', 'ENSG00000165152_TMEM246', 'ENSG00000165156_ZHX1', 'ENSG00000165168_CYBB', 'ENSG00000165169_DYNLT3', 'ENSG00000165171_METTL27', 'ENSG00000165175_MID1IP1', 'ENSG00000165178_NCF1C', 'ENSG00000165182_CXorf58', 'ENSG00000165185_KIAA1958', 'ENSG00000165195_PIGA', 'ENSG00000165209_STRBP', 'ENSG00000165219_GAPVD1', 'ENSG00000165233_CARD19', 'ENSG00000165240_ATP7A', 'ENSG00000165244_ZNF367', 'ENSG00000165259_HDX', 'ENSG00000165264_NDUFB6', 'ENSG00000165271_NOL6', 'ENSG00000165272_AQP3', 'ENSG00000165275_TRMT10B', 'ENSG00000165280_VCP', 'ENSG00000165282_PIGO', 'ENSG00000165283_STOML2', 'ENSG00000165288_BRWD3', 'ENSG00000165300_SLITRK5', 'ENSG00000165304_MELK', 'ENSG00000165312_OTUD1', 'ENSG00000165322_ARHGAP12', 'ENSG00000165325_DEUP1', 'ENSG00000165338_HECTD2', 'ENSG00000165349_SLC7A3', 'ENSG00000165355_FBXO33', 'ENSG00000165359_INTS6L', 'ENSG00000165389_SPTSSA', 'ENSG00000165392_WRN', 'ENSG00000165406_MARCH8', 'ENSG00000165409_TSHR', 'ENSG00000165410_CFL2', 'ENSG00000165416_SUGT1', 'ENSG00000165417_GTF2A1', 'ENSG00000165424_ZCCHC24', 'ENSG00000165434_PGM2L1', 'ENSG00000165449_SLC16A9', 'ENSG00000165457_FOLR2', 'ENSG00000165458_INPPL1', 'ENSG00000165475_CRYL1', 'ENSG00000165476_REEP3', 'ENSG00000165480_SKA3', 'ENSG00000165487_MICU2', 'ENSG00000165490_DDIAS', 'ENSG00000165494_PCF11', 'ENSG00000165501_LRR1', 'ENSG00000165502_RPL36AL', 'ENSG00000165506_DNAAF2', 'ENSG00000165507_DEPP1', 'ENSG00000165511_C10orf25', 'ENSG00000165512_ZNF22', 'ENSG00000165516_KLHDC2', 'ENSG00000165521_EML5', 'ENSG00000165525_NEMF', 'ENSG00000165526_RPUSD4', 'ENSG00000165527_ARF6', 'ENSG00000165533_TTC8', 'ENSG00000165548_TMEM63C', 'ENSG00000165555_NOXRED1', 'ENSG00000165568_AKR1E2', 'ENSG00000165572_KBTBD6', 'ENSG00000165591_FAAH2', 'ENSG00000165609_NUDT5', 'ENSG00000165617_DACT1', 'ENSG00000165626_BEND7', 'ENSG00000165629_ATP5F1C', 'ENSG00000165630_PRPF18', 'ENSG00000165632_TAF3', 'ENSG00000165633_VSTM4', 'ENSG00000165637_VDAC2', 'ENSG00000165644_COMTD1', 'ENSG00000165646_SLC18A2', 'ENSG00000165650_PDZD8', 'ENSG00000165655_ZNF503', 'ENSG00000165660_ABRAXAS2', 'ENSG00000165661_QSOX2', 'ENSG00000165669_FAM204A', 'ENSG00000165671_NSD1', 'ENSG00000165672_PRDX3', 'ENSG00000165675_ENOX2', 'ENSG00000165678_GHITM', 'ENSG00000165682_CLEC1B', 'ENSG00000165684_SNAPC4', 'ENSG00000165685_TMEM52B', 'ENSG00000165688_PMPCA', 'ENSG00000165689_ENTR1', 'ENSG00000165695_AK8', 'ENSG00000165698_SPACA9', 'ENSG00000165699_TSC1', 'ENSG00000165702_GFI1B', 'ENSG00000165704_HPRT1', 'ENSG00000165714_BORCS5', 'ENSG00000165716_FAM69B', 'ENSG00000165724_ZMYND19', 'ENSG00000165730_STOX1', 'ENSG00000165731_RET', 'ENSG00000165732_DDX21', 'ENSG00000165733_BMS1', 'ENSG00000165752_STK32C', 'ENSG00000165757_JCAD', 'ENSG00000165775_FUNDC2', 'ENSG00000165782_PIP4P1', 'ENSG00000165792_METTL17', 'ENSG00000165795_NDRG2', 'ENSG00000165801_ARHGEF40', 'ENSG00000165802_NSMF', 'ENSG00000165804_ZNF219', 'ENSG00000165806_CASP7', 'ENSG00000165807_PPP1R36', 'ENSG00000165813_CCDC186', 'ENSG00000165816_VWA2', 'ENSG00000165819_METTL3', 'ENSG00000165821_SALL2', 'ENSG00000165832_TRUB1', 'ENSG00000165861_ZFYVE1', 'ENSG00000165863_C10orf82', 'ENSG00000165868_HSPA12A', 'ENSG00000165874_SHLD2P1', 'ENSG00000165879_FRAT1', 'ENSG00000165886_UBTD1', 'ENSG00000165887_ANKRD2', 'ENSG00000165891_E2F7', 'ENSG00000165895_ARHGAP42', 'ENSG00000165898_ISCA2', 'ENSG00000165912_PACSIN3', 'ENSG00000165914_TTC7B', 'ENSG00000165915_SLC39A13', 'ENSG00000165916_PSMC3', 'ENSG00000165917_RAPSN', 'ENSG00000165923_AGBL2', 'ENSG00000165929_TC2N', 'ENSG00000165934_CPSF2', 'ENSG00000165935_SMCO2', 'ENSG00000165943_MOAP1', 'ENSG00000165948_IFI27L1', 'ENSG00000165949_IFI27', 'ENSG00000165959_CLMN', 'ENSG00000165966_PDZRN4', 'ENSG00000165972_CCDC38', 'ENSG00000165983_PTER', 'ENSG00000165985_C1QL3', 'ENSG00000165995_CACNB2', 'ENSG00000165996_HACD1', 'ENSG00000165997_ARL5B', 'ENSG00000166002_SMCO4', 'ENSG00000166004_CEP295', 'ENSG00000166012_TAF1D', 'ENSG00000166016_ABTB2', 'ENSG00000166024_R3HCC1L', 'ENSG00000166025_AMOTL1', 'ENSG00000166033_HTRA1', 'ENSG00000166035_LIPC', 'ENSG00000166037_CEP57', 'ENSG00000166046_TCP11L2', 'ENSG00000166068_SPRED1', 'ENSG00000166086_JAM3', 'ENSG00000166091_CMTM5', 'ENSG00000166111_SVOP', 'ENSG00000166123_GPT2', 'ENSG00000166126_AMN', 'ENSG00000166128_RAB8B', 'ENSG00000166130_IKBIP', 'ENSG00000166133_RPUSD2', 'ENSG00000166135_HIF1AN', 'ENSG00000166136_NDUFB8', 'ENSG00000166140_ZFYVE19', 'ENSG00000166145_SPINT1', 'ENSG00000166147_FBN1', 'ENSG00000166153_DEPDC4', 'ENSG00000166164_BRD7', 'ENSG00000166165_CKB', 'ENSG00000166166_TRMT61A', 'ENSG00000166167_BTRC', 'ENSG00000166169_POLL', 'ENSG00000166170_BAG5', 'ENSG00000166171_DPCD', 'ENSG00000166173_LARP6', 'ENSG00000166181_API5', 'ENSG00000166188_ZNF319', 'ENSG00000166189_HPS6', 'ENSG00000166192_SENP8', 'ENSG00000166197_NOLC1', 'ENSG00000166199_ALKBH3', 'ENSG00000166200_COPS2', 'ENSG00000166206_GABRB3', 'ENSG00000166224_SGPL1', 'ENSG00000166225_FRS2', 'ENSG00000166226_CCT2', 'ENSG00000166228_PCBD1', 'ENSG00000166233_ARIH1', 'ENSG00000166246_C16orf71', 'ENSG00000166260_COX11', 'ENSG00000166261_ZNF202', 'ENSG00000166262_FAM227B', 'ENSG00000166263_STXBP4', 'ENSG00000166265_CYYR1', 'ENSG00000166266_CUL5', 'ENSG00000166272_WBP1L', 'ENSG00000166275_BORCS7', 'ENSG00000166278_C2', 'ENSG00000166289_PLEKHF1', 'ENSG00000166295_ANAPC16', 'ENSG00000166311_SMPD1', 'ENSG00000166313_APBB1', 'ENSG00000166317_SYNPO2L', 'ENSG00000166321_NUDT13', 'ENSG00000166323_C11orf65', 'ENSG00000166326_TRIM44', 'ENSG00000166333_ILK', 'ENSG00000166337_TAF10', 'ENSG00000166340_TPP1', 'ENSG00000166341_DCHS1', 'ENSG00000166343_MSS51', 'ENSG00000166347_CYB5A', 'ENSG00000166348_USP54', 'ENSG00000166349_RAG1', 'ENSG00000166352_C11orf74', 'ENSG00000166377_ATP9B', 'ENSG00000166387_PPFIBP2', 'ENSG00000166398_KIAA0355', 'ENSG00000166401_SERPINB8', 'ENSG00000166402_TUB', 'ENSG00000166405_RIC3', 'ENSG00000166407_LMO1', 'ENSG00000166411_IDH3A', 'ENSG00000166415_WDR72', 'ENSG00000166426_CRABP1', 'ENSG00000166428_PLD4', 'ENSG00000166432_ZMAT1', 'ENSG00000166435_XRRA1', 'ENSG00000166436_TRIM66', 'ENSG00000166439_RNF169', 'ENSG00000166441_RPL27A', 'ENSG00000166444_ST5', 'ENSG00000166446_CDYL2', 'ENSG00000166450_PRTG', 'ENSG00000166451_CENPN', 'ENSG00000166452_AKIP1', 'ENSG00000166454_ATMIN', 'ENSG00000166455_C16orf46', 'ENSG00000166471_TMEM41B', 'ENSG00000166477_LEO1', 'ENSG00000166478_ZNF143', 'ENSG00000166479_TMX3', 'ENSG00000166482_MFAP4', 'ENSG00000166483_WEE1', 'ENSG00000166484_MAPK7', 'ENSG00000166501_PRKCB', 'ENSG00000166503_HDGFL3', 'ENSG00000166507_NDST2', 'ENSG00000166508_MCM7', 'ENSG00000166510_CCDC68', 'ENSG00000166526_ZNF3', 'ENSG00000166527_CLEC4D', 'ENSG00000166529_ZSCAN21', 'ENSG00000166532_RIMKLB', 'ENSG00000166548_TK2', 'ENSG00000166557_TMED3', 'ENSG00000166562_SEC11C', 'ENSG00000166575_TMEM135', 'ENSG00000166578_IQCD', 'ENSG00000166579_NDEL1', 'ENSG00000166582_CENPV', 'ENSG00000166595_CIAO2B', 'ENSG00000166598_HSP90B1', 'ENSG00000166619_BLCAP', 'ENSG00000166664_CHRFAM7A', 'ENSG00000166669_ATF7IP2', 'ENSG00000166676_TVP23A', 'ENSG00000166681_BEX3', 'ENSG00000166682_TMPRSS5', 'ENSG00000166685_COG1', 'ENSG00000166689_PLEKHA7', 'ENSG00000166704_ZNF606', 'ENSG00000166707_ZCCHC18', 'ENSG00000166710_B2M', 'ENSG00000166716_ZNF592', 'ENSG00000166734_CASC4', 'ENSG00000166741_NNMT', 'ENSG00000166743_ACSM1', 'ENSG00000166747_AP1G1', 'ENSG00000166750_SLFN5', 'ENSG00000166762_CATSPER2', 'ENSG00000166770_ZNF667-AS1', 'ENSG00000166780_C16orf45', 'ENSG00000166783_MARF1', 'ENSG00000166788_SAAL1', 'ENSG00000166793_YPEL4', 'ENSG00000166794_PPIB', 'ENSG00000166796_LDHC', 'ENSG00000166797_CIAO2A', 'ENSG00000166800_LDHAL6A', 'ENSG00000166801_FAM111A', 'ENSG00000166803_PCLAF', 'ENSG00000166813_KIF7', 'ENSG00000166816_LDHD', 'ENSG00000166821_PEX11A', 'ENSG00000166822_TMEM170A', 'ENSG00000166823_MESP1', 'ENSG00000166825_ANPEP', 'ENSG00000166831_RBPMS2', 'ENSG00000166833_NAV2', 'ENSG00000166839_ANKDD1A', 'ENSG00000166840_GLYATL1', 'ENSG00000166845_C18orf54', 'ENSG00000166847_DCTN5', 'ENSG00000166848_TERF2IP', 'ENSG00000166851_PLK1', 'ENSG00000166855_CLPX', 'ENSG00000166860_ZBTB39', 'ENSG00000166866_MYO1A', 'ENSG00000166881_NEMP1', 'ENSG00000166886_NAB2', 'ENSG00000166887_VPS39', 'ENSG00000166888_STAT6', 'ENSG00000166889_PATL1', 'ENSG00000166896_ATP23', 'ENSG00000166900_STX3', 'ENSG00000166902_MRPL16', 'ENSG00000166908_PIP4K2C', 'ENSG00000166912_MTMR10', 'ENSG00000166913_YWHAB', 'ENSG00000166920_C15orf48', 'ENSG00000166922_SCG5', 'ENSG00000166923_GREM1', 'ENSG00000166924_NYAP1', 'ENSG00000166925_TSC22D4', 'ENSG00000166927_MS4A7', 'ENSG00000166938_DIS3L', 'ENSG00000166946_CCNDBP1', 'ENSG00000166947_EPB42', 'ENSG00000166949_SMAD3', 'ENSG00000166963_MAP1A', 'ENSG00000166965_RCCD1', 'ENSG00000166971_AKTIP', 'ENSG00000166974_MAPRE2', 'ENSG00000166979_EVA1C', 'ENSG00000166986_MARS', 'ENSG00000166987_MBD6', 'ENSG00000166997_CNPY4', 'ENSG00000167004_PDIA3', 'ENSG00000167005_NUDT21', 'ENSG00000167034_NKX3-1', 'ENSG00000167037_SGSM1', 'ENSG00000167046_AL357033.1', 'ENSG00000167065_DUSP18', 'ENSG00000167074_TEF', 'ENSG00000167077_MEI1', 'ENSG00000167081_PBX3', 'ENSG00000167083_GNGT2', 'ENSG00000167085_PHB', 'ENSG00000167088_SNRPD1', 'ENSG00000167100_SAMD14', 'ENSG00000167103_PIP5KL1', 'ENSG00000167105_TMEM92', 'ENSG00000167106_FAM102A', 'ENSG00000167107_ACSF2', 'ENSG00000167110_GOLGA2', 'ENSG00000167112_TRUB2', 'ENSG00000167113_COQ4', 'ENSG00000167114_SLC27A4', 'ENSG00000167117_ANKRD40CL', 'ENSG00000167118_URM1', 'ENSG00000167123_CERCAM', 'ENSG00000167130_DOLPP1', 'ENSG00000167136_ENDOG', 'ENSG00000167173_C15orf39', 'ENSG00000167182_SP2', 'ENSG00000167186_COQ7', 'ENSG00000167193_CRK', 'ENSG00000167196_FBXO22', 'ENSG00000167202_TBC1D2B', 'ENSG00000167207_NOD2', 'ENSG00000167208_SNX20', 'ENSG00000167216_KATNAL2', 'ENSG00000167220_HDHD2', 'ENSG00000167232_ZNF91', 'ENSG00000167257_RNF214', 'ENSG00000167258_CDK12', 'ENSG00000167261_DPEP2', 'ENSG00000167264_DUS2', 'ENSG00000167272_POP5', 'ENSG00000167280_ENGASE', 'ENSG00000167281_RBFOX3', 'ENSG00000167283_ATP5MG', 'ENSG00000167286_CD3D', 'ENSG00000167291_TBC1D16', 'ENSG00000167302_TEPSIN', 'ENSG00000167311_ART5', 'ENSG00000167315_ACAA2', 'ENSG00000167323_STIM1', 'ENSG00000167325_RRM1', 'ENSG00000167333_TRIM68', 'ENSG00000167363_FN3K', 'ENSG00000167371_PRRT2', 'ENSG00000167377_ZNF23', 'ENSG00000167378_IRGQ', 'ENSG00000167380_ZNF226', 'ENSG00000167384_ZNF180', 'ENSG00000167393_PPP2R3B', 'ENSG00000167394_ZNF668', 'ENSG00000167395_ZNF646', 'ENSG00000167397_VKORC1', 'ENSG00000167414_GNG8', 'ENSG00000167419_LPO', 'ENSG00000167447_SMG8', 'ENSG00000167460_TPM4', 'ENSG00000167461_RAB8A', 'ENSG00000167468_GPX4', 'ENSG00000167470_MIDN', 'ENSG00000167476_JSRP1', 'ENSG00000167483_FAM129C', 'ENSG00000167487_KLHL26', 'ENSG00000167491_GATAD2A', 'ENSG00000167508_MVD', 'ENSG00000167513_CDT1', 'ENSG00000167515_TRAPPC2L', 'ENSG00000167522_ANKRD11', 'ENSG00000167523_SPATA33', 'ENSG00000167524_SGK494', 'ENSG00000167525_PROCA1', 'ENSG00000167526_RPL13', 'ENSG00000167528_ZNF641', 'ENSG00000167535_CACNB3', 'ENSG00000167536_DHRS13', 'ENSG00000167543_TP53I13', 'ENSG00000167548_KMT2D', 'ENSG00000167549_CORO6', 'ENSG00000167550_RHEBL1', 'ENSG00000167552_TUBA1A', 'ENSG00000167553_TUBA1C', 'ENSG00000167554_ZNF610', 'ENSG00000167555_ZNF528', 'ENSG00000167562_ZNF701', 'ENSG00000167565_SERTAD3', 'ENSG00000167566_NCKAP5L', 'ENSG00000167578_RAB4B', 'ENSG00000167595_PROSER3', 'ENSG00000167600_CYP2S1', 'ENSG00000167601_AXL', 'ENSG00000167604_NFKBID', 'ENSG00000167608_TMC4', 'ENSG00000167613_LAIR1', 'ENSG00000167614_TTYH1', 'ENSG00000167615_LENG8', 'ENSG00000167617_CDC42EP5', 'ENSG00000167619_TMEM145', 'ENSG00000167625_ZNF526', 'ENSG00000167632_TRAPPC9', 'ENSG00000167635_ZNF146', 'ENSG00000167637_ZNF283', 'ENSG00000167641_PPP1R14A', 'ENSG00000167642_SPINT2', 'ENSG00000167644_C19orf33', 'ENSG00000167645_YIF1B', 'ENSG00000167646_DNAAF3', 'ENSG00000167653_PSCA', 'ENSG00000167657_DAPK3', 'ENSG00000167658_EEF2', 'ENSG00000167664_TMIGD2', 'ENSG00000167670_CHAF1A', 'ENSG00000167671_UBXN6', 'ENSG00000167674_HDGFL2', 'ENSG00000167680_SEMA6B', 'ENSG00000167685_ZNF444', 'ENSG00000167693_NXN', 'ENSG00000167695_FAM57A', 'ENSG00000167699_GLOD4', 'ENSG00000167700_MFSD3', 'ENSG00000167701_GPT', 'ENSG00000167702_KIFC2', 'ENSG00000167703_SLC43A2', 'ENSG00000167705_RILP', 'ENSG00000167711_SERPINF2', 'ENSG00000167716_WDR81', 'ENSG00000167720_SRR', 'ENSG00000167721_TSR1', 'ENSG00000167723_TRPV3', 'ENSG00000167733_HSD11B1L', 'ENSG00000167740_CYB5D2', 'ENSG00000167747_C19orf48', 'ENSG00000167766_ZNF83', 'ENSG00000167768_KRT1', 'ENSG00000167770_OTUB1', 'ENSG00000167771_RCOR2', 'ENSG00000167772_ANGPTL4', 'ENSG00000167775_CD320', 'ENSG00000167778_SPRYD3', 'ENSG00000167779_IGFBP6', 'ENSG00000167785_ZNF558', 'ENSG00000167792_NDUFV1', 'ENSG00000167797_CDK2AP2', 'ENSG00000167799_NUDT8', 'ENSG00000167807_AC011511.1', 'ENSG00000167815_PRDX2', 'ENSG00000167840_ZNF232', 'ENSG00000167842_MIS12', 'ENSG00000167850_CD300C', 'ENSG00000167851_CD300A', 'ENSG00000167862_MRPL58', 'ENSG00000167863_ATP5PD', 'ENSG00000167874_TMEM88', 'ENSG00000167880_EVPL', 'ENSG00000167881_SRP68', 'ENSG00000167895_TMC8', 'ENSG00000167900_TK1', 'ENSG00000167904_TMEM68', 'ENSG00000167912_AC090152.1', 'ENSG00000167920_TMEM99', 'ENSG00000167925_GHDC', 'ENSG00000167930_FAM234A', 'ENSG00000167962_ZNF598', 'ENSG00000167964_RAB26', 'ENSG00000167965_MLST8', 'ENSG00000167967_E4F1', 'ENSG00000167968_DNASE1L2', 'ENSG00000167969_ECI1', 'ENSG00000167971_CASKIN1', 'ENSG00000167972_ABCA3', 'ENSG00000167977_KCTD5', 'ENSG00000167978_SRRM2', 'ENSG00000167981_ZNF597', 'ENSG00000167984_NLRC3', 'ENSG00000167985_SDHAF2', 'ENSG00000167986_DDB1', 'ENSG00000167987_VPS37C', 'ENSG00000167992_VWCE', 'ENSG00000167994_RAB3IL1', 'ENSG00000167995_BEST1', 'ENSG00000167996_FTH1', 'ENSG00000168000_BSCL2', 'ENSG00000168002_POLR2G', 'ENSG00000168003_SLC3A2', 'ENSG00000168004_HRASLS5', 'ENSG00000168005_SPINDOC', 'ENSG00000168010_ATG16L2', 'ENSG00000168014_C2CD3', 'ENSG00000168016_TRANK1', 'ENSG00000168026_TTC21A', 'ENSG00000168028_RPSA', 'ENSG00000168036_CTNNB1', 'ENSG00000168038_ULK4', 'ENSG00000168040_FADD', 'ENSG00000168056_LTBP3', 'ENSG00000168060_NAALADL1', 'ENSG00000168061_SAC3D1', 'ENSG00000168062_BATF2', 'ENSG00000168066_SF1', 'ENSG00000168067_MAP4K2', 'ENSG00000168071_CCDC88B', 'ENSG00000168077_SCARA3', 'ENSG00000168078_PBK', 'ENSG00000168081_PNOC', 'ENSG00000168090_COPS6', 'ENSG00000168092_PAFAH1B2', 'ENSG00000168096_ANKS3', 'ENSG00000168101_NUDT16L1', 'ENSG00000168116_KIAA1586', 'ENSG00000168118_RAB4A', 'ENSG00000168137_SETD5', 'ENSG00000168152_THAP9', 'ENSG00000168159_RNF187', 'ENSG00000168172_HOOK3', 'ENSG00000168175_MAPK1IP1L', 'ENSG00000168209_DDIT4', 'ENSG00000168214_RBPJ', 'ENSG00000168216_LMBRD1', 'ENSG00000168228_ZCCHC4', 'ENSG00000168234_TTC39C', 'ENSG00000168237_GLYCTK', 'ENSG00000168243_GNG4', 'ENSG00000168246_UBTD2', 'ENSG00000168255_POLR2J3', 'ENSG00000168256_NKIRAS2', 'ENSG00000168259_DNAJC7', 'ENSG00000168264_IRF2BP2', 'ENSG00000168268_NT5DC2', 'ENSG00000168273_SMIM4', 'ENSG00000168275_COA6', 'ENSG00000168280_KIF5C', 'ENSG00000168282_MGAT2', 'ENSG00000168283_BMI1', 'ENSG00000168286_THAP11', 'ENSG00000168288_MMADHC', 'ENSG00000168291_PDHB', 'ENSG00000168297_PXK', 'ENSG00000168298_HIST1H1E', 'ENSG00000168300_PCMTD1', 'ENSG00000168301_KCTD6', 'ENSG00000168303_MPLKIP', 'ENSG00000168306_ACOX2', 'ENSG00000168310_IRF2', 'ENSG00000168329_CX3CR1', 'ENSG00000168350_DEGS2', 'ENSG00000168356_SCN11A', 'ENSG00000168374_ARF4', 'ENSG00000168385_SEPT2', 'ENSG00000168386_FILIP1L', 'ENSG00000168389_MFSD2A', 'ENSG00000168393_DTYMK', 'ENSG00000168394_TAP1', 'ENSG00000168395_ING5', 'ENSG00000168397_ATG4B', 'ENSG00000168404_MLKL', 'ENSG00000168405_CMAHP', 'ENSG00000168411_RFWD3', 'ENSG00000168421_RHOH', 'ENSG00000168434_COG7', 'ENSG00000168438_CDC40', 'ENSG00000168439_STIP1', 'ENSG00000168461_RAB31', 'ENSG00000168476_REEP4', 'ENSG00000168477_TNXB', 'ENSG00000168487_BMP1', 'ENSG00000168488_ATXN2L', 'ENSG00000168495_POLR3D', 'ENSG00000168496_FEN1', 'ENSG00000168497_CAVIN2', 'ENSG00000168502_MTCL1', 'ENSG00000168517_HEXIM2', 'ENSG00000168522_FNTA', 'ENSG00000168528_SERINC2', 'ENSG00000168538_TRAPPC11', 'ENSG00000168556_ING2', 'ENSG00000168564_CDKN2AIP', 'ENSG00000168566_SNRNP48', 'ENSG00000168569_TMEM223', 'ENSG00000168575_SLC20A2', 'ENSG00000168589_DYNLRB2', 'ENSG00000168591_TMUB2', 'ENSG00000168610_STAT3', 'ENSG00000168612_ZSWIM1', 'ENSG00000168615_ADAM9', 'ENSG00000168653_NDUFS5', 'ENSG00000168661_ZNF30', 'ENSG00000168671_UGT3A2', 'ENSG00000168672_FAM84B', 'ENSG00000168675_LDLRAD4', 'ENSG00000168676_KCTD19', 'ENSG00000168679_SLC16A4', 'ENSG00000168685_IL7R', 'ENSG00000168701_TMEM208', 'ENSG00000168710_AHCYL1', 'ENSG00000168724_DNAJC21', 'ENSG00000168734_PKIG', 'ENSG00000168754_FAM178B', 'ENSG00000168758_SEMA4C', 'ENSG00000168763_CNNM3', 'ENSG00000168765_GSTM4', 'ENSG00000168769_TET2', 'ENSG00000168778_TCTN2', 'ENSG00000168779_SHOX2', 'ENSG00000168781_PPIP5K1', 'ENSG00000168785_TSPAN5', 'ENSG00000168792_ABHD15', 'ENSG00000168795_ZBTB5', 'ENSG00000168802_CHTF8', 'ENSG00000168803_ADAL', 'ENSG00000168806_LCMT2', 'ENSG00000168807_SNTB2', 'ENSG00000168811_IL12A', 'ENSG00000168813_ZNF507', 'ENSG00000168818_STX18', 'ENSG00000168826_ZBTB49', 'ENSG00000168827_GFM1', 'ENSG00000168852_TPTE2P5', 'ENSG00000168872_DDX19A', 'ENSG00000168876_ANKRD49', 'ENSG00000168883_USP39', 'ENSG00000168884_TNIP2', 'ENSG00000168887_C2orf68', 'ENSG00000168890_TMEM150A', 'ENSG00000168894_RNF181', 'ENSG00000168899_VAMP5', 'ENSG00000168904_LRRC28', 'ENSG00000168906_MAT2A', 'ENSG00000168913_ENHO', 'ENSG00000168916_ZNF608', 'ENSG00000168917_SLC35G2', 'ENSG00000168918_INPP5D', 'ENSG00000168924_LETM1', 'ENSG00000168936_TMEM129', 'ENSG00000168938_PPIC', 'ENSG00000168939_SPRY3', 'ENSG00000168944_CEP120', 'ENSG00000168952_STXBP6', 'ENSG00000168958_MFF', 'ENSG00000168961_LGALS9', 'ENSG00000168970_JMJD7-PLA2G4B', 'ENSG00000168994_PXDC1', 'ENSG00000168995_SIGLEC7', 'ENSG00000169016_E2F6', 'ENSG00000169018_FEM1B', 'ENSG00000169019_COMMD8', 'ENSG00000169020_ATP5ME', 'ENSG00000169021_UQCRFS1', 'ENSG00000169026_SLC49A3', 'ENSG00000169032_MAP2K1', 'ENSG00000169045_HNRNPH1', 'ENSG00000169047_IRS1', 'ENSG00000169057_MECP2', 'ENSG00000169062_UPF3A', 'ENSG00000169071_ROR2', 'ENSG00000169083_AR', 'ENSG00000169084_DHRSX', 'ENSG00000169087_HSPBAP1', 'ENSG00000169093_ASMTL', 'ENSG00000169100_SLC25A6', 'ENSG00000169105_CHST14', 'ENSG00000169116_PARM1', 'ENSG00000169118_CSNK1G1', 'ENSG00000169122_FAM110B', 'ENSG00000169126_ARMC4', 'ENSG00000169129_AFAP1L2', 'ENSG00000169131_ZNF354A', 'ENSG00000169136_ATF5', 'ENSG00000169139_UBE2V2', 'ENSG00000169155_ZBTB43', 'ENSG00000169174_PCSK9', 'ENSG00000169180_XPO6', 'ENSG00000169184_MN1', 'ENSG00000169188_APEX2', 'ENSG00000169189_NSMCE1', 'ENSG00000169193_CCDC126', 'ENSG00000169208_OR10G3', 'ENSG00000169213_RAB3B', 'ENSG00000169217_CD2BP2', 'ENSG00000169220_RGS14', 'ENSG00000169221_TBC1D10B', 'ENSG00000169223_LMAN2', 'ENSG00000169224_GCSAML', 'ENSG00000169228_RAB24', 'ENSG00000169230_PRELID1', 'ENSG00000169231_THBS3', 'ENSG00000169239_CA5B', 'ENSG00000169241_SLC50A1', 'ENSG00000169242_EFNA1', 'ENSG00000169247_SH3TC2', 'ENSG00000169249_ZRSR2', 'ENSG00000169251_NMD3', 'ENSG00000169252_ADRB2', 'ENSG00000169253_AL669983.1', 'ENSG00000169255_B3GALNT1', 'ENSG00000169258_GPRIN1', 'ENSG00000169282_KCNAB1', 'ENSG00000169288_MRPL1', 'ENSG00000169291_SHE', 'ENSG00000169299_PGM2', 'ENSG00000169314_C22orf15', 'ENSG00000169330_KIAA1024', 'ENSG00000169359_SLC33A1', 'ENSG00000169371_SNUPN', 'ENSG00000169372_CRADD', 'ENSG00000169375_SIN3A', 'ENSG00000169379_ARL13B', 'ENSG00000169385_RNASE2', 'ENSG00000169397_RNASE3', 'ENSG00000169398_PTK2', 'ENSG00000169403_PTAFR', 'ENSG00000169410_PTPN9', 'ENSG00000169413_RNASE6', 'ENSG00000169427_KCNK9', 'ENSG00000169429_CXCL8', 'ENSG00000169432_SCN9A', 'ENSG00000169435_RASSF6', 'ENSG00000169439_SDC2', 'ENSG00000169442_CD52', 'ENSG00000169446_MMGT1', 'ENSG00000169490_TM2D2', 'ENSG00000169495_HTRA4', 'ENSG00000169499_PLEKHA2', 'ENSG00000169504_CLIC4', 'ENSG00000169508_GPR183', 'ENSG00000169515_CCDC8', 'ENSG00000169519_METTL15', 'ENSG00000169554_ZEB2', 'ENSG00000169564_PCBP1', 'ENSG00000169567_HINT1', 'ENSG00000169570_DTWD2', 'ENSG00000169575_VPREB1', 'ENSG00000169583_CLIC3', 'ENSG00000169592_INO80E', 'ENSG00000169598_DFFB', 'ENSG00000169599_NFU1', 'ENSG00000169604_ANTXR1', 'ENSG00000169607_CKAP2L', 'ENSG00000169609_C15orf40', 'ENSG00000169612_RAMAC', 'ENSG00000169621_APLF', 'ENSG00000169627_BOLA2B', 'ENSG00000169629_RGPD8', 'ENSG00000169635_HIC2', 'ENSG00000169641_LUZP1', 'ENSG00000169660_HEXDC', 'ENSG00000169668_BCRP2', 'ENSG00000169679_BUB1', 'ENSG00000169682_SPNS1', 'ENSG00000169683_LRRC45', 'ENSG00000169684_CHRNA5', 'ENSG00000169689_CENPX', 'ENSG00000169692_AGPAT2', 'ENSG00000169696_ASPSCR1', 'ENSG00000169704_GP9', 'ENSG00000169710_FASN', 'ENSG00000169714_CNBP', 'ENSG00000169715_MT1E', 'ENSG00000169718_DUS1L', 'ENSG00000169727_GPS1', 'ENSG00000169733_RFNG', 'ENSG00000169738_DCXR', 'ENSG00000169740_ZNF32', 'ENSG00000169750_RAC3', 'ENSG00000169752_NRG4', 'ENSG00000169756_LIMS1', 'ENSG00000169760_NLGN1', 'ENSG00000169762_TAPT1', 'ENSG00000169764_UGP2', 'ENSG00000169813_HNRNPF', 'ENSG00000169814_BTD', 'ENSG00000169826_CSGALNACT2', 'ENSG00000169855_ROBO1', 'ENSG00000169857_AVEN', 'ENSG00000169860_P2RY1', 'ENSG00000169871_TRIM56', 'ENSG00000169877_AHSP', 'ENSG00000169891_REPS2', 'ENSG00000169895_SYAP1', 'ENSG00000169896_ITGAM', 'ENSG00000169902_TPST1', 'ENSG00000169905_TOR1AIP2', 'ENSG00000169908_TM4SF1', 'ENSG00000169914_OTUD3', 'ENSG00000169918_OTUD7A', 'ENSG00000169919_GUSB', 'ENSG00000169925_BRD3', 'ENSG00000169926_KLF13', 'ENSG00000169946_ZFPM2', 'ENSG00000169951_ZNF764', 'ENSG00000169955_ZNF747', 'ENSG00000169957_ZNF768', 'ENSG00000169964_TMEM42', 'ENSG00000169967_MAP3K2', 'ENSG00000169972_PUSL1', 'ENSG00000169976_SF3B5', 'ENSG00000169981_ZNF35', 'ENSG00000169989_TIGD4', 'ENSG00000169991_IFFO2', 'ENSG00000169992_NLGN2', 'ENSG00000170004_CHD3', 'ENSG00000170006_TMEM154', 'ENSG00000170011_MYRIP', 'ENSG00000170017_ALCAM', 'ENSG00000170027_YWHAG', 'ENSG00000170035_UBE2E3', 'ENSG00000170037_CNTROB', 'ENSG00000170043_TRAPPC1', 'ENSG00000170049_KCNAB3', 'ENSG00000170085_SIMC1', 'ENSG00000170088_TMEM192', 'ENSG00000170089_AC106795.1', 'ENSG00000170092_SPDYE5', 'ENSG00000170100_ZNF778', 'ENSG00000170113_NIPA1', 'ENSG00000170142_UBE2E1', 'ENSG00000170144_HNRNPA3', 'ENSG00000170145_SIK2', 'ENSG00000170153_RNF150', 'ENSG00000170160_CCDC144A', 'ENSG00000170161_AL512625.1', 'ENSG00000170175_CHRNB1', 'ENSG00000170180_GYPA', 'ENSG00000170185_USP38', 'ENSG00000170190_SLC16A5', 'ENSG00000170191_NANP', 'ENSG00000170222_ADPRM', 'ENSG00000170234_PWWP2A', 'ENSG00000170242_USP47', 'ENSG00000170248_PDCD6IP', 'ENSG00000170260_ZNF212', 'ENSG00000170264_FAM161A', 'ENSG00000170265_ZNF282', 'ENSG00000170266_GLB1', 'ENSG00000170270_GON7', 'ENSG00000170271_FAXDC2', 'ENSG00000170275_CRTAP', 'ENSG00000170291_ELP5', 'ENSG00000170293_CMTM8', 'ENSG00000170296_GABARAP', 'ENSG00000170298_LGALS9B', 'ENSG00000170310_STX8', 'ENSG00000170312_CDK1', 'ENSG00000170315_UBB', 'ENSG00000170322_NFRKB', 'ENSG00000170325_PRDM10', 'ENSG00000170340_B3GNT2', 'ENSG00000170345_FOS', 'ENSG00000170348_TMED10', 'ENSG00000170364_SETMAR', 'ENSG00000170365_SMAD1', 'ENSG00000170379_TCAF2', 'ENSG00000170381_SEMA3E', 'ENSG00000170382_LRRN2', 'ENSG00000170385_SLC30A1', 'ENSG00000170390_DCLK2', 'ENSG00000170396_ZNF804A', 'ENSG00000170412_GPRC5C', 'ENSG00000170417_TMEM182', 'ENSG00000170421_KRT8', 'ENSG00000170425_ADORA2B', 'ENSG00000170430_MGMT', 'ENSG00000170439_METTL7B', 'ENSG00000170442_KRT86', 'ENSG00000170445_HARS', 'ENSG00000170448_NFXL1', 'ENSG00000170456_DENND5B', 'ENSG00000170458_CD14', 'ENSG00000170464_DNAJC18', 'ENSG00000170468_RIOX1', 'ENSG00000170469_SPATA24', 'ENSG00000170471_RALGAPB', 'ENSG00000170473_PYM1', 'ENSG00000170476_MZB1', 'ENSG00000170482_SLC23A1', 'ENSG00000170500_LONRF2', 'ENSG00000170502_NUDT9', 'ENSG00000170509_HSD17B13', 'ENSG00000170515_PA2G4', 'ENSG00000170522_ELOVL6', 'ENSG00000170525_PFKFB3', 'ENSG00000170540_ARL6IP1', 'ENSG00000170542_SERPINB9', 'ENSG00000170545_SMAGP', 'ENSG00000170558_CDH2', 'ENSG00000170571_EMB', 'ENSG00000170579_DLGAP1', 'ENSG00000170581_STAT2', 'ENSG00000170584_NUDCD2', 'ENSG00000170604_IRF2BP1', 'ENSG00000170606_HSPA4', 'ENSG00000170608_FOXA3', 'ENSG00000170619_COMMD5', 'ENSG00000170627_GTSF1', 'ENSG00000170629_DPY19L2P2', 'ENSG00000170631_ZNF16', 'ENSG00000170632_ARMC10', 'ENSG00000170633_RNF34', 'ENSG00000170634_ACYP2', 'ENSG00000170638_TRABD', 'ENSG00000170653_ATF7', 'ENSG00000170667_RASA4B', 'ENSG00000170677_SOCS6', 'ENSG00000170681_CAVIN4', 'ENSG00000170684_ZNF296', 'ENSG00000170734_POLH', 'ENSG00000170759_KIF5B', 'ENSG00000170776_AKAP13', 'ENSG00000170779_CDCA4', 'ENSG00000170790_OR10A2', 'ENSG00000170791_CHCHD7', 'ENSG00000170801_HTRA3', 'ENSG00000170802_FOXN2', 'ENSG00000170819_BFSP2', 'ENSG00000170832_USP32', 'ENSG00000170835_CEL', 'ENSG00000170836_PPM1D', 'ENSG00000170837_GPR27', 'ENSG00000170846_AC093323.1', 'ENSG00000170852_KBTBD2', 'ENSG00000170854_RIOX2', 'ENSG00000170855_TRIAP1', 'ENSG00000170860_LSM3', 'ENSG00000170871_KIAA0232', 'ENSG00000170873_MTSS1', 'ENSG00000170876_TMEM43', 'ENSG00000170881_RNF139', 'ENSG00000170889_RPS9', 'ENSG00000170890_PLA2G1B', 'ENSG00000170891_CYTL1', 'ENSG00000170892_TSEN34', 'ENSG00000170893_TRH', 'ENSG00000170899_GSTA4', 'ENSG00000170903_MSANTD4', 'ENSG00000170906_NDUFA3', 'ENSG00000170909_OSCAR', 'ENSG00000170915_PAQR8', 'ENSG00000170917_NUDT6', 'ENSG00000170919_TPT1-AS1', 'ENSG00000170921_TANC2', 'ENSG00000170946_DNAJC24', 'ENSG00000170949_ZNF160', 'ENSG00000170954_ZNF415', 'ENSG00000170955_CAVIN3', 'ENSG00000170962_PDGFD', 'ENSG00000170965_PLAC1', 'ENSG00000170989_S1PR1', 'ENSG00000171004_HS6ST2', 'ENSG00000171016_PYGO1', 'ENSG00000171017_LRRC8E', 'ENSG00000171033_PKIA', 'ENSG00000171044_XKR6', 'ENSG00000171045_TSNARE1', 'ENSG00000171049_FPR2', 'ENSG00000171051_FPR1', 'ENSG00000171055_FEZ2', 'ENSG00000171056_SOX7', 'ENSG00000171067_C11orf24', 'ENSG00000171084_FAM86JP', 'ENSG00000171097_KYAT1', 'ENSG00000171100_MTM1', 'ENSG00000171101_SIGLEC17P', 'ENSG00000171103_TRMT61B', 'ENSG00000171105_INSR', 'ENSG00000171109_MFN1', 'ENSG00000171115_GIMAP8', 'ENSG00000171119_NRTN', 'ENSG00000171121_KCNMB3', 'ENSG00000171130_ATP6V0E2', 'ENSG00000171132_PRKCE', 'ENSG00000171135_JAGN1', 'ENSG00000171148_TADA3', 'ENSG00000171150_SOCS5', 'ENSG00000171155_C1GALT1C1', 'ENSG00000171159_C9orf16', 'ENSG00000171160_MORN4', 'ENSG00000171161_ZNF672', 'ENSG00000171163_ZNF692', 'ENSG00000171169_NAIF1', 'ENSG00000171174_RBKS', 'ENSG00000171189_GRIK1', 'ENSG00000171202_TMEM126A', 'ENSG00000171204_TMEM126B', 'ENSG00000171206_TRIM8', 'ENSG00000171208_NETO2', 'ENSG00000171217_CLDN20', 'ENSG00000171219_CDC42BPG', 'ENSG00000171222_SCAND1', 'ENSG00000171223_JUNB', 'ENSG00000171224_FAM241B', 'ENSG00000171227_TMEM37', 'ENSG00000171236_LRG1', 'ENSG00000171241_SHCBP1', 'ENSG00000171262_FAM98B', 'ENSG00000171291_ZNF439', 'ENSG00000171295_ZNF440', 'ENSG00000171298_GAA', 'ENSG00000171302_CANT1', 'ENSG00000171307_ZDHHC16', 'ENSG00000171310_CHST11', 'ENSG00000171311_EXOSC1', 'ENSG00000171314_PGAM1', 'ENSG00000171316_CHD7', 'ENSG00000171320_ESCO2', 'ENSG00000171345_KRT19', 'ENSG00000171357_LURAP1', 'ENSG00000171365_CLCN5', 'ENSG00000171385_KCND3', 'ENSG00000171388_APLN', 'ENSG00000171401_KRT13', 'ENSG00000171408_PDE7B', 'ENSG00000171421_MRPL36', 'ENSG00000171425_ZNF581', 'ENSG00000171428_NAT1', 'ENSG00000171433_GLOD5', 'ENSG00000171443_ZNF524', 'ENSG00000171444_MCC', 'ENSG00000171448_ZBTB26', 'ENSG00000171451_DSEL', 'ENSG00000171453_POLR1C', 'ENSG00000171456_ASXL1', 'ENSG00000171462_DLK2', 'ENSG00000171466_ZNF562', 'ENSG00000171467_ZNF318', 'ENSG00000171469_ZNF561', 'ENSG00000171475_WIPF2', 'ENSG00000171476_HOPX', 'ENSG00000171488_LRRC8C', 'ENSG00000171490_RSL1D1', 'ENSG00000171492_LRRC8D', 'ENSG00000171497_PPID', 'ENSG00000171502_COL24A1', 'ENSG00000171503_ETFDH', 'ENSG00000171509_RXFP1', 'ENSG00000171517_LPAR3', 'ENSG00000171522_PTGER4', 'ENSG00000171530_TBCA', 'ENSG00000171552_BCL2L1', 'ENSG00000171566_PLRG1', 'ENSG00000171574_ZNF584', 'ENSG00000171603_CLSTN1', 'ENSG00000171604_CXXC5', 'ENSG00000171606_ZNF274', 'ENSG00000171608_PIK3CD', 'ENSG00000171612_SLC25A33', 'ENSG00000171617_ENC1', 'ENSG00000171621_SPSB1', 'ENSG00000171631_P2RY6', 'ENSG00000171634_BPTF', 'ENSG00000171643_S100Z', 'ENSG00000171649_ZIK1', 'ENSG00000171657_GPR82', 'ENSG00000171658_NMRAL2P', 'ENSG00000171659_GPR34', 'ENSG00000171681_ATF7IP', 'ENSG00000171695_LKAAEAR1', 'ENSG00000171700_RGS19', 'ENSG00000171703_TCEA2', 'ENSG00000171714_ANO5', 'ENSG00000171720_HDAC3', 'ENSG00000171723_GPHN', 'ENSG00000171729_TMEM51', 'ENSG00000171735_CAMTA1', 'ENSG00000171747_LGALS4', 'ENSG00000171757_LRRC34', 'ENSG00000171763_SPATA5L1', 'ENSG00000171766_GATM', 'ENSG00000171772_SYCE1', 'ENSG00000171777_RASGRP4', 'ENSG00000171786_NHLH1', 'ENSG00000171790_SLFNL1', 'ENSG00000171791_BCL2', 'ENSG00000171792_RHNO1', 'ENSG00000171793_CTPS1', 'ENSG00000171794_UTF1', 'ENSG00000171806_METTL18', 'ENSG00000171811_CFAP46', 'ENSG00000171812_COL8A2', 'ENSG00000171813_PWWP2B', 'ENSG00000171817_ZNF540', 'ENSG00000171823_FBXL14', 'ENSG00000171824_EXOSC10', 'ENSG00000171827_ZNF570', 'ENSG00000171840_NINJ2', 'ENSG00000171843_MLLT3', 'ENSG00000171847_FAM90A1', 'ENSG00000171848_RRM2', 'ENSG00000171853_TRAPPC12', 'ENSG00000171858_RPS21', 'ENSG00000171860_C3AR1', 'ENSG00000171861_MRM3', 'ENSG00000171862_PTEN', 'ENSG00000171863_RPS7', 'ENSG00000171865_RNASEH1', 'ENSG00000171867_PRNP', 'ENSG00000171877_FRMD5', 'ENSG00000171903_CYP4F11', 'ENSG00000171914_TLN2', 'ENSG00000171928_TVP23B', 'ENSG00000171940_ZNF217', 'ENSG00000171943_SRGAP2C', 'ENSG00000171951_SCG2', 'ENSG00000171953_ATPAF2', 'ENSG00000171954_CYP4F22', 'ENSG00000171960_PPIH', 'ENSG00000171962_DRC3', 'ENSG00000171970_ZNF57', 'ENSG00000171984_SHLD1', 'ENSG00000171988_JMJD1C', 'ENSG00000171992_SYNPO', 'ENSG00000172000_ZNF556', 'ENSG00000172006_ZNF554', 'ENSG00000172007_RAB33B', 'ENSG00000172009_THOP1', 'ENSG00000172014_ANKRD20A4', 'ENSG00000172031_EPHX4', 'ENSG00000172037_LAMB2', 'ENSG00000172046_USP19', 'ENSG00000172053_QARS', 'ENSG00000172057_ORMDL3', 'ENSG00000172058_SERF1A', 'ENSG00000172059_KLF11', 'ENSG00000172062_SMN1', 'ENSG00000172071_EIF2AK3', 'ENSG00000172081_MOB3A', 'ENSG00000172086_KRCC1', 'ENSG00000172113_NME6', 'ENSG00000172115_CYCS', 'ENSG00000172116_CD8B', 'ENSG00000172123_SLFN12', 'ENSG00000172137_CALB2', 'ENSG00000172159_FRMD3', 'ENSG00000172164_SNTB1', 'ENSG00000172167_MTBP', 'ENSG00000172171_TEFM', 'ENSG00000172172_MRPL13', 'ENSG00000172175_MALT1', 'ENSG00000172183_ISG20', 'ENSG00000172197_MBOAT1', 'ENSG00000172209_GPR22', 'ENSG00000172216_CEBPB', 'ENSG00000172232_AZU1', 'ENSG00000172236_TPSAB1', 'ENSG00000172239_PAIP1', 'ENSG00000172243_CLEC7A', 'ENSG00000172244_C5orf34', 'ENSG00000172247_C1QTNF4', 'ENSG00000172260_NEGR1', 'ENSG00000172262_ZNF131', 'ENSG00000172264_MACROD2', 'ENSG00000172269_DPAGT1', 'ENSG00000172270_BSG', 'ENSG00000172273_HINFP', 'ENSG00000172292_CERS6', 'ENSG00000172296_SPTLC3', 'ENSG00000172301_COPRS', 'ENSG00000172315_TP53RK', 'ENSG00000172322_CLEC12A', 'ENSG00000172331_BPGM', 'ENSG00000172336_POP7', 'ENSG00000172339_ALG14', 'ENSG00000172340_SUCLG2', 'ENSG00000172345_STARD5', 'ENSG00000172348_RCAN2', 'ENSG00000172349_IL16', 'ENSG00000172354_GNB2', 'ENSG00000172361_CFAP53', 'ENSG00000172366_MCRIP2', 'ENSG00000172375_C2CD2L', 'ENSG00000172379_ARNT2', 'ENSG00000172382_PRSS27', 'ENSG00000172404_DNAJB7', 'ENSG00000172409_CLP1', 'ENSG00000172425_TTC36', 'ENSG00000172426_RSPH9', 'ENSG00000172428_COPS9', 'ENSG00000172432_GTPBP2', 'ENSG00000172456_FGGY', 'ENSG00000172458_IL17D', 'ENSG00000172460_PRSS30P', 'ENSG00000172465_TCEAL1', 'ENSG00000172466_ZNF24', 'ENSG00000172469_MANEA', 'ENSG00000172476_RAB40A', 'ENSG00000172493_AFF1', 'ENSG00000172500_FIBP', 'ENSG00000172508_CARNS1', 'ENSG00000172530_BANP', 'ENSG00000172531_PPP1CA', 'ENSG00000172534_HCFC1', 'ENSG00000172543_CTSW', 'ENSG00000172572_PDE3A', 'ENSG00000172575_RASGRP1', 'ENSG00000172578_KLHL6', 'ENSG00000172586_CHCHD1', 'ENSG00000172590_MRPL52', 'ENSG00000172594_SMPDL3A', 'ENSG00000172602_RND1', 'ENSG00000172613_RAD9A', 'ENSG00000172638_EFEMP2', 'ENSG00000172650_AGAP5', 'ENSG00000172661_WASHC2C', 'ENSG00000172663_TMEM134', 'ENSG00000172667_ZMAT3', 'ENSG00000172671_ZFAND4', 'ENSG00000172687_ZNF738', 'ENSG00000172716_SLFN11', 'ENSG00000172717_FAM71D', 'ENSG00000172725_CORO1B', 'ENSG00000172728_FUT10', 'ENSG00000172731_LRRC20', 'ENSG00000172732_MUS81', 'ENSG00000172733_PURG', 'ENSG00000172738_TMEM217', 'ENSG00000172742_OR4D9', 'ENSG00000172746_RPL12P13', 'ENSG00000172748_ZNF596', 'ENSG00000172752_COL6A5', 'ENSG00000172757_CFL1', 'ENSG00000172765_TMCC1', 'ENSG00000172766_NAA16', 'ENSG00000172771_EFCAB12', 'ENSG00000172775_FAM192A', 'ENSG00000172780_RAB43', 'ENSG00000172785_CBWD1', 'ENSG00000172794_RAB37', 'ENSG00000172795_DCP2', 'ENSG00000172803_SNX32', 'ENSG00000172809_RPL38', 'ENSG00000172817_CYP7B1', 'ENSG00000172819_RARG', 'ENSG00000172824_CES4A', 'ENSG00000172828_CES3', 'ENSG00000172830_SSH3', 'ENSG00000172831_CES2', 'ENSG00000172840_PDP2', 'ENSG00000172845_SP3', 'ENSG00000172867_KRT2', 'ENSG00000172869_DMXL1', 'ENSG00000172878_METAP1D', 'ENSG00000172888_ZNF621', 'ENSG00000172889_EGFL7', 'ENSG00000172890_NADSYN1', 'ENSG00000172893_DHCR7', 'ENSG00000172900_FLJ42102', 'ENSG00000172915_NBEA', 'ENSG00000172922_RNASEH2C', 'ENSG00000172927_MYEOV', 'ENSG00000172932_ANKRD13D', 'ENSG00000172936_MYD88', 'ENSG00000172939_OXSR1', 'ENSG00000172940_SLC22A13', 'ENSG00000172943_PHF8', 'ENSG00000172954_LCLAT1', 'ENSG00000172965_MIR4435-2HG', 'ENSG00000172967_XKR3', 'ENSG00000172974_AC007318.1', 'ENSG00000172977_KAT5', 'ENSG00000172985_SH3RF3', 'ENSG00000172986_GXYLT2', 'ENSG00000172992_DCAKD', 'ENSG00000172995_ARPP21', 'ENSG00000173011_TADA2B', 'ENSG00000173013_CCDC96', 'ENSG00000173020_GRK2', 'ENSG00000173039_RELA', 'ENSG00000173040_EVC2', 'ENSG00000173041_ZNF680', 'ENSG00000173064_HECTD4', 'ENSG00000173065_FAM222B', 'ENSG00000173083_HPSE', 'ENSG00000173085_COQ2', 'ENSG00000173110_HSPA6', 'ENSG00000173113_TRMT112', 'ENSG00000173114_LRRN3', 'ENSG00000173120_KDM2A', 'ENSG00000173124_ACSM6', 'ENSG00000173137_ADCK5', 'ENSG00000173141_MRPL57', 'ENSG00000173145_NOC3L', 'ENSG00000173153_ESRRA', 'ENSG00000173156_RHOD', 'ENSG00000173163_COMMD1', 'ENSG00000173166_RAPH1', 'ENSG00000173171_MTX1', 'ENSG00000173193_PARP14', 'ENSG00000173198_CYSLTR1', 'ENSG00000173200_PARP15', 'ENSG00000173207_CKS1B', 'ENSG00000173209_AHSA2P', 'ENSG00000173214_MFSD4B', 'ENSG00000173218_VANGL1', 'ENSG00000173221_GLRX', 'ENSG00000173226_IQCB1', 'ENSG00000173230_GOLGB1', 'ENSG00000173258_ZNF483', 'ENSG00000173261_PLAC8L1', 'ENSG00000173262_SLC2A14', 'ENSG00000173264_GPR137', 'ENSG00000173272_MZT2A', 'ENSG00000173273_TNKS', 'ENSG00000173275_ZNF449', 'ENSG00000173276_ZBTB21', 'ENSG00000173281_PPP1R3B', 'ENSG00000173295_FAM86B3P', 'ENSG00000173320_STOX2', 'ENSG00000173327_MAP3K11', 'ENSG00000173334_TRIB1', 'ENSG00000173338_KCNK7', 'ENSG00000173349_SFT2D3', 'ENSG00000173402_DAG1', 'ENSG00000173409_ARV1', 'ENSG00000173418_NAA20', 'ENSG00000173436_MINOS1', 'ENSG00000173442_EHBP1L1', 'ENSG00000173451_THAP2', 'ENSG00000173456_RNF26', 'ENSG00000173457_PPP1R14B', 'ENSG00000173465_SSSCA1', 'ENSG00000173473_SMARCC1', 'ENSG00000173480_ZNF417', 'ENSG00000173482_PTPRM', 'ENSG00000173486_FKBP2', 'ENSG00000173511_VEGFB', 'ENSG00000173517_PEAK1', 'ENSG00000173530_TNFRSF10D', 'ENSG00000173531_MST1', 'ENSG00000173535_TNFRSF10C', 'ENSG00000173540_GMPPB', 'ENSG00000173542_MOB1B', 'ENSG00000173545_ZNF622', 'ENSG00000173548_SNX33', 'ENSG00000173559_NABP1', 'ENSG00000173567_ADGRF3', 'ENSG00000173575_CHD2', 'ENSG00000173581_CCDC106', 'ENSG00000173588_CEP83', 'ENSG00000173597_SULT1B1', 'ENSG00000173598_NUDT4', 'ENSG00000173599_PC', 'ENSG00000173611_SCAI', 'ENSG00000173614_NMNAT1', 'ENSG00000173621_LRFN4', 'ENSG00000173638_SLC19A1', 'ENSG00000173653_RCE1', 'ENSG00000173660_UQCRH', 'ENSG00000173674_EIF1AX', 'ENSG00000173681_BCLAF3', 'ENSG00000173692_PSMD1', 'ENSG00000173705_SUSD5', 'ENSG00000173706_HEG1', 'ENSG00000173715_C11orf80', 'ENSG00000173726_TOMM20', 'ENSG00000173727_AP000769.1', 'ENSG00000173744_AGFG1', 'ENSG00000173757_STAT5B', 'ENSG00000173762_CD7', 'ENSG00000173786_CNP', 'ENSG00000173801_JUP', 'ENSG00000173805_HAP1', 'ENSG00000173809_TDRD12', 'ENSG00000173812_EIF1', 'ENSG00000173818_ENDOV', 'ENSG00000173821_RNF213', 'ENSG00000173825_TIGD3', 'ENSG00000173846_PLK3', 'ENSG00000173848_NET1', 'ENSG00000173852_DPY19L1', 'ENSG00000173875_ZNF791', 'ENSG00000173889_PHC3', 'ENSG00000173890_GPR160', 'ENSG00000173894_CBX2', 'ENSG00000173898_SPTBN2', 'ENSG00000173905_GOLIM4', 'ENSG00000173914_RBM4B', 'ENSG00000173915_ATP5MD', 'ENSG00000173917_HOXB2', 'ENSG00000173918_C1QTNF1', 'ENSG00000173926_MARCH3', 'ENSG00000173928_SWSAP1', 'ENSG00000173930_SLCO4C1', 'ENSG00000173933_RBM4', 'ENSG00000173947_PIFO', 'ENSG00000173950_XXYLT1', 'ENSG00000173960_UBXN2A', 'ENSG00000173988_LRRC63', 'ENSG00000173991_TCAP', 'ENSG00000173992_CCS', 'ENSG00000174004_NRROS', 'ENSG00000174007_CEP19', 'ENSG00000174010_KLHL15', 'ENSG00000174013_FBXO45', 'ENSG00000174021_GNG5', 'ENSG00000174028_FAM3C2', 'ENSG00000174032_SLC25A30', 'ENSG00000174059_CD34', 'ENSG00000174080_CTSF', 'ENSG00000174099_MSRB3', 'ENSG00000174106_LEMD3', 'ENSG00000174109_C16orf91', 'ENSG00000174123_TLR10', 'ENSG00000174125_TLR1', 'ENSG00000174130_TLR6', 'ENSG00000174132_FAM174A', 'ENSG00000174137_FAM53A', 'ENSG00000174151_CYB561D1', 'ENSG00000174165_ZDHHC24', 'ENSG00000174173_TRMT10C', 'ENSG00000174175_SELP', 'ENSG00000174177_CTU2', 'ENSG00000174197_MGA', 'ENSG00000174206_C12orf66', 'ENSG00000174227_PIGG', 'ENSG00000174231_PRPF8', 'ENSG00000174233_ADCY6', 'ENSG00000174236_REP15', 'ENSG00000174238_PITPNA', 'ENSG00000174243_DDX23', 'ENSG00000174255_ZNF80', 'ENSG00000174276_ZNHIT2', 'ENSG00000174282_ZBTB4', 'ENSG00000174292_TNK1', 'ENSG00000174306_ZHX3', 'ENSG00000174307_PHLDA3', 'ENSG00000174326_SLC16A11', 'ENSG00000174327_SLC16A13', 'ENSG00000174353_STAG3L3', 'ENSG00000174365_SNHG11', 'ENSG00000174370_C11orf45', 'ENSG00000174371_EXO1', 'ENSG00000174373_RALGAPA1', 'ENSG00000174403_MIR1-1HG-AS1', 'ENSG00000174405_LIG4', 'ENSG00000174428_GTF2IRD2B', 'ENSG00000174437_ATP2A2', 'ENSG00000174442_ZWILCH', 'ENSG00000174444_RPL4', 'ENSG00000174446_SNAPC5', 'ENSG00000174448_STARD6', 'ENSG00000174456_C12orf76', 'ENSG00000174469_CNTNAP2', 'ENSG00000174473_GALNTL6', 'ENSG00000174483_BBS1', 'ENSG00000174485_DENND4A', 'ENSG00000174500_GCSAM', 'ENSG00000174501_ANKRD36C', 'ENSG00000174514_MFSD4A', 'ENSG00000174516_PELI3', 'ENSG00000174529_TMEM81', 'ENSG00000174547_MRPL11', 'ENSG00000174564_IL20RB', 'ENSG00000174567_GOLT1A', 'ENSG00000174574_AKIRIN1', 'ENSG00000174579_MSL2', 'ENSG00000174586_ZNF497', 'ENSG00000174600_CMKLR1', 'ENSG00000174606_ANGEL2', 'ENSG00000174607_UGT8', 'ENSG00000174628_IQCK', 'ENSG00000174652_ZNF266', 'ENSG00000174669_SLC29A2', 'ENSG00000174672_BRSK2', 'ENSG00000174684_B4GAT1', 'ENSG00000174695_TMEM167A', 'ENSG00000174705_SH3PXD2B', 'ENSG00000174715_PPIAP72', 'ENSG00000174718_KIAA1551', 'ENSG00000174720_LARP7', 'ENSG00000174721_FGFBP3', 'ENSG00000174738_NR1D2', 'ENSG00000174740_PABPC5', 'ENSG00000174744_BRMS1', 'ENSG00000174748_RPL15', 'ENSG00000174749_FAM241A', 'ENSG00000174775_HRAS', 'ENSG00000174776_WDR49', 'ENSG00000174780_SRP72', 'ENSG00000174788_PCP2', 'ENSG00000174791_RIN1', 'ENSG00000174796_THAP6', 'ENSG00000174799_CEP135', 'ENSG00000174804_FZD4', 'ENSG00000174808_BTC', 'ENSG00000174837_ADGRE1', 'ENSG00000174839_DENND6A', 'ENSG00000174840_PDE12', 'ENSG00000174842_GLMN', 'ENSG00000174851_YIF1A', 'ENSG00000174871_CNIH2', 'ENSG00000174886_NDUFA11', 'ENSG00000174891_RSRC1', 'ENSG00000174899_PQLC2L', 'ENSG00000174903_RAB1B', 'ENSG00000174912_METTL15P1', 'ENSG00000174915_PTDSS2', 'ENSG00000174917_C19orf70', 'ENSG00000174928_C3orf33', 'ENSG00000174938_SEZ6L2', 'ENSG00000174939_ASPHD1', 'ENSG00000174943_KCTD13', 'ENSG00000174944_P2RY14', 'ENSG00000174945_AMZ1', 'ENSG00000174946_GPR171', 'ENSG00000174951_FUT1', 'ENSG00000174953_DHX36', 'ENSG00000174977_AC026271.1', 'ENSG00000174989_FBXW8', 'ENSG00000174990_CA5A', 'ENSG00000174996_KLC2', 'ENSG00000175029_CTBP2', 'ENSG00000175040_CHST2', 'ENSG00000175048_ZDHHC14', 'ENSG00000175054_ATR', 'ENSG00000175061_LRRC75A-AS1', 'ENSG00000175063_UBE2C', 'ENSG00000175066_GK5', 'ENSG00000175073_VCPIP1', 'ENSG00000175084_DES', 'ENSG00000175087_PDIK1L', 'ENSG00000175093_SPSB4', 'ENSG00000175104_TRAF6', 'ENSG00000175105_ZNF654', 'ENSG00000175106_TVP23C', 'ENSG00000175110_MRPS22', 'ENSG00000175115_PACS1', 'ENSG00000175130_MARCKSL1', 'ENSG00000175137_SH3BP5L', 'ENSG00000175155_YPEL2', 'ENSG00000175161_CADM2', 'ENSG00000175164_ABO', 'ENSG00000175166_PSMD2', 'ENSG00000175175_PPM1E', 'ENSG00000175182_FAM131A', 'ENSG00000175183_CSRP2', 'ENSG00000175193_PARL', 'ENSG00000175197_DDIT3', 'ENSG00000175198_PCCA', 'ENSG00000175203_DCTN2', 'ENSG00000175213_ZNF408', 'ENSG00000175215_CTDSP2', 'ENSG00000175216_CKAP5', 'ENSG00000175220_ARHGAP1', 'ENSG00000175221_MED16', 'ENSG00000175224_ATG13', 'ENSG00000175229_GAL3ST3', 'ENSG00000175262_C1orf127', 'ENSG00000175265_GOLGA8A', 'ENSG00000175274_TP53I11', 'ENSG00000175279_CENPS', 'ENSG00000175283_DOLK', 'ENSG00000175287_PHYHD1', 'ENSG00000175294_CATSPER1', 'ENSG00000175305_CCNE2', 'ENSG00000175309_PHYKPL', 'ENSG00000175315_CST6', 'ENSG00000175322_ZNF519', 'ENSG00000175324_LSM1', 'ENSG00000175334_BANF1', 'ENSG00000175344_CHRNA7', 'ENSG00000175348_TMEM9B', 'ENSG00000175352_NRIP3', 'ENSG00000175354_PTPN2', 'ENSG00000175376_EIF1AD', 'ENSG00000175387_SMAD2', 'ENSG00000175390_EIF3F', 'ENSG00000175395_ZNF25', 'ENSG00000175414_ARL10', 'ENSG00000175416_CLTB', 'ENSG00000175445_LPL', 'ENSG00000175449_RFESD', 'ENSG00000175455_CCDC14', 'ENSG00000175463_TBC1D10C', 'ENSG00000175467_SART1', 'ENSG00000175470_PPP2R2D', 'ENSG00000175471_MCTP1', 'ENSG00000175482_POLD4', 'ENSG00000175489_LRRC25', 'ENSG00000175497_DPP10', 'ENSG00000175505_CLCF1', 'ENSG00000175513_TSGA10IP', 'ENSG00000175518_UBQLNL', 'ENSG00000175536_LIPT2', 'ENSG00000175538_KCNE3', 'ENSG00000175548_ALG10B', 'ENSG00000175550_DRAP1', 'ENSG00000175556_LONRF3', 'ENSG00000175564_UCP3', 'ENSG00000175567_UCP2', 'ENSG00000175573_C11orf68', 'ENSG00000175575_PAAF1', 'ENSG00000175581_MRPL48', 'ENSG00000175582_RAB6A', 'ENSG00000175591_P2RY2', 'ENSG00000175592_FOSL1', 'ENSG00000175595_ERCC4', 'ENSG00000175600_SUGCT', 'ENSG00000175602_CCDC85B', 'ENSG00000175606_TMEM70', 'ENSG00000175611_LINC00476', 'ENSG00000175634_RPS6KB2', 'ENSG00000175643_RMI2', 'ENSG00000175662_TOM1L2', 'ENSG00000175691_ZNF77', 'ENSG00000175701_MTLN', 'ENSG00000175711_B3GNTL1', 'ENSG00000175727_MLXIP', 'ENSG00000175741_RWDD4P2', 'ENSG00000175749_EIF3KP1', 'ENSG00000175756_AURKAIP1', 'ENSG00000175764_TTLL11', 'ENSG00000175768_TOMM5', 'ENSG00000175772_LINC01106', 'ENSG00000175773_AP002986.1', 'ENSG00000175779_C15orf53', 'ENSG00000175782_SLC35E3', 'ENSG00000175787_ZNF169', 'ENSG00000175792_RUVBL1', 'ENSG00000175806_MSRA', 'ENSG00000175820_CCDC168', 'ENSG00000175826_CTDNEP1', 'ENSG00000175832_ETV4', 'ENSG00000175854_SWI5', 'ENSG00000175857_GAPT', 'ENSG00000175866_BAIAP2', 'ENSG00000175886_RPL7AP66', 'ENSG00000175893_ZDHHC21', 'ENSG00000175895_PLEKHF2', 'ENSG00000175899_A2M', 'ENSG00000175906_ARL4D', 'ENSG00000175928_LRRN1', 'ENSG00000175931_UBE2O', 'ENSG00000175938_ORAI3', 'ENSG00000175970_UNC119B', 'ENSG00000175984_DENND2C', 'ENSG00000176014_TUBB6', 'ENSG00000176018_LYSMD3', 'ENSG00000176022_B3GALT6', 'ENSG00000176024_ZNF613', 'ENSG00000176046_NUPR1', 'ENSG00000176049_JAKMIP2', 'ENSG00000176054_RPL23P2', 'ENSG00000176055_MBLAC2', 'ENSG00000176058_TPRN', 'ENSG00000176076_KCNE5', 'ENSG00000176083_ZNF683', 'ENSG00000176087_SLC35A4', 'ENSG00000176092_CRYBG2', 'ENSG00000176095_IP6K1', 'ENSG00000176101_SSNA1', 'ENSG00000176102_CSTF3', 'ENSG00000176105_YES1', 'ENSG00000176108_CHMP6', 'ENSG00000176124_DLEU1', 'ENSG00000176125_UFSP1', 'ENSG00000176142_TMEM39A', 'ENSG00000176148_TCP11L1', 'ENSG00000176155_CCDC57', 'ENSG00000176160_HSF5', 'ENSG00000176170_SPHK1', 'ENSG00000176171_BNIP3', 'ENSG00000176182_MYPOP', 'ENSG00000176208_ATAD5', 'ENSG00000176209_SMIM19', 'ENSG00000176222_ZNF404', 'ENSG00000176225_RTTN', 'ENSG00000176236_C10orf111', 'ENSG00000176244_ACBD7', 'ENSG00000176248_ANAPC2', 'ENSG00000176261_ZBTB8OS', 'ENSG00000176273_SLC35G1', 'ENSG00000176293_ZNF135', 'ENSG00000176302_FOXR1', 'ENSG00000176340_COX8A', 'ENSG00000176343_RPL37AP8', 'ENSG00000176349_AC104129.1', 'ENSG00000176371_ZSCAN2', 'ENSG00000176383_B3GNT4', 'ENSG00000176386_CDC26', 'ENSG00000176387_HSD11B2', 'ENSG00000176390_CRLF3', 'ENSG00000176393_RNPEP', 'ENSG00000176396_EID2', 'ENSG00000176401_EID2B', 'ENSG00000176402_GJC3', 'ENSG00000176406_RIMS2', 'ENSG00000176407_KCMF1', 'ENSG00000176410_DNAJC30', 'ENSG00000176422_SPRYD4', 'ENSG00000176438_SYNE3', 'ENSG00000176444_CLK2', 'ENSG00000176454_LPCAT4', 'ENSG00000176463_SLCO3A1', 'ENSG00000176472_ZNF575', 'ENSG00000176473_WDR25', 'ENSG00000176476_SGF29', 'ENSG00000176485_PLA2G16', 'ENSG00000176490_DIRAS1', 'ENSG00000176531_PHLDB3', 'ENSG00000176532_PRR15', 'ENSG00000176533_GNG7', 'ENSG00000176542_USF3', 'ENSG00000176563_CNTD1', 'ENSG00000176593_AC008969.1', 'ENSG00000176595_KBTBD11', 'ENSG00000176597_B3GNT5', 'ENSG00000176619_LMNB2', 'ENSG00000176623_RMDN1', 'ENSG00000176624_MEX3C', 'ENSG00000176658_MYO1D', 'ENSG00000176659_C20orf197', 'ENSG00000176681_LRRC37A', 'ENSG00000176700_SCAND2P', 'ENSG00000176714_CCDC121', 'ENSG00000176715_ACSF3', 'ENSG00000176723_ZNF843', 'ENSG00000176728_TTTY14', 'ENSG00000176731_C8orf59', 'ENSG00000176732_PFN4', 'ENSG00000176749_CDK5R1', 'ENSG00000176783_RUFY1', 'ENSG00000176788_BASP1', 'ENSG00000176809_LRRC37A3', 'ENSG00000176834_VSIG10', 'ENSG00000176845_METRNL', 'ENSG00000176853_FAM91A1', 'ENSG00000176868_AL358781.1', 'ENSG00000176871_WSB2', 'ENSG00000176890_TYMS', 'ENSG00000176894_PXMP2', 'ENSG00000176896_TCEANC', 'ENSG00000176903_PNMA1', 'ENSG00000176909_MAMSTR', 'ENSG00000176912_TYMSOS', 'ENSG00000176915_ANKLE2', 'ENSG00000176919_C8G', 'ENSG00000176920_FUT2', 'ENSG00000176927_EFCAB5', 'ENSG00000176928_GCNT4', 'ENSG00000176933_TOB2P1', 'ENSG00000176945_MUC20', 'ENSG00000176946_THAP4', 'ENSG00000176953_NFATC2IP', 'ENSG00000176956_LY6H', 'ENSG00000176973_FAM89B', 'ENSG00000176974_SHMT1', 'ENSG00000176978_DPP7', 'ENSG00000176986_SEC24C', 'ENSG00000176994_SMCR8', 'ENSG00000177000_MTHFR', 'ENSG00000177025_C19orf18', 'ENSG00000177030_DEAF1', 'ENSG00000177034_MTX3', 'ENSG00000177042_TMEM80', 'ENSG00000177045_SIX5', 'ENSG00000177051_FBXO46', 'ENSG00000177054_ZDHHC13', 'ENSG00000177058_SLC38A9', 'ENSG00000177076_ACER2', 'ENSG00000177082_WDR73', 'ENSG00000177084_POLE', 'ENSG00000177096_PHETA2', 'ENSG00000177105_RHOG', 'ENSG00000177106_EPS8L2', 'ENSG00000177112_MRVI1-AS1', 'ENSG00000177119_ANO6', 'ENSG00000177125_ZBTB34', 'ENSG00000177133_LINC00982', 'ENSG00000177150_FAM210A', 'ENSG00000177156_TALDO1', 'ENSG00000177169_ULK1', 'ENSG00000177173_NAP1L4P1', 'ENSG00000177181_RIMKLA', 'ENSG00000177189_RPS6KA3', 'ENSG00000177191_B3GNT8', 'ENSG00000177192_PUS1', 'ENSG00000177200_CHD9', 'ENSG00000177225_GATD1', 'ENSG00000177238_TRIM72', 'ENSG00000177239_MAN1B1', 'ENSG00000177272_KCNA3', 'ENSG00000177283_FZD8', 'ENSG00000177302_TOP3A', 'ENSG00000177303_CASKIN2', 'ENSG00000177311_ZBTB38', 'ENSG00000177337_DLGAP1-AS1', 'ENSG00000177340_FLJ13224', 'ENSG00000177352_CCDC71', 'ENSG00000177359_AC024940.1', 'ENSG00000177363_LRRN4CL', 'ENSG00000177370_TIMM22', 'ENSG00000177374_HIC1', 'ENSG00000177380_PPFIA3', 'ENSG00000177383_MAGEF1', 'ENSG00000177398_UMODL1', 'ENSG00000177406_AC021054.1', 'ENSG00000177409_SAMD9L', 'ENSG00000177410_ZFAS1', 'ENSG00000177425_PAWR', 'ENSG00000177426_TGIF1', 'ENSG00000177427_MIEF2', 'ENSG00000177432_NAP1L5', 'ENSG00000177455_CD19', 'ENSG00000177459_ERICH5', 'ENSG00000177462_OR2T8', 'ENSG00000177463_NR2C2', 'ENSG00000177465_ACOT4', 'ENSG00000177469_CAVIN1', 'ENSG00000177479_ARIH2', 'ENSG00000177483_RBM44', 'ENSG00000177485_ZBTB33', 'ENSG00000177489_OR2G2', 'ENSG00000177508_IRX3', 'ENSG00000177542_SLC25A22', 'ENSG00000177548_RABEP2', 'ENSG00000177556_ATOX1', 'ENSG00000177565_TBL1XR1', 'ENSG00000177570_SAMD12', 'ENSG00000177575_CD163', 'ENSG00000177576_C18orf32', 'ENSG00000177595_PIDD1', 'ENSG00000177599_ZNF491', 'ENSG00000177600_RPLP2', 'ENSG00000177602_HASPIN', 'ENSG00000177606_JUN', 'ENSG00000177613_CSTF2T', 'ENSG00000177628_GBA', 'ENSG00000177640_CASC2', 'ENSG00000177646_ACAD9', 'ENSG00000177663_IL17RA', 'ENSG00000177666_PNPLA2', 'ENSG00000177674_AGTRAP', 'ENSG00000177679_SRRM3', 'ENSG00000177683_THAP5', 'ENSG00000177685_CRACR2B', 'ENSG00000177688_SUMO4', 'ENSG00000177692_DNAJC28', 'ENSG00000177694_NAALADL2', 'ENSG00000177697_CD151', 'ENSG00000177700_POLR2L', 'ENSG00000177706_FAM20C', 'ENSG00000177710_SLC35G5', 'ENSG00000177721_ANXA2R', 'ENSG00000177725_AC105206.1', 'ENSG00000177728_TMEM94', 'ENSG00000177731_FLII', 'ENSG00000177732_SOX12', 'ENSG00000177733_HNRNPA0', 'ENSG00000177738_AC025171.1', 'ENSG00000177788_AL162595.1', 'ENSG00000177830_CHID1', 'ENSG00000177842_ZNF620', 'ENSG00000177853_ZNF518A', 'ENSG00000177854_TMEM187', 'ENSG00000177855_CACYBPP2', 'ENSG00000177868_SVBP', 'ENSG00000177873_ZNF619', 'ENSG00000177879_AP3S1', 'ENSG00000177885_GRB2', 'ENSG00000177888_ZBTB41', 'ENSG00000177889_UBE2N', 'ENSG00000177917_ARL6IP6', 'ENSG00000177932_ZNF354C', 'ENSG00000177943_MAMDC4', 'ENSG00000177946_CENPBD1', 'ENSG00000177951_BET1L', 'ENSG00000177954_RPS27', 'ENSG00000177963_RIC8A', 'ENSG00000177971_IMP3', 'ENSG00000177981_ASB8', 'ENSG00000177989_ODF3B', 'ENSG00000177990_DPY19L2', 'ENSG00000177993_ZNRF3-AS1', 'ENSG00000178015_GPR150', 'ENSG00000178026_LRRC75B', 'ENSG00000178028_DMAP1', 'ENSG00000178035_IMPDH2', 'ENSG00000178038_ALS2CL', 'ENSG00000178053_MLF1', 'ENSG00000178057_NDUFAF3', 'ENSG00000178074_C2orf69', 'ENSG00000178075_GRAMD1C', 'ENSG00000178078_STAP2', 'ENSG00000178081_ULK4P3', 'ENSG00000178093_TSSK6', 'ENSG00000178096_BOLA1', 'ENSG00000178104_PDE4DIP', 'ENSG00000178105_DDX10', 'ENSG00000178115_GOLGA8Q', 'ENSG00000178127_NDUFV2', 'ENSG00000178146_AL672207.1', 'ENSG00000178149_DALRD3', 'ENSG00000178150_ZNF114', 'ENSG00000178163_ZNF518B', 'ENSG00000178177_LCORL', 'ENSG00000178184_PARD6G', 'ENSG00000178187_ZNF454', 'ENSG00000178188_SH2B1', 'ENSG00000178199_ZC3H12D', 'ENSG00000178201_VN1R1', 'ENSG00000178202_KDELC2', 'ENSG00000178209_PLEC', 'ENSG00000178222_RNF212', 'ENSG00000178226_PRSS36', 'ENSG00000178229_ZNF543', 'ENSG00000178234_GALNT11', 'ENSG00000178252_WDR6', 'ENSG00000178295_GEN1', 'ENSG00000178297_TMPRSS9', 'ENSG00000178301_AQP11', 'ENSG00000178307_TMEM11', 'ENSG00000178338_ZNF354B', 'ENSG00000178342_KCNG2', 'ENSG00000178381_ZFAND2A', 'ENSG00000178385_PLEKHM3', 'ENSG00000178386_ZNF223', 'ENSG00000178397_FAM220A', 'ENSG00000178404_CEP295NL', 'ENSG00000178409_BEND3', 'ENSG00000178425_NT5DC1', 'ENSG00000178429_RPS3AP5', 'ENSG00000178449_COX14', 'ENSG00000178458_H3F3AP6', 'ENSG00000178460_MCMDC2', 'ENSG00000178464_RPL10P16', 'ENSG00000178467_P4HTM', 'ENSG00000178498_DTX3', 'ENSG00000178502_KLHL11', 'ENSG00000178531_CTXN1', 'ENSG00000178537_SLC25A20', 'ENSG00000178538_CA8', 'ENSG00000178567_EPM2AIP1', 'ENSG00000178573_MAF', 'ENSG00000178585_CTNNBIP1', 'ENSG00000178605_GTPBP6', 'ENSG00000178607_ERN1', 'ENSG00000178623_GPR35', 'ENSG00000178636_AC092656.1', 'ENSG00000178665_ZNF713', 'ENSG00000178685_PARP10', 'ENSG00000178691_SUZ12', 'ENSG00000178694_NSUN3', 'ENSG00000178695_KCTD12', 'ENSG00000178700_DHFR2', 'ENSG00000178715_AL450998.1', 'ENSG00000178718_RPP25', 'ENSG00000178719_GRINA', 'ENSG00000178726_THBD', 'ENSG00000178741_COX5A', 'ENSG00000178752_ERFE', 'ENSG00000178761_FAM219B', 'ENSG00000178764_ZHX2', 'ENSG00000178773_CPNE7', 'ENSG00000178789_CD300LB', 'ENSG00000178802_MPI', 'ENSG00000178809_TRIM73', 'ENSG00000178814_OPLAH', 'ENSG00000178821_TMEM52', 'ENSG00000178852_EFCAB13', 'ENSG00000178860_MSC', 'ENSG00000178878_APOLD1', 'ENSG00000178882_RFLNA', 'ENSG00000178896_EXOSC4', 'ENSG00000178904_DPY19L3', 'ENSG00000178913_TAF7', 'ENSG00000178917_ZNF852', 'ENSG00000178921_PFAS', 'ENSG00000178922_HYI', 'ENSG00000178927_CYBC1', 'ENSG00000178935_ZNF552', 'ENSG00000178950_GAK', 'ENSG00000178951_ZBTB7A', 'ENSG00000178952_TUFM', 'ENSG00000178965_ERICH3', 'ENSG00000178966_RMI1', 'ENSG00000178971_CTC1', 'ENSG00000178974_FBXO34', 'ENSG00000178977_LINC00324', 'ENSG00000178980_SELENOW', 'ENSG00000178982_EIF3K', 'ENSG00000178988_MRFAP1L1', 'ENSG00000178996_SNX18', 'ENSG00000178999_AURKB', 'ENSG00000179010_MRFAP1', 'ENSG00000179021_C3orf38', 'ENSG00000179029_TMEM107', 'ENSG00000179038_AP001885.1', 'ENSG00000179041_RRS1', 'ENSG00000179051_RCC2', 'ENSG00000179057_IGSF22', 'ENSG00000179083_FAM133A', 'ENSG00000179085_DPM3', 'ENSG00000179091_CYC1', 'ENSG00000179094_PER1', 'ENSG00000179097_HTR1F', 'ENSG00000179104_TMTC2', 'ENSG00000179111_HES7', 'ENSG00000179115_FARSA', 'ENSG00000179119_SPTY2D1', 'ENSG00000179134_SAMD4B', 'ENSG00000179144_GIMAP7', 'ENSG00000179151_EDC3', 'ENSG00000179152_TCAIM', 'ENSG00000179157_RPS2P28', 'ENSG00000179163_FUCA1', 'ENSG00000179165_PXT1', 'ENSG00000179178_TMEM125', 'ENSG00000179195_ZNF664', 'ENSG00000179218_CALR', 'ENSG00000179219_LINC00311', 'ENSG00000179222_MAGED1', 'ENSG00000179240_GVQW3', 'ENSG00000179241_LDLRAD3', 'ENSG00000179242_CDH4', 'ENSG00000179253_AL162457.1', 'ENSG00000179262_RAD23A', 'ENSG00000179271_GADD45GIP1', 'ENSG00000179284_DAND5', 'ENSG00000179295_PTPN11', 'ENSG00000179299_NSUN7', 'ENSG00000179314_WSCD1', 'ENSG00000179331_RAB39A', 'ENSG00000179335_CLK3', 'ENSG00000179344_HLA-DQB1', 'ENSG00000179348_GATA2', 'ENSG00000179361_ARID3B', 'ENSG00000179362_HMGN2P46', 'ENSG00000179364_PACS2', 'ENSG00000179387_ELMOD2', 'ENSG00000179397_CATSPERE', 'ENSG00000179399_GPC5', 'ENSG00000179403_VWA1', 'ENSG00000179406_LINC00174', 'ENSG00000179409_GEMIN4', 'ENSG00000179431_FJX1', 'ENSG00000179454_KLHL28', 'ENSG00000179455_MKRN3', 'ENSG00000179456_ZBTB18', 'ENSG00000179476_C14orf28', 'ENSG00000179477_ALOX12B', 'ENSG00000179523_EIF3J-DT', 'ENSG00000179526_SHARPIN', 'ENSG00000179528_LBX2', 'ENSG00000179532_DNHD1', 'ENSG00000179542_SLITRK4', 'ENSG00000179562_GCC1', 'ENSG00000179564_LSMEM2', 'ENSG00000179583_CIITA', 'ENSG00000179588_ZFPM1', 'ENSG00000179593_ALOX15B', 'ENSG00000179598_PLD6', 'ENSG00000179604_CDC42EP4', 'ENSG00000179627_ZBTB42', 'ENSG00000179630_LACC1', 'ENSG00000179632_MAF1', 'ENSG00000179639_FCER1A', 'ENSG00000179673_RPRML', 'ENSG00000179698_WDR97', 'ENSG00000179715_PCED1B', 'ENSG00000179743_FLJ37453', 'ENSG00000179750_APOBEC3B', 'ENSG00000179761_PIPOX', 'ENSG00000179766_ATP8B5P', 'ENSG00000179818_PCBP1-AS1', 'ENSG00000179820_MYADM', 'ENSG00000179832_MROH1', 'ENSG00000179833_SERTAD2', 'ENSG00000179840_PIK3CD-AS1', 'ENSG00000179841_AKAP5', 'ENSG00000179846_NKPD1', 'ENSG00000179855_GIPC3', 'ENSG00000179859_RNF227', 'ENSG00000179862_CITED4', 'ENSG00000179869_ABCA13', 'ENSG00000179886_TIGD5', 'ENSG00000179889_PDXDC1', 'ENSG00000179909_ZNF154', 'ENSG00000179912_R3HDM2', 'ENSG00000179913_B3GNT3', 'ENSG00000179914_ITLN1', 'ENSG00000179918_SEPHS2', 'ENSG00000179922_ZNF784', 'ENSG00000179933_C14orf119', 'ENSG00000179938_GOLGA8J', 'ENSG00000179941_BBS10', 'ENSG00000179943_FIZ1', 'ENSG00000179950_PUF60', 'ENSG00000179954_SSC5D', 'ENSG00000179958_DCTPP1', 'ENSG00000179965_ZNF771', 'ENSG00000179967_PPP1R14BP3', 'ENSG00000179981_TSHZ1', 'ENSG00000179988_PSTK', 'ENSG00000180008_SOCS4', 'ENSG00000180011_ZADH2', 'ENSG00000180015_AC093909.1', 'ENSG00000180035_ZNF48', 'ENSG00000180044_C3orf80', 'ENSG00000180061_TMEM150B', 'ENSG00000180066_C10orf91', 'ENSG00000180071_ANKRD18A', 'ENSG00000180089_TMEM86B', 'ENSG00000180096_SEPT1', 'ENSG00000180098_TRNAU1AP', 'ENSG00000180104_EXOC3', 'ENSG00000180113_TDRD6', 'ENSG00000180139_ACTA2-AS1', 'ENSG00000180152_XIAPP3', 'ENSG00000180155_LYNX1', 'ENSG00000180172_RPS12P23', 'ENSG00000180182_MED14', 'ENSG00000180185_FAHD1', 'ENSG00000180198_RCC1', 'ENSG00000180209_MYLPF', 'ENSG00000180211_FO393411.1', 'ENSG00000180221_TPT1P10', 'ENSG00000180228_PRKRA', 'ENSG00000180229_HERC2P3', 'ENSG00000180233_ZNRF2', 'ENSG00000180257_ZNF816', 'ENSG00000180263_FGD6', 'ENSG00000180304_OAZ2', 'ENSG00000180316_PNPLA1', 'ENSG00000180329_CCDC43', 'ENSG00000180336_MEIOC', 'ENSG00000180340_FZD2', 'ENSG00000180346_TIGD2', 'ENSG00000180353_HCLS1', 'ENSG00000180354_MTURN', 'ENSG00000180357_ZNF609', 'ENSG00000180370_PAK2', 'ENSG00000180376_CCDC66', 'ENSG00000180385_EMC3-AS1', 'ENSG00000180398_MCFD2', 'ENSG00000180423_HARBI1', 'ENSG00000180425_C11orf71', 'ENSG00000180447_GAS1', 'ENSG00000180448_ARHGAP45', 'ENSG00000180458_AC022148.1', 'ENSG00000180479_ZNF571', 'ENSG00000180481_GLIPR1L2', 'ENSG00000180488_MIGA1', 'ENSG00000180509_KCNE1', 'ENSG00000180530_NRIP1', 'ENSG00000180539_C9orf139', 'ENSG00000180543_TSPYL5', 'ENSG00000180549_FUT7', 'ENSG00000180573_HIST1H2AC', 'ENSG00000180574_EIF2S3B', 'ENSG00000180581_SRP9P1', 'ENSG00000180592_SKIDA1', 'ENSG00000180596_HIST1H2BC', 'ENSG00000180611_MB21D2', 'ENSG00000180616_SSTR2', 'ENSG00000180626_ZNF594', 'ENSG00000180628_PCGF5', 'ENSG00000180644_PRF1', 'ENSG00000180660_MAB21L1', 'ENSG00000180662_RPL21P8', 'ENSG00000180667_YOD1', 'ENSG00000180694_TMEM64', 'ENSG00000180747_SMG1P3', 'ENSG00000180758_GPR157', 'ENSG00000180764_PIPSL', 'ENSG00000180767_CHST13', 'ENSG00000180773_SLC36A4', 'ENSG00000180776_ZDHHC20', 'ENSG00000180787_ZFP3', 'ENSG00000180815_MAP3K15', 'ENSG00000180817_PPA1', 'ENSG00000180822_PSMG4', 'ENSG00000180828_BHLHE22', 'ENSG00000180834_MAP6D1', 'ENSG00000180855_ZNF443', 'ENSG00000180867_PDIA3P1', 'ENSG00000180871_CXCR2', 'ENSG00000180879_SSR4', 'ENSG00000180881_CAPS2', 'ENSG00000180884_ZNF792', 'ENSG00000180891_CUEDC1', 'ENSG00000180900_SCRIB', 'ENSG00000180901_KCTD2', 'ENSG00000180902_D2HGDH', 'ENSG00000180914_OXTR', 'ENSG00000180917_CMTR2', 'ENSG00000180921_FAM83H', 'ENSG00000180938_ZNF572', 'ENSG00000180953_ST20', 'ENSG00000180957_PITPNB', 'ENSG00000180964_TCEAL8', 'ENSG00000180979_LRRC57', 'ENSG00000180992_MRPL14', 'ENSG00000180998_GPR137C', 'ENSG00000181004_BBS12', 'ENSG00000181007_ZFP82', 'ENSG00000181009_OR52N5', 'ENSG00000181016_LSMEM1', 'ENSG00000181019_NQO1', 'ENSG00000181026_AEN', 'ENSG00000181027_FKRP', 'ENSG00000181029_TRAPPC5', 'ENSG00000181031_RPH3AL', 'ENSG00000181035_SLC25A42', 'ENSG00000181038_METTL23', 'ENSG00000181045_SLC26A11', 'ENSG00000181061_HIGD1A', 'ENSG00000181072_CHRM2', 'ENSG00000181090_EHMT1', 'ENSG00000181097_BREA2', 'ENSG00000181104_F2R', 'ENSG00000181126_HLA-V', 'ENSG00000181135_ZNF707', 'ENSG00000181163_NPM1', 'ENSG00000181191_PJA1', 'ENSG00000181192_DHTKD1', 'ENSG00000181201_HIST3H2BA', 'ENSG00000181218_HIST3H2A', 'ENSG00000181220_ZNF746', 'ENSG00000181222_POLR2A', 'ENSG00000181240_SLC25A41', 'ENSG00000181264_TMEM136', 'ENSG00000181274_FRAT2', 'ENSG00000181284_TMEM102', 'ENSG00000181315_ZNF322', 'ENSG00000181322_NME9', 'ENSG00000181350_LRRC75A', 'ENSG00000181381_DDX60L', 'ENSG00000181392_SYNE4', 'ENSG00000181396_OGFOD3', 'ENSG00000181404_WASHC1', 'ENSG00000181409_AATK', 'ENSG00000181418_DDN', 'ENSG00000181444_ZNF467', 'ENSG00000181450_ZNF678', 'ENSG00000181458_TMEM45A', 'ENSG00000181467_RAP2B', 'ENSG00000181472_ZBTB2', 'ENSG00000181481_RNF135', 'ENSG00000181513_ACBD4', 'ENSG00000181523_SGSH', 'ENSG00000181524_RPL24P4', 'ENSG00000181544_FANCB', 'ENSG00000181555_SETD2', 'ENSG00000181577_C6orf223', 'ENSG00000181585_TMIE', 'ENSG00000181588_MEX3D', 'ENSG00000181610_MRPS23', 'ENSG00000181619_GPR135', 'ENSG00000181625_SLX1B', 'ENSG00000181631_P2RY13', 'ENSG00000181638_ZFP41', 'ENSG00000181649_PHLDA2', 'ENSG00000181652_ATG9B', 'ENSG00000181656_GPR88', 'ENSG00000181666_HKR1', 'ENSG00000181690_PLAG1', 'ENSG00000181704_YIPF6', 'ENSG00000181722_ZBTB20', 'ENSG00000181744_C3orf58', 'ENSG00000181751_C5orf30', 'ENSG00000181754_AMIGO1', 'ENSG00000181773_GPR3', 'ENSG00000181788_SIAH2', 'ENSG00000181789_COPG1', 'ENSG00000181790_ADGRB1', 'ENSG00000181798_LINC00471', 'ENSG00000181800_CELF2-AS1', 'ENSG00000181804_SLC9A9', 'ENSG00000181817_LSM10', 'ENSG00000181826_RELL1', 'ENSG00000181827_RFX7', 'ENSG00000181830_SLC35C1', 'ENSG00000181852_RNF41', 'ENSG00000181856_SLC2A4', 'ENSG00000181873_IBA57', 'ENSG00000181885_CLDN7', 'ENSG00000181894_ZNF329', 'ENSG00000181896_ZNF101', 'ENSG00000181904_C5orf24', 'ENSG00000181908_AP003774.1', 'ENSG00000181915_ADO', 'ENSG00000181924_COA4', 'ENSG00000181929_PRKAG1', 'ENSG00000181938_GINS3', 'ENSG00000181982_CCDC149', 'ENSG00000181991_MRPS11', 'ENSG00000182004_SNRPE', 'ENSG00000182010_RTKN2', 'ENSG00000182013_PNMA8A', 'ENSG00000182048_TRPC2', 'ENSG00000182054_IDH2', 'ENSG00000182087_TMEM259', 'ENSG00000182093_WRB', 'ENSG00000182095_TNRC18', 'ENSG00000182103_FAM181B', 'ENSG00000182107_TMEM30B', 'ENSG00000182108_DEXI', 'ENSG00000182117_NOP10', 'ENSG00000182118_FAM89A', 'ENSG00000182134_TDRKH', 'ENSG00000182141_ZNF708', 'ENSG00000182149_IST1', 'ENSG00000182150_ERCC6L2', 'ENSG00000182154_MRPL41', 'ENSG00000182158_CREB3L2', 'ENSG00000182162_P2RY8', 'ENSG00000182165_TP53TG1', 'ENSG00000182173_TSEN54', 'ENSG00000182175_RGMA', 'ENSG00000182179_UBA7', 'ENSG00000182180_MRPS16', 'ENSG00000182183_SHISAL2A', 'ENSG00000182185_RAD51B', 'ENSG00000182195_LDOC1', 'ENSG00000182196_ARL6IP4', 'ENSG00000182197_EXT1', 'ENSG00000182199_SHMT2', 'ENSG00000182208_MOB2', 'ENSG00000182220_ATP6AP2', 'ENSG00000182224_CYB5D1', 'ENSG00000182240_BACE2', 'ENSG00000182247_UBE2E2', 'ENSG00000182253_SYNM', 'ENSG00000182257_PRR34', 'ENSG00000182263_FIGN', 'ENSG00000182272_B4GALNT4', 'ENSG00000182287_AP1S2', 'ENSG00000182307_C8orf33', 'ENSG00000182308_DCAF4L1', 'ENSG00000182310_SPACA6', 'ENSG00000182318_ZSCAN22', 'ENSG00000182324_KCNJ14', 'ENSG00000182325_FBXL6', 'ENSG00000182326_C1S', 'ENSG00000182329_KIAA2012', 'ENSG00000182359_KBTBD3', 'ENSG00000182362_YBEY', 'ENSG00000182372_CLN8', 'ENSG00000182376_AC138028.1', 'ENSG00000182378_PLCXD1', 'ENSG00000182379_NXPH4', 'ENSG00000182383_RPL27AP5', 'ENSG00000182389_CACNB4', 'ENSG00000182397_DNM1P46', 'ENSG00000182400_TRAPPC6B', 'ENSG00000182405_PGBD4', 'ENSG00000182446_NPLOC4', 'ENSG00000182472_CAPN12', 'ENSG00000182473_EXOC7', 'ENSG00000182481_KPNA2', 'ENSG00000182484_WASH6P', 'ENSG00000182487_NCF1B', 'ENSG00000182504_CEP97', 'ENSG00000182511_FES', 'ENSG00000182512_GLRX5', 'ENSG00000182518_FAM104B', 'ENSG00000182534_MXRA7', 'ENSG00000182541_LIMK2', 'ENSG00000182544_MFSD5', 'ENSG00000182551_ADI1', 'ENSG00000182552_RWDD4', 'ENSG00000182557_SPNS3', 'ENSG00000182568_SATB1', 'ENSG00000182578_CSF1R', 'ENSG00000182584_ACTL10', 'ENSG00000182585_EPGN', 'ENSG00000182600_SNORC', 'ENSG00000182606_TRAK1', 'ENSG00000182612_TSPAN10', 'ENSG00000182621_PLCB1', 'ENSG00000182628_SKA2', 'ENSG00000182636_NDN', 'ENSG00000182648_LINC01006', 'ENSG00000182670_TTC3', 'ENSG00000182676_PPP1R27', 'ENSG00000182685_BRICD5', 'ENSG00000182687_GALR2', 'ENSG00000182700_IGIP', 'ENSG00000182704_TSKU', 'ENSG00000182712_CMC4', 'ENSG00000182718_ANXA2', 'ENSG00000182732_RGS6', 'ENSG00000182742_HOXB4', 'ENSG00000182747_SLC35D3', 'ENSG00000182749_PAQR7', 'ENSG00000182752_PAPPA', 'ENSG00000182768_NGRN', 'ENSG00000182774_RPS17', 'ENSG00000182795_C1orf116', 'ENSG00000182796_TMEM198B', 'ENSG00000182809_CRIP2', 'ENSG00000182810_DDX28', 'ENSG00000182827_ACBD3', 'ENSG00000182831_C16orf72', 'ENSG00000182841_RRP7BP', 'ENSG00000182858_ALG12', 'ENSG00000182866_LCK', 'ENSG00000182871_COL18A1', 'ENSG00000182872_RBM10', 'ENSG00000182873_PRKCZ-AS1', 'ENSG00000182885_ADGRG3', 'ENSG00000182890_GLUD2', 'ENSG00000182899_RPL35A', 'ENSG00000182903_ZNF721', 'ENSG00000182916_TCEAL7', 'ENSG00000182919_C11orf54', 'ENSG00000182923_CEP63', 'ENSG00000182934_SRPRA', 'ENSG00000182944_EWSR1', 'ENSG00000182952_HMGN4', 'ENSG00000182957_SPATA13', 'ENSG00000182963_GJC1', 'ENSG00000182973_CNOT10', 'ENSG00000182979_MTA1', 'ENSG00000182983_ZNF662', 'ENSG00000182986_ZNF320', 'ENSG00000182993_C12orf60', 'ENSG00000183010_PYCR1', 'ENSG00000183011_NAA38', 'ENSG00000183018_SPNS2', 'ENSG00000183019_MCEMP1', 'ENSG00000183020_AP2A2', 'ENSG00000183023_SLC8A1', 'ENSG00000183032_SLC25A21', 'ENSG00000183044_ABAT', 'ENSG00000183048_SLC25A10', 'ENSG00000183049_CAMK1D', 'ENSG00000183055_FAM133CP', 'ENSG00000183060_LYSMD4', 'ENSG00000183066_WBP2NL', 'ENSG00000183077_AFMID', 'ENSG00000183087_GAS6', 'ENSG00000183091_NEB', 'ENSG00000183092_BEGAIN', 'ENSG00000183098_GPC6', 'ENSG00000183111_ARHGEF37', 'ENSG00000183114_FAM43B', 'ENSG00000183134_PTGDR2', 'ENSG00000183137_CEP57L1', 'ENSG00000183145_RIPPLY3', 'ENSG00000183150_GPR19', 'ENSG00000183154_AC138356.1', 'ENSG00000183155_RABIF', 'ENSG00000183161_FANCF', 'ENSG00000183166_CALN1', 'ENSG00000183172_SMDT1', 'ENSG00000183196_CHST6', 'ENSG00000183199_HSP90AB3P', 'ENSG00000183207_RUVBL2', 'ENSG00000183208_GDPGP1', 'ENSG00000183239_AL109615.1', 'ENSG00000183242_WT1-AS', 'ENSG00000183246_RIMBP3C', 'ENSG00000183248_PRR36', 'ENSG00000183250_LINC01547', 'ENSG00000183255_PTTG1IP', 'ENSG00000183258_DDX41', 'ENSG00000183281_PLGLB1', 'ENSG00000183283_DAZAP2', 'ENSG00000183287_CCBE1', 'ENSG00000183291_SELENOF', 'ENSG00000183298_RPSAP19', 'ENSG00000183307_TMEM121B', 'ENSG00000183308_AC005037.1', 'ENSG00000183309_ZNF623', 'ENSG00000183323_CCDC125', 'ENSG00000183336_BOLA2', 'ENSG00000183337_BCOR', 'ENSG00000183340_JRKL', 'ENSG00000183346_CABCOCO1', 'ENSG00000183354_KIAA2026', 'ENSG00000183386_FHL3', 'ENSG00000183401_CCDC159', 'ENSG00000183426_NPIPA1', 'ENSG00000183431_SF3A3', 'ENSG00000183439_TRIM61', 'ENSG00000183444_OR7E38P', 'ENSG00000183474_GTF2H2C', 'ENSG00000183475_ASB7', 'ENSG00000183484_GPR132', 'ENSG00000183486_MX2', 'ENSG00000183495_EP400', 'ENSG00000183496_MEX3B', 'ENSG00000183506_PI4KAP2', 'ENSG00000183508_TENT5C', 'ENSG00000183513_COA5', 'ENSG00000183520_UTP11', 'ENSG00000183527_PSMG1', 'ENSG00000183530_PRR14L', 'ENSG00000183569_SERHL2', 'ENSG00000183570_PCBP3', 'ENSG00000183576_SETD3', 'ENSG00000183579_ZNRF3', 'ENSG00000183597_TANGO2', 'ENSG00000183598_HIST2H3D', 'ENSG00000183604_SMG1P5', 'ENSG00000183605_SFXN4', 'ENSG00000183617_MRPL54', 'ENSG00000183621_ZNF438', 'ENSG00000183624_HMCES', 'ENSG00000183628_DGCR6', 'ENSG00000183647_ZNF530', 'ENSG00000183648_NDUFB1', 'ENSG00000183655_KLHL25', 'ENSG00000183665_TRMT12', 'ENSG00000183666_GUSBP1', 'ENSG00000183684_ALYREF', 'ENSG00000183688_RFLNB', 'ENSG00000183690_EFHC2', 'ENSG00000183691_NOG', 'ENSG00000183696_UPP1', 'ENSG00000183718_TRIM52', 'ENSG00000183722_LHFPL6', 'ENSG00000183723_CMTM4', 'ENSG00000183726_TMEM50A', 'ENSG00000183734_ASCL2', 'ENSG00000183735_TBK1', 'ENSG00000183741_CBX6', 'ENSG00000183742_MACC1', 'ENSG00000183751_TBL3', 'ENSG00000183762_KREMEN1', 'ENSG00000183763_TRAIP', 'ENSG00000183765_CHEK2', 'ENSG00000183773_AIFM3', 'ENSG00000183780_SLC35F3', 'ENSG00000183784_C9orf66', 'ENSG00000183785_TUBA8', 'ENSG00000183793_NPIPA5', 'ENSG00000183808_RBM12B', 'ENSG00000183813_CCR4', 'ENSG00000183814_LIN9', 'ENSG00000183826_BTBD9', 'ENSG00000183828_NUDT14', 'ENSG00000183837_PNMA3', 'ENSG00000183850_ZNF730', 'ENSG00000183853_KIRREL1', 'ENSG00000183856_IQGAP3', 'ENSG00000183864_TOB2', 'ENSG00000183878_UTY', 'ENSG00000183889_PKD1P1', 'ENSG00000183891_TTC32', 'ENSG00000183914_DNAH2', 'ENSG00000183918_SH2D1A', 'ENSG00000183921_SDR42E2', 'ENSG00000183935_HTR7P1', 'ENSG00000183943_PRKX', 'ENSG00000183955_KMT5A', 'ENSG00000183963_SMTN', 'ENSG00000183971_NPW', 'ENSG00000183978_COA3', 'ENSG00000183979_NPB', 'ENSG00000184005_ST6GALNAC3', 'ENSG00000184007_PTP4A2', 'ENSG00000184009_ACTG1', 'ENSG00000184014_DENND5A', 'ENSG00000184047_DIABLO', 'ENSG00000184056_VPS33B', 'ENSG00000184058_TBX1', 'ENSG00000184060_ADAP2', 'ENSG00000184068_SREBF2-AS1', 'ENSG00000184076_UQCR10', 'ENSG00000184083_FAM120C', 'ENSG00000184100_BRD7P2', 'ENSG00000184110_EIF3C', 'ENSG00000184113_CLDN5', 'ENSG00000184117_NIPSNAP1', 'ENSG00000184139_RPL7AP28', 'ENSG00000184144_CNTN2', 'ENSG00000184154_LRTOMT', 'ENSG00000184160_ADRA2C', 'ENSG00000184162_NR2C2AP', 'ENSG00000184163_C1QTNF12', 'ENSG00000184164_CRELD2', 'ENSG00000184178_SCFD2', 'ENSG00000184182_UBE2F', 'ENSG00000184185_KCNJ12', 'ENSG00000184194_GPR173', 'ENSG00000184203_PPP1R2', 'ENSG00000184205_TSPYL2', 'ENSG00000184206_GOLGA6L4', 'ENSG00000184207_PGP', 'ENSG00000184208_C22orf46', 'ENSG00000184209_SNRNP35', 'ENSG00000184216_IRAK1', 'ENSG00000184220_CMSS1', 'ENSG00000184221_OLIG1', 'ENSG00000184226_PCDH9', 'ENSG00000184232_OAF', 'ENSG00000184260_HIST2H2AC', 'ENSG00000184261_KCNK12', 'ENSG00000184270_HIST2H2AB', 'ENSG00000184271_POU6F1', 'ENSG00000184277_TM2D3', 'ENSG00000184281_TSSC4', 'ENSG00000184292_TACSTD2', 'ENSG00000184293_CLECL1', 'ENSG00000184304_PRKD1', 'ENSG00000184305_CCSER1', 'ENSG00000184307_ZDHHC23', 'ENSG00000184313_MROH7', 'ENSG00000184319_RPL23AP82', 'ENSG00000184344_GDF3', 'ENSG00000184357_HIST1H1B', 'ENSG00000184361_SPATA32', 'ENSG00000184368_MAP7D2', 'ENSG00000184371_CSF1', 'ENSG00000184378_ACTRT3', 'ENSG00000184381_PLA2G6', 'ENSG00000184384_MAML2', 'ENSG00000184385_UMODL1-AS1', 'ENSG00000184402_SS18L1', 'ENSG00000184414_IRS3P', 'ENSG00000184423_RPL23AP38', 'ENSG00000184428_TOP1MT', 'ENSG00000184432_COPB2', 'ENSG00000184436_THAP7', 'ENSG00000184441_AP001062.1', 'ENSG00000184445_KNTC1', 'ENSG00000184451_CCR10', 'ENSG00000184465_WDR27', 'ENSG00000184470_TXNRD2', 'ENSG00000184481_FOXO4', 'ENSG00000184489_PTP4A3', 'ENSG00000184497_TMEM255B', 'ENSG00000184500_PROS1', 'ENSG00000184508_HDDC3', 'ENSG00000184515_BEX5', 'ENSG00000184517_ZFP1', 'ENSG00000184557_SOCS3', 'ENSG00000184566_AC132216.1', 'ENSG00000184574_LPAR5', 'ENSG00000184575_XPOT', 'ENSG00000184584_TMEM173', 'ENSG00000184588_PDE4B', 'ENSG00000184602_SNN', 'ENSG00000184619_KRBA2', 'ENSG00000184634_MED12', 'ENSG00000184635_ZNF93', 'ENSG00000184640_SEPT9', 'ENSG00000184661_CDCA2', 'ENSG00000184669_OR7E14P', 'ENSG00000184675_AMER1', 'ENSG00000184677_ZBTB40', 'ENSG00000184678_HIST2H2BE', 'ENSG00000184698_OR51M1', 'ENSG00000184708_EIF4ENIF1', 'ENSG00000184709_LRRC26', 'ENSG00000184716_SERINC4', 'ENSG00000184719_RNLS', 'ENSG00000184730_APOBR', 'ENSG00000184743_ATL3', 'ENSG00000184752_NDUFA12', 'ENSG00000184785_SMIM10', 'ENSG00000184786_TCTE3', 'ENSG00000184787_UBE2G2', 'ENSG00000184788_SATL1', 'ENSG00000184792_OSBP2', 'ENSG00000184831_APOO', 'ENSG00000184838_PRR16', 'ENSG00000184840_TMED9', 'ENSG00000184857_TMEM186', 'ENSG00000184860_SDR42E1', 'ENSG00000184863_RBM33', 'ENSG00000184867_ARMCX2', 'ENSG00000184887_BTBD6', 'ENSG00000184897_H1FX', 'ENSG00000184898_RBM43', 'ENSG00000184900_SUMO3', 'ENSG00000184903_IMMP2L', 'ENSG00000184905_TCEAL2', 'ENSG00000184916_JAG2', 'ENSG00000184922_FMNL1', 'ENSG00000184924_PTRHD1', 'ENSG00000184925_LCN12', 'ENSG00000184933_OR6A2', 'ENSG00000184937_WT1', 'ENSG00000184939_ZFP90', 'ENSG00000184949_FAM227A', 'ENSG00000184967_NOC4L', 'ENSG00000184979_USP18', 'ENSG00000184983_NDUFA6', 'ENSG00000184986_TMEM121', 'ENSG00000184988_TMEM106A', 'ENSG00000184990_SIVA1', 'ENSG00000184992_BRI3BP', 'ENSG00000185000_DGAT1', 'ENSG00000185009_AP3M1', 'ENSG00000185010_F8', 'ENSG00000185015_CA13', 'ENSG00000185019_UBOX5', 'ENSG00000185022_MAFF', 'ENSG00000185024_BRF1', 'ENSG00000185033_SEMA4B', 'ENSG00000185040_SPDYE16', 'ENSG00000185043_CIB1', 'ENSG00000185046_ANKS1B', 'ENSG00000185049_NELFA', 'ENSG00000185052_SLC24A3', 'ENSG00000185053_SGCZ', 'ENSG00000185055_EFCAB10', 'ENSG00000185065_AC000068.1', 'ENSG00000185085_INTS5', 'ENSG00000185088_RPS27L', 'ENSG00000185090_MANEAL', 'ENSG00000185100_ADSSL1', 'ENSG00000185101_ANO9', 'ENSG00000185104_FAF1', 'ENSG00000185112_FAM43A', 'ENSG00000185115_NSMCE3', 'ENSG00000185122_HSF1', 'ENSG00000185127_C6orf120', 'ENSG00000185129_PURA', 'ENSG00000185130_HIST1H2BL', 'ENSG00000185133_INPP5J', 'ENSG00000185155_MIXL1', 'ENSG00000185158_LRRC37B', 'ENSG00000185163_DDX51', 'ENSG00000185164_NOMO2', 'ENSG00000185187_SIGIRR', 'ENSG00000185189_NRBP2', 'ENSG00000185198_PRSS57', 'ENSG00000185201_IFITM2', 'ENSG00000185203_WASIR1', 'ENSG00000185215_TNFAIP2', 'ENSG00000185219_ZNF445', 'ENSG00000185220_PGBD2', 'ENSG00000185222_TCEAL9', 'ENSG00000185236_RAB11B', 'ENSG00000185238_PRMT3', 'ENSG00000185245_GP1BA', 'ENSG00000185246_PRPF39', 'ENSG00000185250_PPIL6', 'ENSG00000185252_ZNF74', 'ENSG00000185261_KIAA0825', 'ENSG00000185262_UBALD2', 'ENSG00000185267_CDNF', 'ENSG00000185269_NOTUM', 'ENSG00000185271_KLHL33', 'ENSG00000185272_RBM11', 'ENSG00000185278_ZBTB37', 'ENSG00000185290_NUPR2', 'ENSG00000185291_IL3RA', 'ENSG00000185298_CCDC137', 'ENSG00000185304_RGPD2', 'ENSG00000185305_ARL15', 'ENSG00000185324_CDK10', 'ENSG00000185338_SOCS1', 'ENSG00000185339_TCN2', 'ENSG00000185340_GAS2L1', 'ENSG00000185344_ATP6V0A2', 'ENSG00000185345_PRKN', 'ENSG00000185347_TEDC1', 'ENSG00000185359_HGS', 'ENSG00000185361_TNFAIP8L1', 'ENSG00000185379_RAD51D', 'ENSG00000185385_OR7A17', 'ENSG00000185386_MAPK11', 'ENSG00000185404_SP140L', 'ENSG00000185414_MRPL30', 'ENSG00000185418_TARSL2', 'ENSG00000185420_SMYD3', 'ENSG00000185432_METTL7A', 'ENSG00000185437_SH3BGR', 'ENSG00000185442_FAM174B', 'ENSG00000185453_ZSWIM9', 'ENSG00000185475_TMEM179B', 'ENSG00000185477_GPRIN3', 'ENSG00000185480_PARPBP', 'ENSG00000185482_STAC3', 'ENSG00000185485_SDHAP1', 'ENSG00000185495_AC138393.1', 'ENSG00000185499_MUC1', 'ENSG00000185504_FAAP100', 'ENSG00000185507_IRF7', 'ENSG00000185513_L3MBTL1', 'ENSG00000185515_BRCC3', 'ENSG00000185518_SV2B', 'ENSG00000185519_FAM131C', 'ENSG00000185522_LMNTD2', 'ENSG00000185523_SPATA45', 'ENSG00000185527_PDE6G', 'ENSG00000185532_PRKG1', 'ENSG00000185559_DLK1', 'ENSG00000185561_TLCD2', 'ENSG00000185585_OLFML2A', 'ENSG00000185591_SP1', 'ENSG00000185596_WASH3P', 'ENSG00000185608_MRPL40', 'ENSG00000185614_INKA1', 'ENSG00000185619_PCGF3', 'ENSG00000185621_LMLN', 'ENSG00000185624_P4HB', 'ENSG00000185627_PSMD13', 'ENSG00000185630_PBX1', 'ENSG00000185633_NDUFA4L2', 'ENSG00000185634_SHC4', 'ENSG00000185640_KRT79', 'ENSG00000185641_AC034236.1', 'ENSG00000185650_ZFP36L1', 'ENSG00000185651_UBE2L3', 'ENSG00000185658_BRWD1', 'ENSG00000185664_PMEL', 'ENSG00000185666_SYN3', 'ENSG00000185669_SNAI3', 'ENSG00000185670_ZBTB3', 'ENSG00000185674_LYG2', 'ENSG00000185684_EP400P1', 'ENSG00000185697_MYBL1', 'ENSG00000185710_SMG1P4', 'ENSG00000185716_MOSMO', 'ENSG00000185721_DRG1', 'ENSG00000185722_ANKFY1', 'ENSG00000185728_YTHDF3', 'ENSG00000185730_ZNF696', 'ENSG00000185736_ADARB2', 'ENSG00000185745_IFIT1', 'ENSG00000185753_CXorf38', 'ENSG00000185760_KCNQ5', 'ENSG00000185761_ADAMTSL5', 'ENSG00000185787_MORF4L1', 'ENSG00000185798_WDR53', 'ENSG00000185800_DMWD', 'ENSG00000185803_SLC52A2', 'ENSG00000185808_PIGP', 'ENSG00000185811_IKZF1', 'ENSG00000185813_PCYT2', 'ENSG00000185818_NAT8L', 'ENSG00000185825_BCAP31', 'ENSG00000185829_ARL17A', 'ENSG00000185834_RPL12P4', 'ENSG00000185837_HDHD5-AS1', 'ENSG00000185838_GNB1L', 'ENSG00000185842_DNAH14', 'ENSG00000185847_LINC01405', 'ENSG00000185862_EVI2B', 'ENSG00000185864_NPIPB4', 'ENSG00000185869_ZNF829', 'ENSG00000185875_THNSL1', 'ENSG00000185880_TRIM69', 'ENSG00000185883_ATP6V0C', 'ENSG00000185885_IFITM1', 'ENSG00000185896_LAMP1', 'ENSG00000185900_POMK', 'ENSG00000185905_C16orf54', 'ENSG00000185909_KLHDC8B', 'ENSG00000185917_SETD4', 'ENSG00000185920_PTCH1', 'ENSG00000185946_RNPC3', 'ENSG00000185947_ZNF267', 'ENSG00000185950_IRS2', 'ENSG00000185955_C7orf61', 'ENSG00000185963_BICD2', 'ENSG00000185973_TMLHE', 'ENSG00000185986_SDHAP3', 'ENSG00000185989_RASA3', 'ENSG00000186001_LRCH3', 'ENSG00000186010_NDUFA13', 'ENSG00000186017_ZNF566', 'ENSG00000186019_AC021092.1', 'ENSG00000186020_ZNF529', 'ENSG00000186026_ZNF284', 'ENSG00000186047_DLEU7', 'ENSG00000186051_TAL2', 'ENSG00000186056_MATN1-AS1', 'ENSG00000186063_AIDA', 'ENSG00000186073_C15orf41', 'ENSG00000186074_CD300LF', 'ENSG00000186088_GSAP', 'ENSG00000186104_CYP2R1', 'ENSG00000186105_LRRC70', 'ENSG00000186106_ANKRD46', 'ENSG00000186111_PIP5K1C', 'ENSG00000186115_CYP4F2', 'ENSG00000186118_TEX38', 'ENSG00000186130_ZBTB6', 'ENSG00000186132_C2orf76', 'ENSG00000186141_POLR3C', 'ENSG00000186153_WWOX', 'ENSG00000186162_CIDECP', 'ENSG00000186166_CCDC84', 'ENSG00000186174_BCL9L', 'ENSG00000186184_POLR1D', 'ENSG00000186185_KIF18B', 'ENSG00000186187_ZNRF1', 'ENSG00000186188_FFAR4', 'ENSG00000186193_SAPCD2', 'ENSG00000186197_EDARADD', 'ENSG00000186205_MARC1', 'ENSG00000186222_BLOC1S4', 'ENSG00000186230_ZNF749', 'ENSG00000186235_AC016757.1', 'ENSG00000186244_AC091180.1', 'ENSG00000186260_MRTFB', 'ENSG00000186272_ZNF17', 'ENSG00000186280_KDM4D', 'ENSG00000186281_GPAT2', 'ENSG00000186283_TOR3A', 'ENSG00000186298_PPP1CC', 'ENSG00000186300_ZNF555', 'ENSG00000186310_NAP1L3', 'ENSG00000186312_CA5BP1', 'ENSG00000186314_PRELID2', 'ENSG00000186318_BACE1', 'ENSG00000186326_RGS9BP', 'ENSG00000186350_RXRA', 'ENSG00000186352_ANKRD37', 'ENSG00000186364_NUDT17', 'ENSG00000186376_ZNF75D', 'ENSG00000186395_KRT10', 'ENSG00000186399_GOLGA8R', 'ENSG00000186407_CD300E', 'ENSG00000186409_CCDC30', 'ENSG00000186416_NKRF', 'ENSG00000186432_KPNA4', 'ENSG00000186446_ZNF501', 'ENSG00000186448_ZNF197', 'ENSG00000186451_SPATA12', 'ENSG00000186462_NAP1L2', 'ENSG00000186468_RPS23', 'ENSG00000186469_GNG2', 'ENSG00000186470_BTN3A2', 'ENSG00000186472_PCLO', 'ENSG00000186480_INSIG1', 'ENSG00000186481_ANKRD20A5P', 'ENSG00000186487_MYT1L', 'ENSG00000186496_ZNF396', 'ENSG00000186501_TMEM222', 'ENSG00000186517_ARHGAP30', 'ENSG00000186522_SEPT10', 'ENSG00000186523_FAM86B1', 'ENSG00000186529_CYP4F3', 'ENSG00000186532_SMYD4', 'ENSG00000186564_FOXD2', 'ENSG00000186566_GPATCH8', 'ENSG00000186567_CEACAM19', 'ENSG00000186575_NF2', 'ENSG00000186577_SMIM29', 'ENSG00000186591_UBE2H', 'ENSG00000186594_MIR22HG', 'ENSG00000186603_HPDL', 'ENSG00000186615_KTN1-AS1', 'ENSG00000186625_KATNA1', 'ENSG00000186628_FSD2', 'ENSG00000186635_ARAP1', 'ENSG00000186638_KIF24', 'ENSG00000186642_PDE2A', 'ENSG00000186648_CARMIL3', 'ENSG00000186652_PRG2', 'ENSG00000186654_PRR5', 'ENSG00000186660_ZFP91', 'ENSG00000186665_C17orf58', 'ENSG00000186666_BCDIN3D', 'ENSG00000186687_LYRM7', 'ENSG00000186710_CFAP73', 'ENSG00000186714_CCDC73', 'ENSG00000186716_BCR', 'ENSG00000186723_OR10H1', 'ENSG00000186767_SPIN4', 'ENSG00000186777_ZNF732', 'ENSG00000186787_SPIN2B', 'ENSG00000186792_HYAL3', 'ENSG00000186806_VSIG10L', 'ENSG00000186810_CXCR3', 'ENSG00000186812_ZNF397', 'ENSG00000186814_ZSCAN30', 'ENSG00000186815_TPCN1', 'ENSG00000186818_LILRB4', 'ENSG00000186827_TNFRSF4', 'ENSG00000186834_HEXIM1', 'ENSG00000186854_TRABD2A', 'ENSG00000186862_PDZD7', 'ENSG00000186866_POFUT2', 'ENSG00000186868_MAPT', 'ENSG00000186871_ERCC6L', 'ENSG00000186889_TMEM17', 'ENSG00000186891_TNFRSF18', 'ENSG00000186907_RTN4RL2', 'ENSG00000186908_ZDHHC17', 'ENSG00000186918_ZNF395', 'ENSG00000186919_ZACN', 'ENSG00000186951_PPARA', 'ENSG00000186952_TMEM232', 'ENSG00000186976_EFCAB6', 'ENSG00000186998_EMID1', 'ENSG00000187010_RHD', 'ENSG00000187017_ESPN', 'ENSG00000187024_PTRH1', 'ENSG00000187037_GPR141', 'ENSG00000187049_TMEM216', 'ENSG00000187051_RPS19BP1', 'ENSG00000187066_TMEM262', 'ENSG00000187068_C3orf70', 'ENSG00000187091_PLCD1', 'ENSG00000187097_ENTPD5', 'ENSG00000187098_MITF', 'ENSG00000187109_NAP1L1', 'ENSG00000187118_CMC1', 'ENSG00000187123_LYPD6', 'ENSG00000187134_AKR1C1', 'ENSG00000187144_SPATA21', 'ENSG00000187147_RNF220', 'ENSG00000187164_SHTN1', 'ENSG00000187186_AL162231.1', 'ENSG00000187187_ZNF546', 'ENSG00000187189_TSPYL4', 'ENSG00000187193_MT1X', 'ENSG00000187210_GCNT1', 'ENSG00000187231_SESTD1', 'ENSG00000187239_FNBP1', 'ENSG00000187240_DYNC2H1', 'ENSG00000187244_BCAM', 'ENSG00000187257_RSBN1L', 'ENSG00000187260_WDR86', 'ENSG00000187266_EPOR', 'ENSG00000187325_TAF9B', 'ENSG00000187391_MAGI2', 'ENSG00000187446_CHP1', 'ENSG00000187474_FPR3', 'ENSG00000187479_C11orf96', 'ENSG00000187486_KCNJ11', 'ENSG00000187498_COL4A1', 'ENSG00000187513_GJA4', 'ENSG00000187514_PTMA', 'ENSG00000187522_HSPA14', 'ENSG00000187531_SIRT7', 'ENSG00000187534_PRR13P5', 'ENSG00000187535_IFT140', 'ENSG00000187536_TPM3P7', 'ENSG00000187555_USP7', 'ENSG00000187556_NANOS3', 'ENSG00000187566_NHLRC1', 'ENSG00000187595_ZNF385C', 'ENSG00000187601_MAGEH1', 'ENSG00000187605_TET3', 'ENSG00000187607_ZNF286A', 'ENSG00000187608_ISG15', 'ENSG00000187609_EXD3', 'ENSG00000187621_TCL6', 'ENSG00000187624_C17orf97', 'ENSG00000187626_ZKSCAN4', 'ENSG00000187630_DHRS4L2', 'ENSG00000187634_SAMD11', 'ENSG00000187650_VMAC', 'ENSG00000187676_B3GLCT', 'ENSG00000187678_SPRY4', 'ENSG00000187682_ERAS', 'ENSG00000187688_TRPV2', 'ENSG00000187699_C2orf88', 'ENSG00000187713_TMEM203', 'ENSG00000187715_KBTBD12', 'ENSG00000187730_GABRD', 'ENSG00000187735_TCEA1', 'ENSG00000187736_NHEJ1', 'ENSG00000187741_FANCA', 'ENSG00000187742_SECISBP2', 'ENSG00000187753_C9orf153', 'ENSG00000187764_SEMA4D', 'ENSG00000187778_MCRS1', 'ENSG00000187783_TMEM72', 'ENSG00000187790_FANCM', 'ENSG00000187792_ZNF70', 'ENSG00000187796_CARD9', 'ENSG00000187800_PEAR1', 'ENSG00000187801_ZFP69B', 'ENSG00000187808_SOWAHD', 'ENSG00000187815_ZFP69', 'ENSG00000187824_TMEM220', 'ENSG00000187837_HIST1H1C', 'ENSG00000187838_PLSCR3', 'ENSG00000187840_EIF4EBP1', 'ENSG00000187860_CCDC157', 'ENSG00000187862_TTC24', 'ENSG00000187866_FAM122A', 'ENSG00000187902_SHISA7', 'ENSG00000187905_LRRC74B', 'ENSG00000187942_LDLRAD2', 'ENSG00000187951_AC091057.1', 'ENSG00000187953_PMS2CL', 'ENSG00000187954_CYHR1', 'ENSG00000187955_COL14A1', 'ENSG00000187961_KLHL17', 'ENSG00000187987_ZSCAN23', 'ENSG00000187994_RINL', 'ENSG00000187997_C17orf99', 'ENSG00000188002_AC026412.1', 'ENSG00000188010_MORN2', 'ENSG00000188015_S100A3', 'ENSG00000188021_UBQLN2', 'ENSG00000188026_RILPL1', 'ENSG00000188033_ZNF490', 'ENSG00000188037_CLCN1', 'ENSG00000188042_ARL4C', 'ENSG00000188051_TMEM221', 'ENSG00000188060_RAB42', 'ENSG00000188070_C11orf95', 'ENSG00000188092_GPR89B', 'ENSG00000188095_MESP2', 'ENSG00000188107_EYS', 'ENSG00000188112_C6orf132', 'ENSG00000188124_OR2AG2', 'ENSG00000188130_MAPK12', 'ENSG00000188152_NUTM2G', 'ENSG00000188157_AGRN', 'ENSG00000188167_TMPPE', 'ENSG00000188171_ZNF626', 'ENSG00000188177_ZC3H6', 'ENSG00000188185_LINC00265', 'ENSG00000188186_LAMTOR4', 'ENSG00000188191_PRKAR1B', 'ENSG00000188199_NUTM2B', 'ENSG00000188211_NCR3LG1', 'ENSG00000188215_DCUN1D3', 'ENSG00000188227_ZNF793', 'ENSG00000188229_TUBB4B', 'ENSG00000188234_AGAP4', 'ENSG00000188243_COMMD6', 'ENSG00000188266_HYKK', 'ENSG00000188277_C15orf62', 'ENSG00000188283_ZNF383', 'ENSG00000188290_HES4', 'ENSG00000188295_ZNF669', 'ENSG00000188305_PEAK3', 'ENSG00000188306_LRRIQ4', 'ENSG00000188312_CENPP', 'ENSG00000188313_PLSCR1', 'ENSG00000188315_C3orf62', 'ENSG00000188321_ZNF559', 'ENSG00000188322_SBK1', 'ENSG00000188342_GTF2F2', 'ENSG00000188343_FAM92A', 'ENSG00000188352_FOCAD', 'ENSG00000188365_AC092171.1', 'ENSG00000188368_PRR19', 'ENSG00000188372_ZP3', 'ENSG00000188375_H3F3C', 'ENSG00000188396_TCTEX1D4', 'ENSG00000188404_SELL', 'ENSG00000188419_CHM', 'ENSG00000188428_BLOC1S5', 'ENSG00000188451_SRP72P2', 'ENSG00000188452_CERKL', 'ENSG00000188459_WASF4P', 'ENSG00000188467_SLC24A5', 'ENSG00000188483_IER5L', 'ENSG00000188486_H2AFX', 'ENSG00000188493_C19orf54', 'ENSG00000188501_LCTL', 'ENSG00000188511_C22orf34', 'ENSG00000188522_FAM83G', 'ENSG00000188529_SRSF10', 'ENSG00000188536_HBA2', 'ENSG00000188542_DUSP28', 'ENSG00000188549_CCDC9B', 'ENSG00000188554_NBR1', 'ENSG00000188559_RALGAPA2', 'ENSG00000188566_NDOR1', 'ENSG00000188573_FBLL1', 'ENSG00000188580_NKAIN2', 'ENSG00000188585_CLEC20A', 'ENSG00000188596_CFAP54', 'ENSG00000188599_NPIPP1', 'ENSG00000188603_CLN3', 'ENSG00000188610_FAM72B', 'ENSG00000188611_ASAH2', 'ENSG00000188612_SUMO2', 'ENSG00000188613_NANOS1', 'ENSG00000188626_GOLGA8M', 'ENSG00000188636_RTL6', 'ENSG00000188641_DPYD', 'ENSG00000188643_S100A16', 'ENSG00000188647_PTAR1', 'ENSG00000188649_CC2D2B', 'ENSG00000188659_SAXO2', 'ENSG00000188672_RHCE', 'ENSG00000188674_C2orf80', 'ENSG00000188677_PARVB', 'ENSG00000188681_TEKT4P2', 'ENSG00000188687_SLC4A5', 'ENSG00000188690_UROS', 'ENSG00000188693_CYP51A1-AS1', 'ENSG00000188706_ZDHHC9', 'ENSG00000188707_ZBED6CL', 'ENSG00000188710_QRFP', 'ENSG00000188725_SMIM15', 'ENSG00000188732_FAM221A', 'ENSG00000188735_TMEM120B', 'ENSG00000188738_FSIP2', 'ENSG00000188739_RBM34', 'ENSG00000188747_NOXA1', 'ENSG00000188760_TMEM198', 'ENSG00000188761_BCL2L15', 'ENSG00000188763_FZD9', 'ENSG00000188766_SPRED3', 'ENSG00000188785_ZNF548', 'ENSG00000188786_MTF1', 'ENSG00000188807_TMEM201', 'ENSG00000188811_NHLRC3', 'ENSG00000188818_ZDHHC11', 'ENSG00000188820_CALHM6', 'ENSG00000188822_CNR2', 'ENSG00000188825_LINC00910', 'ENSG00000188827_SLX4', 'ENSG00000188846_RPL14', 'ENSG00000188848_BEND4', 'ENSG00000188859_FAM78B', 'ENSG00000188868_ZNF563', 'ENSG00000188873_RPL10AP2', 'ENSG00000188878_FBF1', 'ENSG00000188883_KLRG2', 'ENSG00000188895_MSL1', 'ENSG00000188897_AC099489.1', 'ENSG00000188906_LRRK2', 'ENSG00000188916_INSYN2', 'ENSG00000188917_TRMT2B', 'ENSG00000188921_HACD4', 'ENSG00000188933_USP32P1', 'ENSG00000188938_FAM120AOS', 'ENSG00000188976_NOC2L', 'ENSG00000188981_MSANTD1', 'ENSG00000188985_DHFRP1', 'ENSG00000188986_NELFB', 'ENSG00000188994_ZNF292', 'ENSG00000188996_HUS1B', 'ENSG00000188997_KCTD21', 'ENSG00000189007_ADAT2', 'ENSG00000189042_ZNF567', 'ENSG00000189043_NDUFA4', 'ENSG00000189045_ANKDD1B', 'ENSG00000189046_ALKBH2', 'ENSG00000189050_RNFT1', 'ENSG00000189051_RNF222', 'ENSG00000189057_FAM111B', 'ENSG00000189060_H1F0', 'ENSG00000189067_LITAF', 'ENSG00000189068_VSTM1', 'ENSG00000189077_TMEM120A', 'ENSG00000189079_ARID2', 'ENSG00000189091_SF3B3', 'ENSG00000189114_BLOC1S3', 'ENSG00000189134_NKAPL', 'ENSG00000189136_UBE2Q2P1', 'ENSG00000189144_ZNF573', 'ENSG00000189149_CRYM-AS1', 'ENSG00000189152_GRAPL', 'ENSG00000189157_FAM47E', 'ENSG00000189159_JPT1', 'ENSG00000189164_ZNF527', 'ENSG00000189171_S100A13', 'ENSG00000189180_ZNF33A', 'ENSG00000189190_ZNF600', 'ENSG00000189195_BTBD8', 'ENSG00000189212_DPY19L2P1', 'ENSG00000189221_MAOA', 'ENSG00000189223_PAX8-AS1', 'ENSG00000189227_C15orf61', 'ENSG00000189229_AC069277.1', 'ENSG00000189241_TSPYL1', 'ENSG00000189266_PNRC2', 'ENSG00000189269_DRICH1', 'ENSG00000189283_FHIT', 'ENSG00000189292_ALKAL2', 'ENSG00000189298_ZKSCAN3', 'ENSG00000189306_RRP7A', 'ENSG00000189308_LIN54', 'ENSG00000189316_AC073349.1', 'ENSG00000189319_FAM53B', 'ENSG00000189337_KAZN', 'ENSG00000189339_SLC35E2B', 'ENSG00000189343_RPS2P46', 'ENSG00000189350_TOGARAM2', 'ENSG00000189362_NEMP2', 'ENSG00000189366_ALG1L', 'ENSG00000189369_GSPT2', 'ENSG00000189376_C8orf76', 'ENSG00000189401_OTUD6A', 'ENSG00000189403_HMGB1', 'ENSG00000189410_SH2D5', 'ENSG00000189419_SPATA41', 'ENSG00000189420_ZFP92', 'ENSG00000189423_USP32P3', 'ENSG00000189430_NCR1', 'ENSG00000196071_OR2L13', 'ENSG00000196072_BLOC1S2', 'ENSG00000196074_SYCP2', 'ENSG00000196081_ZNF724', 'ENSG00000196083_IL1RAP', 'ENSG00000196104_SPOCK3', 'ENSG00000196110_ZNF699', 'ENSG00000196114_AL031577.1', 'ENSG00000196116_TDRD7', 'ENSG00000196118_CCDC189', 'ENSG00000196123_KIAA0895L', 'ENSG00000196126_HLA-DRB1', 'ENSG00000196132_MYT1', 'ENSG00000196139_AKR1C3', 'ENSG00000196141_SPATS2L', 'ENSG00000196150_ZNF250', 'ENSG00000196151_WDSUB1', 'ENSG00000196152_ZNF79', 'ENSG00000196154_S100A4', 'ENSG00000196155_PLEKHG4', 'ENSG00000196159_FAT4', 'ENSG00000196166_C8orf86', 'ENSG00000196167_COLCA1', 'ENSG00000196169_KIF19', 'ENSG00000196172_ZNF681', 'ENSG00000196177_ACADSB', 'ENSG00000196182_STK40', 'ENSG00000196187_TMEM63A', 'ENSG00000196188_CTSE', 'ENSG00000196189_SEMA4A', 'ENSG00000196199_MPHOSPH8', 'ENSG00000196204_RNF216P1', 'ENSG00000196205_EEF1A1P5', 'ENSG00000196208_GREB1', 'ENSG00000196209_SIRPB2', 'ENSG00000196214_ZNF766', 'ENSG00000196218_RYR1', 'ENSG00000196220_SRGAP3', 'ENSG00000196227_FAM217B', 'ENSG00000196230_TUBB', 'ENSG00000196233_LCOR', 'ENSG00000196235_SUPT5H', 'ENSG00000196236_XPNPEP3', 'ENSG00000196242_OR2C3', 'ENSG00000196247_ZNF107', 'ENSG00000196262_PPIA', 'ENSG00000196263_ZNF471', 'ENSG00000196267_ZNF836', 'ENSG00000196268_ZNF493', 'ENSG00000196275_GTF2IRD2', 'ENSG00000196284_SUPT3H', 'ENSG00000196290_NIF3L1', 'ENSG00000196295_GARS-DT', 'ENSG00000196296_ATP2A1', 'ENSG00000196305_IARS', 'ENSG00000196312_MFSD14C', 'ENSG00000196313_POM121', 'ENSG00000196323_ZBTB44', 'ENSG00000196329_GIMAP5', 'ENSG00000196338_NLGN3', 'ENSG00000196345_ZKSCAN7', 'ENSG00000196352_CD55', 'ENSG00000196357_ZNF565', 'ENSG00000196358_NTNG2', 'ENSG00000196361_ELAVL3', 'ENSG00000196363_WDR5', 'ENSG00000196365_LONP1', 'ENSG00000196366_C9orf163', 'ENSG00000196367_TRRAP', 'ENSG00000196368_NUDT11', 'ENSG00000196369_SRGAP2B', 'ENSG00000196371_FUT4', 'ENSG00000196372_ASB13', 'ENSG00000196378_ZNF34', 'ENSG00000196381_ZNF781', 'ENSG00000196387_ZNF140', 'ENSG00000196391_ZNF774', 'ENSG00000196396_PTPN1', 'ENSG00000196405_EVL', 'ENSG00000196411_EPHB4', 'ENSG00000196415_PRTN3', 'ENSG00000196417_ZNF765', 'ENSG00000196418_ZNF124', 'ENSG00000196419_XRCC6', 'ENSG00000196420_S100A5', 'ENSG00000196421_C20orf204', 'ENSG00000196422_PPP1R26', 'ENSG00000196428_TSC22D2', 'ENSG00000196431_CRYBA4', 'ENSG00000196436_NPIPB15', 'ENSG00000196437_ZNF569', 'ENSG00000196440_ARMCX4', 'ENSG00000196449_YRDC', 'ENSG00000196453_ZNF777', 'ENSG00000196455_PIK3R4', 'ENSG00000196456_ZNF775', 'ENSG00000196458_ZNF605', 'ENSG00000196459_TRAPPC2', 'ENSG00000196460_RFX8', 'ENSG00000196465_MYL6B', 'ENSG00000196466_ZNF799', 'ENSG00000196468_FGF16', 'ENSG00000196470_SIAH1', 'ENSG00000196476_C20orf96', 'ENSG00000196497_IPO4', 'ENSG00000196498_NCOR2', 'ENSG00000196502_SULT1A1', 'ENSG00000196503_ARL9', 'ENSG00000196504_PRPF40A', 'ENSG00000196505_GDAP2', 'ENSG00000196507_TCEAL3', 'ENSG00000196510_ANAPC7', 'ENSG00000196511_TPK1', 'ENSG00000196517_SLC6A9', 'ENSG00000196526_AFAP1', 'ENSG00000196531_NACA', 'ENSG00000196535_MYO18A', 'ENSG00000196544_BORCS6', 'ENSG00000196547_MAN2A2', 'ENSG00000196549_MME', 'ENSG00000196550_FAM72A', 'ENSG00000196562_SULF2', 'ENSG00000196565_HBG2', 'ENSG00000196569_LAMA2', 'ENSG00000196576_PLXNB2', 'ENSG00000196584_XRCC2', 'ENSG00000196586_MYO6', 'ENSG00000196588_MRTFA', 'ENSG00000196591_HDAC2', 'ENSG00000196597_ZNF782', 'ENSG00000196605_ZNF846', 'ENSG00000196628_TCF4', 'ENSG00000196632_WNK3', 'ENSG00000196636_SDHAF3', 'ENSG00000196639_HRH1', 'ENSG00000196642_RABL6', 'ENSG00000196646_ZNF136', 'ENSG00000196652_ZKSCAN5', 'ENSG00000196653_ZNF502', 'ENSG00000196655_TRAPPC4', 'ENSG00000196656_AC004057.1', 'ENSG00000196659_TTC30B', 'ENSG00000196663_TECPR2', 'ENSG00000196664_TLR7', 'ENSG00000196666_FAM180B', 'ENSG00000196668_LINC00173', 'ENSG00000196670_ZFP62', 'ENSG00000196678_ERI2', 'ENSG00000196683_TOMM7', 'ENSG00000196684_HSH2D', 'ENSG00000196689_TRPV1', 'ENSG00000196693_ZNF33B', 'ENSG00000196696_PDXDC2P-NPIPB14P', 'ENSG00000196700_ZNF512B', 'ENSG00000196704_AMZ2', 'ENSG00000196705_ZNF431', 'ENSG00000196712_NF1', 'ENSG00000196715_VKORC1L1', 'ENSG00000196724_ZNF418', 'ENSG00000196730_DAPK1', 'ENSG00000196735_HLA-DQA1', 'ENSG00000196739_COL27A1', 'ENSG00000196741_LINC01560', 'ENSG00000196743_GM2A', 'ENSG00000196747_HIST1H2AI', 'ENSG00000196748_CLPSL2', 'ENSG00000196754_S100A2', 'ENSG00000196756_SNHG17', 'ENSG00000196757_ZNF700', 'ENSG00000196776_CD47', 'ENSG00000196781_TLE1', 'ENSG00000196782_MAML3', 'ENSG00000196787_HIST1H2AG', 'ENSG00000196792_STRN3', 'ENSG00000196793_ZNF239', 'ENSG00000196810_CTBP1-DT', 'ENSG00000196811_CHRNG', 'ENSG00000196812_ZSCAN16', 'ENSG00000196814_MVB12B', 'ENSG00000196821_C6orf106', 'ENSG00000196832_OR11G2', 'ENSG00000196839_ADA', 'ENSG00000196843_ARID5A', 'ENSG00000196850_PPTC7', 'ENSG00000196860_TOMM20L', 'ENSG00000196865_NHLRC2', 'ENSG00000196866_HIST1H2AD', 'ENSG00000196867_ZFP28', 'ENSG00000196872_KIAA1211L', 'ENSG00000196873_CBWD3', 'ENSG00000196876_SCN8A', 'ENSG00000196878_LAMB3', 'ENSG00000196890_HIST3H2BB', 'ENSG00000196911_KPNA5', 'ENSG00000196912_ANKRD36B', 'ENSG00000196914_ARHGEF12', 'ENSG00000196922_ZNF252P', 'ENSG00000196923_PDLIM7', 'ENSG00000196924_FLNA', 'ENSG00000196933_RPS26P11', 'ENSG00000196935_SRGAP1', 'ENSG00000196937_FAM3C', 'ENSG00000196943_NOP9', 'ENSG00000196950_SLC39A10', 'ENSG00000196951_SCOC-AS1', 'ENSG00000196954_CASP4', 'ENSG00000196961_AP2A1', 'ENSG00000196967_ZNF585A', 'ENSG00000196968_FUT11', 'ENSG00000196972_SMIM10L2B', 'ENSG00000196975_ANXA4', 'ENSG00000196976_LAGE3', 'ENSG00000196981_WDR5B', 'ENSG00000196998_WDR45', 'ENSG00000197006_METTL9', 'ENSG00000197008_ZNF138', 'ENSG00000197013_ZNF429', 'ENSG00000197016_ZNF470', 'ENSG00000197019_SERTAD1', 'ENSG00000197020_ZNF100', 'ENSG00000197021_CXorf40B', 'ENSG00000197024_ZNF398', 'ENSG00000197037_ZSCAN25', 'ENSG00000197043_ANXA6', 'ENSG00000197044_ZNF441', 'ENSG00000197045_GMFB', 'ENSG00000197046_SIGLEC15', 'ENSG00000197050_ZNF420', 'ENSG00000197054_ZNF763', 'ENSG00000197056_ZMYM1', 'ENSG00000197061_HIST1H4C', 'ENSG00000197062_ZSCAN26', 'ENSG00000197063_MAFG', 'ENSG00000197070_ARRDC1', 'ENSG00000197077_KIAA1671', 'ENSG00000197081_IGF2R', 'ENSG00000197093_GAL3ST4', 'ENSG00000197099_AC068631.1', 'ENSG00000197102_DYNC1H1', 'ENSG00000197111_PCBP2', 'ENSG00000197114_ZGPAT', 'ENSG00000197119_SLC25A29', 'ENSG00000197121_PGAP1', 'ENSG00000197122_SRC', 'ENSG00000197124_ZNF682', 'ENSG00000197128_ZNF772', 'ENSG00000197134_ZNF257', 'ENSG00000197136_PCNX3', 'ENSG00000197140_ADAM32', 'ENSG00000197142_ACSL5', 'ENSG00000197147_LRRC8B', 'ENSG00000197149_AC107956.1', 'ENSG00000197150_ABCB8', 'ENSG00000197153_HIST1H3J', 'ENSG00000197157_SND1', 'ENSG00000197162_ZNF785', 'ENSG00000197165_SULT1A2', 'ENSG00000197168_NEK5', 'ENSG00000197170_PSMD12', 'ENSG00000197180_CH17-340M24.3', 'ENSG00000197182_MIRLET7BHG', 'ENSG00000197183_NOL4L', 'ENSG00000197191_CYSRT1', 'ENSG00000197208_SLC22A4', 'ENSG00000197217_ENTPD4', 'ENSG00000197223_C1D', 'ENSG00000197226_TBC1D9B', 'ENSG00000197238_HIST1H4J', 'ENSG00000197245_FAM110D', 'ENSG00000197249_SERPINA1', 'ENSG00000197253_TPSB2', 'ENSG00000197256_KANK2', 'ENSG00000197258_EIF4BP6', 'ENSG00000197261_C6orf141', 'ENSG00000197265_GTF2E2', 'ENSG00000197275_RAD54B', 'ENSG00000197279_ZNF165', 'ENSG00000197283_SYNGAP1', 'ENSG00000197291_RAMP2-AS1', 'ENSG00000197296_FITM2', 'ENSG00000197299_BLM', 'ENSG00000197301_AC090673.1', 'ENSG00000197302_ZNF720', 'ENSG00000197312_DDI2', 'ENSG00000197321_SVIL', 'ENSG00000197323_TRIM33', 'ENSG00000197324_LRP10', 'ENSG00000197329_PELI1', 'ENSG00000197332_AC008543.1', 'ENSG00000197343_ZNF655', 'ENSG00000197345_MRPL21', 'ENSG00000197355_UAP1L1', 'ENSG00000197360_ZNF98', 'ENSG00000197361_FBXL22', 'ENSG00000197362_ZNF786', 'ENSG00000197363_ZNF517', 'ENSG00000197372_ZNF675', 'ENSG00000197375_SLC22A5', 'ENSG00000197380_DACT3', 'ENSG00000197381_ADARB1', 'ENSG00000197386_HTT', 'ENSG00000197405_C5AR1', 'ENSG00000197415_VEPH1', 'ENSG00000197417_SHPK', 'ENSG00000197429_IPP', 'ENSG00000197442_MAP3K5', 'ENSG00000197444_OGDHL', 'ENSG00000197448_GSTK1', 'ENSG00000197451_HNRNPAB', 'ENSG00000197454_OR2L5', 'ENSG00000197457_STMN3', 'ENSG00000197461_PDGFA', 'ENSG00000197465_GYPE', 'ENSG00000197471_SPN', 'ENSG00000197472_ZNF695', 'ENSG00000197483_ZNF628', 'ENSG00000197496_SLC2A10', 'ENSG00000197497_ZNF665', 'ENSG00000197498_RPF2', 'ENSG00000197506_SLC28A3', 'ENSG00000197530_MIB2', 'ENSG00000197535_MYO5A', 'ENSG00000197536_C5orf56', 'ENSG00000197548_ATG7', 'ENSG00000197550_AL359955.1', 'ENSG00000197555_SIPA1L1', 'ENSG00000197557_TTC30A', 'ENSG00000197558_SSPO', 'ENSG00000197561_ELANE', 'ENSG00000197562_RAB40C', 'ENSG00000197563_PIGN', 'ENSG00000197566_ZNF624', 'ENSG00000197568_HHLA3', 'ENSG00000197575_RPS17P2', 'ENSG00000197576_HOXA4', 'ENSG00000197579_TOPORS', 'ENSG00000197580_BCO2', 'ENSG00000197582_GPX1P1', 'ENSG00000197586_ENTPD6', 'ENSG00000197599_CCDC154', 'ENSG00000197601_FAR1', 'ENSG00000197603_CPLANE1', 'ENSG00000197608_ZNF841', 'ENSG00000197619_ZNF615', 'ENSG00000197620_CXorf40A', 'ENSG00000197622_CDC42SE1', 'ENSG00000197629_MPEG1', 'ENSG00000197632_SERPINB2', 'ENSG00000197635_DPP4', 'ENSG00000197647_ZNF433', 'ENSG00000197653_DNAH10', 'ENSG00000197670_AL157838.1', 'ENSG00000197694_SPTAN1', 'ENSG00000197696_NMB', 'ENSG00000197712_FAM114A1', 'ENSG00000197713_RPE', 'ENSG00000197714_ZNF460', 'ENSG00000197721_CR1L', 'ENSG00000197724_PHF2', 'ENSG00000197728_RPS26', 'ENSG00000197734_C14orf178', 'ENSG00000197744_PTMAP2', 'ENSG00000197746_PSAP', 'ENSG00000197747_S100A10', 'ENSG00000197748_CFAP43', 'ENSG00000197753_LHFPL5', 'ENSG00000197756_RPL37A', 'ENSG00000197763_TXNRD3', 'ENSG00000197766_CFD', 'ENSG00000197771_MCMBP', 'ENSG00000197774_EME2', 'ENSG00000197776_KLHDC1', 'ENSG00000197779_ZNF81', 'ENSG00000197780_TAF13', 'ENSG00000197782_ZNF780A', 'ENSG00000197785_ATAD3A', 'ENSG00000197798_FAM118B', 'ENSG00000197808_ZNF461', 'ENSG00000197813_AC011450.1', 'ENSG00000197816_CCDC180', 'ENSG00000197818_SLC9A8', 'ENSG00000197822_OCLN', 'ENSG00000197837_HIST4H4', 'ENSG00000197841_ZNF181', 'ENSG00000197847_SLC22A20P', 'ENSG00000197857_ZNF44', 'ENSG00000197858_GPAA1', 'ENSG00000197860_SGTB', 'ENSG00000197863_ZNF790', 'ENSG00000197872_FAM49A', 'ENSG00000197879_MYO1C', 'ENSG00000197885_NKIRAS1', 'ENSG00000197889_MEIG1', 'ENSG00000197892_KIF13B', 'ENSG00000197894_ADH5', 'ENSG00000197903_HIST1H2BK', 'ENSG00000197905_TEAD4', 'ENSG00000197912_SPG7', 'ENSG00000197921_HES5', 'ENSG00000197927_C2orf27A', 'ENSG00000197928_ZNF677', 'ENSG00000197930_ERO1A', 'ENSG00000197933_ZNF823', 'ENSG00000197935_ZNF311', 'ENSG00000197937_ZNF347', 'ENSG00000197943_PLCG2', 'ENSG00000197948_FCHSD1', 'ENSG00000197951_ZNF71', 'ENSG00000197956_S100A6', 'ENSG00000197958_RPL12', 'ENSG00000197959_DNM3', 'ENSG00000197961_ZNF121', 'ENSG00000197965_MPZL1', 'ENSG00000197969_VPS13A', 'ENSG00000197971_MBP', 'ENSG00000197976_AKAP17A', 'ENSG00000197978_GOLGA6L9', 'ENSG00000197980_LEKR1', 'ENSG00000197982_C1orf122', 'ENSG00000197989_SNHG12', 'ENSG00000197992_CLEC9A', 'ENSG00000197993_KEL', 'ENSG00000198000_NOL8', 'ENSG00000198001_IRAK4', 'ENSG00000198003_CCDC151', 'ENSG00000198015_MRPL42', 'ENSG00000198018_ENTPD7', 'ENSG00000198019_FCGR1B', 'ENSG00000198026_ZNF335', 'ENSG00000198028_ZNF560', 'ENSG00000198034_RPS4X', 'ENSG00000198039_ZNF273', 'ENSG00000198040_ZNF84', 'ENSG00000198042_MAK16', 'ENSG00000198046_ZNF667', 'ENSG00000198053_SIRPA', 'ENSG00000198055_GRK6', 'ENSG00000198056_PRIM1', 'ENSG00000198060_MARCH5', 'ENSG00000198064_NPIPB13', 'ENSG00000198075_SULT1C4', 'ENSG00000198081_ZBTB14', 'ENSG00000198087_CD2AP', 'ENSG00000198088_NUP62CL', 'ENSG00000198089_SFI1', 'ENSG00000198093_ZNF649', 'ENSG00000198105_ZNF248', 'ENSG00000198106_SNX29P2', 'ENSG00000198108_CHSY3', 'ENSG00000198113_TOR4A', 'ENSG00000198121_LPAR1', 'ENSG00000198128_OR2L3', 'ENSG00000198130_HIBCH', 'ENSG00000198131_ZNF544', 'ENSG00000198133_TMEM229B', 'ENSG00000198142_SOWAHC', 'ENSG00000198146_ZNF770', 'ENSG00000198153_ZNF849P', 'ENSG00000198155_ZNF876P', 'ENSG00000198157_HMGN5', 'ENSG00000198160_MIER1', 'ENSG00000198162_MAN1A2', 'ENSG00000198168_SVIP', 'ENSG00000198169_ZNF251', 'ENSG00000198171_DDRGK1', 'ENSG00000198176_TFDP1', 'ENSG00000198178_CLEC4C', 'ENSG00000198182_ZNF607', 'ENSG00000198185_ZNF334', 'ENSG00000198189_HSD17B11', 'ENSG00000198198_SZT2', 'ENSG00000198203_SULT1C2', 'ENSG00000198205_ZXDA', 'ENSG00000198208_RPS6KL1', 'ENSG00000198218_QRICH1', 'ENSG00000198223_CSF2RA', 'ENSG00000198225_FKBP1C', 'ENSG00000198231_DDX42', 'ENSG00000198237_AC131392.1', 'ENSG00000198242_RPL23A', 'ENSG00000198246_SLC29A3', 'ENSG00000198252_STYX', 'ENSG00000198258_UBL5', 'ENSG00000198265_HELZ', 'ENSG00000198270_TMEM116', 'ENSG00000198276_UCKL1', 'ENSG00000198286_CARD11', 'ENSG00000198298_ZNF485', 'ENSG00000198300_PEG3', 'ENSG00000198301_SDAD1', 'ENSG00000198315_ZKSCAN8', 'ENSG00000198324_PHETA1', 'ENSG00000198331_HYLS1', 'ENSG00000198336_MYL4', 'ENSG00000198342_ZNF442', 'ENSG00000198346_ZNF813', 'ENSG00000198355_PIM3', 'ENSG00000198356_ASNA1', 'ENSG00000198363_ASPH', 'ENSG00000198369_SPRED2', 'ENSG00000198373_WWP2', 'ENSG00000198380_GFPT1', 'ENSG00000198382_UVRAG', 'ENSG00000198393_ZNF26', 'ENSG00000198399_ITSN2', 'ENSG00000198400_NTRK1', 'ENSG00000198406_BZW1P2', 'ENSG00000198408_OGA', 'ENSG00000198416_ZNF658B', 'ENSG00000198417_MT1F', 'ENSG00000198420_TCAF1', 'ENSG00000198429_ZNF69', 'ENSG00000198431_TXNRD1', 'ENSG00000198435_NRARP', 'ENSG00000198440_ZNF583', 'ENSG00000198452_OR14L1P', 'ENSG00000198453_ZNF568', 'ENSG00000198455_ZXDB', 'ENSG00000198464_ZNF480', 'ENSG00000198466_ZNF587', 'ENSG00000198467_TPM2', 'ENSG00000198468_FLVCR1-DT', 'ENSG00000198478_SH3BGRL2', 'ENSG00000198482_ZNF808', 'ENSG00000198483_ANKRD35', 'ENSG00000198492_YTHDF2', 'ENSG00000198496_NBR2', 'ENSG00000198498_TMA16', 'ENSG00000198502_HLA-DRB5', 'ENSG00000198513_ATL1', 'ENSG00000198515_CNGA1', 'ENSG00000198517_MAFK', 'ENSG00000198520_ARMH1', 'ENSG00000198521_ZNF43', 'ENSG00000198522_GPN1', 'ENSG00000198538_ZNF28', 'ENSG00000198546_ZNF511', 'ENSG00000198547_C20orf203', 'ENSG00000198551_ZNF627', 'ENSG00000198553_KCNRG', 'ENSG00000198554_WDHD1', 'ENSG00000198556_ZNF789', 'ENSG00000198561_CTNND1', 'ENSG00000198563_DDX39B', 'ENSG00000198576_ARC', 'ENSG00000198585_NUDT16', 'ENSG00000198586_TLK1', 'ENSG00000198589_LRBA', 'ENSG00000198590_C3orf35', 'ENSG00000198598_MMP17', 'ENSG00000198604_BAZ1A', 'ENSG00000198612_COPS8', 'ENSG00000198618_PPIAP22', 'ENSG00000198624_CCDC69', 'ENSG00000198625_MDM4', 'ENSG00000198633_ZNF534', 'ENSG00000198642_KLHL9', 'ENSG00000198646_NCOA6', 'ENSG00000198648_STK39', 'ENSG00000198663_C6orf89', 'ENSG00000198668_CALM1', 'ENSG00000198673_FAM19A2', 'ENSG00000198677_TTC37', 'ENSG00000198680_TUSC1', 'ENSG00000198682_PAPSS2', 'ENSG00000198685_LINC01565', 'ENSG00000198689_SLC9A6', 'ENSG00000198690_FAN1', 'ENSG00000198692_EIF1AY', 'ENSG00000198695_MT-ND6', 'ENSG00000198700_IPO9', 'ENSG00000198707_CEP290', 'ENSG00000198711_SSBP3-AS1', 'ENSG00000198712_MT-CO2', 'ENSG00000198715_GLMP', 'ENSG00000198718_TOGARAM1', 'ENSG00000198719_DLL1', 'ENSG00000198720_ANKRD13B', 'ENSG00000198721_ECI2', 'ENSG00000198722_UNC13B', 'ENSG00000198727_MT-CYB', 'ENSG00000198728_LDB1', 'ENSG00000198729_PPP1R14C', 'ENSG00000198730_CTR9', 'ENSG00000198734_F5', 'ENSG00000198736_MSRB1', 'ENSG00000198740_ZNF652', 'ENSG00000198742_SMURF1', 'ENSG00000198743_SLC5A3', 'ENSG00000198744_MTCO3P12', 'ENSG00000198746_GPATCH3', 'ENSG00000198752_CDC42BPB', 'ENSG00000198754_OXCT2', 'ENSG00000198755_RPL10A', 'ENSG00000198756_COLGALT2', 'ENSG00000198763_MT-ND2', 'ENSG00000198771_RCSD1', 'ENSG00000198774_RASSF9', 'ENSG00000198780_FAM169A', 'ENSG00000198783_ZNF830', 'ENSG00000198785_GRIN3A', 'ENSG00000198786_MT-ND5', 'ENSG00000198791_CNOT7', 'ENSG00000198792_TMEM184B', 'ENSG00000198793_MTOR', 'ENSG00000198794_SCAMP5', 'ENSG00000198795_ZNF521', 'ENSG00000198799_LRIG2', 'ENSG00000198804_MT-CO1', 'ENSG00000198805_PNP', 'ENSG00000198814_GK', 'ENSG00000198815_FOXJ3', 'ENSG00000198816_ZNF358', 'ENSG00000198818_SFT2D1', 'ENSG00000198821_CD247', 'ENSG00000198824_CHAMP1', 'ENSG00000198825_INPP5F', 'ENSG00000198826_ARHGAP11A', 'ENSG00000198829_SUCNR1', 'ENSG00000198830_HMGN2', 'ENSG00000198832_SELENOM', 'ENSG00000198833_UBE2J1', 'ENSG00000198835_GJC2', 'ENSG00000198836_OPA1', 'ENSG00000198837_DENND4B', 'ENSG00000198838_RYR3', 'ENSG00000198839_ZNF277', 'ENSG00000198840_MT-ND3', 'ENSG00000198841_KTI12', 'ENSG00000198842_DUSP27', 'ENSG00000198843_SELENOT', 'ENSG00000198846_TOX', 'ENSG00000198848_CES1', 'ENSG00000198851_CD3E', 'ENSG00000198853_RUSC2', 'ENSG00000198855_FICD', 'ENSG00000198856_OSTC', 'ENSG00000198858_R3HDM4', 'ENSG00000198860_TSEN15', 'ENSG00000198862_LTN1', 'ENSG00000198863_RUNDC1', 'ENSG00000198865_CCDC152', 'ENSG00000198870_STKLD1', 'ENSG00000198873_GRK5', 'ENSG00000198874_TYW1', 'ENSG00000198876_DCAF12', 'ENSG00000198879_SFMBT2', 'ENSG00000198881_ASB12', 'ENSG00000198885_ITPRIPL1', 'ENSG00000198886_MT-ND4', 'ENSG00000198887_SMC5', 'ENSG00000198888_MT-ND1', 'ENSG00000198890_PRMT6', 'ENSG00000198892_SHISA4', 'ENSG00000198894_CIPC', 'ENSG00000198898_CAPZA2', 'ENSG00000198899_MT-ATP6', 'ENSG00000198900_TOP1', 'ENSG00000198901_PRC1', 'ENSG00000198908_BHLHB9', 'ENSG00000198909_MAP3K3', 'ENSG00000198911_SREBF2', 'ENSG00000198912_C1orf174', 'ENSG00000198915_RASGEF1A', 'ENSG00000198917_SPOUT1', 'ENSG00000198918_RPL39', 'ENSG00000198919_DZIP3', 'ENSG00000198920_KIAA0753', 'ENSG00000198924_DCLRE1A', 'ENSG00000198925_ATG9A', 'ENSG00000198929_NOS1AP', 'ENSG00000198931_APRT', 'ENSG00000198932_GPRASP1', 'ENSG00000198933_TBKBP1', 'ENSG00000198934_MAGEE1', 'ENSG00000198937_CCDC167', 'ENSG00000198938_MT-CO3', 'ENSG00000198939_ZFP2', 'ENSG00000198945_L3MBTL3', 'ENSG00000198947_DMD', 'ENSG00000198948_MFAP3L', 'ENSG00000198951_NAGA', 'ENSG00000198952_SMG5', 'ENSG00000198954_KIF1BP', 'ENSG00000198959_TGM2', 'ENSG00000198960_ARMCX6', 'ENSG00000198961_PJA2', 'ENSG00000198964_SGMS1', 'ENSG00000198967_OR10Z1', 'ENSG00000199325_RNU4-39P', 'ENSG00000199377_RNU5F-1', 'ENSG00000199472_RF00019', 'ENSG00000199477_SNORA31', 'ENSG00000199568_RNU5A-1', 'ENSG00000199730_RN7SKP95', 'ENSG00000199804_RNA5SP383', 'ENSG00000199805_RNU1-134P', 'ENSG00000199874_RNA5SP452', 'ENSG00000199879_RNU1-120P', 'ENSG00000199883_RN7SKP90', 'ENSG00000199933_RNY1P16', 'ENSG00000199977_RF00045', 'ENSG00000199990_VTRNA1-1', 'ENSG00000200087_SNORA73B', 'ENSG00000200090_RF00019', 'ENSG00000200091_RN7SKP163', 'ENSG00000200156_RNU5B-1', 'ENSG00000200169_RNU5D-1', 'ENSG00000200183_RNU6-238P', 'ENSG00000200397_RF00019', 'ENSG00000200403_RNU6-1099P', 'ENSG00000200502_RF00019', 'ENSG00000200534_SNORA33', 'ENSG00000200731_RNU1-124P', 'ENSG00000200741_RNA5SP161', 'ENSG00000200795_RNU4-1', 'ENSG00000200834_RF00019', 'ENSG00000200840_RNU6-82P', 'ENSG00000200882_RNU6-681P', 'ENSG00000200972_RNU5A-8P', 'ENSG00000200997_RNU1-85P', 'ENSG00000201098_RNY1', 'ENSG00000201183_RNVU1-3', 'ENSG00000201221_RNU4-40P', 'ENSG00000201367_RNU6-522P', 'ENSG00000201388_SNORA68B', 'ENSG00000201542_RF00091', 'ENSG00000201558_RNVU1-6', 'ENSG00000201600_RN7SKP124', 'ENSG00000201620_RNA5SP51', 'ENSG00000201674_RF00012', 'ENSG00000201801_RNU5E-4P', 'ENSG00000201821_RNU4-9P', 'ENSG00000201863_RF00432', 'ENSG00000201944_RF00139', 'ENSG00000202058_RN7SKP80', 'ENSG00000202078_RF00019', 'ENSG00000202198_RF00100', 'ENSG00000202222_RF00019', 'ENSG00000202343_RF00410', 'ENSG00000202347_RNU1-16P', 'ENSG00000202374_RF00091', 'ENSG00000202392_RN7SKP292', 'ENSG00000202408_RNU1-122P', 'ENSG00000202515_VTRNA1-3', 'ENSG00000202538_RNU4-2', 'ENSG00000203279_AL590705.1', 'ENSG00000203286_RF00017', 'ENSG00000203288_TDRKH-AS1', 'ENSG00000203321_C9orf41-AS1', 'ENSG00000203325_AL445248.1', 'ENSG00000203326_ZNF525', 'ENSG00000203327_AC012358.1', 'ENSG00000203362_POLH-AS1', 'ENSG00000203414_BTBD7P1', 'ENSG00000203435_E2F3P2', 'ENSG00000203441_LINC00449', 'ENSG00000203469_AL354956.1', 'ENSG00000203485_INF2', 'ENSG00000203497_PDCD4-AS1', 'ENSG00000203499_IQANK1', 'ENSG00000203546_AL139353.1', 'ENSG00000203644_AC083799.1', 'ENSG00000203650_LINC01285', 'ENSG00000203663_OR2L2', 'ENSG00000203666_EFCAB2', 'ENSG00000203667_COX20', 'ENSG00000203668_CHML', 'ENSG00000203684_IBA57-DT', 'ENSG00000203685_STUM', 'ENSG00000203705_TATDN3', 'ENSG00000203709_MIR29B2CHG', 'ENSG00000203710_CR1', 'ENSG00000203711_C6orf99', 'ENSG00000203724_C1orf53', 'ENSG00000203734_ECT2L', 'ENSG00000203739_AL645568.1', 'ENSG00000203760_CENPW', 'ENSG00000203778_FAM229B', 'ENSG00000203780_FANK1', 'ENSG00000203791_EEF1AKMT2', 'ENSG00000203797_DDO', 'ENSG00000203799_CCDC162P', 'ENSG00000203804_ADAMTSL4-AS1', 'ENSG00000203805_PLPP4', 'ENSG00000203814_HIST2H2BF', 'ENSG00000203865_ATP1A1-AS1', 'ENSG00000203872_C6orf163', 'ENSG00000203875_SNHG5', 'ENSG00000203876_ADD3-AS1', 'ENSG00000203877_RIPPLY2', 'ENSG00000203879_GDI1', 'ENSG00000203880_PCMTD2', 'ENSG00000203883_SOX18', 'ENSG00000203930_LINC00632', 'ENSG00000203943_SAMD13', 'ENSG00000203950_RTL8A', 'ENSG00000203963_C1orf141', 'ENSG00000203965_EFCAB7', 'ENSG00000203993_ARRDC1-AS1', 'ENSG00000204001_LCN8', 'ENSG00000204020_LIPN', 'ENSG00000204025_TRPC5OS', 'ENSG00000204054_LINC00963', 'ENSG00000204060_FOXO6', 'ENSG00000204065_TCEAL5', 'ENSG00000204070_SYS1', 'ENSG00000204084_INPP5B', 'ENSG00000204086_RPA4', 'ENSG00000204103_MAFB', 'ENSG00000204104_TRAF3IP1', 'ENSG00000204116_CHIC1', 'ENSG00000204118_NAP1L6', 'ENSG00000204120_GIGYF2', 'ENSG00000204128_C2orf72', 'ENSG00000204130_RUFY2', 'ENSG00000204131_NHSL2', 'ENSG00000204136_GGTA1P', 'ENSG00000204138_PHACTR4', 'ENSG00000204147_ASAH2B', 'ENSG00000204149_AGAP6', 'ENSG00000204152_TIMM23B', 'ENSG00000204160_ZDHHC18', 'ENSG00000204161_TMEM273', 'ENSG00000204172_AGAP9', 'ENSG00000204176_SYT15', 'ENSG00000204177_BMS1P1', 'ENSG00000204178_MACO1', 'ENSG00000204179_PTPN20', 'ENSG00000204186_ZDBF2', 'ENSG00000204209_DAXX', 'ENSG00000204217_BMPR2', 'ENSG00000204219_TCEA3', 'ENSG00000204220_PFDN6', 'ENSG00000204227_RING1', 'ENSG00000204228_HSD17B8', 'ENSG00000204231_RXRB', 'ENSG00000204237_OXLD1', 'ENSG00000204241_AP000911.1', 'ENSG00000204252_HLA-DOA', 'ENSG00000204253_HNRNPCP2', 'ENSG00000204256_BRD2', 'ENSG00000204257_HLA-DMA', 'ENSG00000204261_PSMB8-AS1', 'ENSG00000204262_COL5A2', 'ENSG00000204264_PSMB8', 'ENSG00000204267_TAP2', 'ENSG00000204271_SPIN3', 'ENSG00000204272_NBDY', 'ENSG00000204277_LINC01993', 'ENSG00000204282_TNRC6C-AS1', 'ENSG00000204287_HLA-DRA', 'ENSG00000204291_COL15A1', 'ENSG00000204301_NOTCH4', 'ENSG00000204304_PBX2', 'ENSG00000204305_AGER', 'ENSG00000204308_RNF5', 'ENSG00000204310_AGPAT1', 'ENSG00000204311_PJVK', 'ENSG00000204314_PRRT1', 'ENSG00000204315_FKBPL', 'ENSG00000204316_MRPL38', 'ENSG00000204323_SMIM5', 'ENSG00000204344_STK19', 'ENSG00000204348_DXO', 'ENSG00000204351_SKIV2L', 'ENSG00000204356_NELFE', 'ENSG00000204366_ZBTB12', 'ENSG00000204370_SDHD', 'ENSG00000204371_EHMT2', 'ENSG00000204381_LAYN', 'ENSG00000204385_SLC44A4', 'ENSG00000204386_NEU1', 'ENSG00000204387_C6orf48', 'ENSG00000204388_HSPA1B', 'ENSG00000204389_HSPA1A', 'ENSG00000204390_HSPA1L', 'ENSG00000204392_LSM2', 'ENSG00000204394_VARS', 'ENSG00000204396_VWA7', 'ENSG00000204397_CARD16', 'ENSG00000204406_MBD5', 'ENSG00000204410_MSH5', 'ENSG00000204420_MPIG6B', 'ENSG00000204421_LY6G6C', 'ENSG00000204427_ABHD16A', 'ENSG00000204428_LY6G5C', 'ENSG00000204435_CSNK2B', 'ENSG00000204438_GPANK1', 'ENSG00000204439_C6orf47', 'ENSG00000204444_APOM', 'ENSG00000204463_BAG6', 'ENSG00000204469_PRRC2A', 'ENSG00000204472_AIF1', 'ENSG00000204475_NCR3', 'ENSG00000204482_LST1', 'ENSG00000204498_NFKBIL1', 'ENSG00000204514_ZNF814', 'ENSG00000204516_MICB', 'ENSG00000204519_ZNF551', 'ENSG00000204520_MICA', 'ENSG00000204524_ZNF805', 'ENSG00000204525_HLA-C', 'ENSG00000204531_POU5F1', 'ENSG00000204536_CCHCR1', 'ENSG00000204540_PSORS1C1', 'ENSG00000204556_AL450124.1', 'ENSG00000204560_DHX16', 'ENSG00000204564_C6orf136', 'ENSG00000204568_MRPS18B', 'ENSG00000204569_PPP1R10', 'ENSG00000204574_ABCF1', 'ENSG00000204576_PRR3', 'ENSG00000204577_LILRB3', 'ENSG00000204580_DDR1', 'ENSG00000204584_FLJ45513', 'ENSG00000204588_LINC01123', 'ENSG00000204590_GNL1', 'ENSG00000204592_HLA-E', 'ENSG00000204599_TRIM39', 'ENSG00000204603_LINC01257', 'ENSG00000204604_ZNF468', 'ENSG00000204610_TRIM15', 'ENSG00000204611_ZNF616', 'ENSG00000204613_TRIM10', 'ENSG00000204619_PPP1R11', 'ENSG00000204620_AC115618.1', 'ENSG00000204622_HLA-J', 'ENSG00000204623_ZNRD1ASP', 'ENSG00000204625_HCG9', 'ENSG00000204628_RACK1', 'ENSG00000204632_HLA-G', 'ENSG00000204634_TBC1D8', 'ENSG00000204642_HLA-F', 'ENSG00000204644_ZFP57', 'ENSG00000204650_LINC02210', 'ENSG00000204652_RPS26P8', 'ENSG00000204659_CBY3', 'ENSG00000204673_AKT1S1', 'ENSG00000204681_GABBR1', 'ENSG00000204682_CASC10', 'ENSG00000204685_STARD7-AS1', 'ENSG00000204706_MAMDC2-AS1', 'ENSG00000204710_SPDYC', 'ENSG00000204713_TRIM27', 'ENSG00000204740_MALRD1', 'ENSG00000204758_AC008429.1', 'ENSG00000204764_RANBP17', 'ENSG00000204778_CBWD4P', 'ENSG00000204789_ZNF204P', 'ENSG00000204802_AL590399.1', 'ENSG00000204815_TTC25', 'ENSG00000204822_MRPL53', 'ENSG00000204832_ST8SIA6-AS1', 'ENSG00000204837_FGF7P3', 'ENSG00000204839_MROH6', 'ENSG00000204842_ATXN2', 'ENSG00000204843_DCTN1', 'ENSG00000204852_TCTN1', 'ENSG00000204856_FAM216A', 'ENSG00000204859_ZBTB48', 'ENSG00000204860_FAM201A', 'ENSG00000204866_IGFL2', 'ENSG00000204872_NAT8B', 'ENSG00000204899_MZT1', 'ENSG00000204904_LINC01545', 'ENSG00000204920_ZNF155', 'ENSG00000204922_UQCC3', 'ENSG00000204923_FBXO48', 'ENSG00000204934_ATP6V0E2-AS1', 'ENSG00000204946_ZNF783', 'ENSG00000204947_ZNF425', 'ENSG00000204954_C12orf73', 'ENSG00000204959_ARHGEF34P', 'ENSG00000204977_TRIM13', 'ENSG00000204991_SPIRE2', 'ENSG00000205002_AARD', 'ENSG00000205018_AC092384.1', 'ENSG00000205022_PABPN1L', 'ENSG00000205041_AC118344.1', 'ENSG00000205045_SLFN12L', 'ENSG00000205060_SLC35B4', 'ENSG00000205078_SYCE1L', 'ENSG00000205084_TMEM231', 'ENSG00000205085_FAM71F2', 'ENSG00000205090_TMEM240', 'ENSG00000205111_CDKL4', 'ENSG00000205129_C4orf47', 'ENSG00000205133_TRIQK', 'ENSG00000205138_SDHAF1', 'ENSG00000205155_PSENEN', 'ENSG00000205181_LINC00654', 'ENSG00000205189_ZBTB10', 'ENSG00000205208_C4orf46', 'ENSG00000205212_CCDC144NL', 'ENSG00000205213_LGR4', 'ENSG00000205220_PSMB10', 'ENSG00000205250_E2F4', 'ENSG00000205268_PDE7A', 'ENSG00000205269_TMEM170B', 'ENSG00000205277_MUC12', 'ENSG00000205293_LINC01602', 'ENSG00000205300_AL356414.1', 'ENSG00000205302_SNX2', 'ENSG00000205307_SAP25', 'ENSG00000205309_NT5M', 'ENSG00000205323_SARNP', 'ENSG00000205336_ADGRG1', 'ENSG00000205339_IPO7', 'ENSG00000205352_PRR13', 'ENSG00000205356_TECPR1', 'ENSG00000205358_MT1H', 'ENSG00000205403_CFI', 'ENSG00000205413_SAMD9', 'ENSG00000205423_CNEP1R1', 'ENSG00000205426_KRT81', 'ENSG00000205436_EXOC3L4', 'ENSG00000205456_TP53TG3D', 'ENSG00000205464_ATP6AP1L', 'ENSG00000205476_CCDC85C', 'ENSG00000205485_AC004980.1', 'ENSG00000205531_NAP1L4', 'ENSG00000205534_SMG1P2', 'ENSG00000205542_TMSB4X', 'ENSG00000205544_TMEM256', 'ENSG00000205559_CHKB-DT', 'ENSG00000205560_CPT1B', 'ENSG00000205571_SMN2', 'ENSG00000205572_SERF1B', 'ENSG00000205581_HMGN1', 'ENSG00000205583_STAG3L1', 'ENSG00000205592_MUC19', 'ENSG00000205593_DENND6B', 'ENSG00000205609_EIF3CL', 'ENSG00000205611_LINC01597', 'ENSG00000205622_AP001043.1', 'ENSG00000205629_LCMT1', 'ENSG00000205639_MFSD2B', 'ENSG00000205643_CDPF1', 'ENSG00000205659_LIN52', 'ENSG00000205670_SMIM11A', 'ENSG00000205682_AC020741.1', 'ENSG00000205683_DPF3', 'ENSG00000205702_CYP2D7', 'ENSG00000205704_LINC00634', 'ENSG00000205707_ETFRF1', 'ENSG00000205710_C17orf107', 'ENSG00000205726_ITSN1', 'ENSG00000205730_ITPRIPL2', 'ENSG00000205740_AL359878.1', 'ENSG00000205744_DENND1C', 'ENSG00000205755_CRLF2', 'ENSG00000205758_CRYZL1', 'ENSG00000205763_RP9P', 'ENSG00000205765_C5orf51', 'ENSG00000205771_CATSPER2P1', 'ENSG00000205784_ARRDC5', 'ENSG00000205790_DPP9-AS1', 'ENSG00000205791_LOH12CR2', 'ENSG00000205794_AC098591.1', 'ENSG00000205808_PLPP6', 'ENSG00000205846_CLEC6A', 'ENSG00000205853_RFPL3S', 'ENSG00000205861_PCOTH', 'ENSG00000205871_RPS3AP47', 'ENSG00000205885_C1RL-AS1', 'ENSG00000205903_ZNF316', 'ENSG00000205913_SRRM2-AS1', 'ENSG00000205918_PDPK2P', 'ENSG00000205929_C21orf62', 'ENSG00000205930_C21orf62-AS1', 'ENSG00000205937_RNPS1', 'ENSG00000205940_HSP90AB2P', 'ENSG00000205959_AC105345.1', 'ENSG00000205978_NYNRIN', 'ENSG00000205981_DNAJC19', 'ENSG00000206028_Z99774.1', 'ENSG00000206053_JPT2', 'ENSG00000206120_EGFEM1P', 'ENSG00000206127_GOLGA8O', 'ENSG00000206140_TMEM191C', 'ENSG00000206145_P2RX6P', 'ENSG00000206149_HERC2P9', 'ENSG00000206159_GYG2P1', 'ENSG00000206172_HBA1', 'ENSG00000206177_HBM', 'ENSG00000206190_ATP10A', 'ENSG00000206195_DUXAP8', 'ENSG00000206337_HCP5', 'ENSG00000206341_HLA-H', 'ENSG00000206344_HCG27', 'ENSG00000206417_H1FX-AS1', 'ENSG00000206418_RAB12', 'ENSG00000206503_HLA-A', 'ENSG00000206527_HACD2', 'ENSG00000206530_CFAP44', 'ENSG00000206531_CD200R1L', 'ENSG00000206535_LNP1', 'ENSG00000206557_TRIM71', 'ENSG00000206559_ZCWPW2', 'ENSG00000206560_ANKRD28', 'ENSG00000206561_COLQ', 'ENSG00000206562_METTL6', 'ENSG00000206567_AC022007.1', 'ENSG00000206573_THUMPD3-AS1', 'ENSG00000206605_RNU6-946P', 'ENSG00000206633_SNORA80B', 'ENSG00000206687_RNU1-109P', 'ENSG00000206698_RNU1-73P', 'ENSG00000206763_RNU6-10P', 'ENSG00000206772_RNU6-44P', 'ENSG00000206828_RF00003', 'ENSG00000206897_SNORA9B', 'ENSG00000206913_RF00409', 'ENSG00000206941_SNORD15A', 'ENSG00000206976_RF00409', 'ENSG00000206989_SNORD63', 'ENSG00000207003_RNU6-611P', 'ENSG00000207067_SNORA72', 'ENSG00000207110_RNU1-106P', 'ENSG00000207175_RNU1-67P', 'ENSG00000207205_RNVU1-15', 'ENSG00000207233_SNORA37', 'ENSG00000207313_SNORA2B', 'ENSG00000207392_SNORA20', 'ENSG00000207751_AP000553.1', 'ENSG00000209082_MT-TL1', 'ENSG00000210049_MT-TF', 'ENSG00000210077_MT-TV', 'ENSG00000210082_MT-RNR2', 'ENSG00000210100_MT-TI', 'ENSG00000210107_MT-TQ', 'ENSG00000210112_MT-TM', 'ENSG00000210117_MT-TW', 'ENSG00000210127_MT-TA', 'ENSG00000210135_MT-TN', 'ENSG00000210140_MT-TC', 'ENSG00000210144_MT-TY', 'ENSG00000210151_MT-TS1', 'ENSG00000210154_MT-TD', 'ENSG00000210156_MT-TK', 'ENSG00000210164_MT-TG', 'ENSG00000210174_MT-TR', 'ENSG00000210176_MT-TH', 'ENSG00000210184_MT-TS2', 'ENSG00000210191_MT-TL2', 'ENSG00000210194_MT-TE', 'ENSG00000210195_MT-TT', 'ENSG00000210196_MT-TP', 'ENSG00000211445_GPX3', 'ENSG00000211450_SELENOH', 'ENSG00000211451_GNRHR2', 'ENSG00000211454_AKR7L', 'ENSG00000211455_STK38L', 'ENSG00000211456_SACM1L', 'ENSG00000211459_MT-RNR1', 'ENSG00000211460_TSN', 'ENSG00000211584_SLC48A1', 'ENSG00000211592_IGKC', 'ENSG00000211689_TRGC1', 'ENSG00000211695_TRGV9', 'ENSG00000211697_TRGV5', 'ENSG00000211714_TRBV7-3', 'ENSG00000211747_TRBV20-1', 'ENSG00000211751_TRBC1', 'ENSG00000211753_TRBV28', 'ENSG00000211772_TRBC2', 'ENSG00000211791_TRAV13-2', 'ENSG00000211802_TRAV22', 'ENSG00000211815_TRAV36DV7', 'ENSG00000211821_TRDV2', 'ENSG00000211829_TRDC', 'ENSG00000211898_IGHD', 'ENSG00000211899_IGHM', 'ENSG00000212123_PRR22', 'ENSG00000212124_TAS2R19', 'ENSG00000212125_TAS2R15P', 'ENSG00000212127_TAS2R14', 'ENSG00000212128_TAS2R13', 'ENSG00000212163_SNORD91A', 'ENSG00000212190_RNU6-298P', 'ENSG00000212195_RF00012', 'ENSG00000212402_SNORA74B', 'ENSG00000212539_RF00012', 'ENSG00000212567_RF00191', 'ENSG00000212664_AC064799.1', 'ENSG00000212694_LINC01089', 'ENSG00000212719_C17orf51', 'ENSG00000212743_AL137145.1', 'ENSG00000212747_RTL8B', 'ENSG00000212802_RPL15P3', 'ENSG00000212864_RNF208', 'ENSG00000212907_MT-ND4L', 'ENSG00000212916_MAP10', 'ENSG00000212939_Z97192.1', 'ENSG00000212961_HNRNPA1P40', 'ENSG00000212978_AC016747.1', 'ENSG00000213005_PTTG3P', 'ENSG00000213015_ZNF580', 'ENSG00000213018_AL590762.1', 'ENSG00000213020_ZNF611', 'ENSG00000213023_SYT3', 'ENSG00000213024_NUP62', 'ENSG00000213025_COX20P1', 'ENSG00000213033_AURKAP1', 'ENSG00000213047_DENND1B', 'ENSG00000213049_HNRNPA1P34', 'ENSG00000213050_TPM3P1', 'ENSG00000213051_RPL5P5', 'ENSG00000213055_EEF1B2P7', 'ENSG00000213057_C1orf220', 'ENSG00000213058_AL365357.1', 'ENSG00000213062_AL021068.1', 'ENSG00000213064_SFT2D2', 'ENSG00000213066_FGFR1OP', 'ENSG00000213071_LPAL2', 'ENSG00000213073_AL353625.1', 'ENSG00000213079_SCAF8', 'ENSG00000213080_AL354714.2', 'ENSG00000213085_CFAP45', 'ENSG00000213087_AL356535.1', 'ENSG00000213088_ACKR1', 'ENSG00000213090_AC007256.1', 'ENSG00000213096_ZNF254', 'ENSG00000213121_AL590867.1', 'ENSG00000213123_TCTEX1D2', 'ENSG00000213131_YWHAZP4', 'ENSG00000213139_CRYGS', 'ENSG00000213144_AC084880.1', 'ENSG00000213145_CRIP1', 'ENSG00000213152_RPL7AP60', 'ENSG00000213160_KLHL23', 'ENSG00000213169_AC069218.1', 'ENSG00000213178_RPL22P1', 'ENSG00000213185_FAM24B', 'ENSG00000213186_TRIM59', 'ENSG00000213189_BTF3L4P2', 'ENSG00000213190_MLLT11', 'ENSG00000213199_ASIC3', 'ENSG00000213203_GIMAP1', 'ENSG00000213204_AL049697.1', 'ENSG00000213212_NCLP1', 'ENSG00000213213_CCDC183', 'ENSG00000213214_ARHGEF35', 'ENSG00000213221_DNLZ', 'ENSG00000213225_NOC2LP1', 'ENSG00000213232_PPP1R2P10', 'ENSG00000213236_YWHAZP2', 'ENSG00000213246_SUPT4H1', 'ENSG00000213260_YWHAZP5', 'ENSG00000213261_EEF1B2P6', 'ENSG00000213265_TSGA13', 'ENSG00000213269_AC004386.1', 'ENSG00000213270_RPL6P25', 'ENSG00000213279_Z97192.2', 'ENSG00000213280_AC090114.1', 'ENSG00000213281_NRAS', 'ENSG00000213293_AC012618.1', 'ENSG00000213300_HNRNPA3P6', 'ENSG00000213304_AC008481.2', 'ENSG00000213305_HNRNPCP6', 'ENSG00000213309_RPL9P18', 'ENSG00000213315_AL122020.1', 'ENSG00000213316_LTC4S', 'ENSG00000213326_RPS7P11', 'ENSG00000213337_ANKRD39', 'ENSG00000213339_QTRT1', 'ENSG00000213341_CHUK', 'ENSG00000213347_MXD3', 'ENSG00000213358_AC092933.1', 'ENSG00000213363_RPS3P6', 'ENSG00000213366_GSTM2', 'ENSG00000213371_NAP1L1P3', 'ENSG00000213380_COG8', 'ENSG00000213385_AC105052.2', 'ENSG00000213390_ARHGAP19', 'ENSG00000213397_HAUS7', 'ENSG00000213398_LCAT', 'ENSG00000213399_AC022210.1', 'ENSG00000213411_RBM22P2', 'ENSG00000213420_GPC2', 'ENSG00000213430_HSPD1P1', 'ENSG00000213432_RPL17P34', 'ENSG00000213433_RPLP1P6', 'ENSG00000213440_H2AFZP1', 'ENSG00000213442_RPL18AP3', 'ENSG00000213445_SIPA1', 'ENSG00000213453_FTH1P3', 'ENSG00000213462_ERV3-1', 'ENSG00000213463_SYNJ2BP', 'ENSG00000213465_ARL2', 'ENSG00000213468_FIRRE', 'ENSG00000213484_EIF4A1P8', 'ENSG00000213492_NT5C3AP1', 'ENSG00000213509_PPIAP16', 'ENSG00000213512_GBP7', 'ENSG00000213516_RBMXL1', 'ENSG00000213523_SRA1', 'ENSG00000213533_STIMATE', 'ENSG00000213542_AC007000.1', 'ENSG00000213551_DNAJC9', 'ENSG00000213553_RPLP0P6', 'ENSG00000213563_C8orf82', 'ENSG00000213585_VDAC1', 'ENSG00000213590_AL807752.1', 'ENSG00000213593_TMX2', 'ENSG00000213598_AL049873.1', 'ENSG00000213600_U73169.1', 'ENSG00000213613_RPL11P3', 'ENSG00000213614_HEXA', 'ENSG00000213619_NDUFS3', 'ENSG00000213625_LEPROT', 'ENSG00000213626_LBH', 'ENSG00000213638_ADAT3', 'ENSG00000213639_PPP1CB', 'ENSG00000213654_GPSM3', 'ENSG00000213658_LAT', 'ENSG00000213669_AL137074.1', 'ENSG00000213672_NCKIPSD', 'ENSG00000213676_ATF6B', 'ENSG00000213683_AC002056.1', 'ENSG00000213684_LDHBP2', 'ENSG00000213693_SEC14L1P1', 'ENSG00000213694_S1PR3', 'ENSG00000213699_SLC35F6', 'ENSG00000213700_RPL17P50', 'ENSG00000213703_AL138847.1', 'ENSG00000213707_HMGB1P10', 'ENSG00000213713_PIGCP1', 'ENSG00000213714_FAM209B', 'ENSG00000213719_CLIC1', 'ENSG00000213722_DDAH2', 'ENSG00000213726_RPS2P52', 'ENSG00000213740_SERBP1P1', 'ENSG00000213741_RPS29', 'ENSG00000213742_ZNF337-AS1', 'ENSG00000213753_CENPBD1P1', 'ENSG00000213754_AL356317.1', 'ENSG00000213759_UGT2B11', 'ENSG00000213760_ATP6V1G2', 'ENSG00000213762_ZNF134', 'ENSG00000213780_GTF2H4', 'ENSG00000213782_DDX47', 'ENSG00000213790_OLA1P1', 'ENSG00000213793_ZNF888', 'ENSG00000213799_ZNF845', 'ENSG00000213801_ZNF321P', 'ENSG00000213830_CFL1P5', 'ENSG00000213842_SUGT1P2', 'ENSG00000213846_AC098614.1', 'ENSG00000213853_EMP2', 'ENSG00000213856_VDAC1P2', 'ENSG00000213859_KCTD11', 'ENSG00000213860_RPL21P75', 'ENSG00000213862_AC044787.1', 'ENSG00000213863_AL731661.1', 'ENSG00000213864_EEF1B2P2', 'ENSG00000213865_C8orf44', 'ENSG00000213866_YBX1P10', 'ENSG00000213872_AC092798.1', 'ENSG00000213885_RPL13AP7', 'ENSG00000213888_LINC01521', 'ENSG00000213889_PPM1N', 'ENSG00000213901_SLC23A3', 'ENSG00000213903_LTB4R', 'ENSG00000213904_LIPE-AS1', 'ENSG00000213906_LTB4R2', 'ENSG00000213917_RPL5P8', 'ENSG00000213918_DNASE1', 'ENSG00000213920_MDP1', 'ENSG00000213923_CSNK1E', 'ENSG00000213928_IRF9', 'ENSG00000213930_GALT', 'ENSG00000213934_HBG1', 'ENSG00000213935_RPL22P16', 'ENSG00000213937_CLDN9', 'ENSG00000213939_AC091153.1', 'ENSG00000213949_ITGA1', 'ENSG00000213963_AC019080.1', 'ENSG00000213965_NUDT19', 'ENSG00000213967_ZNF726', 'ENSG00000213973_ZNF99', 'ENSG00000213976_AC010615.1', 'ENSG00000213977_TAX1BP3', 'ENSG00000213983_AP1G2', 'ENSG00000213985_AC078899.1', 'ENSG00000213988_ZNF90', 'ENSG00000213995_NAXD', 'ENSG00000214013_GANC', 'ENSG00000214018_RRM2P3', 'ENSG00000214019_AL034370.1', 'ENSG00000214021_TTLL3', 'ENSG00000214022_REPIN1', 'ENSG00000214026_MRPL23', 'ENSG00000214029_ZNF891', 'ENSG00000214046_SMIM7', 'ENSG00000214049_UCA1', 'ENSG00000214050_FBXO16', 'ENSG00000214063_TSPAN4', 'ENSG00000214067_AL360182.1', 'ENSG00000214078_CPNE1', 'ENSG00000214087_ARL16', 'ENSG00000214093_AL096701.1', 'ENSG00000214106_PAXIP1-AS2', 'ENSG00000214110_LDHAP4', 'ENSG00000214113_LYRM4', 'ENSG00000214114_MYCBP', 'ENSG00000214121_PRDX1P1', 'ENSG00000214135_AC132008.2', 'ENSG00000214140_PRCD', 'ENSG00000214142_RPL7P60', 'ENSG00000214145_LINC00887', 'ENSG00000214160_ALG3', 'ENSG00000214174_AMZ2P1', 'ENSG00000214176_PLEKHM1P1', 'ENSG00000214182_PTMAP5', 'ENSG00000214184_GCC2-AS1', 'ENSG00000214188_ST7-OT4', 'ENSG00000214189_ZNF788P', 'ENSG00000214192_UBE2V1P2', 'ENSG00000214193_SH3D21', 'ENSG00000214194_SMIM30', 'ENSG00000214198_TTC41P', 'ENSG00000214203_RPS4XP1', 'ENSG00000214212_C19orf38', 'ENSG00000214223_HNRNPA1P10', 'ENSG00000214226_C17orf67', 'ENSG00000214248_AC010336.1', 'ENSG00000214253_FIS1', 'ENSG00000214262_ANKRD36BP1', 'ENSG00000214273_AGGF1P1', 'ENSG00000214274_ANG', 'ENSG00000214278_AC010442.1', 'ENSG00000214279_SCART1', 'ENSG00000214280_AC046134.1', 'ENSG00000214283_RAD51AP1P1', 'ENSG00000214293_APTR', 'ENSG00000214300_SPDYE3', 'ENSG00000214309_MBLAC1', 'ENSG00000214331_AC009053.1', 'ENSG00000214353_VAC14-AS1', 'ENSG00000214357_NEURL1B', 'ENSG00000214359_RPL18P10', 'ENSG00000214360_EFCAB9', 'ENSG00000214367_HAUS3', 'ENSG00000214389_RPS3AP26', 'ENSG00000214391_TUBAP2', 'ENSG00000214401_KANSL1-AS1', 'ENSG00000214413_BBIP1', 'ENSG00000214425_LRRC37A4P', 'ENSG00000214432_VPS33B-DT', 'ENSG00000214433_GOLGA2P8', 'ENSG00000214439_FAM185BP', 'ENSG00000214455_RCN1P2', 'ENSG00000214456_PLIN5', 'ENSG00000214465_SMARCE1P6', 'ENSG00000214485_RPL7P1', 'ENSG00000214517_PPME1', 'ENSG00000214530_STARD10', 'ENSG00000214534_ZNF705E', 'ENSG00000214535_RPS15AP1', 'ENSG00000214544_GTF2IRD2P1', 'ENSG00000214548_MEG3', 'ENSG00000214559_AC019077.1', 'ENSG00000214562_NUTM2D', 'ENSG00000214575_CPEB1', 'ENSG00000214595_EML6', 'ENSG00000214612_RPS19P1', 'ENSG00000214626_POLR3DP1', 'ENSG00000214629_RPSAP6', 'ENSG00000214650_AC073592.1', 'ENSG00000214652_ZNF727', 'ENSG00000214654_B3GNT10', 'ENSG00000214655_ZSWIM8', 'ENSG00000214688_C10orf105', 'ENSG00000214694_ARHGEF33', 'ENSG00000214706_IFRD2', 'ENSG00000214708_AC116407.1', 'ENSG00000214711_CAPN14', 'ENSG00000214717_ZBED1', 'ENSG00000214719_SMURF2P1-LRRC37BP1', 'ENSG00000214725_CDIPTOSP', 'ENSG00000214753_HNRNPUL2', 'ENSG00000214756_CSKMT', 'ENSG00000214760_RPL21P1', 'ENSG00000214765_SEPT7P2', 'ENSG00000214770_AL161756.1', 'ENSG00000214773_AC112512.1', 'ENSG00000214776_AC092821.1', 'ENSG00000214783_POLR2J4', 'ENSG00000214784_AC010468.1', 'ENSG00000214796_AC098934.1', 'ENSG00000214803_AC090921.1', 'ENSG00000214826_DDX12P', 'ENSG00000214827_MTCP1', 'ENSG00000214832_UPF3AP2', 'ENSG00000214837_LINC01347', 'ENSG00000214855_APOC1P1', 'ENSG00000214870_AC004540.1', 'ENSG00000214881_TMEM14DP', 'ENSG00000214894_LINC00243', 'ENSG00000214900_LINC01588', 'ENSG00000214917_AC011825.1', 'ENSG00000214922_HLA-F-AS1', 'ENSG00000214941_ZSWIM7', 'ENSG00000214944_ARHGEF28', 'ENSG00000214954_LRRC69', 'ENSG00000214960_ISPD', 'ENSG00000214999_AC129492.1', 'ENSG00000215007_DNAJA1P3', 'ENSG00000215012_RTL10', 'ENSG00000215014_AL645728.1', 'ENSG00000215016_RPL24P7', 'ENSG00000215021_PHB2', 'ENSG00000215022_AL008729.1', 'ENSG00000215030_RPL13P12', 'ENSG00000215032_GNL3LP1', 'ENSG00000215039_CD27-AS1', 'ENSG00000215041_NEURL4', 'ENSG00000215067_ALOX12-AS1', 'ENSG00000215068_AC025171.2', 'ENSG00000215105_TTC3P1', 'ENSG00000215114_UBXN2B', 'ENSG00000215126_CBWD6', 'ENSG00000215146_BX322639.1', 'ENSG00000215154_AC141586.1', 'ENSG00000215158_AC138409.2', 'ENSG00000215183_MSMP', 'ENSG00000215184_RPS12P16', 'ENSG00000215190_LINC00680', 'ENSG00000215193_PEX26', 'ENSG00000215221_UBA52P6', 'ENSG00000215237_AL592293.1', 'ENSG00000215241_LINC02449', 'ENSG00000215244_AL137145.2', 'ENSG00000215246_AC116351.1', 'ENSG00000215251_FASTKD5', 'ENSG00000215252_GOLGA8B', 'ENSG00000215256_DHRS4-AS1', 'ENSG00000215271_HOMEZ', 'ENSG00000215284_AL512633.1', 'ENSG00000215301_DDX3X', 'ENSG00000215302_AC127502.1', 'ENSG00000215304_AC135983.1', 'ENSG00000215305_VPS16', 'ENSG00000215374_FAM66B', 'ENSG00000215375_MYL5', 'ENSG00000215386_MIR99AHG', 'ENSG00000215414_PSMA6P1', 'ENSG00000215417_MIR17HG', 'ENSG00000215421_ZNF407', 'ENSG00000215424_MCM3AP-AS1', 'ENSG00000215440_NPEPL1', 'ENSG00000215458_AATBC', 'ENSG00000215467_RPL27AP', 'ENSG00000215472_RPL17-C18orf32', 'ENSG00000215475_SIAH3', 'ENSG00000215481_BCRP3', 'ENSG00000215515_IFIT1P1', 'ENSG00000215548_FRG1JP', 'ENSG00000215559_ANKRD20A11P', 'ENSG00000215580_BCORP1', 'ENSG00000215630_GUSBP9', 'ENSG00000215712_TMEM242', 'ENSG00000215717_TMEM167B', 'ENSG00000215769_ARHGAP27P1-BPTFP1-KPNA2P3', 'ENSG00000215784_FAM72D', 'ENSG00000215788_TNFRSF25', 'ENSG00000215790_SLC35E2A', 'ENSG00000215796_AL512637.1', 'ENSG00000215838_AL451074.1', 'ENSG00000215840_AL592295.1', 'ENSG00000215841_AP002761.1', 'ENSG00000215845_TSTD1', 'ENSG00000215867_KRT18P57', 'ENSG00000215869_AC092506.1', 'ENSG00000215878_MARCKSL1P2', 'ENSG00000215883_CYB5RL', 'ENSG00000215908_CROCCP2', 'ENSG00000215912_TTC34', 'ENSG00000215915_ATAD3C', 'ENSG00000216316_AL022722.1', 'ENSG00000216331_HIST1H1PS1', 'ENSG00000216490_IFI30', 'ENSG00000216613_AL023284.2', 'ENSG00000216642_AL136116.1', 'ENSG00000216657_GLRX3P2', 'ENSG00000216775_AL109918.1', 'ENSG00000216809_AL589993.1', 'ENSG00000216866_RPS2P55', 'ENSG00000216895_AC009403.1', 'ENSG00000216937_CCDC7', 'ENSG00000216977_RPL21P65', 'ENSG00000217027_TPT1P4', 'ENSG00000217128_FNIP1', 'ENSG00000217130_AL139100.1', 'ENSG00000217165_ANKRD18EP', 'ENSG00000217181_AL139039.2', 'ENSG00000217231_AL109755.1', 'ENSG00000217239_AL136968.2', 'ENSG00000217241_CBX3P9', 'ENSG00000217275_AL031777.1', 'ENSG00000217325_PRELID1P1', 'ENSG00000217334_AL355615.1', 'ENSG00000217416_ISCA1P1', 'ENSG00000217442_SYCE3', 'ENSG00000217527_RPS16P5', 'ENSG00000217555_CKLF', 'ENSG00000217643_PTGES3P2', 'ENSG00000217646_HIST1H2BPS2', 'ENSG00000217648_AL136116.3', 'ENSG00000217702_AC073263.1', 'ENSG00000217767_NDUFAB1P1', 'ENSG00000217801_AL390719.1', 'ENSG00000217896_ZNF839P1', 'ENSG00000217930_PAM16', 'ENSG00000218018_AL109955.1', 'ENSG00000218052_ADAMTS7P4', 'ENSG00000218175_AC016739.1', 'ENSG00000218208_RPS27AP11', 'ENSG00000218227_AC136632.1', 'ENSG00000218265_RPS4XP7', 'ENSG00000218283_MORF4L1P1', 'ENSG00000218313_AL139274.1', 'ENSG00000218358_RAET1K', 'ENSG00000218424_NDUFS5P1', 'ENSG00000218426_AL590867.2', 'ENSG00000218510_LINC00339', 'ENSG00000218537_MIF-AS1', 'ENSG00000218565_AL592429.1', 'ENSG00000218596_AL162578.1', 'ENSG00000218631_AL117344.1', 'ENSG00000218676_BRD7P4', 'ENSG00000218682_AC064847.1', 'ENSG00000218690_HIST1H2APS4', 'ENSG00000218739_CEBPZOS', 'ENSG00000218757_AL121952.1', 'ENSG00000218819_TDRD15', 'ENSG00000218823_PAPOLB', 'ENSG00000218890_NUFIP1P', 'ENSG00000218891_ZNF579', 'ENSG00000218896_TUBB8P2', 'ENSG00000218996_ARL4AP5', 'ENSG00000219023_AL033519.2', 'ENSG00000219085_NPM1P37', 'ENSG00000219133_AL592114.1', 'ENSG00000219163_HMGB1P20', 'ENSG00000219200_RNASEK', 'ENSG00000219201_AC138392.1', 'ENSG00000219222_RPL12P47', 'ENSG00000219355_RPL31P52', 'ENSG00000219409_AL590704.1', 'ENSG00000219433_BTBD10P2', 'ENSG00000219438_FAM19A5', 'ENSG00000219451_RPL23P8', 'ENSG00000219470_AL355802.1', 'ENSG00000219481_NBPF1', 'ENSG00000219487_AL603766.1', 'ENSG00000219507_FTH1P8', 'ENSG00000219529_AP000580.1', 'ENSG00000219545_UMAD1', 'ENSG00000219553_AL031133.1', 'ENSG00000219607_PPP1R3G', 'ENSG00000219626_FAM228B', 'ENSG00000219665_ZNF433-AS1', 'ENSG00000219703_RAP1BP3', 'ENSG00000219712_AL357054.1', 'ENSG00000219747_AL133260.1', 'ENSG00000219755_AL137784.1', 'ENSG00000219891_ZSCAN12P1', 'ENSG00000219902_RPL35P3', 'ENSG00000219928_AL161787.1', 'ENSG00000220008_LINGO3', 'ENSG00000220161_LINC02076', 'ENSG00000220201_ZGLP1', 'ENSG00000220205_VAMP2', 'ENSG00000220305_HNRNPH1P1', 'ENSG00000220323_HIST2H2BD', 'ENSG00000220370_AL078595.1', 'ENSG00000220472_AL139095.2', 'ENSG00000220506_AL136310.1', 'ENSG00000220517_ASS1P1', 'ENSG00000220583_RPL35P2', 'ENSG00000220660_AL023284.3', 'ENSG00000220685_AL139094.1', 'ENSG00000220744_RPL5P18', 'ENSG00000220749_RPL21P28', 'ENSG00000220771_BOLA2P3', 'ENSG00000220785_MTMR9LP', 'ENSG00000220793_RPL21P119', 'ENSG00000220804_LINC01881', 'ENSG00000220848_RPS18P9', 'ENSG00000220875_HIST1H3PS1', 'ENSG00000220920_AL023807.1', 'ENSG00000221059_RNU6ATAC6P', 'ENSG00000221164_SNORA11F', 'ENSG00000221184_MIR1254-1', 'ENSG00000221216_RNU6ATAC27P', 'ENSG00000221676_RNU6ATAC', 'ENSG00000221817_PPP3CB-AS1', 'ENSG00000221821_C6orf226', 'ENSG00000221823_PPP3R1', 'ENSG00000221829_FANCG', 'ENSG00000221838_AP4M1', 'ENSG00000221866_PLXNA4', 'ENSG00000221869_CEBPD', 'ENSG00000221882_OR3A2', 'ENSG00000221883_ARIH2OS', 'ENSG00000221886_ZBED8', 'ENSG00000221890_NPTXR', 'ENSG00000221909_FAM200A', 'ENSG00000221914_PPP2R2A', 'ENSG00000221916_C19orf73', 'ENSG00000221923_ZNF880', 'ENSG00000221926_TRIM16', 'ENSG00000221930_FAM45BP', 'ENSG00000221944_TIGD1', 'ENSG00000221946_FXYD7', 'ENSG00000221947_XKR9', 'ENSG00000221949_LINC01465', 'ENSG00000221955_SLC12A8', 'ENSG00000221962_TMEM14EP', 'ENSG00000221963_APOL6', 'ENSG00000221968_FADS3', 'ENSG00000221978_CCNL2', 'ENSG00000221983_UBA52', 'ENSG00000221988_PPT2', 'ENSG00000221990_EXOC3-AS1', 'ENSG00000221994_ZNF630', 'ENSG00000221995_TIAF1', 'ENSG00000222005_LINC01118', 'ENSG00000222009_BTBD19', 'ENSG00000222011_FAM185A', 'ENSG00000222019_URAHP', 'ENSG00000222020_AC062017.1', 'ENSG00000222041_CYTOR', 'ENSG00000222043_AC079305.1', 'ENSG00000222046_DCDC2B', 'ENSG00000222047_C10orf55', 'ENSG00000222057_RNU4-62P', 'ENSG00000222112_RN7SKP16', 'ENSG00000222222_RNU2-17P', 'ENSG00000222588_RF00413', 'ENSG00000222614_RF00019', 'ENSG00000222714_RN7SKP38', 'ENSG00000222724_RNU2-63P', 'ENSG00000222726_RNU2-7P', 'ENSG00000222743_RNU6-1190P', 'ENSG00000222810_RNU2-68P', 'ENSG00000222872_RNU4-78P', 'ENSG00000223138_RNA5SP450', 'ENSG00000223305_RN7SKP30', 'ENSG00000223343_AC137630.1', 'ENSG00000223361_FTH1P10', 'ENSG00000223392_CLDN10-AS1', 'ENSG00000223396_RPS10P7', 'ENSG00000223416_RPS26P15', 'ENSG00000223450_AL590632.1', 'ENSG00000223460_GAPDHP69', 'ENSG00000223466_LINC01825', 'ENSG00000223478_AL441992.1', 'ENSG00000223482_NUTM2A-AS1', 'ENSG00000223486_AC092198.1', 'ENSG00000223496_EXOSC6', 'ENSG00000223501_VPS52', 'ENSG00000223502_AL731537.1', 'ENSG00000223508_RPL23AP53', 'ENSG00000223509_AC135983.2', 'ENSG00000223516_AFF2-IT1', 'ENSG00000223525_RABGAP1L-IT1', 'ENSG00000223528_AL359094.1', 'ENSG00000223546_LINC00630', 'ENSG00000223547_ZNF844', 'ENSG00000223551_TMSB4XP4', 'ENSG00000223571_DHRSX-IT1', 'ENSG00000223573_TINCR', 'ENSG00000223575_RBMX2P3', 'ENSG00000223583_AL513365.1', 'ENSG00000223584_TVP23CP1', 'ENSG00000223599_AL513523.1', 'ENSG00000223609_HBD', 'ENSG00000223635_AL121990.1', 'ENSG00000223653_AL078459.1', 'ENSG00000223658_C1GALT1C1L', 'ENSG00000223685_LINC00571', 'ENSG00000223692_DIP2A-IT1', 'ENSG00000223695_FO393418.1', 'ENSG00000223697_AF230666.1', 'ENSG00000223705_NSUN5P1', 'ENSG00000223714_LINC02601', 'ENSG00000223724_RAD17P2', 'ENSG00000223727_AC034195.1', 'ENSG00000223745_CCDC18-AS1', 'ENSG00000223749_MIR503HG', 'ENSG00000223756_TSSC2', 'ENSG00000223764_LINC02593', 'ENSG00000223768_LINC00205', 'ENSG00000223773_CD99P1', 'ENSG00000223776_LGALS8-AS1', 'ENSG00000223791_UBE2E1-AS1', 'ENSG00000223797_ENTPD3-AS1', 'ENSG00000223799_IL10RB-DT', 'ENSG00000223802_CERS1', 'ENSG00000223803_RPS20P14', 'ENSG00000223804_AC244669.1', 'ENSG00000223813_AC007255.1', 'ENSG00000223820_CFL1P1', 'ENSG00000223849_AL354893.1', 'ENSG00000223861_AL365436.1', 'ENSG00000223865_HLA-DPB1', 'ENSG00000223886_AC073073.1', 'ENSG00000223891_OSER1-DT', 'ENSG00000223901_AP001469.1', 'ENSG00000223916_AC097638.1', 'ENSG00000223922_ASS1P2', 'ENSG00000223959_AFG3L1P', 'ENSG00000223960_AC009948.1', 'ENSG00000223969_AC002456.1', 'ENSG00000223974_BNIP3P42', 'ENSG00000223979_SMCR2', 'ENSG00000223984_HNRNPRP1', 'ENSG00000224003_YES1P1', 'ENSG00000224019_RPL21P32', 'ENSG00000224020_MIR181A2HG', 'ENSG00000224023_FLJ37035', 'ENSG00000224032_EPB41L4A-AS1', 'ENSG00000224043_CCNT2-AS1', 'ENSG00000224046_AC005076.1', 'ENSG00000224051_CPTP', 'ENSG00000224063_AC007319.1', 'ENSG00000224066_AL049795.1', 'ENSG00000224078_SNHG14', 'ENSG00000224083_MTCO1P11', 'ENSG00000224086_AC245452.1', 'ENSG00000224094_RPS24P8', 'ENSG00000224101_ELMO1-AS1', 'ENSG00000224109_CENPVL3', 'ENSG00000224114_AL591846.1', 'ENSG00000224126_UBE2SP2', 'ENSG00000224152_AC009506.1', 'ENSG00000224157_HCG14', 'ENSG00000224165_DNAJC27-AS1', 'ENSG00000224184_MIR3681HG', 'ENSG00000224186_C5orf66', 'ENSG00000224208_AL590762.3', 'ENSG00000224255_PDCL3P6', 'ENSG00000224259_LINC01133', 'ENSG00000224261_RPSAP18', 'ENSG00000224281_SLC25A5-AS1', 'ENSG00000224287_MSL3P1', 'ENSG00000224292_AF196972.1', 'ENSG00000224307_AL161785.1', 'ENSG00000224315_RPL7P7', 'ENSG00000224324_THAP5P1', 'ENSG00000224331_AC019181.1', 'ENSG00000224349_AL365226.1', 'ENSG00000224356_AL356966.1', 'ENSG00000224358_AL451074.2', 'ENSG00000224376_AC017104.1', 'ENSG00000224383_PRR29', 'ENSG00000224397_SMIM25', 'ENSG00000224400_AC010880.1', 'ENSG00000224407_AL136988.1', 'ENSG00000224411_HSP90AA2P', 'ENSG00000224415_AC007683.1', 'ENSG00000224418_STK24-AS1', 'ENSG00000224420_ADM5', 'ENSG00000224424_PRKAR2A-AS1', 'ENSG00000224429_LINC00539', 'ENSG00000224430_MKRN5P', 'ENSG00000224442_AC017035.1', 'ENSG00000224451_ATP5PBP1', 'ENSG00000224464_PGAM1P6', 'ENSG00000224470_ATXN1L', 'ENSG00000224478_AL356417.1', 'ENSG00000224490_TTC21B-AS1', 'ENSG00000224505_AC138150.1', 'ENSG00000224531_SMIM13', 'ENSG00000224533_TMLHE-AS1', 'ENSG00000224536_AC096677.1', 'ENSG00000224543_SNRPGP15', 'ENSG00000224546_EIF4BP3', 'ENSG00000224550_AC114491.1', 'ENSG00000224551_HMGB3P21', 'ENSG00000224553_AC008065.1', 'ENSG00000224557_HLA-DPB2', 'ENSG00000224578_HNRNPA1P48', 'ENSG00000224593_AC092427.1', 'ENSG00000224596_ZMIZ1-AS1', 'ENSG00000224597_SVIL-AS1', 'ENSG00000224598_RPS5P2', 'ENSG00000224609_HSD52', 'ENSG00000224614_TNK2-AS1', 'ENSG00000224616_RTCA-AS1', 'ENSG00000224629_AC004975.1', 'ENSG00000224631_RPS27AP16', 'ENSG00000224632_Z73361.1', 'ENSG00000224635_AL391095.1', 'ENSG00000224660_SH3BP5-AS1', 'ENSG00000224672_RPL17P10', 'ENSG00000224687_RASAL2-AS1', 'ENSG00000224699_LAMTOR5-AS1', 'ENSG00000224707_E2F3-IT1', 'ENSG00000224712_NPIPA3', 'ENSG00000224722_AC020688.1', 'ENSG00000224727_FCF1P7', 'ENSG00000224738_AC099850.1', 'ENSG00000224739_AC016735.1', 'ENSG00000224745_AC063965.1', 'ENSG00000224746_AC015987.1', 'ENSG00000224775_BRAFP1', 'ENSG00000224786_CETN4P', 'ENSG00000224790_AP000704.1', 'ENSG00000224794_AL022326.1', 'ENSG00000224796_RPL32P1', 'ENSG00000224802_TUBB4BP2', 'ENSG00000224805_LINC00853', 'ENSG00000224822_THRB-IT1', 'ENSG00000224830_OR2X1P', 'ENSG00000224837_GCSHP5', 'ENSG00000224843_LINC00240', 'ENSG00000224846_AL133351.1', 'ENSG00000224856_AL035398.1', 'ENSG00000224858_RPL29P11', 'ENSG00000224861_YBX1P1', 'ENSG00000224870_AL391244.1', 'ENSG00000224877_NDUFAF8', 'ENSG00000224885_EIPR1-IT1', 'ENSG00000224888_AC138028.2', 'ENSG00000224892_RPS4XP16', 'ENSG00000224895_VPS26BP1', 'ENSG00000224897_POT1-AS1', 'ENSG00000224903_AC005534.1', 'ENSG00000224905_AP001347.1', 'ENSG00000224914_LINC00863', 'ENSG00000224934_AL391684.1', 'ENSG00000224940_PRRT4', 'ENSG00000224950_AL390066.1', 'ENSG00000224958_PGM5-AS1', 'ENSG00000224975_INE1', 'ENSG00000224977_AL121983.1', 'ENSG00000224985_AL590714.1', 'ENSG00000224992_AL445645.1', 'ENSG00000225008_AC234781.2', 'ENSG00000225014_KCTD9P1', 'ENSG00000225022_UBE2D3P1', 'ENSG00000225026_AC091492.1', 'ENSG00000225032_AL162586.1', 'ENSG00000225051_HMGB3P22', 'ENSG00000225057_AC012485.1', 'ENSG00000225062_CATIP-AS1', 'ENSG00000225067_RPL23AP2', 'ENSG00000225071_AC004552.1', 'ENSG00000225075_AL603832.1', 'ENSG00000225077_LINC00337', 'ENSG00000225078_AL365338.1', 'ENSG00000225092_AL355336.1', 'ENSG00000225093_RPL3P7', 'ENSG00000225131_PSME2P2', 'ENSG00000225137_DYNC1I2P1', 'ENSG00000225138_SLC9A3-AS1', 'ENSG00000225151_GOLGA2P7', 'ENSG00000225159_NPM1P39', 'ENSG00000225163_LINC00618', 'ENSG00000225170_AL049737.1', 'ENSG00000225171_DUTP6', 'ENSG00000225173_AL662890.1', 'ENSG00000225177_FLJ46906', 'ENSG00000225178_RPSAP58', 'ENSG00000225190_PLEKHM1', 'ENSG00000225192_ZNF33BP1', 'ENSG00000225193_RPS12P26', 'ENSG00000225194_LINC00092', 'ENSG00000225200_AC246787.1', 'ENSG00000225205_AC078883.1', 'ENSG00000225210_DUXAP9', 'ENSG00000225213_AC073367.1', 'ENSG00000225215_SMARCE1P1', 'ENSG00000225224_RPS27AP12', 'ENSG00000225231_LINC02470', 'ENSG00000225234_TRAPPC12-AS1', 'ENSG00000225235_INTS6L-AS1', 'ENSG00000225259_ST13P6', 'ENSG00000225264_ZNRF2P2', 'ENSG00000225265_TAF1A-AS1', 'ENSG00000225282_AP000350.2', 'ENSG00000225300_AL591623.1', 'ENSG00000225331_LINC01678', 'ENSG00000225335_AC016027.1', 'ENSG00000225338_RPL23AP18', 'ENSG00000225339_AL354740.1', 'ENSG00000225342_AC079630.1', 'ENSG00000225361_PPP1R26-AS1', 'ENSG00000225377_NRSN2-AS1', 'ENSG00000225411_CR786580.1', 'ENSG00000225416_AC104843.1', 'ENSG00000225419_RPL21P3', 'ENSG00000225434_LINC01504', 'ENSG00000225439_BOLA3-AS1', 'ENSG00000225442_MPRIP-AS1', 'ENSG00000225447_RPS15AP10', 'ENSG00000225450_AL021707.1', 'ENSG00000225465_RFPL1S', 'ENSG00000225470_JPX', 'ENSG00000225484_NUTM2B-AS1', 'ENSG00000225486_AL663058.1', 'ENSG00000225489_AL354707.1', 'ENSG00000225492_GBP1P1', 'ENSG00000225505_AC104332.1', 'ENSG00000225506_CYP4A22-AS1', 'ENSG00000225511_LINC00475', 'ENSG00000225513_AL158824.1', 'ENSG00000225518_LINC01703', 'ENSG00000225526_MKRN2OS', 'ENSG00000225528_Z82206.1', 'ENSG00000225558_UBE2D3P2', 'ENSG00000225568_AC093155.1', 'ENSG00000225569_CCT4P2', 'ENSG00000225573_RPL35P5', 'ENSG00000225578_NCBP2-AS1', 'ENSG00000225580_AL358942.1', 'ENSG00000225591_BX248409.1', 'ENSG00000225611_LINC02158', 'ENSG00000225614_ZNF469', 'ENSG00000225630_MTND2P28', 'ENSG00000225643_AL606491.1', 'ENSG00000225648_SBDSP1', 'ENSG00000225663_MCRIP1', 'ENSG00000225670_CADM3-AS1', 'ENSG00000225673_AC104164.1', 'ENSG00000225675_LINC01771', 'ENSG00000225693_LAGE3P1', 'ENSG00000225697_SLC26A6', 'ENSG00000225706_PTPRD-AS1', 'ENSG00000225712_ATP5MC2P1', 'ENSG00000225721_AL592166.1', 'ENSG00000225726_AC007000.2', 'ENSG00000225733_FGD5-AS1', 'ENSG00000225742_LINC02036', 'ENSG00000225746_MEG8', 'ENSG00000225761_AL596247.1', 'ENSG00000225766_DHRS4L1', 'ENSG00000225774_SIRPAP1', 'ENSG00000225783_MIAT', 'ENSG00000225791_TRAM2-AS1', 'ENSG00000225792_AC004540.2', 'ENSG00000225793_AL080250.1', 'ENSG00000225806_AL121917.1', 'ENSG00000225808_DNAJC19P5', 'ENSG00000225828_FAM229A', 'ENSG00000225830_ERCC6', 'ENSG00000225850_AL355490.1', 'ENSG00000225855_RUSC1-AS1', 'ENSG00000225871_AC245100.2', 'ENSG00000225872_LINC01529', 'ENSG00000225873_C3orf86', 'ENSG00000225880_LINC00115', 'ENSG00000225885_AC023590.1', 'ENSG00000225889_AC012368.1', 'ENSG00000225900_HSPE1P13', 'ENSG00000225912_AL121871.1', 'ENSG00000225914_HCG23', 'ENSG00000225920_RIMKLBP2', 'ENSG00000225921_NOL7', 'ENSG00000225930_LINC02249', 'ENSG00000225933_AC133965.1', 'ENSG00000225934_AL592310.1', 'ENSG00000225936_AL731557.1', 'ENSG00000225937_PCA3', 'ENSG00000225963_AC009950.1', 'ENSG00000225964_NRIR', 'ENSG00000225968_ELFN1', 'ENSG00000225969_ABHD11-AS1', 'ENSG00000225972_MTND1P23', 'ENSG00000225973_PIGBOS1', 'ENSG00000225975_LINC01534', 'ENSG00000225979_AC010746.1', 'ENSG00000225986_UBXN10-AS1', 'ENSG00000225991_RPL23AP34', 'ENSG00000226007_BX005266.2', 'ENSG00000226009_KCNIP2-AS1', 'ENSG00000226015_CCT8P1', 'ENSG00000226029_LINC01772', 'ENSG00000226049_TLK2P1', 'ENSG00000226054_MEMO1P1', 'ENSG00000226055_PAICSP1', 'ENSG00000226056_MTND4P32', 'ENSG00000226067_LINC00623', 'ENSG00000226085_UQCRFS1P1', 'ENSG00000226091_LINC00937', 'ENSG00000226107_AC004383.1', 'ENSG00000226121_AHCTF1P1', 'ENSG00000226124_FTCDNL1', 'ENSG00000226137_BAIAP2-DT', 'ENSG00000226167_AP4B1-AS1', 'ENSG00000226174_TEX22', 'ENSG00000226179_LINC00685', 'ENSG00000226180_AC010536.1', 'ENSG00000226200_SGMS1-AS1', 'ENSG00000226210_WASH8P', 'ENSG00000226221_RPL26P19', 'ENSG00000226241_Z75746.1', 'ENSG00000226243_RPL37AP1', 'ENSG00000226252_AL135960.1', 'ENSG00000226253_MRPL35P3', 'ENSG00000226259_GTF2H2B', 'ENSG00000226266_AC009961.1', 'ENSG00000226276_AC093382.1', 'ENSG00000226278_PSPHP1', 'ENSG00000226279_RPL12P10', 'ENSG00000226281_AL031123.1', 'ENSG00000226287_TMEM191A', 'ENSG00000226310_AL022157.1', 'ENSG00000226312_CFLAR-AS1', 'ENSG00000226314_ZNF192P1', 'ENSG00000226318_RPS3AP38', 'ENSG00000226324_AL358453.1', 'ENSG00000226328_NUP50-DT', 'ENSG00000226352_PSPC1-AS2', 'ENSG00000226360_RPL10AP6', 'ENSG00000226361_TERF1P5', 'ENSG00000226377_AC084809.1', 'ENSG00000226380_LINC-PINT', 'ENSG00000226396_AL031727.1', 'ENSG00000226415_TPI1P1', 'ENSG00000226419_SLC16A1-AS1', 'ENSG00000226435_ANKRD18DP', 'ENSG00000226465_AL390198.1', 'ENSG00000226471_Z93930.2', 'ENSG00000226472_AC008013.1', 'ENSG00000226478_UPF3AP1', 'ENSG00000226479_TMEM185B', 'ENSG00000226493_RPL26P27', 'ENSG00000226496_LINC00323', 'ENSG00000226498_RPSAP21', 'ENSG00000226499_AL136380.1', 'ENSG00000226506_AC007463.1', 'ENSG00000226508_LINC01918', 'ENSG00000226510_UPK1A-AS1', 'ENSG00000226525_RPS7P10', 'ENSG00000226530_AL158055.1', 'ENSG00000226571_AL592429.2', 'ENSG00000226581_AC092634.3', 'ENSG00000226608_FTLP3', 'ENSG00000226609_AL390067.1', 'ENSG00000226624_AC005099.1', 'ENSG00000226632_UBE2V1P1', 'ENSG00000226644_AL121899.1', 'ENSG00000226645_AP006216.1', 'ENSG00000226650_KIF4B', 'ENSG00000226659_AC021028.1', 'ENSG00000226686_LINC01535', 'ENSG00000226688_ENTPD1-AS1', 'ENSG00000226696_LENG8-AS1', 'ENSG00000226701_RPL15P14', 'ENSG00000226711_FAM66C', 'ENSG00000226721_EEF1DP2', 'ENSG00000226742_HSBP1L1', 'ENSG00000226746_SMCR5', 'ENSG00000226752_CUTALP', 'ENSG00000226754_AL606760.1', 'ENSG00000226756_AC007365.1', 'ENSG00000226758_DISC1-IT1', 'ENSG00000226761_TAS2R46', 'ENSG00000226763_SRRM5', 'ENSG00000226777_FAM30A', 'ENSG00000226780_AC244035.1', 'ENSG00000226781_TBCAP1', 'ENSG00000226791_AC109826.1', 'ENSG00000226801_OSTCP8', 'ENSG00000226803_ZNF451-AS1', 'ENSG00000226806_AC011893.1', 'ENSG00000226816_AC005082.1', 'ENSG00000226822_AL390036.1', 'ENSG00000226823_SUGT1P1', 'ENSG00000226824_AC006001.2', 'ENSG00000226833_AC092164.1', 'ENSG00000226849_AL109811.1', 'ENSG00000226853_AC010894.2', 'ENSG00000226856_THORLNC', 'ENSG00000226862_AC104463.2', 'ENSG00000226864_ATE1-AS1', 'ENSG00000226887_ERVMER34-1', 'ENSG00000226889_AL359541.1', 'ENSG00000226891_LINC01359', 'ENSG00000226928_RPS14P4', 'ENSG00000226937_CEP164P1', 'ENSG00000226942_IL9RP3', 'ENSG00000226950_DANCR', 'ENSG00000226976_COX6A1P2', 'ENSG00000226982_CENPCP1', 'ENSG00000226986_AC092017.1', 'ENSG00000226987_AL157938.1', 'ENSG00000227008_AL009174.1', 'ENSG00000227014_AC007285.1', 'ENSG00000227017_AC007036.1', 'ENSG00000227028_SLC8A1-AS1', 'ENSG00000227032_RPS2P36', 'ENSG00000227034_AL445433.1', 'ENSG00000227036_LINC00511', 'ENSG00000227039_ITGB2-AS1', 'ENSG00000227053_AC105446.1', 'ENSG00000227055_AC009961.2', 'ENSG00000227057_WDR46', 'ENSG00000227060_LINC00629', 'ENSG00000227070_AC104170.1', 'ENSG00000227071_FOCAD-AS1', 'ENSG00000227077_AC107983.1', 'ENSG00000227081_AC005912.1', 'ENSG00000227097_RPS28P7', 'ENSG00000227124_ZNF717', 'ENSG00000227159_DDX11L16', 'ENSG00000227165_WDR11-AS1', 'ENSG00000227189_AC092535.1', 'ENSG00000227191_TRGC2', 'ENSG00000227199_ST7-AS1', 'ENSG00000227213_SPATA13-AS1', 'ENSG00000227214_HCG15', 'ENSG00000227252_AC105760.2', 'ENSG00000227256_MIS18A-AS1', 'ENSG00000227258_SMIM2-AS1', 'ENSG00000227262_HCG4B', 'ENSG00000227268_KLLN', 'ENSG00000227304_AC067942.1', 'ENSG00000227309_AC140076.1', 'ENSG00000227331_AC005042.1', 'ENSG00000227337_RPL23AP43', 'ENSG00000227344_HAUS6P1', 'ENSG00000227345_PARG', 'ENSG00000227347_HNRNPKP2', 'ENSG00000227354_RBM26-AS1', 'ENSG00000227355_AL359644.1', 'ENSG00000227359_AC017074.1', 'ENSG00000227370_AC254562.1', 'ENSG00000227372_TP73-AS1', 'ENSG00000227373_AL121983.2', 'ENSG00000227374_AL157832.1', 'ENSG00000227376_FTH1P16', 'ENSG00000227382_EIF4A2P2', 'ENSG00000227383_AL353662.1', 'ENSG00000227388_AL133410.1', 'ENSG00000227394_AC007386.1', 'ENSG00000227398_KIF9-AS1', 'ENSG00000227403_LINC01806', 'ENSG00000227449_FGF7P6', 'ENSG00000227456_LINC00310', 'ENSG00000227474_RPL6P24', 'ENSG00000227486_AL445472.1', 'ENSG00000227500_SCAMP4', 'ENSG00000227502_LINC01268', 'ENSG00000227507_LTB', 'ENSG00000227518_AL928970.1', 'ENSG00000227523_RPS20P15', 'ENSG00000227525_RPL7P6', 'ENSG00000227533_SLC2A1-AS1', 'ENSG00000227536_SOCS5P4', 'ENSG00000227540_DNAJC9-AS1', 'ENSG00000227543_SPAG5-AS1', 'ENSG00000227560_RPS15AP30', 'ENSG00000227591_AL031316.1', 'ENSG00000227598_Z94721.1', 'ENSG00000227615_AP001324.1', 'ENSG00000227617_CERS6-AS1', 'ENSG00000227627_AL080276.2', 'ENSG00000227630_LINC01132', 'ENSG00000227638_HNRNPA1P14', 'ENSG00000227640_SOX21-AS1', 'ENSG00000227671_AL390728.4', 'ENSG00000227688_HNRNPA3P2', 'ENSG00000227694_RPL23AP74', 'ENSG00000227698_AP001619.1', 'ENSG00000227704_AL354892.1', 'ENSG00000227706_AL713998.1', 'ENSG00000227714_MTND6P18', 'ENSG00000227741_AL121987.2', 'ENSG00000227748_AL138916.1', 'ENSG00000227755_AP000344.1', 'ENSG00000227766_AL671277.1', 'ENSG00000227773_ASH1L-IT1', 'ENSG00000227775_AL031282.1', 'ENSG00000227782_AC002553.1', 'ENSG00000227799_AC012358.2', 'ENSG00000227805_RPL21P90', 'ENSG00000227811_INKA2-AS1', 'ENSG00000227825_SLC9A7P1', 'ENSG00000227836_AC008850.1', 'ENSG00000227848_SUCLA2-AS1', 'ENSG00000227855_DPY19L2P3', 'ENSG00000227857_AL358075.2', 'ENSG00000227885_AL590652.1', 'ENSG00000227896_AL731569.1', 'ENSG00000227908_FLJ31104', 'ENSG00000227910_AC092634.4', 'ENSG00000227939_RPL3P2', 'ENSG00000227946_AC007383.2', 'ENSG00000227953_LINC01341', 'ENSG00000227963_RBM15-AS1', 'ENSG00000227973_PIN4P1', 'ENSG00000227986_TRIM60P18', 'ENSG00000227992_AC108463.1', 'ENSG00000228008_AC105935.1', 'ENSG00000228010_AC073343.2', 'ENSG00000228013_IL6R-AS1', 'ENSG00000228014_ZNF680P1', 'ENSG00000228036_HSPD1P9', 'ENSG00000228043_AC114763.1', 'ENSG00000228049_POLR2J2', 'ENSG00000228057_SEC63P1', 'ENSG00000228063_LYPLAL1-DT', 'ENSG00000228065_LINC01515', 'ENSG00000228071_RPL7P47', 'ENSG00000228084_AC118553.1', 'ENSG00000228106_AL392172.1', 'ENSG00000228107_AP000692.1', 'ENSG00000228109_MELTF-AS1', 'ENSG00000228110_ST13P19', 'ENSG00000228113_AC003991.1', 'ENSG00000228126_FALEC', 'ENSG00000228137_AP001469.2', 'ENSG00000228146_CASP16P', 'ENSG00000228149_RPL3P1', 'ENSG00000228150_AL357140.2', 'ENSG00000228166_MTND1P11', 'ENSG00000228172_AL020996.1', 'ENSG00000228187_AC093433.1', 'ENSG00000228192_AL512353.1', 'ENSG00000228201_AL022341.1', 'ENSG00000228203_RNF144A-AS1', 'ENSG00000228205_AC131235.1', 'ENSG00000228223_HCG11', 'ENSG00000228224_NACA4P', 'ENSG00000228242_AC093495.1', 'ENSG00000228251_AC012442.1', 'ENSG00000228252_COL6A4P2', 'ENSG00000228253_MT-ATP8', 'ENSG00000228274_AL021707.2', 'ENSG00000228280_AL731568.1', 'ENSG00000228283_KATNBL1P6', 'ENSG00000228288_PCAT6', 'ENSG00000228300_C19orf24', 'ENSG00000228302_AL512770.1', 'ENSG00000228305_AC016734.1', 'ENSG00000228308_LINC01209', 'ENSG00000228314_CYP4F29P', 'ENSG00000228315_GUSBP11', 'ENSG00000228327_AL669831.1', 'ENSG00000228335_AC073063.1', 'ENSG00000228336_OR9H1P', 'ENSG00000228340_MIR646HG', 'ENSG00000228363_AC015971.1', 'ENSG00000228382_ITPKB-IT1', 'ENSG00000228393_LINC01004', 'ENSG00000228395_AL356481.1', 'ENSG00000228397_LINC01635', 'ENSG00000228399_AL109741.2', 'ENSG00000228401_HSPC324', 'ENSG00000228408_AL031056.1', 'ENSG00000228409_CCT6P1', 'ENSG00000228415_PTMAP1', 'ENSG00000228427_AL590764.1', 'ENSG00000228434_AC004951.1', 'ENSG00000228436_AL139260.1', 'ENSG00000228439_TSTD3', 'ENSG00000228444_AL137244.1', 'ENSG00000228446_AC073052.1', 'ENSG00000228451_SDAD1P1', 'ENSG00000228463_AP006222.1', 'ENSG00000228474_OST4', 'ENSG00000228477_AL663070.1', 'ENSG00000228486_C2orf92', 'ENSG00000228492_RAB11FIP1P1', 'ENSG00000228496_AC106875.1', 'ENSG00000228506_AL513550.1', 'ENSG00000228519_AC097263.1', 'ENSG00000228526_MIR34AHG', 'ENSG00000228541_AC093159.1', 'ENSG00000228544_CCDC183-AS1', 'ENSG00000228546_AC091390.2', 'ENSG00000228548_ITPKB-AS1', 'ENSG00000228549_BX284668.2', 'ENSG00000228559_AL033519.3', 'ENSG00000228589_SPCS2P4', 'ENSG00000228590_MIR4432HG', 'ENSG00000228594_FNDC10', 'ENSG00000228599_RPL7P52', 'ENSG00000228606_AL139011.1', 'ENSG00000228623_ZNF883', 'ENSG00000228624_HDAC2-AS2', 'ENSG00000228643_AC079779.2', 'ENSG00000228649_SNHG26', 'ENSG00000228661_AC090587.1', 'ENSG00000228665_BX679664.1', 'ENSG00000228672_PROB1', 'ENSG00000228686_AL590723.1', 'ENSG00000228696_ARL17B', 'ENSG00000228701_TNKS2-AS1', 'ENSG00000228702_AL645998.1', 'ENSG00000228705_LINC00659', 'ENSG00000228716_DHFR', 'ENSG00000228748_AL450306.1', 'ENSG00000228775_WEE2-AS1', 'ENSG00000228778_AL513542.1', 'ENSG00000228779_Z69666.1', 'ENSG00000228782_MRPL45P2', 'ENSG00000228784_LINC00954', 'ENSG00000228791_THRB-AS1', 'ENSG00000228794_LINC01128', 'ENSG00000228800_AL590068.1', 'ENSG00000228801_AC064807.1', 'ENSG00000228808_HMGB3P4', 'ENSG00000228817_BACH1-IT2', 'ENSG00000228824_MIR4500HG', 'ENSG00000228828_TLK2P2', 'ENSG00000228830_AL160408.2', 'ENSG00000228834_AL445189.2', 'ENSG00000228847_ATP5MC2P4', 'ENSG00000228857_AC104653.1', 'ENSG00000228878_SEPT7-AS1', 'ENSG00000228886_AL138963.1', 'ENSG00000228887_EEF1DP1', 'ENSG00000228889_UBAC2-AS1', 'ENSG00000228903_RASA4CP', 'ENSG00000228906_AL353804.1', 'ENSG00000228925_AC016722.2', 'ENSG00000228929_RPS13P2', 'ENSG00000228939_AKT3-IT1', 'ENSG00000228956_SATB1-AS1', 'ENSG00000228960_OR2A9P', 'ENSG00000229007_EXOSC3P1', 'ENSG00000229018_PMS2P7', 'ENSG00000229019_AL161457.1', 'ENSG00000229023_AC067945.1', 'ENSG00000229036_VDAC1P8', 'ENSG00000229043_AC091729.3', 'ENSG00000229044_AL451070.1', 'ENSG00000229047_AF127577.1', 'ENSG00000229052_AL449283.1', 'ENSG00000229054_RPS29P14', 'ENSG00000229056_AC020571.1', 'ENSG00000229083_PSMA6P2', 'ENSG00000229087_AC007738.1', 'ENSG00000229097_CALM2P2', 'ENSG00000229117_RPL41', 'ENSG00000229119_AC026403.1', 'ENSG00000229124_VIM-AS1', 'ENSG00000229127_AC007038.1', 'ENSG00000229132_EIF4A1P10', 'ENSG00000229133_RPS7P4', 'ENSG00000229140_CCDC26', 'ENSG00000229152_ANKRD10-IT1', 'ENSG00000229153_EPHA1-AS1', 'ENSG00000229169_AL096701.2', 'ENSG00000229172_AC073065.1', 'ENSG00000229178_AC233280.1', 'ENSG00000229180_AC006001.3', 'ENSG00000229196_AC087071.1', 'ENSG00000229204_PTGES3P3', 'ENSG00000229222_KRT18P4', 'ENSG00000229227_AL356056.1', 'ENSG00000229241_PNPT1P1', 'ENSG00000229245_AL359636.1', 'ENSG00000229267_AC016708.1', 'ENSG00000229278_AL133353.1', 'ENSG00000229299_AL121845.1', 'ENSG00000229320_KRT8P12', 'ENSG00000229325_ACAP2-IT1', 'ENSG00000229331_GK-IT1', 'ENSG00000229334_AC046143.1', 'ENSG00000229336_AP000568.1', 'ENSG00000229344_MTCO2P12', 'ENSG00000229348_HYI-AS1', 'ENSG00000229358_DPY19L1P1', 'ENSG00000229388_LINC01715', 'ENSG00000229391_HLA-DRB6', 'ENSG00000229418_AL136181.1', 'ENSG00000229419_RALGAPA1P1', 'ENSG00000229425_AJ009632.2', 'ENSG00000229447_AC114495.2', 'ENSG00000229453_SPINK8', 'ENSG00000229473_RGS17P1', 'ENSG00000229474_PATL2', 'ENSG00000229503_AC092155.2', 'ENSG00000229525_AC053503.2', 'ENSG00000229539_AL353194.1', 'ENSG00000229557_LINC00379', 'ENSG00000229582_AL358074.1', 'ENSG00000229585_RPL21P44', 'ENSG00000229587_AL158825.2', 'ENSG00000229591_AC006017.1', 'ENSG00000229605_RPL21P93', 'ENSG00000229619_MBNL1-AS1', 'ENSG00000229622_MTND5P2', 'ENSG00000229638_RPL4P4', 'ENSG00000229644_NAMPTP1', 'ENSG00000229659_RPL26P6', 'ENSG00000229666_MAST4-AS1', 'ENSG00000229671_LINC01150', 'ENSG00000229676_ZNF492', 'ENSG00000229677_AC018644.1', 'ENSG00000229689_AC009237.3', 'ENSG00000229692_SOS1-IT1', 'ENSG00000229704_EIF2S2P2', 'ENSG00000229717_AC110615.1', 'ENSG00000229719_MIR194-2HG', 'ENSG00000229721_AC104115.1', 'ENSG00000229728_AL136531.1', 'ENSG00000229754_CXCR2P1', 'ENSG00000229794_MTCYBP32', 'ENSG00000229806_RPS15P5', 'ENSG00000229807_XIST', 'ENSG00000229808_AL391825.1', 'ENSG00000229809_ZNF688', 'ENSG00000229816_DDX50P1', 'ENSG00000229820_CR391992.1', 'ENSG00000229833_PET100', 'ENSG00000229851_ARSD-AS1', 'ENSG00000229852_AC019205.1', 'ENSG00000229855_AC008568.1', 'ENSG00000229880_IMMTP1', 'ENSG00000229891_LINC01315', 'ENSG00000229893_AC005091.1', 'ENSG00000229896_AL157373.2', 'ENSG00000229897_SEPT7P7', 'ENSG00000229917_RPL7P46', 'ENSG00000229920_RPS4XP5', 'ENSG00000229939_AL589880.1', 'ENSG00000229944_EIF4EP2', 'ENSG00000229954_MTND2P2', 'ENSG00000229956_ZRANB2-AS2', 'ENSG00000229980_TOB1-AS1', 'ENSG00000229988_HBBP1', 'ENSG00000229989_MIR181A1HG', 'ENSG00000229994_RPL5P4', 'ENSG00000229999_AL022238.2', 'ENSG00000230002_ALMS1-IT1', 'ENSG00000230006_ANKRD36BP2', 'ENSG00000230021_AL669831.3', 'ENSG00000230042_AK3P3', 'ENSG00000230043_TMSB4XP6', 'ENSG00000230061_TRPM2-AS', 'ENSG00000230067_HSPD1P6', 'ENSG00000230068_CDC42-IT1', 'ENSG00000230071_RPL4P6', 'ENSG00000230074_AL162231.2', 'ENSG00000230076_RPL10P6', 'ENSG00000230082_PRRT3-AS1', 'ENSG00000230084_AC006059.1', 'ENSG00000230091_TMEM254-AS1', 'ENSG00000230092_AL669831.4', 'ENSG00000230105_AL354793.1', 'ENSG00000230124_ACBD6', 'ENSG00000230148_HOXB-AS1', 'ENSG00000230155_FO393401.1', 'ENSG00000230176_LINC01433', 'ENSG00000230177_AL080317.1', 'ENSG00000230184_SMYD3-IT1', 'ENSG00000230185_C9orf147', 'ENSG00000230202_AL450405.1', 'ENSG00000230207_RPL4P5', 'ENSG00000230215_LINC01840', 'ENSG00000230216_HSPB1P2', 'ENSG00000230223_ATXN8OS', 'ENSG00000230225_MTND5P14', 'ENSG00000230262_LINC02603', 'ENSG00000230266_XXYLT1-AS2', 'ENSG00000230267_HERC2P4', 'ENSG00000230291_AC078817.1', 'ENSG00000230305_AC004980.3', 'ENSG00000230319_AL022476.1', 'ENSG00000230325_AL359921.1', 'ENSG00000230337_AL109811.2', 'ENSG00000230359_TPI1P2', 'ENSG00000230366_DSCR9', 'ENSG00000230368_FAM41C', 'ENSG00000230373_GOLGA6L5P', 'ENSG00000230383_AC009245.1', 'ENSG00000230393_AC092667.1', 'ENSG00000230409_TCEA1P2', 'ENSG00000230415_LINC01786', 'ENSG00000230417_LINC00856', 'ENSG00000230424_AL035413.1', 'ENSG00000230426_ERVMER61-1', 'ENSG00000230432_AC114803.1', 'ENSG00000230438_SERPINB9P1', 'ENSG00000230449_RPL7P4', 'ENSG00000230453_ANKRD18B', 'ENSG00000230454_U73166.1', 'ENSG00000230457_PA2G4P4', 'ENSG00000230480_AC093142.1', 'ENSG00000230482_ATP5MC2P3', 'ENSG00000230487_PSMG3-AS1', 'ENSG00000230489_VAV3-AS1', 'ENSG00000230510_PPP5D1', 'ENSG00000230513_THAP7-AS1', 'ENSG00000230521_AL645929.1', 'ENSG00000230530_LIMD1-AS1', 'ENSG00000230542_LINC00102', 'ENSG00000230551_AC021078.1', 'ENSG00000230555_AL450326.1', 'ENSG00000230562_FAM133DP', 'ENSG00000230572_AC027612.2', 'ENSG00000230580_AC021016.1', 'ENSG00000230583_GTF2IRD1P1', 'ENSG00000230587_LINC02580', 'ENSG00000230590_FTX', 'ENSG00000230596_GPAA1P2', 'ENSG00000230606_AC092683.1', 'ENSG00000230613_HM13-AS1', 'ENSG00000230615_AL139220.2', 'ENSG00000230623_AC104461.1', 'ENSG00000230629_RPS23P8', 'ENSG00000230630_DNM3OS', 'ENSG00000230641_USP12-AS2', 'ENSG00000230658_KLHL7-DT', 'ENSG00000230666_CEACAM22P', 'ENSG00000230673_PABPC1P3', 'ENSG00000230679_ENO1-AS1', 'ENSG00000230684_AL158207.2', 'ENSG00000230699_AL645608.2', 'ENSG00000230701_FBXW4P1', 'ENSG00000230732_AC016949.1', 'ENSG00000230733_AC092171.2', 'ENSG00000230734_RPL10P3', 'ENSG00000230747_AC021188.1', 'ENSG00000230749_MEIS1-AS2', 'ENSG00000230751_AC007036.2', 'ENSG00000230756_RHOQP3', 'ENSG00000230769_Z98048.1', 'ENSG00000230783_AC009961.3', 'ENSG00000230793_SMARCE1P5', 'ENSG00000230795_HLA-K', 'ENSG00000230797_YY2', 'ENSG00000230832_AC241584.1', 'ENSG00000230844_ZNF674-AS1', 'ENSG00000230869_AGAP10P', 'ENSG00000230896_AL604028.1', 'ENSG00000230897_RPS18P12', 'ENSG00000230946_HNRNPA1P68', 'ENSG00000230955_AL929472.2', 'ENSG00000230979_AC079250.1', 'ENSG00000230987_AL359976.1', 'ENSG00000230989_HSBP1', 'ENSG00000231006_RPL7P32', 'ENSG00000231007_CDC20P1', 'ENSG00000231028_LINC00271', 'ENSG00000231043_AC007238.1', 'ENSG00000231047_GCNT1P3', 'ENSG00000231050_AL109917.1', 'ENSG00000231064_AC234582.1', 'ENSG00000231066_NPM1P9', 'ENSG00000231073_AL590133.1', 'ENSG00000231074_HCG18', 'ENSG00000231084_RPL22P24', 'ENSG00000231096_NDUFB4P3', 'ENSG00000231113_AL035587.1', 'ENSG00000231119_AL031666.1', 'ENSG00000231125_AF129075.1', 'ENSG00000231128_AL137856.1', 'ENSG00000231131_LNCAROD', 'ENSG00000231154_MORF4L2-AS1', 'ENSG00000231160_KLF3-AS1', 'ENSG00000231164_RPL7P56', 'ENSG00000231165_TRBV26OR9-2', 'ENSG00000231167_YBX1P2', 'ENSG00000231169_EEF1B2P1', 'ENSG00000231177_LINC00852', 'ENSG00000231181_AL954705.1', 'ENSG00000231187_AL356056.2', 'ENSG00000231205_ZNF826P', 'ENSG00000231212_AL590399.3', 'ENSG00000231241_RPS3AP3', 'ENSG00000231245_C1DP1', 'ENSG00000231249_ITPR1-DT', 'ENSG00000231256_CFAP97D1', 'ENSG00000231296_AL050341.1', 'ENSG00000231304_SGO1-AS1', 'ENSG00000231312_AC007388.1', 'ENSG00000231327_LINC01816', 'ENSG00000231341_VDAC1P6', 'ENSG00000231344_AL020997.1', 'ENSG00000231345_BEND3P1', 'ENSG00000231365_AL359915.2', 'ENSG00000231384_AC007919.2', 'ENSG00000231389_HLA-DPA1', 'ENSG00000231409_AC018868.1', 'ENSG00000231414_AC016700.2', 'ENSG00000231416_AL358472.1', 'ENSG00000231439_WASIR2', 'ENSG00000231466_AL022324.2', 'ENSG00000231473_RB1-DT', 'ENSG00000231485_AL357078.1', 'ENSG00000231494_AC104634.1', 'ENSG00000231500_RPS18', 'ENSG00000231503_PTMAP4', 'ENSG00000231527_CR769775.1', 'ENSG00000231535_LINC00278', 'ENSG00000231549_ATP5MDP1', 'ENSG00000231551_AC245100.4', 'ENSG00000231564_EIF4A1P11', 'ENSG00000231566_LINC02595', 'ENSG00000231584_FAHD2CP', 'ENSG00000231595_AC005224.1', 'ENSG00000231607_DLEU2', 'ENSG00000231609_AC007098.1', 'ENSG00000231615_AL645568.2', 'ENSG00000231621_AC013264.1', 'ENSG00000231636_AGBL5-AS1', 'ENSG00000231646_FSIP2-AS1', 'ENSG00000231652_AL590428.1', 'ENSG00000231655_AC011742.2', 'ENSG00000231663_AL355472.1', 'ENSG00000231672_DIRC3', 'ENSG00000231697_NANOGP5', 'ENSG00000231705_AL451069.2', 'ENSG00000231707_PABPC1P1', 'ENSG00000231711_LINC00899', 'ENSG00000231721_LINC-PINT', 'ENSG00000231728_TMSB15B-AS1', 'ENSG00000231744_AL669818.1', 'ENSG00000231747_AC079922.1', 'ENSG00000231752_EMBP1', 'ENSG00000231760_AL355312.2', 'ENSG00000231767_AL136454.1', 'ENSG00000231769_AL035701.1', 'ENSG00000231770_TMEM44-AS1', 'ENSG00000231784_DBIL5P', 'ENSG00000231789_PIK3CD-AS2', 'ENSG00000231793_DOC2GP', 'ENSG00000231799_PA2G4P6', 'ENSG00000231806_PCAT7', 'ENSG00000231841_FAM192BP', 'ENSG00000231848_AC012354.2', 'ENSG00000231856_AL162377.1', 'ENSG00000231871_IPO9-AS1', 'ENSG00000231875_AL359885.1', 'ENSG00000231877_AL359551.1', 'ENSG00000231878_SNRPFP1', 'ENSG00000231880_KF459542.1', 'ENSG00000231884_NDUFB1P1', 'ENSG00000231887_PRH1', 'ENSG00000231889_TRAF3IP2-AS1', 'ENSG00000231890_DARS-AS1', 'ENSG00000231903_AC079354.3', 'ENSG00000231908_IDH1-AS1', 'ENSG00000231916_AC006033.1', 'ENSG00000231925_TAPBP', 'ENSG00000231940_RPS7P3', 'ENSG00000231943_PGM5P4-AS1', 'ENSG00000231952_DPY19L1P2', 'ENSG00000231955_AC007383.3', 'ENSG00000231969_AC007364.1', 'ENSG00000231999_LRRC8C-DT', 'ENSG00000232000_CLCN3P1', 'ENSG00000232010_AP001059.1', 'ENSG00000232021_LEF1-AS1', 'ENSG00000232022_FAAHP1', 'ENSG00000232024_LSM12P1', 'ENSG00000232031_AL078599.2', 'ENSG00000232037_AL590556.1', 'ENSG00000232043_AL133230.1', 'ENSG00000232063_AL691447.2', 'ENSG00000232070_TMEM253', 'ENSG00000232093_DCST1-AS1', 'ENSG00000232098_AC012313.1', 'ENSG00000232104_RFX3-AS1', 'ENSG00000232112_TMA7', 'ENSG00000232118_BACH1-AS1', 'ENSG00000232119_MCTS1', 'ENSG00000232124_AP001057.1', 'ENSG00000232125_DYTN', 'ENSG00000232134_RPS15AP12', 'ENSG00000232149_FERP1', 'ENSG00000232150_ST13P4', 'ENSG00000232160_RAP2C-AS1', 'ENSG00000232176_AL161909.1', 'ENSG00000232196_MTRNR2L4', 'ENSG00000232215_OR2L6P', 'ENSG00000232220_AC008440.2', 'ENSG00000232228_AC092431.2', 'ENSG00000232229_LINC00865', 'ENSG00000232233_LINC02043', 'ENSG00000232234_AL355499.1', 'ENSG00000232265_AC239803.1', 'ENSG00000232268_OR52I1', 'ENSG00000232295_AL589935.1', 'ENSG00000232300_FAM215B', 'ENSG00000232303_DFFBP1', 'ENSG00000232310_AL078590.3', 'ENSG00000232342_AC022540.1', 'ENSG00000232346_Z74021.1', 'ENSG00000232347_AL390728.5', 'ENSG00000232362_ATP5MGP2', 'ENSG00000232373_MTCYBP3', 'ENSG00000232386_AC015712.1', 'ENSG00000232387_SKA2P1', 'ENSG00000232388_SMIM26', 'ENSG00000232392_AC002366.1', 'ENSG00000232412_AL121601.1', 'ENSG00000232422_KNOP1P4', 'ENSG00000232434_AJM1', 'ENSG00000232437_RPS26P42', 'ENSG00000232439_RPL18AP7', 'ENSG00000232442_MHENCR', 'ENSG00000232445_AC006329.1', 'ENSG00000232450_AL133517.1', 'ENSG00000232453_AC105277.1', 'ENSG00000232454_AL138752.1', 'ENSG00000232472_EEF1B2P3', 'ENSG00000232485_AC098820.1', 'ENSG00000232486_AL592437.2', 'ENSG00000232489_MFAP1P1', 'ENSG00000232499_AL391058.1', 'ENSG00000232527_AC245595.1', 'ENSG00000232533_AC093673.1', 'ENSG00000232536_AL365436.2', 'ENSG00000232545_AC253536.3', 'ENSG00000232564_MRTFA-AS1', 'ENSG00000232573_RPL3P4', 'ENSG00000232586_KIAA1614-AS1', 'ENSG00000232587_EEF1A1P3', 'ENSG00000232593_KANTR', 'ENSG00000232611_AL683813.1', 'ENSG00000232613_LINC02576', 'ENSG00000232626_AC099336.1', 'ENSG00000232629_HLA-DQB2', 'ENSG00000232640_AL354892.2', 'ENSG00000232645_LINC01431', 'ENSG00000232648_AC107214.1', 'ENSG00000232653_GOLGA8N', 'ENSG00000232671_AL391069.2', 'ENSG00000232677_LINC00665', 'ENSG00000232682_AL592430.1', 'ENSG00000232684_ATP11A-AS1', 'ENSG00000232693_AC012370.1', 'ENSG00000232712_KIZ-AS1', 'ENSG00000232713_AC010733.1', 'ENSG00000232721_AC239800.2', 'ENSG00000232729_AC211433.1', 'ENSG00000232732_AC097717.1', 'ENSG00000232748_AC135050.1', 'ENSG00000232767_AC016825.1', 'ENSG00000232768_AL356320.1', 'ENSG00000232774_FLJ22447', 'ENSG00000232788_ITGA6-AS1', 'ENSG00000232807_AL137186.2', 'ENSG00000232810_TNF', 'ENSG00000232811_AL360270.1', 'ENSG00000232818_RPS2P32', 'ENSG00000232828_AC231533.1', 'ENSG00000232832_LMLN-AS1', 'ENSG00000232838_PET117', 'ENSG00000232850_PTGES2-AS1', 'ENSG00000232858_RPL34P27', 'ENSG00000232859_LYRM9', 'ENSG00000232860_SMG7-AS1', 'ENSG00000232871_SEC1P', 'ENSG00000232872_CTAGE3P', 'ENSG00000232874_AC080129.2', 'ENSG00000232876_AL353596.1', 'ENSG00000232884_AF127936.2', 'ENSG00000232888_RPS11P5', 'ENSG00000232907_DLGAP4-AS1', 'ENSG00000232928_DDX3P1', 'ENSG00000232931_LINC00342', 'ENSG00000232934_AL157786.1', 'ENSG00000232936_AL157400.2', 'ENSG00000232938_RPL23AP87', 'ENSG00000232940_HCG25', 'ENSG00000232952_AL512844.1', 'ENSG00000232956_SNHG15', 'ENSG00000232973_CYP1B1-AS1', 'ENSG00000232977_LINC00327', 'ENSG00000232995_RGS5', 'ENSG00000233006_MIR3936HG', 'ENSG00000233008_LINC01725', 'ENSG00000233016_SNHG7', 'ENSG00000233040_FAM204BP', 'ENSG00000233045_AC097523.1', 'ENSG00000233058_LINC00884', 'ENSG00000233061_TTLL7-IT1', 'ENSG00000233070_ZFY-AS1', 'ENSG00000233072_RPS15AP6', 'ENSG00000233098_CCDC144NL-AS1', 'ENSG00000233101_HOXB-AS3', 'ENSG00000233122_CTAGE7P', 'ENSG00000233131_AC096649.2', 'ENSG00000233133_AC104451.1', 'ENSG00000233170_AC138356.2', 'ENSG00000233175_AC008105.1', 'ENSG00000233178_AL161457.2', 'ENSG00000233184_AC093157.1', 'ENSG00000233189_RPL12P29', 'ENSG00000233203_DHCR24-DT', 'ENSG00000233220_LINC00167', 'ENSG00000233223_AC016876.1', 'ENSG00000233230_AC079807.1', 'ENSG00000233236_LINC02573', 'ENSG00000233246_AL513327.2', 'ENSG00000233254_RPL21P134', 'ENSG00000233261_FAM238A', 'ENSG00000233264_AC006042.3', 'ENSG00000233266_HMGB1P31', 'ENSG00000233268_AL691449.1', 'ENSG00000233270_SNRPEP4', 'ENSG00000233276_GPX1', 'ENSG00000233325_MIPEPP3', 'ENSG00000233327_USP32P2', 'ENSG00000233328_PFN1P1', 'ENSG00000233330_AL078581.1', 'ENSG00000233355_CHRM3-AS2', 'ENSG00000233369_GTF2IP4', 'ENSG00000233382_NKAPP1', 'ENSG00000233393_AP000688.2', 'ENSG00000233396_LINC01719', 'ENSG00000233406_AL162430.1', 'ENSG00000233410_LINC01222', 'ENSG00000233416_AC012065.2', 'ENSG00000233421_LINC01783', 'ENSG00000233426_EIF3FP3', 'ENSG00000233429_HOTAIRM1', 'ENSG00000233430_AC239798.1', 'ENSG00000233435_AGGF1P2', 'ENSG00000233436_BTBD18', 'ENSG00000233452_STXBP5-AS1', 'ENSG00000233461_AL445524.1', 'ENSG00000233478_AL031280.1', 'ENSG00000233483_AC008105.2', 'ENSG00000233493_TMEM238', 'ENSG00000233514_AL356653.1', 'ENSG00000233527_ZNF529-AS1', 'ENSG00000233538_AC017104.3', 'ENSG00000233547_AL158212.2', 'ENSG00000233554_B4GALT1-AS1', 'ENSG00000233558_AL050331.1', 'ENSG00000233559_LINC00513', 'ENSG00000233560_KRT8P39', 'ENSG00000233578_EIF4EP1', 'ENSG00000233579_KRT8P15', 'ENSG00000233585_AC231533.2', 'ENSG00000233588_CYP51A1P2', 'ENSG00000233593_AL590094.1', 'ENSG00000233597_AC133435.1', 'ENSG00000233621_LINC01137', 'ENSG00000233622_CYP2T1P', 'ENSG00000233654_AC108047.1', 'ENSG00000233661_SPIN4-AS1', 'ENSG00000233668_AL353662.2', 'ENSG00000233672_RNASEH2B-AS1', 'ENSG00000233690_EBAG9P1', 'ENSG00000233695_GAS6-AS1', 'ENSG00000233716_AC074367.1', 'ENSG00000233718_MYCNOS', 'ENSG00000233723_LINC01122', 'ENSG00000233730_LINC01765', 'ENSG00000233747_RPL36AP13', 'ENSG00000233757_AC092835.1', 'ENSG00000233760_AC004947.1', 'ENSG00000233762_AC007969.1', 'ENSG00000233766_AC098617.1', 'ENSG00000233776_LINC01251', 'ENSG00000233783_AP001442.1', 'ENSG00000233806_LINC01237', 'ENSG00000233818_AP000695.2', 'ENSG00000233820_AL589843.2', 'ENSG00000233822_HIST1H2BN', 'ENSG00000233830_EIF4HP1', 'ENSG00000233834_AC005083.1', 'ENSG00000233836_AC139769.1', 'ENSG00000233844_KCNQ5-IT1', 'ENSG00000233851_LATS2-AS1', 'ENSG00000233864_TTTY15', 'ENSG00000233868_AC009302.1', 'ENSG00000233871_DLG5-AS1', 'ENSG00000233885_YEATS2-AS1', 'ENSG00000233892_PAIP1P1', 'ENSG00000233901_LINC01503', 'ENSG00000233903_Z83851.1', 'ENSG00000233912_AC026202.2', 'ENSG00000233913_RPL10P9', 'ENSG00000233922_LINC01694', 'ENSG00000233924_RPSAP13', 'ENSG00000233927_RPS28', 'ENSG00000233929_MT1XP1', 'ENSG00000233937_CTC-338M12.4', 'ENSG00000233954_UQCRHL', 'ENSG00000233956_BTF3P6', 'ENSG00000233966_UBE2SP1', 'ENSG00000233967_AL359715.1', 'ENSG00000233968_AL157895.1', 'ENSG00000233971_RPS20P10', 'ENSG00000233975_LINC02574', 'ENSG00000233990_AL353754.1', 'ENSG00000234006_DDX39B-AS1', 'ENSG00000234009_RPL5P34', 'ENSG00000234028_AC062029.1', 'ENSG00000234031_RPS3AP44', 'ENSG00000234036_TXNP6', 'ENSG00000234040_RPL10P12', 'ENSG00000234043_NUDT9P1', 'ENSG00000234062_AL390879.1', 'ENSG00000234072_AC074117.1', 'ENSG00000234073_AC011816.1', 'ENSG00000234080_AL596275.1', 'ENSG00000234084_AL049552.1', 'ENSG00000234093_RPS15AP11', 'ENSG00000234118_RPL13AP6', 'ENSG00000234127_TRIM26', 'ENSG00000234129_AC073529.1', 'ENSG00000234134_AL158835.2', 'ENSG00000234141_AC006042.4', 'ENSG00000234142_LINC01675', 'ENSG00000234147_AL035446.1', 'ENSG00000234171_RNASEH1-AS1', 'ENSG00000234177_LINC01114', 'ENSG00000234183_LINC01952', 'ENSG00000234187_AIMP1P1', 'ENSG00000234215_AC006012.1', 'ENSG00000234219_CDCA4P4', 'ENSG00000234222_LIX1L-AS1', 'ENSG00000234232_AC243772.3', 'ENSG00000234233_KCNH1-IT1', 'ENSG00000234241_AL109618.1', 'ENSG00000234264_DEPDC1-AS1', 'ENSG00000234268_AP000936.3', 'ENSG00000234274_COX7BP2', 'ENSG00000234277_LINC01641', 'ENSG00000234283_LINC01731', 'ENSG00000234284_ZNF879', 'ENSG00000234287_AC099560.2', 'ENSG00000234297_AL592293.2', 'ENSG00000234311_AL451069.3', 'ENSG00000234327_AC012146.1', 'ENSG00000234329_AL604028.2', 'ENSG00000234335_RPS4XP11', 'ENSG00000234336_JAZF1-AS1', 'ENSG00000234354_RPS26P47', 'ENSG00000234369_TATDN1P1', 'ENSG00000234377_RNF219-AS1', 'ENSG00000234380_LINC01426', 'ENSG00000234390_USP27X-AS1', 'ENSG00000234409_CCDC188', 'ENSG00000234420_ZNF37BP', 'ENSG00000234432_AC092171.3', 'ENSG00000234444_ZNF736', 'ENSG00000234456_MAGI2-AS3', 'ENSG00000234465_PINLYP', 'ENSG00000234473_ZNF101P2', 'ENSG00000234476_AC092811.1', 'ENSG00000234492_RPL34-AS1', 'ENSG00000234493_RHOXF1P1', 'ENSG00000234494_SP2-AS1', 'ENSG00000234498_RPL13AP20', 'ENSG00000234506_LINC01506', 'ENSG00000234509_AP000253.1', 'ENSG00000234513_AC073072.2', 'ENSG00000234518_PTGES3P1', 'ENSG00000234545_FAM133B', 'ENSG00000234546_LNCTAM34A', 'ENSG00000234572_LINC01800', 'ENSG00000234585_CCT6P3', 'ENSG00000234607_AL355994.4', 'ENSG00000234608_MAPKAPK5-AS1', 'ENSG00000234616_JRK', 'ENSG00000234617_SNRK-AS1', 'ENSG00000234618_RPSAP9', 'ENSG00000234624_AC016894.1', 'ENSG00000234636_MED14OS', 'ENSG00000234663_LINC01934', 'ENSG00000234664_HMGN2P5', 'ENSG00000234665_AL512625.3', 'ENSG00000234676_IFT74-AS1', 'ENSG00000234678_ELF3-AS1', 'ENSG00000234684_SDCBP2-AS1', 'ENSG00000234690_AC106869.1', 'ENSG00000234694_AL139289.2', 'ENSG00000234705_HMGA1P4', 'ENSG00000234709_UPF3AP3', 'ENSG00000234719_NPIPB2', 'ENSG00000234737_KRT18P15', 'ENSG00000234740_AL162386.2', 'ENSG00000234741_GAS5', 'ENSG00000234742_AC144530.1', 'ENSG00000234745_HLA-B', 'ENSG00000234751_AP002381.1', 'ENSG00000234753_FOXP4-AS1', 'ENSG00000234771_SLC25A25-AS1', 'ENSG00000234772_LINC00412', 'ENSG00000234773_AC012618.3', 'ENSG00000234782_TPT1P9', 'ENSG00000234785_EEF1GP5', 'ENSG00000234788_HSPA8P3', 'ENSG00000234793_AC114730.2', 'ENSG00000234797_RPS3AP6', 'ENSG00000234816_HIST1H2APS5', 'ENSG00000234825_XRCC6P2', 'ENSG00000234851_RPL23AP42', 'ENSG00000234882_EIF3EP1', 'ENSG00000234883_MIR155HG', 'ENSG00000234886_MTND5P26', 'ENSG00000234902_AC007879.3', 'ENSG00000234906_APOC2', 'ENSG00000234911_TEX21P', 'ENSG00000234912_SNHG20', 'ENSG00000234913_AC016027.2', 'ENSG00000234915_AL360091.3', 'ENSG00000234917_AC098484.2', 'ENSG00000234925_ATP5PDP4', 'ENSG00000234928_LINC01659', 'ENSG00000234936_AC010883.1', 'ENSG00000234937_AL139128.1', 'ENSG00000234945_GTF3C2-AS1', 'ENSG00000234964_FABP5P7', 'ENSG00000234996_AC098934.2', 'ENSG00000235001_EIF4A1P2', 'ENSG00000235016_SEMA3F-AS1', 'ENSG00000235018_AL137077.1', 'ENSG00000235020_AL390783.1', 'ENSG00000235026_DPP10-AS1', 'ENSG00000235034_C19orf81', 'ENSG00000235036_AL035456.1', 'ENSG00000235043_TECRP1', 'ENSG00000235065_RPL24P2', 'ENSG00000235078_AC231981.1', 'ENSG00000235079_ZRANB2-AS1', 'ENSG00000235082_SUMO1P3', 'ENSG00000235098_ANKRD65', 'ENSG00000235106_BRD3OS', 'ENSG00000235109_ZSCAN31', 'ENSG00000235111_Z97192.3', 'ENSG00000235119_AL138895.1', 'ENSG00000235121_AL645504.1', 'ENSG00000235128_AC013474.1', 'ENSG00000235145_RPSAP16', 'ENSG00000235159_AL121672.2', 'ENSG00000235162_C12orf75', 'ENSG00000235169_SMIM1', 'ENSG00000235172_LINC01366', 'ENSG00000235173_HGH1', 'ENSG00000235174_RPL39P3', 'ENSG00000235175_RPL26P37', 'ENSG00000235183_SRP14P3', 'ENSG00000235192_AC009495.3', 'ENSG00000235194_PPP1R3E', 'ENSG00000235217_TSPY26P', 'ENSG00000235236_AC137630.2', 'ENSG00000235244_DANT2', 'ENSG00000235245_AL360181.2', 'ENSG00000235248_OR13C1P', 'ENSG00000235257_ITGA9-AS1', 'ENSG00000235262_KDM5C-IT1', 'ENSG00000235272_RAMACL', 'ENSG00000235280_MCF2L-AS1', 'ENSG00000235297_FAUP1', 'ENSG00000235298_AL354733.3', 'ENSG00000235313_HM13-IT1', 'ENSG00000235314_LINC00957', 'ENSG00000235316_DUSP8P5', 'ENSG00000235319_AC012360.1', 'ENSG00000235330_RPL12P25', 'ENSG00000235354_RPS29P16', 'ENSG00000235358_AC093151.3', 'ENSG00000235363_SNRPGP10', 'ENSG00000235374_SSR4P1', 'ENSG00000235381_AL596202.1', 'ENSG00000235423_AC068768.1', 'ENSG00000235436_DPY19L2P4', 'ENSG00000235437_LINC01278', 'ENSG00000235445_AC016027.3', 'ENSG00000235449_AC098934.3', 'ENSG00000235453_SMIM27', 'ENSG00000235459_RPS26P31', 'ENSG00000235482_RPL21P135', 'ENSG00000235488_JARID2-AS1', 'ENSG00000235489_DBF4P1', 'ENSG00000235499_AC073046.1', 'ENSG00000235501_AC105942.1', 'ENSG00000235505_CASP17P', 'ENSG00000235508_RPS2P7', 'ENSG00000235513_AL035681.1', 'ENSG00000235522_AC010978.1', 'ENSG00000235527_HIPK1-AS1', 'ENSG00000235530_AC087294.1', 'ENSG00000235545_AC103923.1', 'ENSG00000235552_RPL6P27', 'ENSG00000235554_AC005822.1', 'ENSG00000235559_NOP56P1', 'ENSG00000235560_AC002310.1', 'ENSG00000235568_NFAM1', 'ENSG00000235578_AC007731.3', 'ENSG00000235582_AL365258.2', 'ENSG00000235587_GAPDHP65', 'ENSG00000235590_GNAS-AS1', 'ENSG00000235605_AL355472.2', 'ENSG00000235609_AF127577.4', 'ENSG00000235613_NSRP1P1', 'ENSG00000235618_FAM21EP', 'ENSG00000235619_RPL36AP33', 'ENSG00000235636_NUS1P1', 'ENSG00000235651_AC064850.1', 'ENSG00000235652_AL356599.1', 'ENSG00000235655_H3F3AP4', 'ENSG00000235660_LINC00345', 'ENSG00000235663_SAPCD1-AS1', 'ENSG00000235672_AC090286.2', 'ENSG00000235703_LINC00894', 'ENSG00000235706_DICER1-AS1', 'ENSG00000235724_AC009299.3', 'ENSG00000235748_SEPT14P12', 'ENSG00000235749_AL390860.1', 'ENSG00000235750_KIAA0040', 'ENSG00000235763_SNRPGP5', 'ENSG00000235776_AC000089.1', 'ENSG00000235782_AL031429.1', 'ENSG00000235786_ZNRF3-IT1', 'ENSG00000235795_AC093157.2', 'ENSG00000235821_IFITM4P', 'ENSG00000235823_OLMALINC', 'ENSG00000235831_BHLHE40-AS1', 'ENSG00000235833_AC017099.1', 'ENSG00000235843_AL390961.2', 'ENSG00000235852_AC005540.1', 'ENSG00000235859_AC006978.1', 'ENSG00000235863_B3GALT4', 'ENSG00000235865_GSN-AS1', 'ENSG00000235888_AF064858.1', 'ENSG00000235897_TM4SF19-AS1', 'ENSG00000235903_CPB2-AS1', 'ENSG00000235908_RHOA-IT1', 'ENSG00000235912_AL031729.1', 'ENSG00000235916_AC233279.1', 'ENSG00000235919_ASH1L-AS1', 'ENSG00000235920_AC073109.1', 'ENSG00000235927_NEXN-AS1', 'ENSG00000235931_LINC01553', 'ENSG00000235944_ZNF815P', 'ENSG00000235945_AC002543.1', 'ENSG00000235947_EGOT', 'ENSG00000235954_TTC28-AS1', 'ENSG00000235957_COX7CP1', 'ENSG00000235961_PNMA6A', 'ENSG00000235978_AC018816.1', 'ENSG00000235989_MORC2-AS1', 'ENSG00000236008_LINC01814', 'ENSG00000236017_ASMTL-AS1', 'ENSG00000236018_AC004898.1', 'ENSG00000236021_AL359265.3', 'ENSG00000236048_AC013470.3', 'ENSG00000236051_MYCBP2-AS1', 'ENSG00000236058_RPL17P36', 'ENSG00000236060_HSPB1P1', 'ENSG00000236065_AL020995.1', 'ENSG00000236081_ELFN1-AS1', 'ENSG00000236086_HMGN2P28', 'ENSG00000236088_COX10-AS1', 'ENSG00000236104_ZBTB22', 'ENSG00000236107_AC010127.1', 'ENSG00000236137_AL445231.1', 'ENSG00000236140_AC245014.1', 'ENSG00000236144_TMEM147-AS1', 'ENSG00000236152_MRPS36P1', 'ENSG00000236154_AL450311.2', 'ENSG00000236155_AL355877.1', 'ENSG00000236184_TCEA1P4', 'ENSG00000236194_AC099811.1', 'ENSG00000236200_KDM4A-AS1', 'ENSG00000236204_LINC01376', 'ENSG00000236206_AL356441.1', 'ENSG00000236255_AC009404.1', 'ENSG00000236264_RPL26P30', 'ENSG00000236266_Z98884.1', 'ENSG00000236274_AC004865.1', 'ENSG00000236278_PEBP1P3', 'ENSG00000236287_ZBED5', 'ENSG00000236296_GUSBP5', 'ENSG00000236297_AC048351.1', 'ENSG00000236304_AP001189.1', 'ENSG00000236305_SLC12A9-AS1', 'ENSG00000236307_EEF1E1P1', 'ENSG00000236312_RPL34P34', 'ENSG00000236320_SLFN14', 'ENSG00000236325_AC005300.1', 'ENSG00000236330_RPL5P9', 'ENSG00000236333_TRHDE-AS1', 'ENSG00000236334_PPIAL4G', 'ENSG00000236337_FMR1-IT1', 'ENSG00000236352_AC005220.1', 'ENSG00000236377_AC084809.2', 'ENSG00000236383_CCDC200', 'ENSG00000236388_ITCH-AS1', 'ENSG00000236397_DDX11L2', 'ENSG00000236404_VLDLR-AS1', 'ENSG00000236409_NRADDP', 'ENSG00000236411_NDUFAF4P3', 'ENSG00000236423_LINC01134', 'ENSG00000236432_AC097662.1', 'ENSG00000236438_FAM157A', 'ENSG00000236439_AC099336.2', 'ENSG00000236444_UBE2L5', 'ENSG00000236456_AL035458.1', 'ENSG00000236478_AC012513.2', 'ENSG00000236489_AC133473.1', 'ENSG00000236493_EIF2S2P3', 'ENSG00000236498_AC107081.2', 'ENSG00000236499_LINC00896', 'ENSG00000236514_AL135791.1', 'ENSG00000236519_LINC01424', 'ENSG00000236523_NPM1P40', 'ENSG00000236526_AL035448.1', 'ENSG00000236528_AL033528.2', 'ENSG00000236533_AC009413.1', 'ENSG00000236534_H3F3BP1', 'ENSG00000236535_RC3H1-IT1', 'ENSG00000236539_HNRNPA1P54', 'ENSG00000236540_AC006547.1', 'ENSG00000236548_RNF217-AS1', 'ENSG00000236552_RPL13AP5', 'ENSG00000236576_AC241520.1', 'ENSG00000236577_SNRPGP14', 'ENSG00000236609_ZNF853', 'ENSG00000236617_AC127070.1', 'ENSG00000236618_PITPNA-AS1', 'ENSG00000236636_AL627308.2', 'ENSG00000236670_KRT18P5', 'ENSG00000236698_EIF1AXP1', 'ENSG00000236709_DAPK1-IT1', 'ENSG00000236723_AL606760.2', 'ENSG00000236735_RPL31P63', 'ENSG00000236753_MKLN1-AS', 'ENSG00000236756_DNAJC9-AS1', 'ENSG00000236762_RPL19P16', 'ENSG00000236764_COX7A2P2', 'ENSG00000236778_INTS6-AS1', 'ENSG00000236782_AL391650.1', 'ENSG00000236792_AL513175.2', 'ENSG00000236801_RPL24P8', 'ENSG00000236804_RPS3AP12', 'ENSG00000236809_SNX25P1', 'ENSG00000236810_ELOA-AS1', 'ENSG00000236814_AC046176.1', 'ENSG00000236824_BCYRN1', 'ENSG00000236829_Z97634.1', 'ENSG00000236830_CBR3-AS1', 'ENSG00000236852_BX322784.1', 'ENSG00000236859_NIFK-AS1', 'ENSG00000236871_LINC00106', 'ENSG00000236901_MIR600HG', 'ENSG00000236911_AL137789.1', 'ENSG00000236913_AC025750.2', 'ENSG00000236928_AC008267.4', 'ENSG00000236935_AP003774.4', 'ENSG00000236939_BAALC-AS2', 'ENSG00000236940_AL589765.3', 'ENSG00000236947_AL139412.1', 'ENSG00000236977_ANKRD44-IT1', 'ENSG00000236986_AL157938.2', 'ENSG00000236991_EDRF1-AS1', 'ENSG00000236992_RPL12P12', 'ENSG00000237001_WASF3-AS1', 'ENSG00000237003_AC126124.2', 'ENSG00000237004_ZNRF2P1', 'ENSG00000237007_KRT18P52', 'ENSG00000237013_LINC01812', 'ENSG00000237015_AL031186.1', 'ENSG00000237017_AC245052.4', 'ENSG00000237036_ZEB1-AS1', 'ENSG00000237037_NDUFA6-DT', 'ENSG00000237054_PRMT5-AS1', 'ENSG00000237065_NANOGP4', 'ENSG00000237073_AL162727.2', 'ENSG00000237080_EHMT2-AS1', 'ENSG00000237082_COX5BP6', 'ENSG00000237094_AL732372.2', 'ENSG00000237106_FABP5P15', 'ENSG00000237118_CYP2F2P', 'ENSG00000237125_HAND2-AS1', 'ENSG00000237135_DDX10P1', 'ENSG00000237149_ZNF503-AS2', 'ENSG00000237161_AC068446.1', 'ENSG00000237166_LINC01792', 'ENSG00000237170_RPS7P15', 'ENSG00000237172_B3GNT9', 'ENSG00000237181_AC147651.3', 'ENSG00000237186_AC092418.1', 'ENSG00000237190_CDKN2AIPNL', 'ENSG00000237214_AL080243.2', 'ENSG00000237232_ZNF295-AS1', 'ENSG00000237238_BMS1P10', 'ENSG00000237248_LINC00987', 'ENSG00000237273_RSL24D1P8', 'ENSG00000237278_RLIMP2', 'ENSG00000237280_AL136982.3', 'ENSG00000237296_SMG1P1', 'ENSG00000237298_TTN-AS1', 'ENSG00000237300_MTCO1P19', 'ENSG00000237301_AL121992.1', 'ENSG00000237307_SRRM1P3', 'ENSG00000237310_GS1-124K5.4', 'ENSG00000237311_AL034397.2', 'ENSG00000237321_AL354936.1', 'ENSG00000237343_AC246785.3', 'ENSG00000237349_BX679664.2', 'ENSG00000237350_CDC42P6', 'ENSG00000237352_LINC01358', 'ENSG00000237356_AL365295.1', 'ENSG00000237357_BX088651.4', 'ENSG00000237359_AL354977.2', 'ENSG00000237372_UNQ6494', 'ENSG00000237373_BRWD1-IT1', 'ENSG00000237382_RPL21P121', 'ENSG00000237399_PITRM1-AS1', 'ENSG00000237402_CAMTA1-IT1', 'ENSG00000237406_NDUFA9P1', 'ENSG00000237409_AL513302.1', 'ENSG00000237424_FOXD2-AS1', 'ENSG00000237429_BX293535.1', 'ENSG00000237436_CAMTA1-DT', 'ENSG00000237438_CECR7', 'ENSG00000237440_ZNF737', 'ENSG00000237441_RGL2', 'ENSG00000237476_LINC01637', 'ENSG00000237481_AL117350.1', 'ENSG00000237484_LINC01684', 'ENSG00000237489_C10orf143', 'ENSG00000237491_AL669831.5', 'ENSG00000237493_AC034102.1', 'ENSG00000237499_AL357060.1', 'ENSG00000237505_PKN2-AS1', 'ENSG00000237512_UNC5B-AS1', 'ENSG00000237513_AC007384.1', 'ENSG00000237520_AL391832.1', 'ENSG00000237522_NONOP2', 'ENSG00000237531_AL672277.1', 'ENSG00000237541_HLA-DQA2', 'ENSG00000237550_RPL9P9', 'ENSG00000237551_AC096775.1', 'ENSG00000237575_PYY2', 'ENSG00000237576_LINC01888', 'ENSG00000237594_AP000251.1', 'ENSG00000237595_AL161937.2', 'ENSG00000237605_AL591846.2', 'ENSG00000237628_MTCO2P19', 'ENSG00000237637_FRY-AS1', 'ENSG00000237649_KIFC1', 'ENSG00000237651_C2orf74', 'ENSG00000237668_RPS15AP38', 'ENSG00000237676_RPL30P4', 'ENSG00000237686_AL109615.3', 'ENSG00000237693_IRGM', 'ENSG00000237718_AC009095.1', 'ENSG00000237719_Z95152.1', 'ENSG00000237721_AF064858.3', 'ENSG00000237729_AC002075.2', 'ENSG00000237732_AC010980.1', 'ENSG00000237748_UQCRBP1', 'ENSG00000237749_AL034379.1', 'ENSG00000237753_FLJ42351', 'ENSG00000237765_FAM200B', 'ENSG00000237773_AC073332.1', 'ENSG00000237775_DDR1-DT', 'ENSG00000237788_AL162615.1', 'ENSG00000237797_AL161935.3', 'ENSG00000237803_LINC00211', 'ENSG00000237818_RPS3AP29', 'ENSG00000237819_AC002454.1', 'ENSG00000237821_AC083873.1', 'ENSG00000237827_RPS15AP29', 'ENSG00000237840_FAM21FP', 'ENSG00000237842_AL157713.1', 'ENSG00000237846_AL773545.3', 'ENSG00000237851_AL023584.2', 'ENSG00000237852_AC119800.1', 'ENSG00000237853_NFIA-AS1', 'ENSG00000237854_LINC00674', 'ENSG00000237877_LINC01473', 'ENSG00000237883_DGUOK-AS1', 'ENSG00000237886_NALT1', 'ENSG00000237887_RPL23AP32', 'ENSG00000237892_KLF7-IT1', 'ENSG00000237903_AC004000.1', 'ENSG00000237940_LINC01238', 'ENSG00000237943_PRKCQ-AS1', 'ENSG00000237945_LINC00649', 'ENSG00000237950_AL357079.1', 'ENSG00000237953_AC013267.1', 'ENSG00000237973_MTCO1P12', 'ENSG00000237975_FLG-AS1', 'ENSG00000237976_AL391069.3', 'ENSG00000237984_PTENP1', 'ENSG00000237988_OR2I1P', 'ENSG00000237991_RPL35P1', 'ENSG00000238000_AC116347.1', 'ENSG00000238005_AL391832.2', 'ENSG00000238009_AL627309.1', 'ENSG00000238018_AC093110.1', 'ENSG00000238045_AC009133.1', 'ENSG00000238055_Z98742.3', 'ENSG00000238057_ZEB2-AS1', 'ENSG00000238058_AL355574.1', 'ENSG00000238061_AL356273.2', 'ENSG00000238082_AC009948.2', 'ENSG00000238083_LRRC37A2', 'ENSG00000238085_AL590682.1', 'ENSG00000238098_ABCA17P', 'ENSG00000238099_LINC01625', 'ENSG00000238103_RPL9P7', 'ENSG00000238105_GOLGA2P5', 'ENSG00000238107_AC245100.6', 'ENSG00000238113_LINC01410', 'ENSG00000238120_LINC01589', 'ENSG00000238121_LINC00426', 'ENSG00000238123_MID1IP1-AS1', 'ENSG00000238140_AC104170.2', 'ENSG00000238142_BX284668.5', 'ENSG00000238164_TNFRSF14-AS1', 'ENSG00000238168_AC137055.1', 'ENSG00000238171_AC068196.1', 'ENSG00000238172_RPS2P35', 'ENSG00000238184_CD81-AS1', 'ENSG00000238197_PAXBP1-AS1', 'ENSG00000238198_AL357055.3', 'ENSG00000238225_CRIP1P2', 'ENSG00000238227_TMEM250', 'ENSG00000238228_OR7E7P', 'ENSG00000238241_CCR12P', 'ENSG00000238243_OR2W3', 'ENSG00000238251_AL133477.1', 'ENSG00000238260_AL513320.1', 'ENSG00000238269_PAGE2B', 'ENSG00000238287_AL603839.3', 'ENSG00000238390_RF01241', 'ENSG00000238754_SCARNA18B', 'ENSG00000238825_RNU1-13P', 'ENSG00000239005_RF01225', 'ENSG00000239039_SNORD13', 'ENSG00000239040_RF00019', 'ENSG00000239183_SNORA84', 'ENSG00000239213_NCK1-DT', 'ENSG00000239218_RPS20P22', 'ENSG00000239221_RN7SL442P', 'ENSG00000239246_AC008026.1', 'ENSG00000239247_RN7SL589P', 'ENSG00000239254_AC009220.2', 'ENSG00000239263_RBM43P1', 'ENSG00000239264_TXNDC5', 'ENSG00000239272_RPL21P10', 'ENSG00000239280_AC108693.1', 'ENSG00000239282_CASTOR1', 'ENSG00000239291_AC002558.1', 'ENSG00000239305_RNF103', 'ENSG00000239306_RBM14', 'ENSG00000239317_AC091959.1', 'ENSG00000239322_ATP6V1B1-AS1', 'ENSG00000239332_LINC01119', 'ENSG00000239344_AC090686.1', 'ENSG00000239382_ALKBH6', 'ENSG00000239396_RN7SL414P', 'ENSG00000239407_Z68871.1', 'ENSG00000239415_AP001469.3', 'ENSG00000239419_RN7SL535P', 'ENSG00000239445_ST3GAL6-AS1', 'ENSG00000239453_SIDT1-AS1', 'ENSG00000239467_AC007405.3', 'ENSG00000239470_AC011979.1', 'ENSG00000239481_RPS3AP41', 'ENSG00000239483_RPS15AP16', 'ENSG00000239486_AC091390.3', 'ENSG00000239494_RN7SL333P', 'ENSG00000239521_CASTOR3', 'ENSG00000239523_MYLK-AS1', 'ENSG00000239528_RPS14P8', 'ENSG00000239559_RPL37P2', 'ENSG00000239569_KMT2E-AS1', 'ENSG00000239570_SETP11', 'ENSG00000239572_AC108749.1', 'ENSG00000239577_RN7SL388P', 'ENSG00000239593_AL513122.2', 'ENSG00000239602_AC091959.2', 'ENSG00000239617_AC073610.1', 'ENSG00000239636_AC004865.2', 'ENSG00000239653_PSMD6-AS2', 'ENSG00000239665_AL157392.3', 'ENSG00000239672_NME1', 'ENSG00000239697_TNFSF12', 'ENSG00000239704_CDRT4', 'ENSG00000239713_APOBEC3G', 'ENSG00000239726_RN7SL688P', 'ENSG00000239763_AC009120.1', 'ENSG00000239779_WBP1', 'ENSG00000239789_MRPS17', 'ENSG00000239791_AC002310.2', 'ENSG00000239801_DENND6A-AS1', 'ENSG00000239809_AC008026.2', 'ENSG00000239827_SUGT1P3', 'ENSG00000239857_GET4', 'ENSG00000239872_RPL35AP19', 'ENSG00000239883_PARGP1', 'ENSG00000239884_RN7SL608P', 'ENSG00000239887_C1orf226', 'ENSG00000239899_RN7SL674P', 'ENSG00000239900_ADSL', 'ENSG00000239910_RN7SL530P', 'ENSG00000239911_PRKAG2-AS1', 'ENSG00000239917_RPS10P16', 'ENSG00000239920_AC104389.4', 'ENSG00000239941_AC108718.1', 'ENSG00000239942_RN7SL394P', 'ENSG00000239948_RN7SL368P', 'ENSG00000239953_RN7SL273P', 'ENSG00000239969_AC091390.4', 'ENSG00000239998_LILRA2', 'ENSG00000240005_AC106047.1', 'ENSG00000240024_LINC00888', 'ENSG00000240036_AC104563.1', 'ENSG00000240038_AMY2B', 'ENSG00000240053_LY6G5B', 'ENSG00000240057_AC078785.1', 'ENSG00000240065_PSMB9', 'ENSG00000240098_RN7SL351P', 'ENSG00000240106_RN7SL146P', 'ENSG00000240132_ETF1P2', 'ENSG00000240137_ERICH6-AS1', 'ENSG00000240143_AL023653.1', 'ENSG00000240160_RN7SL263P', 'ENSG00000240167_RPS7P7', 'ENSG00000240204_SMKR1', 'ENSG00000240207_AC080013.1', 'ENSG00000240211_AC092849.1', 'ENSG00000240225_ZNF542P', 'ENSG00000240230_COX19', 'ENSG00000240233_RN7SL587P', 'ENSG00000240280_TCAM1P', 'ENSG00000240288_GHRLOS', 'ENSG00000240291_AL450384.2', 'ENSG00000240303_ACAD11', 'ENSG00000240322_RN7SL481P', 'ENSG00000240327_RN7SL93P', 'ENSG00000240328_AC091805.1', 'ENSG00000240342_RPS2P5', 'ENSG00000240344_PPIL3', 'ENSG00000240350_AC017002.3', 'ENSG00000240356_RPL23AP7', 'ENSG00000240370_RPL13P5', 'ENSG00000240371_RPS4XP13', 'ENSG00000240376_AC010343.1', 'ENSG00000240385_RPS29P20', 'ENSG00000240392_RPL9P3', 'ENSG00000240399_AC004801.2', 'ENSG00000240401_AC012358.3', 'ENSG00000240405_SAMMSON', 'ENSG00000240418_AC020917.1', 'ENSG00000240429_LRRFIP1P1', 'ENSG00000240454_RPL39P26', 'ENSG00000240457_RN7SL472P', 'ENSG00000240463_RPS19P3', 'ENSG00000240477_AC022494.1', 'ENSG00000240489_SETP14', 'ENSG00000240497_AC092919.1', 'ENSG00000240509_RPL34P18', 'ENSG00000240521_AC092979.1', 'ENSG00000240522_RPL7AP10', 'ENSG00000240531_RPL21P123', 'ENSG00000240541_TM4SF1-AS1', 'ENSG00000240563_L1TD1', 'ENSG00000240583_AQP1', 'ENSG00000240591_AL096701.3', 'ENSG00000240616_RPS6P25', 'ENSG00000240634_AC145285.1', 'ENSG00000240652_AP001024.1', 'ENSG00000240674_AC106872.2', 'ENSG00000240682_ISY1', 'ENSG00000240694_PNMA2', 'ENSG00000240695_AC117382.1', 'ENSG00000240710_AL512306.3', 'ENSG00000240718_RN7SL851P', 'ENSG00000240720_LRRD1', 'ENSG00000240729_AC146507.2', 'ENSG00000240747_KRBOX1', 'ENSG00000240750_RN7SL559P', 'ENSG00000240751_AC026348.1', 'ENSG00000240767_RN7SL288P', 'ENSG00000240771_ARHGEF25', 'ENSG00000240808_AC126389.1', 'ENSG00000240823_RN7SL23P', 'ENSG00000240828_RPL21P4', 'ENSG00000240849_TMEM189', 'ENSG00000240857_RDH14', 'ENSG00000240859_AC093627.4', 'ENSG00000240870_RPL19P14', 'ENSG00000240875_LINC00886', 'ENSG00000240877_RN7SL521P', 'ENSG00000240889_NDUFB2-AS1', 'ENSG00000240891_PLCXD2', 'ENSG00000240898_AC132942.1', 'ENSG00000240905_RN7SL798P', 'ENSG00000240914_RPL15P2', 'ENSG00000240919_AC022034.2', 'ENSG00000240950_AC021074.1', 'ENSG00000240966_RN7SL681P', 'ENSG00000240972_MIF', 'ENSG00000240990_HOXA11-AS', 'ENSG00000241007_SEPT7P6', 'ENSG00000241015_TPM3P9', 'ENSG00000241030_RPL29P24', 'ENSG00000241058_NSUN6', 'ENSG00000241061_RPL5P1', 'ENSG00000241067_RPL17P40', 'ENSG00000241081_RPL22P2', 'ENSG00000241095_CYP51A1P1', 'ENSG00000241106_HLA-DOB', 'ENSG00000241112_RPL29P14', 'ENSG00000241120_HMGN1P8', 'ENSG00000241127_YAE1', 'ENSG00000241155_ARHGAP31-AS1', 'ENSG00000241157_AC104763.1', 'ENSG00000241163_LINC00877', 'ENSG00000241170_AP001992.1', 'ENSG00000241175_RN7SL494P', 'ENSG00000241180_AL645608.4', 'ENSG00000241187_AC008379.1', 'ENSG00000241217_RN7SL809P', 'ENSG00000241230_RN7SL801P', 'ENSG00000241243_RN7SL629P', 'ENSG00000241255_AL136126.1', 'ENSG00000241258_CRCP', 'ENSG00000241269_AC093620.1', 'ENSG00000241282_RPL34P33', 'ENSG00000241288_AC092902.2', 'ENSG00000241293_PPATP1', 'ENSG00000241295_ZBTB20-AS2', 'ENSG00000241322_CDRT1', 'ENSG00000241333_RN7SL385P', 'ENSG00000241343_RPL36A', 'ENSG00000241352_AC007688.1', 'ENSG00000241360_PDXP', 'ENSG00000241361_SLC25A24P1', 'ENSG00000241370_RPP21', 'ENSG00000241388_HNF1A-AS1', 'ENSG00000241399_CD302', 'ENSG00000241404_EGFL8', 'ENSG00000241420_RN7SL505P', 'ENSG00000241429_EEF1A1P25', 'ENSG00000241431_RPL37P6', 'ENSG00000241461_RN7SL182P', 'ENSG00000241468_ATP5MF', 'ENSG00000241472_PTPRG-AS1', 'ENSG00000241484_ARHGAP8', 'ENSG00000241494_AL355032.1', 'ENSG00000241506_PSMC1P1', 'ENSG00000241511_RPS15AP24', 'ENSG00000241520_AC098820.3', 'ENSG00000241525_AC141424.1', 'ENSG00000241532_AGGF1P3', 'ENSG00000241535_CBX5P1', 'ENSG00000241537_AC134050.1', 'ENSG00000241549_GUSBP2', 'ENSG00000241553_ARPC4', 'ENSG00000241556_AC018475.1', 'ENSG00000241563_CORT', 'ENSG00000241587_RN7SL482P', 'ENSG00000241612_AC114728.1', 'ENSG00000241634_AC069499.1', 'ENSG00000241640_AC092757.1', 'ENSG00000241641_RPS23P6', 'ENSG00000241666_AL031733.2', 'ENSG00000241678_AC091564.1', 'ENSG00000241680_RPL31P49', 'ENSG00000241685_ARPC1A', 'ENSG00000241697_TMEFF1', 'ENSG00000241728_AP001062.3', 'ENSG00000241735_FABP5P3', 'ENSG00000241738_ZNF90P1', 'ENSG00000241741_RPL7AP30', 'ENSG00000241743_XACT', 'ENSG00000241749_RPSAP52', 'ENSG00000241764_AC002467.1', 'ENSG00000241769_LINC00893', 'ENSG00000241772_AC092620.2', 'ENSG00000241782_AP002812.1', 'ENSG00000241837_ATP5PO', 'ENSG00000241839_PLEKHO2', 'ENSG00000241852_C8orf58', 'ENSG00000241859_ANOS2P', 'ENSG00000241860_AL627309.5', 'ENSG00000241878_PISD', 'ENSG00000241886_AC112496.1', 'ENSG00000241911_TRBVB', 'ENSG00000241923_RPL14P3', 'ENSG00000241939_RN7SL517P', 'ENSG00000241941_RPL32P26', 'ENSG00000241945_PWP2', 'ENSG00000241959_RN7SL76P', 'ENSG00000241962_AC079447.1', 'ENSG00000241963_RN7SL655P', 'ENSG00000241973_PI4KA', 'ENSG00000241975_ELOCP19', 'ENSG00000241978_AKAP2', 'ENSG00000241983_RN7SL566P', 'ENSG00000241990_PRR34-AS1', 'ENSG00000241993_RPL38P1', 'ENSG00000242028_HYPK', 'ENSG00000242048_AC093583.1', 'ENSG00000242071_RPL7AP6', 'ENSG00000242085_RPS20P33', 'ENSG00000242086_MUC20-OT1', 'ENSG00000242094_FOXP1-IT1', 'ENSG00000242100_RPL9P32', 'ENSG00000242110_AMACR', 'ENSG00000242114_MTFP1', 'ENSG00000242125_SNHG3', 'ENSG00000242134_RPL5P13', 'ENSG00000242142_SERBP1P3', 'ENSG00000242154_AC004884.2', 'ENSG00000242163_AL121769.1', 'ENSG00000242170_RN7SL329P', 'ENSG00000242173_ARHGDIG', 'ENSG00000242182_RN7SL745P', 'ENSG00000242193_CRYZL2P', 'ENSG00000242197_AC098869.1', 'ENSG00000242220_TCP10L', 'ENSG00000242241_RN7SL306P', 'ENSG00000242247_ARFGAP3', 'ENSG00000242251_RN7SL20P', 'ENSG00000242252_BGLAP', 'ENSG00000242256_RN7SL57P', 'ENSG00000242258_LINC00996', 'ENSG00000242259_C22orf39', 'ENSG00000242261_AC018635.1', 'ENSG00000242262_AC092597.1', 'ENSG00000242265_PEG10', 'ENSG00000242268_LINC02082', 'ENSG00000242282_AC108488.1', 'ENSG00000242285_RPL6P8', 'ENSG00000242294_STAG3L5P', 'ENSG00000242299_AC073861.1', 'ENSG00000242318_AC058823.1', 'ENSG00000242325_RPS12P31', 'ENSG00000242327_AC023906.1', 'ENSG00000242338_BMS1P4', 'ENSG00000242358_RPS21P4', 'ENSG00000242372_EIF6', 'ENSG00000242405_AC007537.2', 'ENSG00000242412_DBIL5P2', 'ENSG00000242445_RPL7AP11', 'ENSG00000242474_AC093627.5', 'ENSG00000242477_AC091429.1', 'ENSG00000242479_AC109992.1', 'ENSG00000242485_MRPL20', 'ENSG00000242488_AF107885.1', 'ENSG00000242493_RN7SL37P', 'ENSG00000242498_ARPIN', 'ENSG00000242516_LINC00960', 'ENSG00000242539_AC007620.2', 'ENSG00000242550_SERPINB10', 'ENSG00000242553_AP001432.1', 'ENSG00000242574_HLA-DMB', 'ENSG00000242588_AC108010.1', 'ENSG00000242600_MBL1P', 'ENSG00000242602_AC008953.1', 'ENSG00000242607_RPS3AP34', 'ENSG00000242612_DECR2', 'ENSG00000242615_AC022415.1', 'ENSG00000242616_GNG10', 'ENSG00000242622_AC092910.3', 'ENSG00000242670_RPL22P13', 'ENSG00000242683_RPL12P21', 'ENSG00000242689_CNTF', 'ENSG00000242692_RPS27AP1', 'ENSG00000242697_RPL5P12', 'ENSG00000242707_RN7SL362P', 'ENSG00000242715_CCDC169', 'ENSG00000242732_RTL5', 'ENSG00000242759_LINC00882', 'ENSG00000242767_ZBTB20-AS4', 'ENSG00000242779_ZNF702P', 'ENSG00000242781_LINC02050', 'ENSG00000242797_GLYCTK-AS1', 'ENSG00000242798_AC073842.2', 'ENSG00000242802_AP5Z1', 'ENSG00000242808_SOX2-OT', 'ENSG00000242810_MRPL42P6', 'ENSG00000242814_AC113398.1', 'ENSG00000242818_RN7SL846P', 'ENSG00000242829_RPS26P21', 'ENSG00000242852_ZNF709', 'ENSG00000242861_AL591895.1', 'ENSG00000242876_RN7SL812P', 'ENSG00000242882_RPL5P11', 'ENSG00000242889_RN7SL449P', 'ENSG00000242931_RPL7P49', 'ENSG00000242951_AC007182.2', 'ENSG00000242960_FTH1P23', 'ENSG00000242970_AC068522.1', 'ENSG00000242971_RN7SL233P', 'ENSG00000242999_RN7SL239P', 'ENSG00000243004_AC005062.1', 'ENSG00000243014_PTMAP8', 'ENSG00000243029_RN7SL635P', 'ENSG00000243056_EIF4EBP3', 'ENSG00000243064_ABCC13', 'ENSG00000243069_ARHGEF26-AS1', 'ENSG00000243071_AC107032.1', 'ENSG00000243094_AC079203.1', 'ENSG00000243103_RN7SL452P', 'ENSG00000243107_AC000120.1', 'ENSG00000243147_MRPL33', 'ENSG00000243150_AC106707.1', 'ENSG00000243155_AL162431.2', 'ENSG00000243156_MICAL3', 'ENSG00000243175_RPSAP36', 'ENSG00000243176_AC092944.1', 'ENSG00000243181_AC087343.1', 'ENSG00000243193_AC006387.1', 'ENSG00000243199_AC115223.1', 'ENSG00000243224_AC006252.1', 'ENSG00000243227_RN7SL55P', 'ENSG00000243244_STON1', 'ENSG00000243260_RN7SL558P', 'ENSG00000243279_PRAF2', 'ENSG00000243280_AC093663.2', 'ENSG00000243284_VSIG8', 'ENSG00000243297_RPL31P61', 'ENSG00000243302_AC018638.4', 'ENSG00000243303_AC103987.1', 'ENSG00000243304_AC008494.1', 'ENSG00000243305_AC026347.1', 'ENSG00000243313_RN7SL285P', 'ENSG00000243314_AC106707.2', 'ENSG00000243317_STMP1', 'ENSG00000243323_PTPRVP', 'ENSG00000243333_RN7SL174P', 'ENSG00000243335_KCTD7', 'ENSG00000243353_RPS29P19', 'ENSG00000243364_EFNA4', 'ENSG00000243370_RN7SL775P', 'ENSG00000243398_RN7SL141P', 'ENSG00000243403_AC090543.2', 'ENSG00000243404_RPL35AP32', 'ENSG00000243406_MRPS31P5', 'ENSG00000243410_PSMD6-AS1', 'ENSG00000243414_TICAM2', 'ENSG00000243422_RPL23AP49', 'ENSG00000243431_RPL5P30', 'ENSG00000243437_RN7SL370P', 'ENSG00000243445_AC106820.1', 'ENSG00000243449_C4orf48', 'ENSG00000243468_INGX', 'ENSG00000243477_NAA80', 'ENSG00000243478_AOX2P', 'ENSG00000243504_RPS23P1', 'ENSG00000243508_AC108688.1', 'ENSG00000243517_AC024940.2', 'ENSG00000243532_RN7SL19P', 'ENSG00000243538_RPS26P28', 'ENSG00000243544_RN7SL172P', 'ENSG00000243547_HNRNPKP4', 'ENSG00000243554_AC004967.1', 'ENSG00000243560_RN7SL364P', 'ENSG00000243568_AC020779.1', 'ENSG00000243609_RPS2P44', 'ENSG00000243642_RN7SL526P', 'ENSG00000243646_IL10RB', 'ENSG00000243650_RN7SL834P', 'ENSG00000243660_ZNF487', 'ENSG00000243667_WDR92', 'ENSG00000243678_NME2', 'ENSG00000243679_AC018638.5', 'ENSG00000243680_RPL37P23', 'ENSG00000243696_AC006254.1', 'ENSG00000243701_DUBR', 'ENSG00000243702_RN7SL638P', 'ENSG00000243716_NPIPB5', 'ENSG00000243725_TTC4', 'ENSG00000243742_RPLP0P2', 'ENSG00000243749_TMEM35B', 'ENSG00000243753_HLA-L', 'ENSG00000243759_ST13P15', 'ENSG00000243762_AC006547.2', 'ENSG00000243770_RN7SL65P', 'ENSG00000243775_OSTCP1', 'ENSG00000243779_AP001086.1', 'ENSG00000243806_RPL7P18', 'ENSG00000243811_APOBEC3D', 'ENSG00000243819_RN7SL832P', 'ENSG00000243829_AC011495.1', 'ENSG00000243845_RN7SL30P', 'ENSG00000243847_RN7SL610P', 'ENSG00000243859_RPL5P17', 'ENSG00000243871_RN7SL487P', 'ENSG00000243883_RN7SL419P', 'ENSG00000243911_RN7SL430P', 'ENSG00000243926_TIPARP-AS1', 'ENSG00000243927_MRPS6', 'ENSG00000243943_ZNF512', 'ENSG00000243951_RN7SL308P', 'ENSG00000243954_RN7SL743P', 'ENSG00000243960_AL390195.1', 'ENSG00000243964_RPL23AP65', 'ENSG00000243970_PPIEL', 'ENSG00000243977_AC125604.1', 'ENSG00000243979_AC087752.1', 'ENSG00000243989_ACY1', 'ENSG00000244004_AC097493.2', 'ENSG00000244005_NFS1', 'ENSG00000244026_FAM86DP', 'ENSG00000244036_AC073320.1', 'ENSG00000244038_DDOST', 'ENSG00000244040_IL12A-AS1', 'ENSG00000244041_LINC01011', 'ENSG00000244045_TMEM199', 'ENSG00000244053_RPL13AP2', 'ENSG00000244055_AC007566.1', 'ENSG00000244060_RPS2P41', 'ENSG00000244073_RPS4XP6', 'ENSG00000244089_HMGB1P30', 'ENSG00000244115_DNAJC25-GNG10', 'ENSG00000244119_PDCL3P4', 'ENSG00000244131_RPSAP51', 'ENSG00000244165_P2RY11', 'ENSG00000244184_AC091153.3', 'ENSG00000244187_TMEM141', 'ENSG00000244192_AC113367.1', 'ENSG00000244198_AC004889.1', 'ENSG00000244219_TMEM225B', 'ENSG00000244229_RPL26P35', 'ENSG00000244232_RN7SL698P', 'ENSG00000244239_AC007009.1', 'ENSG00000244256_RN7SL130P', 'ENSG00000244267_RPL34P22', 'ENSG00000244273_PGBD4P1', 'ENSG00000244274_DBNDD2', 'ENSG00000244280_ECEL1P2', 'ENSG00000244300_GATA2-AS1', 'ENSG00000244301_AOX3P', 'ENSG00000244306_DUXAP10', 'ENSG00000244313_AC024293.1', 'ENSG00000244314_RN7SL36P', 'ENSG00000244327_AC109992.2', 'ENSG00000244331_AC008677.2', 'ENSG00000244332_AL138759.1', 'ENSG00000244346_AC092953.1', 'ENSG00000244363_RPL7P23', 'ENSG00000244378_RPS2P45', 'ENSG00000244389_RN7SL242P', 'ENSG00000244391_RN7SL330P', 'ENSG00000244398_AC116533.1', 'ENSG00000244405_ETV5', 'ENSG00000244414_CFHR1', 'ENSG00000244459_AC147067.1', 'ENSG00000244462_RBM12', 'ENSG00000244479_OR2A1-AS1', 'ENSG00000244480_AC005154.2', 'ENSG00000244490_RWDD4P1', 'ENSG00000244491_AL021707.5', 'ENSG00000244509_APOBEC3C', 'ENSG00000244513_AC109587.1', 'ENSG00000244535_AL049714.1', 'ENSG00000244556_ODCP', 'ENSG00000244558_KCNK15-AS1', 'ENSG00000244560_AC004890.2', 'ENSG00000244582_RPL21P120', 'ENSG00000244604_AC025518.1', 'ENSG00000244607_CCDC13', 'ENSG00000244617_ASPRV1', 'ENSG00000244620_AC246787.2', 'ENSG00000244625_MIATNB', 'ENSG00000244627_TPTEP2', 'ENSG00000244671_RN7SL280P', 'ENSG00000244675_AC108676.1', 'ENSG00000244682_FCGR2C', 'ENSG00000244687_UBE2V1', 'ENSG00000244692_RN7SL724P', 'ENSG00000244701_AC004918.1', 'ENSG00000244710_RN7SL47P', 'ENSG00000244716_BX679664.3', 'ENSG00000244720_AC055748.1', 'ENSG00000244733_AL132656.2', 'ENSG00000244734_HBB', 'ENSG00000244752_CRYBB2', 'ENSG00000244754_N4BP2L2', 'ENSG00000244879_GABPB1-AS1', 'ENSG00000244921_MTCYBP18', 'ENSG00000244926_ALKBH3-AS1', 'ENSG00000244952_AC123768.2', 'ENSG00000245008_AP001122.1', 'ENSG00000245017_LINC02453', 'ENSG00000245025_AC107959.1', 'ENSG00000245059_AC092718.1', 'ENSG00000245060_LINC00847', 'ENSG00000245067_IGFBP7-AS1', 'ENSG00000245080_MIR3150BHG', 'ENSG00000245105_A2M-AS1', 'ENSG00000245112_SMARCA5-AS1', 'ENSG00000245146_MALINC1', 'ENSG00000245148_ARAP1-AS2', 'ENSG00000245149_RNF139-AS1', 'ENSG00000245156_AP001107.1', 'ENSG00000245164_LINC00861', 'ENSG00000245213_AC105285.1', 'ENSG00000245248_USP2-AS1', 'ENSG00000245275_SAP30L-AS1', 'ENSG00000245281_AC124242.1', 'ENSG00000245293_AC096564.1', 'ENSG00000245311_ARNTL2-AS1', 'ENSG00000245317_AC008393.1', 'ENSG00000245330_AP005717.1', 'ENSG00000245468_LINC02447', 'ENSG00000245479_LINC01585', 'ENSG00000245498_AP000866.1', 'ENSG00000245522_AC026250.1', 'ENSG00000245532_NEAT1', 'ENSG00000245534_RORA-AS1', 'ENSG00000245552_AP000787.1', 'ENSG00000245556_SCAMP1-AS1', 'ENSG00000245571_FAM111A-DT', 'ENSG00000245573_BDNF-AS', 'ENSG00000245614_DDX11-AS1', 'ENSG00000245648_AC022075.1', 'ENSG00000245662_LINC02211', 'ENSG00000245667_AC006064.1', 'ENSG00000245680_ZNF585B', 'ENSG00000245685_FRG1-DT', 'ENSG00000245694_CRNDE', 'ENSG00000245748_AC097382.2', 'ENSG00000245750_DRAIC', 'ENSG00000245768_AC092378.1', 'ENSG00000245848_CEBPA', 'ENSG00000245849_RAD51-AS1', 'ENSG00000245857_GS1-24F4.2', 'ENSG00000245888_FLJ21408', 'ENSG00000245904_AC025164.1', 'ENSG00000245910_SNHG6', 'ENSG00000245937_LINC01184', 'ENSG00000245958_AC093752.1', 'ENSG00000245970_AP003352.1', 'ENSG00000245975_AC090515.2', 'ENSG00000246016_LINC01513', 'ENSG00000246067_RAB30-AS1', 'ENSG00000246082_NUDT16P1', 'ENSG00000246089_AC016065.1', 'ENSG00000246090_AP002026.1', 'ENSG00000246100_LINC00900', 'ENSG00000246145_RRS1-AS1', 'ENSG00000246174_KCTD21-AS1', 'ENSG00000246223_LINC01550', 'ENSG00000246225_AC006299.1', 'ENSG00000246250_AC087521.2', 'ENSG00000246263_UBR5-AS1', 'ENSG00000246273_SBF2-AS1', 'ENSG00000246283_AC090510.1', 'ENSG00000246308_AC116535.1', 'ENSG00000246323_AC113382.1', 'ENSG00000246334_PRR7-AS1', 'ENSG00000246339_EXTL3-AS1', 'ENSG00000246350_AL049543.1', 'ENSG00000246366_LACTB2-AS1', 'ENSG00000246422_AC008781.2', 'ENSG00000246451_AL049840.1', 'ENSG00000246465_AC138904.1', 'ENSG00000246477_AF131216.1', 'ENSG00000246526_LINC002481', 'ENSG00000246528_AC079089.1', 'ENSG00000246548_LINC02288', 'ENSG00000246560_AC018797.2', 'ENSG00000246582_AC100861.1', 'ENSG00000246596_AC139795.1', 'ENSG00000246627_CACNA1C-AS1', 'ENSG00000246662_LINC00535', 'ENSG00000246695_RASSF8-AS1', 'ENSG00000246705_H2AFJ', 'ENSG00000246731_MGC16275', 'ENSG00000246777_AC044802.1', 'ENSG00000246792_AC106038.1', 'ENSG00000246851_AL157938.3', 'ENSG00000246859_STARD4-AS1', 'ENSG00000246877_DNM1P35', 'ENSG00000246889_AP000487.1', 'ENSG00000246898_LINC00920', 'ENSG00000246922_UBAP1L', 'ENSG00000246982_Z84485.1', 'ENSG00000246985_SOCS2-AS1', 'ENSG00000247033_AC099508.1', 'ENSG00000247077_PGAM5', 'ENSG00000247081_BAALC-AS1', 'ENSG00000247092_SNHG10', 'ENSG00000247095_MIR210HG', 'ENSG00000247121_AC009126.1', 'ENSG00000247131_AC025263.1', 'ENSG00000247134_AC090204.1', 'ENSG00000247137_AP000873.2', 'ENSG00000247151_CSTF3-DT', 'ENSG00000247157_LINC01252', 'ENSG00000247240_UBL7-AS1', 'ENSG00000247271_ZBED5-AS1', 'ENSG00000247287_AL359220.1', 'ENSG00000247315_ZCCHC3', 'ENSG00000247317_LY6E-DT', 'ENSG00000247324_AC010547.1', 'ENSG00000247345_AC092343.1', 'ENSG00000247363_AC090061.1', 'ENSG00000247373_AC055713.1', 'ENSG00000247400_DNAJC3-DT', 'ENSG00000247498_GPRC5D-AS1', 'ENSG00000247516_MIR4458HG', 'ENSG00000247556_OIP5-AS1', 'ENSG00000247572_CKMT2-AS1', 'ENSG00000247595_SPTY2D1OS', 'ENSG00000247596_TWF2', 'ENSG00000247624_CPEB2-DT', 'ENSG00000247626_MARS2', 'ENSG00000247627_MTND4P12', 'ENSG00000247675_LRP4-AS1', 'ENSG00000247679_AC139795.2', 'ENSG00000247708_STX18-AS1', 'ENSG00000247728_AC091057.2', 'ENSG00000247735_AC120114.1', 'ENSG00000247746_USP51', 'ENSG00000247765_AC068446.2', 'ENSG00000247774_PCED1B-AS1', 'ENSG00000247775_SNCA-AS1', 'ENSG00000247796_AC008966.1', 'ENSG00000247828_TMEM161B-AS1', 'ENSG00000247853_AC006064.2', 'ENSG00000247877_AC021086.1', 'ENSG00000247903_AC024896.1', 'ENSG00000247934_AC022364.1', 'ENSG00000247950_SEC24B-AS1', 'ENSG00000247982_LINC00926', 'ENSG00000248008_NRAV', 'ENSG00000248015_AC005329.1', 'ENSG00000248019_FAM13A-AS1', 'ENSG00000248049_UBA6-AS1', 'ENSG00000248079_DPH6-DT', 'ENSG00000248092_NNT-AS1', 'ENSG00000248098_BCKDHA', 'ENSG00000248099_INSL3', 'ENSG00000248115_AC023154.1', 'ENSG00000248124_RRN3P1', 'ENSG00000248161_AC098487.1', 'ENSG00000248200_AC093770.1', 'ENSG00000248240_AC114956.1', 'ENSG00000248243_LINC02014', 'ENSG00000248275_TRIM52-AS1', 'ENSG00000248278_SUMO2P17', 'ENSG00000248283_CCNL2P1', 'ENSG00000248309_MEF2C-AS1', 'ENSG00000248318_AC104958.1', 'ENSG00000248323_LUCAT1', 'ENSG00000248333_CDK11B', 'ENSG00000248334_WHAMMP2', 'ENSG00000248360_LINC00504', 'ENSG00000248367_AC008610.1', 'ENSG00000248373_AC096577.1', 'ENSG00000248429_FAM198B-AS1', 'ENSG00000248445_SEMA6A-AS1', 'ENSG00000248455_LINC02217', 'ENSG00000248468_AC107027.1', 'ENSG00000248473_LINC01962', 'ENSG00000248476_BACH1-IT1', 'ENSG00000248483_POU5F2', 'ENSG00000248487_ABHD14A', 'ENSG00000248489_LINC02062', 'ENSG00000248503_AL356235.1', 'ENSG00000248508_SRP14-AS1', 'ENSG00000248527_MTATP6P1', 'ENSG00000248538_AC022784.1', 'ENSG00000248544_AC008676.1', 'ENSG00000248559_AC109454.2', 'ENSG00000248585_AC084024.2', 'ENSG00000248592_TMEM110-MUSTN1', 'ENSG00000248593_DSTNP2', 'ENSG00000248626_GAPDHP40', 'ENSG00000248632_AC106872.5', 'ENSG00000248636_AC002070.1', 'ENSG00000248643_RBM14-RBM4', 'ENSG00000248663_LINC00992', 'ENSG00000248664_AC010273.1', 'ENSG00000248668_OXCT1-AS1', 'ENSG00000248671_ALG1L9P', 'ENSG00000248697_TOX4P1', 'ENSG00000248703_LINC02415', 'ENSG00000248712_CCDC153', 'ENSG00000248714_AC091180.2', 'ENSG00000248734_AC008906.1', 'ENSG00000248774_AC097534.1', 'ENSG00000248787_AC092903.2', 'ENSG00000248791_AC010627.1', 'ENSG00000248794_AC026436.1', 'ENSG00000248803_AC092349.1', 'ENSG00000248840_AL645949.1', 'ENSG00000248858_FLJ46284', 'ENSG00000248863_AC097376.1', 'ENSG00000248866_USP46-AS1', 'ENSG00000248874_C5orf17', 'ENSG00000248881_AC010245.1', 'ENSG00000248885_AC118465.1', 'ENSG00000248890_HHIP-AS1', 'ENSG00000248905_FMN1', 'ENSG00000248925_HRAT5', 'ENSG00000248932_AC097103.2', 'ENSG00000248956_HMGB1P44', 'ENSG00000248968_AC012640.1', 'ENSG00000248971_KRT8P46', 'ENSG00000249014_HMGN2P4', 'ENSG00000249020_SNORA58', 'ENSG00000249042_AC008771.1', 'ENSG00000249057_MAST4-IT1', 'ENSG00000249068_AC008417.1', 'ENSG00000249069_LINC01033', 'ENSG00000249087_ZNF436-AS1', 'ENSG00000249115_HAUS5', 'ENSG00000249119_MTND6P4', 'ENSG00000249129_SUDS3P1', 'ENSG00000249141_AL159163.1', 'ENSG00000249142_AC074134.1', 'ENSG00000249193_HSPD1P5', 'ENSG00000249212_ATP1B1P1', 'ENSG00000249222_ATP5MGL', 'ENSG00000249242_TMEM150C', 'ENSG00000249249_AC010226.1', 'ENSG00000249258_AC079193.2', 'ENSG00000249267_LINC00939', 'ENSG00000249274_PDLIM1P4', 'ENSG00000249307_LINC01088', 'ENSG00000249348_UGDH-AS1', 'ENSG00000249353_NPM1P27', 'ENSG00000249359_AC093274.1', 'ENSG00000249375_CASC11', 'ENSG00000249437_NAIP', 'ENSG00000249459_ZNF286B', 'ENSG00000249464_LINC01091', 'ENSG00000249471_ZNF324B', 'ENSG00000249476_AC008467.1', 'ENSG00000249483_AC026726.1', 'ENSG00000249485_RBBP4P1', 'ENSG00000249494_AC008629.1', 'ENSG00000249502_AC006160.1', 'ENSG00000249565_SERBP1P5', 'ENSG00000249572_AC034231.1', 'ENSG00000249592_AC139887.2', 'ENSG00000249593_AC011405.1', 'ENSG00000249604_AC096564.2', 'ENSG00000249635_AC109361.1', 'ENSG00000249637_AC008438.1', 'ENSG00000249639_AC022092.1', 'ENSG00000249649_MRPS33P2', 'ENSG00000249655_AC008434.1', 'ENSG00000249669_CARMN', 'ENSG00000249673_NOP14-AS1', 'ENSG00000249684_AC106795.2', 'ENSG00000249690_AC110813.1', 'ENSG00000249700_SRD5A3-AS1', 'ENSG00000249709_ZNF564', 'ENSG00000249713_AC026725.1', 'ENSG00000249736_LINC02242', 'ENSG00000249741_AC093890.1', 'ENSG00000249751_ECSCR', 'ENSG00000249753_AC084357.2', 'ENSG00000249784_SCARNA22', 'ENSG00000249786_EAF1-AS1', 'ENSG00000249790_AC092490.1', 'ENSG00000249797_LINC02147', 'ENSG00000249846_LINC02021', 'ENSG00000249855_EEF1A1P19', 'ENSG00000249859_PVT1', 'ENSG00000249863_AC021106.1', 'ENSG00000249876_AC010285.2', 'ENSG00000249898_MCPH1-AS1', 'ENSG00000249915_PDCD6', 'ENSG00000249921_AC034207.1', 'ENSG00000249931_GOLGA8K', 'ENSG00000249936_RAC1P2', 'ENSG00000249987_RPS4XP20', 'ENSG00000249992_TMEM158', 'ENSG00000250011_HMGB1P3', 'ENSG00000250030_AC104806.1', 'ENSG00000250031_AC009927.1', 'ENSG00000250033_SLC7A11-AS1', 'ENSG00000250057_AC114781.2', 'ENSG00000250067_YJEFN3', 'ENSG00000250069_AC011379.1', 'ENSG00000250072_SH3TC2-DT', 'ENSG00000250073_AP000866.2', 'ENSG00000250075_AC104806.2', 'ENSG00000250081_AC025176.1', 'ENSG00000250091_DNAH10OS', 'ENSG00000250116_AC018682.1', 'ENSG00000250125_LINC02232', 'ENSG00000250131_AC078881.1', 'ENSG00000250132_AC004803.1', 'ENSG00000250138_AC139495.3', 'ENSG00000250155_AC008957.1', 'ENSG00000250159_AC106791.1', 'ENSG00000250170_RASA2-IT1', 'ENSG00000250182_EEF1A1P13', 'ENSG00000250186_AC091180.3', 'ENSG00000250189_AC097504.1', 'ENSG00000250197_HMGN1P15', 'ENSG00000250220_AC053527.1', 'ENSG00000250222_AC008443.4', 'ENSG00000250240_AC008840.1', 'ENSG00000250251_PKD1P6', 'ENSG00000250280_AC026124.1', 'ENSG00000250290_NCAPGP1', 'ENSG00000250299_MRPS31P4', 'ENSG00000250303_AP002884.1', 'ENSG00000250305_TRMT9B', 'ENSG00000250312_ZNF718', 'ENSG00000250317_SMIM20', 'ENSG00000250321_AC079140.2', 'ENSG00000250326_AC104596.1', 'ENSG00000250329_KDELC1P1', 'ENSG00000250337_PURPL', 'ENSG00000250361_GYPB', 'ENSG00000250365_AL139353.2', 'ENSG00000250378_AC114296.1', 'ENSG00000250397_AP006623.1', 'ENSG00000250461_AC122718.1', 'ENSG00000250462_LRRC37BP1', 'ENSG00000250471_GMPSP1', 'ENSG00000250474_WBP1LP2', 'ENSG00000250479_CHCHD10', 'ENSG00000250486_FAM218A', 'ENSG00000250493_AP004147.1', 'ENSG00000250497_AC007126.1', 'ENSG00000250508_AP000808.1', 'ENSG00000250510_GPR162', 'ENSG00000250536_ABHD17AP3', 'ENSG00000250562_RPL38P4', 'ENSG00000250565_ATP6V1E2', 'ENSG00000250568_AC098591.2', 'ENSG00000250569_NTAN1P2', 'ENSG00000250571_GLI4', 'ENSG00000250615_AC008581.1', 'ENSG00000250616_AC012645.1', 'ENSG00000250641_LY6G6F-LY6G6D', 'ENSG00000250644_AC068580.4', 'ENSG00000250654_AC023794.3', 'ENSG00000250687_AC146944.2', 'ENSG00000250696_AC111000.4', 'ENSG00000250714_AC100861.2', 'ENSG00000250722_SELENOP', 'ENSG00000250731_TPM3P6', 'ENSG00000250734_AL391335.1', 'ENSG00000250739_LINC01262', 'ENSG00000250771_AC106865.1', 'ENSG00000250786_SNHG18', 'ENSG00000250790_AC127070.2', 'ENSG00000250802_ZBED3-AS1', 'ENSG00000250869_AC087359.1', 'ENSG00000250878_METTL21EP', 'ENSG00000250889_LINC01336', 'ENSG00000250899_AC125807.2', 'ENSG00000250900_AC008443.5', 'ENSG00000250903_GMDS-DT', 'ENSG00000250917_AL035458.2', 'ENSG00000250938_AC108866.1', 'ENSG00000250959_GLUD1P3', 'ENSG00000250966_AC023886.2', 'ENSG00000250980_AC113155.1', 'ENSG00000250988_SNHG21', 'ENSG00000250999_AC136604.3', 'ENSG00000251000_AC008592.3', 'ENSG00000251002_AC244502.1', 'ENSG00000251015_SLC25A30-AS1', 'ENSG00000251022_THAP9-AS1', 'ENSG00000251023_AC114980.1', 'ENSG00000251034_AC037459.2', 'ENSG00000251050_AC112184.1', 'ENSG00000251056_ANKRD20A17P', 'ENSG00000251072_LMNB1-DT', 'ENSG00000251073_NUDT19P5', 'ENSG00000251079_BMS1P2', 'ENSG00000251131_AC025171.3', 'ENSG00000251136_AF117829.1', 'ENSG00000251141_MRPS30-DT', 'ENSG00000251143_AP002490.1', 'ENSG00000251161_AC020661.1', 'ENSG00000251179_TMEM92-AS1', 'ENSG00000251192_ZNF674', 'ENSG00000251194_AL133330.1', 'ENSG00000251209_LINC00923', 'ENSG00000251215_GOLGA5P1', 'ENSG00000251221_LINC01337', 'ENSG00000251229_AL645924.2', 'ENSG00000251230_MIR3945HG', 'ENSG00000251247_ZNF345', 'ENSG00000251259_AC004069.1', 'ENSG00000251273_LINC02228', 'ENSG00000251287_ALG1L2', 'ENSG00000251301_LINC02384', 'ENSG00000251322_SHANK3', 'ENSG00000251323_AP003086.1', 'ENSG00000251330_AC114939.1', 'ENSG00000251348_HSPD1P11', 'ENSG00000251354_AC024451.1', 'ENSG00000251359_WWC2-AS2', 'ENSG00000251364_AC107884.1', 'ENSG00000251369_ZNF550', 'ENSG00000251381_LINC00958', 'ENSG00000251396_LINC01301', 'ENSG00000251411_AC093827.3', 'ENSG00000251417_AC145285.2', 'ENSG00000251429_AC098679.2', 'ENSG00000251432_AC108062.1', 'ENSG00000251442_LINC01094', 'ENSG00000251443_LINC02160', 'ENSG00000251455_AC092611.1', 'ENSG00000251474_RPL32P3', 'ENSG00000251503_CENPS-CORT', 'ENSG00000251555_AC096711.2', 'ENSG00000251562_MALAT1', 'ENSG00000251580_LINC02482', 'ENSG00000251595_ABCA11P', 'ENSG00000251600_AC139713.2', 'ENSG00000251602_AL928654.1', 'ENSG00000251615_AC104825.1', 'ENSG00000251634_AC145138.1', 'ENSG00000251636_LINC01218', 'ENSG00000251660_AC007036.3', 'ENSG00000251661_AC136475.1', 'ENSG00000251666_ZNF346-IT1', 'ENSG00000251667_BRCC3P1', 'ENSG00000251669_FAM86EP', 'ENSG00000251682_AC122718.2', 'ENSG00000251867_AC009812.1', 'ENSG00000251994_RNU2-27P', 'ENSG00000252010_SCARNA5', 'ENSG00000252118_RNU6ATAC39P', 'ENSG00000252122_RF00598', 'ENSG00000252198_RF00019', 'ENSG00000252213_SNORA74D', 'ENSG00000252305_RF00090', 'ENSG00000252311_RNU1-103P', 'ENSG00000252316_RNY4', 'ENSG00000252361_RNU6-118P', 'ENSG00000252473_RF00272', 'ENSG00000252498_RNU6-1016P', 'ENSG00000252657_RF00156', 'ENSG00000252690_AC105339.2', 'ENSG00000253058_RNA5SP437', 'ENSG00000253106_AC090198.1', 'ENSG00000253133_AC009630.1', 'ENSG00000253140_AC026904.1', 'ENSG00000253154_AC100801.1', 'ENSG00000253159_PCDHGA12', 'ENSG00000253174_AC009630.2', 'ENSG00000253180_AC104986.1', 'ENSG00000253187_HOXA10-AS', 'ENSG00000253190_AC084082.1', 'ENSG00000253194_AL137009.1', 'ENSG00000253200_AC037459.3', 'ENSG00000253203_GUSBP3', 'ENSG00000253210_AC040970.1', 'ENSG00000253213_AC010306.1', 'ENSG00000253218_KLF3P1', 'ENSG00000253250_C8orf88', 'ENSG00000253251_SHLD3', 'ENSG00000253276_CCDC71L', 'ENSG00000253284_AC092828.1', 'ENSG00000253293_HOXA10', 'ENSG00000253320_AZIN1-AS1', 'ENSG00000253327_RAD21-AS1', 'ENSG00000253330_AC024451.2', 'ENSG00000253341_AC115837.1', 'ENSG00000253352_TUG1', 'ENSG00000253368_TRNP1', 'ENSG00000253372_AC016405.1', 'ENSG00000253384_AC124242.2', 'ENSG00000253385_AP003696.1', 'ENSG00000253392_AC119403.1', 'ENSG00000253404_AC034243.1', 'ENSG00000253408_AC083973.1', 'ENSG00000253431_SRPK2P', 'ENSG00000253438_PCAT1', 'ENSG00000253463_HMGB1P19', 'ENSG00000253475_AC103769.1', 'ENSG00000253485_PCDHGA5', 'ENSG00000253488_SINHCAFP3', 'ENSG00000253506_NACA2', 'ENSG00000253516_HMGB1P41', 'ENSG00000253520_AC136628.3', 'ENSG00000253522_MIR3142HG', 'ENSG00000253540_FAM86HP', 'ENSG00000253549_CA3-AS1', 'ENSG00000253552_HOXA-AS2', 'ENSG00000253553_AC090578.1', 'ENSG00000253558_AC024568.1', 'ENSG00000253559_OSGEPL1-AS1', 'ENSG00000253582_AC090579.1', 'ENSG00000253586_AC067817.1', 'ENSG00000253598_SLC10A5', 'ENSG00000253619_AC068413.1', 'ENSG00000253626_EIF5AL1', 'ENSG00000253633_AP002852.1', 'ENSG00000253636_AC022893.1', 'ENSG00000253645_AC108863.1', 'ENSG00000253667_AC100821.1', 'ENSG00000253669_GASAL1', 'ENSG00000253683_AC027309.2', 'ENSG00000253696_KBTBD11-OT1', 'ENSG00000253704_AC023632.2', 'ENSG00000253710_ALG11', 'ENSG00000253716_MINCR', 'ENSG00000253719_ATXN7L3B', 'ENSG00000253729_PRKDC', 'ENSG00000253731_PCDHGA6', 'ENSG00000253738_OTUD6B-AS1', 'ENSG00000253764_AC019257.1', 'ENSG00000253770_HMGB1P23', 'ENSG00000253771_TPTE2P1', 'ENSG00000253773_C8orf37-AS1', 'ENSG00000253796_AC104248.1', 'ENSG00000253797_UTP14C', 'ENSG00000253816_AC138866.1', 'ENSG00000253829_AC067817.2', 'ENSG00000253833_AC022868.1', 'ENSG00000253846_PCDHGA10', 'ENSG00000253848_AC010834.2', 'ENSG00000253851_AC025370.1', 'ENSG00000253854_AC010834.3', 'ENSG00000253865_AC131025.1', 'ENSG00000253878_AC087752.3', 'ENSG00000253882_AC099548.2', 'ENSG00000253893_FAM85B', 'ENSG00000253919_THAP12P7', 'ENSG00000253923_AP002981.1', 'ENSG00000253948_AC104986.2', 'ENSG00000253954_HMGN1P38', 'ENSG00000253958_CLDN23', 'ENSG00000253966_AC008514.2', 'ENSG00000253967_AC022730.4', 'ENSG00000253982_AC100810.1', 'ENSG00000254003_AC003991.2', 'ENSG00000254004_ZNF260', 'ENSG00000254006_AC104232.1', 'ENSG00000254017_IGHEP2', 'ENSG00000254027_AC009902.2', 'ENSG00000254034_INTS9-AS1', 'ENSG00000254051_AC011853.1', 'ENSG00000254087_LYN', 'ENSG00000254090_MTND2P32', 'ENSG00000254092_AC015468.3', 'ENSG00000254093_PINX1', 'ENSG00000254101_LINC02055', 'ENSG00000254109_RBPMS-AS1', 'ENSG00000254126_CD8B2', 'ENSG00000254139_AC104051.2', 'ENSG00000254153_AC103957.2', 'ENSG00000254162_AC009812.3', 'ENSG00000254165_AC090739.1', 'ENSG00000254170_AC008802.1', 'ENSG00000254172_RNU5A-3P', 'ENSG00000254186_AC113414.1', 'ENSG00000254198_AC113191.1', 'ENSG00000254206_NPIPB11', 'ENSG00000254244_PAICSP4', 'ENSG00000254273_AC018620.1', 'ENSG00000254275_LINC00824', 'ENSG00000254285_KRT8P3', 'ENSG00000254319_AC246817.2', 'ENSG00000254325_AC018607.1', 'ENSG00000254332_AF201337.1', 'ENSG00000254333_NDST1-AS1', 'ENSG00000254343_AC091563.1', 'ENSG00000254363_AC011379.2', 'ENSG00000254369_HOXA-AS3', 'ENSG00000254373_AC112191.2', 'ENSG00000254388_DUTP2', 'ENSG00000254389_RHPN1-AS1', 'ENSG00000254396_AL355432.1', 'ENSG00000254397_AC132192.1', 'ENSG00000254402_LRRC24', 'ENSG00000254409_AC087521.3', 'ENSG00000254415_SIGLEC14', 'ENSG00000254427_AC103736.1', 'ENSG00000254428_AP003392.1', 'ENSG00000254433_AP001001.1', 'ENSG00000254438_AC022240.1', 'ENSG00000254450_ALG9-IT1', 'ENSG00000254452_AP001107.2', 'ENSG00000254453_NAV2-AS2', 'ENSG00000254454_RCC2P6', 'ENSG00000254463_PPIAP41', 'ENSG00000254469_AP002495.1', 'ENSG00000254470_AP5B1', 'ENSG00000254473_AL354920.1', 'ENSG00000254480_AC015689.1', 'ENSG00000254484_AP002336.1', 'ENSG00000254501_AP003068.1', 'ENSG00000254503_AC010319.1', 'ENSG00000254505_CHMP4A', 'ENSG00000254521_SIGLEC12', 'ENSG00000254531_FLJ20021', 'ENSG00000254535_PABPC4L', 'ENSG00000254556_AF131215.2', 'ENSG00000254559_AC069287.2', 'ENSG00000254577_AC087276.1', 'ENSG00000254584_AL035078.1', 'ENSG00000254598_CSNK2A3', 'ENSG00000254602_AP000662.1', 'ENSG00000254612_AP001000.1', 'ENSG00000254614_AP003068.2', 'ENSG00000254615_AC027031.2', 'ENSG00000254634_SMG1P6', 'ENSG00000254635_WAC-AS1', 'ENSG00000254665_AC091053.1', 'ENSG00000254676_AP000873.4', 'ENSG00000254682_AP002387.1', 'ENSG00000254685_FPGT', 'ENSG00000254694_AP001893.1', 'ENSG00000254703_SENCR', 'ENSG00000254712_AC087280.1', 'ENSG00000254718_AL157756.1', 'ENSG00000254726_MEX3A', 'ENSG00000254731_AP003059.1', 'ENSG00000254759_NAP1L1P1', 'ENSG00000254772_EEF1G', 'ENSG00000254783_AP003084.1', 'ENSG00000254786_AP001636.1', 'ENSG00000254791_FAR1-IT1', 'ENSG00000254793_FDPSP4', 'ENSG00000254802_AC022182.2', 'ENSG00000254827_SLC22A18AS', 'ENSG00000254837_AP001372.2', 'ENSG00000254838_GVINP1', 'ENSG00000254843_XIAPP2', 'ENSG00000254858_MPV17L2', 'ENSG00000254859_AC067930.4', 'ENSG00000254860_TMEM9B-AS1', 'ENSG00000254873_AP001267.1', 'ENSG00000254876_SUGT1P4-STRA6LP', 'ENSG00000254877_AP001636.2', 'ENSG00000254884_PRR13P2', 'ENSG00000254887_AC010247.1', 'ENSG00000254893_AC113404.3', 'ENSG00000254901_BORCS8', 'ENSG00000254911_SCARNA9', 'ENSG00000254928_AP001372.3', 'ENSG00000254929_AL591684.2', 'ENSG00000254936_AF131215.4', 'ENSG00000254985_RSF1-IT2', 'ENSG00000254986_DPP3', 'ENSG00000254995_STX16-NPEPL1', 'ENSG00000254996_ANKHD1-EIF4EBP3', 'ENSG00000254999_BRK1', 'ENSG00000255008_AP000442.1', 'ENSG00000255031_AP002807.1', 'ENSG00000255040_MORF4L1P3', 'ENSG00000255042_AC109635.4', 'ENSG00000255045_AP000866.5', 'ENSG00000255046_AC069185.1', 'ENSG00000255057_AP003041.1', 'ENSG00000255062_AP001318.2', 'ENSG00000255081_AP003168.2', 'ENSG00000255100_AP003119.2', 'ENSG00000255112_CHMP1B', 'ENSG00000255114_AP003392.3', 'ENSG00000255118_AP003306.2', 'ENSG00000255121_AP003392.4', 'ENSG00000255129_AP000880.1', 'ENSG00000255135_AP002360.1', 'ENSG00000255139_AP000442.2', 'ENSG00000255141_HNRNPA1P76', 'ENSG00000255145_STX17-AS1', 'ENSG00000255150_EID3', 'ENSG00000255152_MSH5-SAPCD1', 'ENSG00000255153_TOLLIP-AS1', 'ENSG00000255154_HTD2', 'ENSG00000255165_AC134775.1', 'ENSG00000255182_AC084125.2', 'ENSG00000255185_PDXDC2P', 'ENSG00000255189_GLYATL1P1', 'ENSG00000255197_AC090559.1', 'ENSG00000255198_SNHG9', 'ENSG00000255200_PGAM1P8', 'ENSG00000255224_AC109322.1', 'ENSG00000255237_AC138230.1', 'ENSG00000255240_AP001636.3', 'ENSG00000255248_MIR100HG', 'ENSG00000255284_AP006621.3', 'ENSG00000255302_EID1', 'ENSG00000255303_OR5BA1P', 'ENSG00000255306_AC004923.4', 'ENSG00000255310_AF131215.5', 'ENSG00000255319_ENPP7P8', 'ENSG00000255320_AP000759.1', 'ENSG00000255326_AP001922.5', 'ENSG00000255328_AC136475.5', 'ENSG00000255337_AP001830.1', 'ENSG00000255347_AC021820.1', 'ENSG00000255351_AC023946.1', 'ENSG00000255358_AC019227.1', 'ENSG00000255363_AP001189.5', 'ENSG00000255374_TAS2R43', 'ENSG00000255384_AP001267.2', 'ENSG00000255389_Z97989.1', 'ENSG00000255397_AC022182.3', 'ENSG00000255409_RSF1-IT1', 'ENSG00000255423_EBLN2', 'ENSG00000255435_AP001267.3', 'ENSG00000255441_AC008750.2', 'ENSG00000255455_AP003486.1', 'ENSG00000255458_AC108471.2', 'ENSG00000255467_AP002433.1', 'ENSG00000255468_AP001107.9', 'ENSG00000255476_AC011092.2', 'ENSG00000255495_AC145124.1', 'ENSG00000255508_AP002990.1', 'ENSG00000255517_AP002748.3', 'ENSG00000255529_POLR2M', 'ENSG00000255537_AP000708.1', 'ENSG00000255559_ZNF252P-AS1', 'ENSG00000255561_FDXACB1', 'ENSG00000255566_AC135279.1', 'ENSG00000255568_BRWD1-AS2', 'ENSG00000255571_MIR9-3HG', 'ENSG00000255581_AC006511.2', 'ENSG00000255585_AL590627.1', 'ENSG00000255587_RAB44', 'ENSG00000255624_AC073585.1', 'ENSG00000255642_PABPC1P4', 'ENSG00000255647_AC093510.1', 'ENSG00000255680_AC091564.6', 'ENSG00000255717_SNHG1', 'ENSG00000255725_TDGP1', 'ENSG00000255730_AC011462.1', 'ENSG00000255737_AGAP2-AS1', 'ENSG00000255769_GOLGA2P10', 'ENSG00000255823_MTRNR2L8', 'ENSG00000255833_TIFAB', 'ENSG00000255836_AC131206.1', 'ENSG00000255837_TAS2R20', 'ENSG00000255856_AC069503.1', 'ENSG00000255857_PXN-AS1', 'ENSG00000255875_AC008813.1', 'ENSG00000255882_AC091814.1', 'ENSG00000255909_PDCD5P1', 'ENSG00000255920_CCND2-AS1', 'ENSG00000255949_RPS6KB2-AS1', 'ENSG00000255959_AP000777.2', 'ENSG00000256006_AC084117.1', 'ENSG00000256019_TAS2R63P', 'ENSG00000256037_MRPL40P1', 'ENSG00000256043_CTSO', 'ENSG00000256053_APOPT1', 'ENSG00000256060_TRAPPC2B', 'ENSG00000256061_DNAAF4', 'ENSG00000256073_URB1-AS1', 'ENSG00000256087_ZNF432', 'ENSG00000256092_AC137767.1', 'ENSG00000256101_AC092745.1', 'ENSG00000256116_AP001453.1', 'ENSG00000256152_AC027290.1', 'ENSG00000256188_TAS2R30', 'ENSG00000256210_AC005255.1', 'ENSG00000256222_MTRNR2L3', 'ENSG00000256223_ZNF10', 'ENSG00000256229_ZNF486', 'ENSG00000256235_SMIM3', 'ENSG00000256238_SUPT16HP1', 'ENSG00000256262_USP30-AS1', 'ENSG00000256269_HMBS', 'ENSG00000256271_CACNA1C-AS2', 'ENSG00000256274_TAS2R64P', 'ENSG00000256278_AC048382.1', 'ENSG00000256294_ZNF225', 'ENSG00000256325_AC025423.1', 'ENSG00000256331_NIFKP3', 'ENSG00000256338_RPL41P2', 'ENSG00000256341_AP006333.1', 'ENSG00000256361_AC027544.1', 'ENSG00000256364_AC069234.2', 'ENSG00000256427_AC010175.1', 'ENSG00000256433_AC005840.2', 'ENSG00000256436_TAS2R31', 'ENSG00000256442_AC010186.1', 'ENSG00000256448_AP000763.3', 'ENSG00000256453_DND1', 'ENSG00000256464_YWHABP2', 'ENSG00000256525_POLG2', 'ENSG00000256537_SMIM10L1', 'ENSG00000256546_AC156455.1', 'ENSG00000256552_AC092745.2', 'ENSG00000256576_LINC02361', 'ENSG00000256591_AP003108.2', 'ENSG00000256594_AC010186.2', 'ENSG00000256603_AP003170.4', 'ENSG00000256612_CYP2B7P', 'ENSG00000256618_MTRNR2L1', 'ENSG00000256625_AC092747.2', 'ENSG00000256628_ZBTB11-AS1', 'ENSG00000256633_AP005019.1', 'ENSG00000256651_AC006518.1', 'ENSG00000256660_CLEC12B', 'ENSG00000256663_AC112777.1', 'ENSG00000256667_KLRA1P', 'ENSG00000256682_AC006518.2', 'ENSG00000256683_ZNF350', 'ENSG00000256690_AP001160.1', 'ENSG00000256705_AL137779.1', 'ENSG00000256742_AC145422.1', 'ENSG00000256745_AP002784.2', 'ENSG00000256771_ZNF253', 'ENSG00000256806_C17orf100', 'ENSG00000256812_CAPNS2', 'ENSG00000256826_ATP5MFP4', 'ENSG00000256928_AP000763.4', 'ENSG00000256940_AP001453.2', 'ENSG00000256948_AC026369.3', 'ENSG00000256950_AC069503.2', 'ENSG00000256973_AC053513.1', 'ENSG00000256981_AC134349.2', 'ENSG00000256982_AC135782.1', 'ENSG00000257017_HP', 'ENSG00000257027_AC010186.3', 'ENSG00000257037_RARSP1', 'ENSG00000257038_AP002761.3', 'ENSG00000257086_AP001453.3', 'ENSG00000257093_KIAA1147', 'ENSG00000257103_LSM14A', 'ENSG00000257108_NHLRC4', 'ENSG00000257122_RRN3P3', 'ENSG00000257135_AC007249.2', 'ENSG00000257151_PWAR6', 'ENSG00000257159_AC084033.1', 'ENSG00000257167_TMPO-AS1', 'ENSG00000257169_AC125612.1', 'ENSG00000257176_AC009318.1', 'ENSG00000257178_AC103702.1', 'ENSG00000257181_AC025423.4', 'ENSG00000257210_NACAP8', 'ENSG00000257218_GATC', 'ENSG00000257221_AC007569.1', 'ENSG00000257222_AC079907.1', 'ENSG00000257225_AC079601.1', 'ENSG00000257239_AC090630.1', 'ENSG00000257243_AC020612.1', 'ENSG00000257246_PHBP19', 'ENSG00000257258_AC012150.1', 'ENSG00000257261_AC008014.1', 'ENSG00000257267_ZNF271P', 'ENSG00000257279_AC127164.1', 'ENSG00000257285_AL132780.1', 'ENSG00000257298_AC008147.2', 'ENSG00000257303_AC073896.2', 'ENSG00000257315_ZBED6', 'ENSG00000257335_MGAM', 'ENSG00000257337_AC068888.1', 'ENSG00000257342_AC025165.2', 'ENSG00000257354_AC048341.1', 'ENSG00000257365_FNTB', 'ENSG00000257368_AC063924.1', 'ENSG00000257390_AC023055.1', 'ENSG00000257391_AC126763.1', 'ENSG00000257433_AC004241.1', 'ENSG00000257446_ZNF878', 'ENSG00000257475_AC068888.2', 'ENSG00000257489_AC010203.1', 'ENSG00000257497_AC121761.1', 'ENSG00000257509_AC073487.1', 'ENSG00000257511_AC084824.1', 'ENSG00000257526_AC107032.2', 'ENSG00000257527_AC126755.3', 'ENSG00000257531_AC008147.3', 'ENSG00000257550_AC023509.2', 'ENSG00000257551_HLX-AS1', 'ENSG00000257553_AC034102.4', 'ENSG00000257556_LINC02298', 'ENSG00000257557_PPP1R12A-AS1', 'ENSG00000257576_HSPD1P4', 'ENSG00000257582_LINC01475', 'ENSG00000257591_ZNF625', 'ENSG00000257594_GALNT4', 'ENSG00000257595_LINC02356', 'ENSG00000257596_AC078778.1', 'ENSG00000257599_OVCH1-AS1', 'ENSG00000257604_AC027288.2', 'ENSG00000257605_AC073611.1', 'ENSG00000257607_AC073957.2', 'ENSG00000257613_LINC01481', 'ENSG00000257621_PSMA3-AS1', 'ENSG00000257654_AC125603.1', 'ENSG00000257660_AC117498.2', 'ENSG00000257666_CBX3P5', 'ENSG00000257681_AC025265.1', 'ENSG00000257698_AC084033.3', 'ENSG00000257702_LBX2-AS1', 'ENSG00000257704_INAFM1', 'ENSG00000257718_CPNE8-AS1', 'ENSG00000257727_CNPY2', 'ENSG00000257732_AC089983.1', 'ENSG00000257740_AC073896.3', 'ENSG00000257764_AC020656.1', 'ENSG00000257769_AC026401.1', 'ENSG00000257800_FNBP1P1', 'ENSG00000257815_LINC01481', 'ENSG00000257818_C1GALT1P1', 'ENSG00000257831_AL136418.1', 'ENSG00000257839_AC011611.4', 'ENSG00000257878_AC007298.2', 'ENSG00000257880_AC078789.1', 'ENSG00000257883_AC125603.2', 'ENSG00000257894_AC027288.3', 'ENSG00000257913_DDN-AS1', 'ENSG00000257923_CUX1', 'ENSG00000257941_AC011611.5', 'ENSG00000257950_P2RX5-TAX1BP3', 'ENSG00000257954_AC125611.2', 'ENSG00000257956_NOP56P3', 'ENSG00000257966_OLA1P3', 'ENSG00000258001_AC126614.1', 'ENSG00000258016_HIGD1AP1', 'ENSG00000258017_AC011603.2', 'ENSG00000258044_AC073569.1', 'ENSG00000258056_AC009779.2', 'ENSG00000258057_BCDIN3D-AS1', 'ENSG00000258082_AL391832.3', 'ENSG00000258092_AC005841.1', 'ENSG00000258099_ATXN2-AS', 'ENSG00000258101_AC010173.1', 'ENSG00000258102_MAP1LC3B2', 'ENSG00000258111_AC079316.1', 'ENSG00000258122_AC044802.2', 'ENSG00000258137_AC079313.2', 'ENSG00000258153_HSPE1P4', 'ENSG00000258181_AC008083.2', 'ENSG00000258199_AC073896.4', 'ENSG00000258210_AC144548.1', 'ENSG00000258227_CLEC5A', 'ENSG00000258230_AC063950.1', 'ENSG00000258232_AC125611.3', 'ENSG00000258274_AC012085.2', 'ENSG00000258289_CHURC1', 'ENSG00000258301_VASH1-AS1', 'ENSG00000258302_AC025034.1', 'ENSG00000258303_AC012464.2', 'ENSG00000258311_BLOC1S1-RDH5', 'ENSG00000258315_C17orf49', 'ENSG00000258317_AC034102.6', 'ENSG00000258325_ITFG2-AS1', 'ENSG00000258337_AC130895.1', 'ENSG00000258344_AC078778.2', 'ENSG00000258359_PCNPP1', 'ENSG00000258365_AC073655.2', 'ENSG00000258366_RTEL1', 'ENSG00000258376_AC004846.1', 'ENSG00000258377_AL139099.1', 'ENSG00000258384_AC068831.1', 'ENSG00000258388_PPT2-EGFL8', 'ENSG00000258405_ZNF578', 'ENSG00000258422_AL160191.1', 'ENSG00000258427_RBM8B', 'ENSG00000258429_PDF', 'ENSG00000258430_AL583722.2', 'ENSG00000258441_LINC00641', 'ENSG00000258446_AL133153.1', 'ENSG00000258448_AL442663.2', 'ENSG00000258450_AL139099.2', 'ENSG00000258457_AL132780.2', 'ENSG00000258458_AL160314.2', 'ENSG00000258461_AC012651.1', 'ENSG00000258469_CHMP4BP1', 'ENSG00000258471_AL161668.3', 'ENSG00000258472_AC005726.1', 'ENSG00000258476_LINC02207', 'ENSG00000258484_SPESP1', 'ENSG00000258500_AL845552.1', 'ENSG00000258525_AL049830.3', 'ENSG00000258559_AC005519.1', 'ENSG00000258560_AL157912.1', 'ENSG00000258561_AL359232.1', 'ENSG00000258565_BLZF2P', 'ENSG00000258568_RHOQP1', 'ENSG00000258572_AL133467.1', 'ENSG00000258581_AL157871.3', 'ENSG00000258599_AL355922.2', 'ENSG00000258603_AC005225.2', 'ENSG00000258608_DNAJC19P9', 'ENSG00000258626_COX7A2P1', 'ENSG00000258632_AC135626.1', 'ENSG00000258634_AL160006.1', 'ENSG00000258640_RPL21P5', 'ENSG00000258644_SYNJ2BP-COX16', 'ENSG00000258645_HSPE1P2', 'ENSG00000258646_AL049780.1', 'ENSG00000258651_SEC23A-AS1', 'ENSG00000258653_AC005520.1', 'ENSG00000258655_ARHGAP5-AS1', 'ENSG00000258667_HIF1A-AS2', 'ENSG00000258682_AL132989.1', 'ENSG00000258701_LINC00638', 'ENSG00000258702_AL137786.1', 'ENSG00000258704_SRP54-AS1', 'ENSG00000258708_SLC25A21-AS1', 'ENSG00000258711_AL358334.2', 'ENSG00000258725_PRC1-AS1', 'ENSG00000258727_AL135999.1', 'ENSG00000258730_ITPK1-AS1', 'ENSG00000258731_AL356020.1', 'ENSG00000258733_LINC02328', 'ENSG00000258738_AL121603.2', 'ENSG00000258741_H2AFVP1', 'ENSG00000258757_AL133453.1', 'ENSG00000258768_AL356019.2', 'ENSG00000258791_LINC00520', 'ENSG00000258798_AL133153.2', 'ENSG00000258813_AL442663.3', 'ENSG00000258818_RNASE4', 'ENSG00000258839_MC1R', 'ENSG00000258843_AL133485.1', 'ENSG00000258875_AL135818.1', 'ENSG00000258881_AC007040.2', 'ENSG00000258890_CEP95', 'ENSG00000258891_AC005480.1', 'ENSG00000258896_SCOCP1', 'ENSG00000258900_HNRNPCP1', 'ENSG00000258904_AL157871.5', 'ENSG00000258920_FOXN3-AS1', 'ENSG00000258938_AL162311.3', 'ENSG00000258940_AL132639.2', 'ENSG00000258944_AC004846.2', 'ENSG00000258947_TUBB3', 'ENSG00000258957_AL359317.2', 'ENSG00000258968_AL049830.4', 'ENSG00000258982_AL133523.1', 'ENSG00000258998_LINC02302', 'ENSG00000259001_AL355075.4', 'ENSG00000259003_AC243965.1', 'ENSG00000259004_LINC02285', 'ENSG00000259013_AL118558.2', 'ENSG00000259015_AL442663.4', 'ENSG00000259020_AL049872.1', 'ENSG00000259024_TVP23C-CDRT4', 'ENSG00000259030_FPGT-TNNI3K', 'ENSG00000259042_AC244502.3', 'ENSG00000259049_AL139317.3', 'ENSG00000259054_LINC02332', 'ENSG00000259062_ACTN1-AS1', 'ENSG00000259065_AC005520.2', 'ENSG00000259071_AL359397.2', 'ENSG00000259073_FOXN3-AS2', 'ENSG00000259083_AL132639.3', 'ENSG00000259086_AL136298.3', 'ENSG00000259088_AL137779.2', 'ENSG00000259090_SEPT7P1', 'ENSG00000259118_AL139022.1', 'ENSG00000259120_SMIM6', 'ENSG00000259134_LINC00924', 'ENSG00000259137_AL109766.1', 'ENSG00000259138_AL049780.2', 'ENSG00000259151_CAP2P1', 'ENSG00000259153_AC004816.1', 'ENSG00000259158_ADAM20P1', 'ENSG00000259172_AC023024.1', 'ENSG00000259185_AC090971.1', 'ENSG00000259188_AC025040.1', 'ENSG00000259205_PRKXP1', 'ENSG00000259207_ITGB3', 'ENSG00000259209_AC004943.1', 'ENSG00000259212_AC103739.1', 'ENSG00000259215_AC027237.2', 'ENSG00000259232_AC105129.1', 'ENSG00000259238_AC092755.2', 'ENSG00000259244_AC048382.2', 'ENSG00000259248_USP3-AS1', 'ENSG00000259251_AC104590.1', 'ENSG00000259254_AC022405.1', 'ENSG00000259274_AC107241.1', 'ENSG00000259286_AC087639.1', 'ENSG00000259287_AC010809.1', 'ENSG00000259291_ZNF710-AS1', 'ENSG00000259295_CSPG4P12', 'ENSG00000259305_ZHX1-C8orf76', 'ENSG00000259308_AC024270.1', 'ENSG00000259315_ACTG1P17', 'ENSG00000259316_AC087632.1', 'ENSG00000259318_AL356801.1', 'ENSG00000259319_AF111167.2', 'ENSG00000259321_AL136295.2', 'ENSG00000259326_AC116158.1', 'ENSG00000259330_INAFM2', 'ENSG00000259335_HNRNPMP1', 'ENSG00000259343_TMC3-AS1', 'ENSG00000259347_AC087482.1', 'ENSG00000259349_AC011921.1', 'ENSG00000259357_AL590133.2', 'ENSG00000259363_AC090825.1', 'ENSG00000259366_AC108449.2', 'ENSG00000259370_AC103740.1', 'ENSG00000259378_DCAF13P3', 'ENSG00000259396_AC087721.1', 'ENSG00000259397_AC021231.2', 'ENSG00000259407_AC021739.2', 'ENSG00000259408_AC010809.2', 'ENSG00000259415_AC012291.1', 'ENSG00000259424_IRAIN', 'ENSG00000259426_AC027237.3', 'ENSG00000259429_UBE2Q2P2', 'ENSG00000259431_THTPA', 'ENSG00000259446_AC055874.1', 'ENSG00000259448_LINC02352', 'ENSG00000259456_ADNP-AS1', 'ENSG00000259462_CPEB1-AS1', 'ENSG00000259467_NDUFAF4P1', 'ENSG00000259469_AC084757.2', 'ENSG00000259479_SORD2P', 'ENSG00000259488_AC023355.1', 'ENSG00000259494_MRPL46', 'ENSG00000259498_TPM1-AS', 'ENSG00000259511_UBE2Q2L', 'ENSG00000259512_HNRNPA1P5', 'ENSG00000259516_ANP32AP1', 'ENSG00000259520_AC051619.5', 'ENSG00000259529_AL136295.5', 'ENSG00000259539_AC051619.6', 'ENSG00000259553_AC140725.1', 'ENSG00000259562_AC090607.2', 'ENSG00000259577_CERNA1', 'ENSG00000259581_TYRO3P', 'ENSG00000259583_AC015712.2', 'ENSG00000259585_RBM17P4', 'ENSG00000259605_AC074212.1', 'ENSG00000259623_AC125257.1', 'ENSG00000259630_AC104046.1', 'ENSG00000259635_AC100830.1', 'ENSG00000259642_ST20-AS1', 'ENSG00000259648_AL132640.2', 'ENSG00000259658_AC027559.1', 'ENSG00000259659_AC009996.1', 'ENSG00000259660_DNM1P47', 'ENSG00000259673_IQCH-AS1', 'ENSG00000259677_AC027176.2', 'ENSG00000259682_AC091231.1', 'ENSG00000259687_LINC01220', 'ENSG00000259699_HMGB1P8', 'ENSG00000259704_AC124248.1', 'ENSG00000259706_HSP90B2P', 'ENSG00000259712_AC023906.5', 'ENSG00000259713_AC013391.2', 'ENSG00000259715_AC022087.1', 'ENSG00000259728_LINC00933', 'ENSG00000259735_AC092868.2', 'ENSG00000259736_CRTC3-AS1', 'ENSG00000259746_HSPE1P3', 'ENSG00000259755_AC090907.2', 'ENSG00000259767_AC022558.1', 'ENSG00000259768_AC004943.2', 'ENSG00000259771_AC092756.1', 'ENSG00000259775_AL138976.2', 'ENSG00000259776_AC093426.1', 'ENSG00000259781_HMGB1P6', 'ENSG00000259782_AC008915.1', 'ENSG00000259790_ANP32BP1', 'ENSG00000259797_AC020978.1', 'ENSG00000259802_AC012640.2', 'ENSG00000259803_SLC22A31', 'ENSG00000259804_AC027682.1', 'ENSG00000259818_AL606760.3', 'ENSG00000259820_AC083843.2', 'ENSG00000259826_AC072061.1', 'ENSG00000259834_AL365361.1', 'ENSG00000259845_HERC2P10', 'ENSG00000259865_AL390728.6', 'ENSG00000259877_AC009113.1', 'ENSG00000259881_AC092384.2', 'ENSG00000259891_AC107375.1', 'ENSG00000259895_AC106820.2', 'ENSG00000259915_AC017071.1', 'ENSG00000259920_AC007938.1', 'ENSG00000259921_AC022819.1', 'ENSG00000259924_AC011939.1', 'ENSG00000259926_AC007342.1', 'ENSG00000259932_AC051619.7', 'ENSG00000259935_AC009754.1', 'ENSG00000259939_AC022167.1', 'ENSG00000259940_AC109449.1', 'ENSG00000259941_AC084782.1', 'ENSG00000259943_AL050341.2', 'ENSG00000259945_AC027682.2', 'ENSG00000259952_AC009133.2', 'ENSG00000259953_AL138756.1', 'ENSG00000259956_RBM15B', 'ENSG00000259959_AC107068.1', 'ENSG00000259972_AC009120.2', 'ENSG00000259985_AC017100.1', 'ENSG00000259994_AL353796.1', 'ENSG00000259999_AC009054.1', 'ENSG00000260000_AL133338.1', 'ENSG00000260001_TGFBR3L', 'ENSG00000260005_AC027601.1', 'ENSG00000260007_AC107871.1', 'ENSG00000260011_AC132938.1', 'ENSG00000260018_AC040169.1', 'ENSG00000260025_AC009414.2', 'ENSG00000260027_HOXB7', 'ENSG00000260032_NORAD', 'ENSG00000260034_LCMT1-AS2', 'ENSG00000260035_AC051619.8', 'ENSG00000260036_AC013355.1', 'ENSG00000260037_AC104938.1', 'ENSG00000260038_AC009090.1', 'ENSG00000260052_AC023813.3', 'ENSG00000260059_AC092620.3', 'ENSG00000260060_AC009088.1', 'ENSG00000260063_AL512408.1', 'ENSG00000260064_AC092332.1', 'ENSG00000260077_AC104794.2', 'ENSG00000260078_AC007342.3', 'ENSG00000260081_AF274858.1', 'ENSG00000260083_MIR762HG', 'ENSG00000260086_AC007611.1', 'ENSG00000260088_AL445483.1', 'ENSG00000260091_AC093752.3', 'ENSG00000260093_AC034111.1', 'ENSG00000260095_AC106820.3', 'ENSG00000260100_AL512604.3', 'ENSG00000260101_AC008074.2', 'ENSG00000260103_AC012435.1', 'ENSG00000260111_AC012184.1', 'ENSG00000260121_AC138028.4', 'ENSG00000260122_AC068987.2', 'ENSG00000260128_ULK4P2', 'ENSG00000260132_AL032819.1', 'ENSG00000260136_AC008915.2', 'ENSG00000260160_AC011468.1', 'ENSG00000260167_AC093249.2', 'ENSG00000260179_AL162741.1', 'ENSG00000260188_AC002464.1', 'ENSG00000260190_AL807752.5', 'ENSG00000260193_AL138781.1', 'ENSG00000260196_AC124798.1', 'ENSG00000260197_AC010889.1', 'ENSG00000260212_AL356432.3', 'ENSG00000260213_AC092718.2', 'ENSG00000260219_AC106782.2', 'ENSG00000260229_PPIAP51', 'ENSG00000260230_FRRS1L', 'ENSG00000260231_KDM7A-DT', 'ENSG00000260233_SSSCA1-AS1', 'ENSG00000260236_AC099778.1', 'ENSG00000260238_PMF1-BGLAP', 'ENSG00000260244_AC104083.1', 'ENSG00000260246_AC000032.1', 'ENSG00000260249_AC007608.3', 'ENSG00000260257_AL035071.1', 'ENSG00000260259_LINC02166', 'ENSG00000260260_SNHG19', 'ENSG00000260261_AC124944.3', 'ENSG00000260267_AC026471.1', 'ENSG00000260273_AL359711.2', 'ENSG00000260274_AC068338.2', 'ENSG00000260276_AC022167.2', 'ENSG00000260278_AC098818.2', 'ENSG00000260279_AC137932.1', 'ENSG00000260285_AL133367.1', 'ENSG00000260288_AC019294.2', 'ENSG00000260290_AC092115.1', 'ENSG00000260293_AC106820.4', 'ENSG00000260296_AC095057.3', 'ENSG00000260300_AC009119.2', 'ENSG00000260306_AC092375.2', 'ENSG00000260314_MRC1', 'ENSG00000260316_AL008727.1', 'ENSG00000260317_AC009812.4', 'ENSG00000260318_COX6CP1', 'ENSG00000260325_HSPB9', 'ENSG00000260329_AC007541.1', 'ENSG00000260331_AC079148.1', 'ENSG00000260339_HEXA-AS1', 'ENSG00000260349_AC087190.1', 'ENSG00000260352_AC092287.1', 'ENSG00000260360_AL353708.1', 'ENSG00000260361_AC106028.3', 'ENSG00000260367_AC109460.1', 'ENSG00000260368_AC027373.1', 'ENSG00000260369_AC120024.1', 'ENSG00000260386_LINC01225', 'ENSG00000260388_LINC00562', 'ENSG00000260398_AC068700.1', 'ENSG00000260400_AL513534.1', 'ENSG00000260401_AP002761.4', 'ENSG00000260404_AC110079.1', 'ENSG00000260409_AC012414.5', 'ENSG00000260417_AC092127.1', 'ENSG00000260418_AL023284.4', 'ENSG00000260423_LINC02367', 'ENSG00000260428_SCX', 'ENSG00000260442_ATP2A1-AS1', 'ENSG00000260448_LCMT1-AS1', 'ENSG00000260452_TPRKBP2', 'ENSG00000260456_C16orf95', 'ENSG00000260461_AL133355.1', 'ENSG00000260464_AL049796.1', 'ENSG00000260465_AC018557.1', 'ENSG00000260475_AL353719.1', 'ENSG00000260479_AC009145.1', 'ENSG00000260490_MYL12BP1', 'ENSG00000260493_AC011773.4', 'ENSG00000260495_AC009148.1', 'ENSG00000260498_AC126696.3', 'ENSG00000260507_AC133919.1', 'ENSG00000260509_AL590787.1', 'ENSG00000260517_AC009093.2', 'ENSG00000260518_BMS1P8', 'ENSG00000260526_AC109347.1', 'ENSG00000260528_FAM157C', 'ENSG00000260549_MT1L', 'ENSG00000260552_AC023043.1', 'ENSG00000260558_AC018557.2', 'ENSG00000260563_AC132872.1', 'ENSG00000260565_ERVK13-1', 'ENSG00000260570_AC133550.1', 'ENSG00000260572_AC069224.1', 'ENSG00000260583_LINC00515', 'ENSG00000260588_AC027702.1', 'ENSG00000260589_STAM-AS1', 'ENSG00000260592_AC130456.3', 'ENSG00000260613_Z98885.2', 'ENSG00000260621_AC092140.1', 'ENSG00000260625_AC026471.2', 'ENSG00000260629_BGLT3', 'ENSG00000260630_SNAI3-AS1', 'ENSG00000260641_AC114811.2', 'ENSG00000260643_AC092718.3', 'ENSG00000260645_AL359715.2', 'ENSG00000260646_AL031705.1', 'ENSG00000260647_AC127537.1', 'ENSG00000260648_AC020658.3', 'ENSG00000260651_AF213884.3', 'ENSG00000260657_AC107871.2', 'ENSG00000260664_AC004158.1', 'ENSG00000260668_AC093536.1', 'ENSG00000260669_AL096870.2', 'ENSG00000260685_AC027104.1', 'ENSG00000260686_AC008669.1', 'ENSG00000260689_HNRNPA3P11', 'ENSG00000260693_AC026150.1', 'ENSG00000260698_AL591848.3', 'ENSG00000260708_AL118516.1', 'ENSG00000260711_AL121839.2', 'ENSG00000260714_AC133552.1', 'ENSG00000260727_SLC7A5P1', 'ENSG00000260733_AC046158.1', 'ENSG00000260735_AC139256.1', 'ENSG00000260740_AC026471.3', 'ENSG00000260742_AC009962.1', 'ENSG00000260743_AC007823.1', 'ENSG00000260747_AC022968.1', 'ENSG00000260751_AC008870.2', 'ENSG00000260755_AC010542.2', 'ENSG00000260774_AC021087.2', 'ENSG00000260778_AC009065.4', 'ENSG00000260782_AC007225.1', 'ENSG00000260784_AC026150.2', 'ENSG00000260790_AC092338.2', 'ENSG00000260793_AC003102.1', 'ENSG00000260796_AC145285.3', 'ENSG00000260802_SERTM2', 'ENSG00000260804_LINC01963', 'ENSG00000260805_AC092803.2', 'ENSG00000260806_AL163051.1', 'ENSG00000260807_AC009041.2', 'ENSG00000260816_AC027279.1', 'ENSG00000260822_AC004656.1', 'ENSG00000260830_AL135744.1', 'ENSG00000260852_FBXL19-AS1', 'ENSG00000260853_AC109460.2', 'ENSG00000260855_AL591848.4', 'ENSG00000260865_AC010287.1', 'ENSG00000260886_TAT-AS1', 'ENSG00000260898_ADPGK-AS1', 'ENSG00000260907_AC015818.2', 'ENSG00000260908_AC009093.3', 'ENSG00000260911_AC135050.3', 'ENSG00000260912_AL158206.1', 'ENSG00000260916_CCPG1', 'ENSG00000260917_AL158212.3', 'ENSG00000260918_AC107398.3', 'ENSG00000260920_AL031985.3', 'ENSG00000260923_LINC02193', 'ENSG00000260924_LINC01311', 'ENSG00000260927_AC009107.2', 'ENSG00000260934_AC130456.5', 'ENSG00000260942_CAPN10-DT', 'ENSG00000260947_AL356489.2', 'ENSG00000260948_AL390195.2', 'ENSG00000260949_AP006545.1', 'ENSG00000260954_AL133297.1', 'ENSG00000260955_AC100821.2', 'ENSG00000260966_AP001486.2', 'ENSG00000260988_AC090260.1', 'ENSG00000260997_AC004847.1', 'ENSG00000261000_AC244034.2', 'ENSG00000261003_AL008628.1', 'ENSG00000261008_LINC01572', 'ENSG00000261019_AC010132.4', 'ENSG00000261033_AC005730.2', 'ENSG00000261040_WFDC21P', 'ENSG00000261052_SULT1A3', 'ENSG00000261056_AC079416.1', 'ENSG00000261060_AL160286.2', 'ENSG00000261061_AC092718.4', 'ENSG00000261064_LINC02256', 'ENSG00000261071_AL441883.1', 'ENSG00000261072_AC084783.1', 'ENSG00000261079_AC009053.2', 'ENSG00000261080_RUNX2-AS1', 'ENSG00000261087_AP003469.4', 'ENSG00000261093_AC141586.3', 'ENSG00000261094_AC007066.2', 'ENSG00000261096_AC073476.3', 'ENSG00000261098_AP000766.1', 'ENSG00000261101_AC234775.3', 'ENSG00000261102_ATP5MFP6', 'ENSG00000261105_LMO7-AS1', 'ENSG00000261114_AC012181.1', 'ENSG00000261118_AC092123.1', 'ENSG00000261123_AC009065.5', 'ENSG00000261126_RBFADN', 'ENSG00000261131_AC012186.2', 'ENSG00000261135_AL137802.2', 'ENSG00000261136_AC023908.3', 'ENSG00000261140_AC093525.4', 'ENSG00000261141_AC092718.5', 'ENSG00000261158_AC109597.2', 'ENSG00000261159_AC112484.3', 'ENSG00000261167_AC107027.3', 'ENSG00000261173_AC018845.3', 'ENSG00000261174_HMGB1P33', 'ENSG00000261177_AC135012.1', 'ENSG00000261183_SPINT1-AS1', 'ENSG00000261186_LINC01238', 'ENSG00000261187_AC079322.1', 'ENSG00000261188_Z95115.1', 'ENSG00000261200_AC136944.2', 'ENSG00000261202_Z83847.1', 'ENSG00000261211_AL031123.2', 'ENSG00000261216_AC007216.2', 'ENSG00000261218_AC099524.1', 'ENSG00000261220_AC103706.1', 'ENSG00000261221_ZNF865', 'ENSG00000261226_AC092384.3', 'ENSG00000261229_AC021483.2', 'ENSG00000261236_BOP1', 'ENSG00000261242_AL136038.3', 'ENSG00000261248_AC009120.4', 'ENSG00000261251_Z97055.2', 'ENSG00000261302_AC106779.1', 'ENSG00000261308_FIGNL2', 'ENSG00000261312_AC002550.1', 'ENSG00000261315_LARP4P', 'ENSG00000261324_AC010168.2', 'ENSG00000261326_LINC01355', 'ENSG00000261335_AC005837.1', 'ENSG00000261336_EIF4BP5', 'ENSG00000261338_AC021016.2', 'ENSG00000261341_AC010325.1', 'ENSG00000261342_AC006538.1', 'ENSG00000261349_AL031432.2', 'ENSG00000261351_AC116913.1', 'ENSG00000261359_PYCARD-AS1', 'ENSG00000261360_AC010491.1', 'ENSG00000261366_MANEA-DT', 'ENSG00000261367_AC012645.2', 'ENSG00000261371_PECAM1', 'ENSG00000261373_VPS9D1-AS1', 'ENSG00000261377_PDCD6IPP2', 'ENSG00000261386_AC027682.4', 'ENSG00000261390_MAFTRR', 'ENSG00000261404_AC138627.1', 'ENSG00000261407_AC013565.3', 'ENSG00000261408_TEN1-CDK3', 'ENSG00000261416_AC012645.3', 'ENSG00000261420_AL022069.1', 'ENSG00000261423_TMEM202-AS1', 'ENSG00000261428_AC097461.1', 'ENSG00000261430_AL031600.2', 'ENSG00000261431_AL023803.1', 'ENSG00000261433_AC002347.1', 'ENSG00000261438_AL157394.1', 'ENSG00000261441_AC124068.2', 'ENSG00000261448_AC109446.3', 'ENSG00000261451_AC104964.2', 'ENSG00000261455_LINC01003', 'ENSG00000261460_AC009690.2', 'ENSG00000261465_AC099518.4', 'ENSG00000261468_AC096921.2', 'ENSG00000261474_AC026471.4', 'ENSG00000261480_GOLGA8M', 'ENSG00000261481_AC022167.4', 'ENSG00000261485_PAN3-AS1', 'ENSG00000261487_AC135048.1', 'ENSG00000261490_AC005674.2', 'ENSG00000261504_LINC01686', 'ENSG00000261505_AL031714.1', 'ENSG00000261512_AC092368.3', 'ENSG00000261519_AC010542.4', 'ENSG00000261526_AC012615.1', 'ENSG00000261534_AL596244.1', 'ENSG00000261542_AC011978.2', 'ENSG00000261544_AC011939.2', 'ENSG00000261546_AC135782.3', 'ENSG00000261553_AL137782.1', 'ENSG00000261556_SMG1P7', 'ENSG00000261560_AC007216.3', 'ENSG00000261575_AC005829.1', 'ENSG00000261584_AL513548.1', 'ENSG00000261586_AC068987.4', 'ENSG00000261587_TMEM249', 'ENSG00000261596_AC005632.2', 'ENSG00000261604_AC114947.2', 'ENSG00000261609_GAN', 'ENSG00000261610_AP000265.1', 'ENSG00000261613_AC093525.6', 'ENSG00000261618_AC083837.1', 'ENSG00000261628_AC087481.2', 'ENSG00000261635_AC103988.1', 'ENSG00000261641_AL031600.3', 'ENSG00000261645_DISC1FP1', 'ENSG00000261649_GOLGA6L7', 'ENSG00000261652_C15orf65', 'ENSG00000261654_AL360270.2', 'ENSG00000261659_Z92544.2', 'ENSG00000261662_AL359752.1', 'ENSG00000261663_AC009065.8', 'ENSG00000261668_AC093591.2', 'ENSG00000261669_AC008731.1', 'ENSG00000261671_AL158211.1', 'ENSG00000261673_AC009075.1', 'ENSG00000261684_AC018362.1', 'ENSG00000261714_AC105137.1', 'ENSG00000261716_HIST2H2BC', 'ENSG00000261732_AL031708.1', 'ENSG00000261737_AL049597.2', 'ENSG00000261739_GOLGA8S', 'ENSG00000261740_BOLA2-SMG1P6', 'ENSG00000261758_AC117382.2', 'ENSG00000261759_AC099518.5', 'ENSG00000261760_AC140479.4', 'ENSG00000261764_KRT18P18', 'ENSG00000261766_AC133550.2', 'ENSG00000261770_AC006504.1', 'ENSG00000261775_AC012435.2', 'ENSG00000261777_AC012184.3', 'ENSG00000261783_AC009054.2', 'ENSG00000261794_GOLGA8H', 'ENSG00000261798_AL033527.3', 'ENSG00000261799_AC007406.5', 'ENSG00000261801_LOXL1-AS1', 'ENSG00000261821_AC090826.1', 'ENSG00000261822_AC018362.2', 'ENSG00000261823_AC084782.2', 'ENSG00000261824_LINC00662', 'ENSG00000261832_AC138894.1', 'ENSG00000261839_AL358933.1', 'ENSG00000261840_AC093249.6', 'ENSG00000261845_AC124283.1', 'ENSG00000261868_MFSD1P1', 'ENSG00000261879_AC087500.1', 'ENSG00000261884_AC040162.1', 'ENSG00000261886_AC005670.1', 'ENSG00000261889_AC108134.2', 'ENSG00000261916_AC027796.1', 'ENSG00000261924_AC127496.1', 'ENSG00000261970_AL391840.1', 'ENSG00000261971_MMP25-AS1', 'ENSG00000261997_AC007336.1', 'ENSG00000262001_DLGAP1-AS2', 'ENSG00000262020_AC007014.1', 'ENSG00000262038_AC007599.2', 'ENSG00000262049_AC139530.1', 'ENSG00000262050_AC005696.1', 'ENSG00000262074_SNORD3B-2', 'ENSG00000262075_DKFZP434A062', 'ENSG00000262079_AC007638.1', 'ENSG00000262089_AC040977.1', 'ENSG00000262136_AC092115.3', 'ENSG00000262155_LINC02175', 'ENSG00000262160_AC020978.5', 'ENSG00000262165_AC233723.1', 'ENSG00000262180_OCLM', 'ENSG00000262185_AC005736.1', 'ENSG00000262202_AC007952.4', 'ENSG00000262227_AC004771.3', 'ENSG00000262228_AC087392.3', 'ENSG00000262246_CORO7', 'ENSG00000262248_AC027796.2', 'ENSG00000262265_AC002558.3', 'ENSG00000262304_AC027796.3', 'ENSG00000262312_AC004494.1', 'ENSG00000262333_HNRNPA1P16', 'ENSG00000262362_AC004233.1', 'ENSG00000262370_AC108134.3', 'ENSG00000262380_AC026401.2', 'ENSG00000262402_MCUR1P1', 'ENSG00000262410_AC024361.1', 'ENSG00000262413_AC145207.2', 'ENSG00000262420_AC007613.1', 'ENSG00000262429_AC004771.4', 'ENSG00000262454_MIR193BHG', 'ENSG00000262456_AC006435.1', 'ENSG00000262468_LINC01569', 'ENSG00000262477_AC021224.1', 'ENSG00000262482_AC004034.1', 'ENSG00000262484_CCER2', 'ENSG00000262503_AC027763.1', 'ENSG00000262514_AC020978.6', 'ENSG00000262528_AL022341.2', 'ENSG00000262587_AC133552.2', 'ENSG00000262621_AC025283.2', 'ENSG00000262652_AC124283.3', 'ENSG00000262655_SPON1', 'ENSG00000262663_AC087222.1', 'ENSG00000262686_GLIS2-AS1', 'ENSG00000262691_AC040160.1', 'ENSG00000262700_AC133552.3', 'ENSG00000262712_AC012676.1', 'ENSG00000262728_AC123768.3', 'ENSG00000262732_AC010401.1', 'ENSG00000262766_AC135050.5', 'ENSG00000262777_AC032044.1', 'ENSG00000262803_AL354943.1', 'ENSG00000262814_MRPL12', 'ENSG00000262823_AC127521.1', 'ENSG00000262831_AC145207.3', 'ENSG00000262873_AC127496.5', 'ENSG00000262874_C19orf84', 'ENSG00000262877_AC110285.2', 'ENSG00000262879_AC068152.1', 'ENSG00000262899_AC004232.2', 'ENSG00000262902_MTCO1P40', 'ENSG00000262904_TMPOP2', 'ENSG00000262919_CCNQ', 'ENSG00000262943_ALOX12P2', 'ENSG00000262967_AC005921.2', 'ENSG00000262979_AC124319.1', 'ENSG00000263001_GTF2I', 'ENSG00000263002_ZNF234', 'ENSG00000263004_AC007114.1', 'ENSG00000263015_AC015853.2', 'ENSG00000263033_AC007220.1', 'ENSG00000263063_AC024361.2', 'ENSG00000263069_AC124319.2', 'ENSG00000263072_ZNF213-AS1', 'ENSG00000263080_AC009121.2', 'ENSG00000263081_AC211486.2', 'ENSG00000263089_AC007114.2', 'ENSG00000263096_AC007638.2', 'ENSG00000263105_AC009171.2', 'ENSG00000263120_AC004584.3', 'ENSG00000263142_LRRC37A17P', 'ENSG00000263179_HNRNPCP4', 'ENSG00000263220_AC015727.1', 'ENSG00000263235_AC006111.2', 'ENSG00000263264_AC119396.1', 'ENSG00000263266_RPS7P1', 'ENSG00000263272_AC004148.2', 'ENSG00000263276_AC020978.7', 'ENSG00000263280_AC003965.1', 'ENSG00000263293_THCAT158', 'ENSG00000263307_AC007216.4', 'ENSG00000263327_TAPT1-AS1', 'ENSG00000263331_AC008551.1', 'ENSG00000263335_AF001548.2', 'ENSG00000263400_TMEM220-AS1', 'ENSG00000263412_AC004477.1', 'ENSG00000263427_AC129492.2', 'ENSG00000263465_SRSF8', 'ENSG00000263513_FAM72C', 'ENSG00000263528_IKBKE', 'ENSG00000263531_AC130324.1', 'ENSG00000263535_AK4P1', 'ENSG00000263563_UBBP4', 'ENSG00000263585_AC145207.4', 'ENSG00000263603_AC127024.2', 'ENSG00000263606_AP000919.1', 'ENSG00000263627_PPP4R1-AS1', 'ENSG00000263639_MSMB', 'ENSG00000263657_AC023389.1', 'ENSG00000263724_DLGAP1-AS3', 'ENSG00000263731_AC145207.5', 'ENSG00000263753_LINC00667', 'ENSG00000263755_RN7SL498P', 'ENSG00000263766_AC025682.1', 'ENSG00000263781_AC024619.4', 'ENSG00000263786_AC022211.1', 'ENSG00000263812_LINC00908', 'ENSG00000263818_RDM1P5', 'ENSG00000263823_AC009831.1', 'ENSG00000263826_AC112907.3', 'ENSG00000263829_SINHCAFP1', 'ENSG00000263843_AC022211.2', 'ENSG00000263847_AP005899.1', 'ENSG00000263859_AC145207.6', 'ENSG00000263874_LINC00672', 'ENSG00000263883_EEF1DP7', 'ENSG00000263884_AP000845.1', 'ENSG00000263905_RN7SL555P', 'ENSG00000263916_AC100778.2', 'ENSG00000263924_AC022960.1', 'ENSG00000263934_SNORD3A', 'ENSG00000263941_RN7SL32P', 'ENSG00000263956_NBPF11', 'ENSG00000263961_RHEX', 'ENSG00000263968_RN7SL381P', 'ENSG00000263986_AC087393.2', 'ENSG00000264007_AC104564.2', 'ENSG00000264017_RN7SL336P', 'ENSG00000264047_RN7SL455P', 'ENSG00000264066_AC024267.1', 'ENSG00000264071_RN7SL531P', 'ENSG00000264078_AC114488.3', 'ENSG00000264107_AC138207.2', 'ENSG00000264112_AC015813.1', 'ENSG00000264176_MAGOH2P', 'ENSG00000264204_AGAP7P', 'ENSG00000264207_AC239868.1', 'ENSG00000264229_RNU4ATAC', 'ENSG00000264235_AP005329.1', 'ENSG00000264247_LINC00909', 'ENSG00000264254_AP001496.1', 'ENSG00000264275_RN7SL753P', 'ENSG00000264278_ZNF236-DT', 'ENSG00000264290_AC104564.3', 'ENSG00000264322_RN7SL448P', 'ENSG00000264343_NOTCH2NLA', 'ENSG00000264350_SNRPGP2', 'ENSG00000264364_DYNLL2', 'ENSG00000264365_AC023983.1', 'ENSG00000264384_RN7SL431P', 'ENSG00000264391_RN7SL208P', 'ENSG00000264443_AL445686.2', 'ENSG00000264451_AC036222.1', 'ENSG00000264456_AC138207.4', 'ENSG00000264522_OTUD7B', 'ENSG00000264538_SUZ12P1', 'ENSG00000264546_AC008026.3', 'ENSG00000264548_AC132872.2', 'ENSG00000264554_RN7SL793P', 'ENSG00000264558_AC015674.1', 'ENSG00000264569_AC137723.1', 'ENSG00000264575_LINC00526', 'ENSG00000264577_AC010761.1', 'ENSG00000264596_AP000897.1', 'ENSG00000264608_AC005726.3', 'ENSG00000264635_AP001020.2', 'ENSG00000264666_AC020558.1', 'ENSG00000264695_AC007922.1', 'ENSG00000264707_L3MBTL4-AS1', 'ENSG00000264739_AC093484.2', 'ENSG00000264743_DPRXP4', 'ENSG00000264772_AC016876.2', 'ENSG00000264808_AC068025.1', 'ENSG00000264812_AC132938.2', 'ENSG00000264853_AC011933.2', 'ENSG00000264885_AC026271.3', 'ENSG00000264895_AC006141.1', 'ENSG00000264916_RN7SL230P', 'ENSG00000264920_AC018521.5', 'ENSG00000264937_AC100830.2', 'ENSG00000264940_SNORD3C', 'ENSG00000264954_PRR29-AS1', 'ENSG00000264968_AC090844.2', 'ENSG00000265008_AC011825.4', 'ENSG00000265018_AGAP12P', 'ENSG00000265055_AC145343.1', 'ENSG00000265093_RN7SL246P', 'ENSG00000265094_AC007922.2', 'ENSG00000265100_AC005332.1', 'ENSG00000265123_RN7SL200P', 'ENSG00000265136_AC124283.4', 'ENSG00000265148_TSPOAP1-AS1', 'ENSG00000265158_LRRC37A7P', 'ENSG00000265168_AC005726.4', 'ENSG00000265185_SNORD3B-1', 'ENSG00000265188_AP001496.3', 'ENSG00000265194_AL359922.2', 'ENSG00000265206_AC004687.1', 'ENSG00000265241_RBM8A', 'ENSG00000265254_AC015917.2', 'ENSG00000265287_AC005726.5', 'ENSG00000265293_ARGFXP2', 'ENSG00000265354_TIMM23', 'ENSG00000265366_GLUD1P2', 'ENSG00000265369_PCAT18', 'ENSG00000265401_AC093484.3', 'ENSG00000265413_AP001094.2', 'ENSG00000265415_AC099850.3', 'ENSG00000265451_AC012447.1', 'ENSG00000265458_AC132938.3', 'ENSG00000265478_AC107982.3', 'ENSG00000265479_DTX2P1-UPK3BP1-PMS2P11', 'ENSG00000265485_LINC01915', 'ENSG00000265490_AP001178.2', 'ENSG00000265491_RNF115', 'ENSG00000265618_AC002094.2', 'ENSG00000265625_AC104564.5', 'ENSG00000265666_RARA-AS1', 'ENSG00000265681_RPL17', 'ENSG00000265683_SYPL1P2', 'ENSG00000265684_RN7SL378P', 'ENSG00000265688_MAFG-DT', 'ENSG00000265692_LINC01970', 'ENSG00000265713_AC023389.2', 'ENSG00000265727_RN7SL648P', 'ENSG00000265739_AC104984.3', 'ENSG00000265745_RN7SL375P', 'ENSG00000265749_AC135178.3', 'ENSG00000265750_AC090772.3', 'ENSG00000265763_ZNF488', 'ENSG00000265778_AC018413.1', 'ENSG00000265791_AC127024.4', 'ENSG00000265798_AC138207.6', 'ENSG00000265800_AC022211.3', 'ENSG00000265801_AC069366.2', 'ENSG00000265802_RN7SL49P', 'ENSG00000265808_SEC22B', 'ENSG00000265817_FSBP', 'ENSG00000265882_RN7SL73P', 'ENSG00000265912_AC037487.1', 'ENSG00000265943_AC090912.1', 'ENSG00000265967_AC100830.3', 'ENSG00000265972_TXNIP', 'ENSG00000265975_AC002091.1', 'ENSG00000266028_SRGAP2', 'ENSG00000266053_NDUFV2-AS1', 'ENSG00000266066_POLRMTP1', 'ENSG00000266074_BAHCC1', 'ENSG00000266075_RN7SL574P', 'ENSG00000266079_SNORA59B', 'ENSG00000266086_AC015813.2', 'ENSG00000266088_AC004585.1', 'ENSG00000266094_RASSF5', 'ENSG00000266111_AC068025.2', 'ENSG00000266126_AC005730.3', 'ENSG00000266127_ZNF415P1', 'ENSG00000266160_RN7SL612P', 'ENSG00000266171_AP001020.3', 'ENSG00000266173_STRADA', 'ENSG00000266185_RN7SL804P', 'ENSG00000266208_AC080112.1', 'ENSG00000266236_NARF-IT1', 'ENSG00000266237_AC121320.1', 'ENSG00000266258_LINC01909', 'ENSG00000266282_UBL5P2', 'ENSG00000266289_AC012213.3', 'ENSG00000266313_AC026254.2', 'ENSG00000266338_NBPF15', 'ENSG00000266340_AC138207.7', 'ENSG00000266369_AC090774.2', 'ENSG00000266371_AC079915.1', 'ENSG00000266378_AC005224.3', 'ENSG00000266385_AC005899.5', 'ENSG00000266389_AC002091.2', 'ENSG00000266401_AP002478.1', 'ENSG00000266402_SNHG25', 'ENSG00000266405_CBX3P2', 'ENSG00000266412_NCOA4', 'ENSG00000266420_RN7SL118P', 'ENSG00000266441_AP005205.2', 'ENSG00000266445_NARF-AS1', 'ENSG00000266467_RN7SL220P', 'ENSG00000266469_AC005288.1', 'ENSG00000266472_MRPS21', 'ENSG00000266473_AC007448.3', 'ENSG00000266490_AC127024.5', 'ENSG00000266495_AC011731.1', 'ENSG00000266498_AC055811.3', 'ENSG00000266501_AC025198.1', 'ENSG00000266524_GDF10', 'ENSG00000266527_AC005697.2', 'ENSG00000266538_AC005838.2', 'ENSG00000266598_AC037487.2', 'ENSG00000266642_AC024267.6', 'ENSG00000266644_AC103810.5', 'ENSG00000266648_SETP3', 'ENSG00000266651_AC093484.4', 'ENSG00000266677_AC087164.1', 'ENSG00000266680_AL135905.1', 'ENSG00000266709_MGC12916', 'ENSG00000266714_MYO15B', 'ENSG00000266718_AC079336.5', 'ENSG00000266744_AC005304.3', 'ENSG00000266777_SH3GL1P1', 'ENSG00000266783_AP005136.2', 'ENSG00000266801_AC009137.2', 'ENSG00000266805_AP005432.1', 'ENSG00000266821_AC018521.7', 'ENSG00000266839_AC008088.1', 'ENSG00000266850_AC090912.2', 'ENSG00000266865_AC138207.8', 'ENSG00000266896_AL354892.3', 'ENSG00000266900_AC027514.1', 'ENSG00000266904_LINC00663', 'ENSG00000266910_AC008507.1', 'ENSG00000266916_ZNF793-AS1', 'ENSG00000266918_AC091132.3', 'ENSG00000266921_AC006213.1', 'ENSG00000266962_AC067852.2', 'ENSG00000266964_FXYD1', 'ENSG00000266965_AC090220.1', 'ENSG00000266967_AARSD1', 'ENSG00000266969_AP002449.1', 'ENSG00000266970_AC061992.1', 'ENSG00000266983_AC011444.1', 'ENSG00000266987_AC104984.6', 'ENSG00000266990_AC004528.1', 'ENSG00000266993_AL050343.1', 'ENSG00000266998_AC111182.1', 'ENSG00000267002_AC060780.1', 'ENSG00000267006_AC008507.2', 'ENSG00000267007_AC012615.2', 'ENSG00000267011_AC011498.1', 'ENSG00000267023_LRRC37A16P', 'ENSG00000267024_AC008747.1', 'ENSG00000267033_AC020911.1', 'ENSG00000267040_AC027097.1', 'ENSG00000267041_ZNF850', 'ENSG00000267042_AC100793.2', 'ENSG00000267058_AC006213.2', 'ENSG00000267060_PTGES3L', 'ENSG00000267062_AC018761.2', 'ENSG00000267064_UXT-AS1', 'ENSG00000267072_NAGPA-AS1', 'ENSG00000267079_AP001269.2', 'ENSG00000267080_ASB16-AS1', 'ENSG00000267083_KRT18P61', 'ENSG00000267088_AC087683.1', 'ENSG00000267096_AC008735.1', 'ENSG00000267100_ILF3-DT', 'ENSG00000267102_AC060766.1', 'ENSG00000267105_AC011511.3', 'ENSG00000267106_ZNF561-AS1', 'ENSG00000267112_AC098848.1', 'ENSG00000267114_AC011481.1', 'ENSG00000267115_AC022148.2', 'ENSG00000267119_RPL10P15', 'ENSG00000267121_AC008105.3', 'ENSG00000267123_LINC02081', 'ENSG00000267127_AC090360.1', 'ENSG00000267130_AC008738.2', 'ENSG00000267135_AD000091.1', 'ENSG00000267136_AP005131.1', 'ENSG00000267152_AC093227.1', 'ENSG00000267160_AC091152.2', 'ENSG00000267169_AC022098.1', 'ENSG00000267174_AC011472.2', 'ENSG00000267185_PTP4A2P1', 'ENSG00000267198_AC091132.4', 'ENSG00000267199_AP001029.2', 'ENSG00000267201_LINC01775', 'ENSG00000267213_AC007773.1', 'ENSG00000267216_AC020915.1', 'ENSG00000267221_C17orf113', 'ENSG00000267222_AC107993.1', 'ENSG00000267226_AC104971.1', 'ENSG00000267244_AC012615.6', 'ENSG00000267248_AC025048.2', 'ENSG00000267249_AP005482.3', 'ENSG00000267253_AC055813.1', 'ENSG00000267254_ZNF790-AS1', 'ENSG00000267260_AC020928.1', 'ENSG00000267265_AC011476.3', 'ENSG00000267270_PARD6G-AS1', 'ENSG00000267272_LINC01140', 'ENSG00000267274_AC008770.3', 'ENSG00000267277_AC024575.1', 'ENSG00000267278_MAP3K14-AS1', 'ENSG00000267279_AC090409.1', 'ENSG00000267281_AC023509.3', 'ENSG00000267282_AC011481.2', 'ENSG00000267283_AC005306.1', 'ENSG00000267289_AC008752.2', 'ENSG00000267293_AC012569.1', 'ENSG00000267296_CEBPA-DT', 'ENSG00000267298_AC006116.5', 'ENSG00000267302_RNFT1-DT', 'ENSG00000267308_LINC01764', 'ENSG00000267309_AC092295.2', 'ENSG00000267311_AC007673.1', 'ENSG00000267312_AC015911.4', 'ENSG00000267317_AC027307.2', 'ENSG00000267321_LINC02001', 'ENSG00000267322_SNHG22', 'ENSG00000267340_AC060780.2', 'ENSG00000267342_AC087289.2', 'ENSG00000267348_GEMIN7-AS1', 'ENSG00000267353_AC020928.2', 'ENSG00000267364_AC022706.1', 'ENSG00000267368_UPK3BL1', 'ENSG00000267370_AC008752.3', 'ENSG00000267374_AC016205.1', 'ENSG00000267383_AC011447.3', 'ENSG00000267385_AC011498.4', 'ENSG00000267390_AC036176.1', 'ENSG00000267397_AC090229.1', 'ENSG00000267402_TCF4-AS2', 'ENSG00000267412_AC092068.2', 'ENSG00000267416_AC025048.4', 'ENSG00000267419_AC011477.1', 'ENSG00000267421_AC005498.2', 'ENSG00000267422_AC016582.2', 'ENSG00000267423_AC005616.1', 'ENSG00000267430_AC036176.2', 'ENSG00000267439_AD000671.3', 'ENSG00000267453_LINC01835', 'ENSG00000267454_ZNF582-AS1', 'ENSG00000267457_AC004223.2', 'ENSG00000267458_AC092069.1', 'ENSG00000267469_AC005944.1', 'ENSG00000267470_ZNF571-AS1', 'ENSG00000267474_AC008569.2', 'ENSG00000267475_AC008736.1', 'ENSG00000267481_AC011477.2', 'ENSG00000267484_AC027319.1', 'ENSG00000267493_CIRBP-AS1', 'ENSG00000267498_AC007786.1', 'ENSG00000267500_ZNF887P', 'ENSG00000267506_AC021683.2', 'ENSG00000267508_ZNF285', 'ENSG00000267510_AC011451.1', 'ENSG00000267512_AC011446.1', 'ENSG00000267515_AP001029.3', 'ENSG00000267519_AC020916.1', 'ENSG00000267523_AC008735.2', 'ENSG00000267526_AC005702.2', 'ENSG00000267532_MIR497HG', 'ENSG00000267534_S1PR2', 'ENSG00000267541_MTCO2P2', 'ENSG00000267544_AC007229.1', 'ENSG00000267546_AC015802.4', 'ENSG00000267547_AC060766.4', 'ENSG00000267549_AC006116.8', 'ENSG00000267551_AC005264.1', 'ENSG00000267563_AC011471.2', 'ENSG00000267571_AC104532.2', 'ENSG00000267575_AC006504.5', 'ENSG00000267576_AC011472.3', 'ENSG00000267580_AC008738.3', 'ENSG00000267586_LINC00907', 'ENSG00000267598_AC011446.2', 'ENSG00000267607_AC011511.5', 'ENSG00000267632_AC067852.3', 'ENSG00000267633_AC008686.1', 'ENSG00000267637_AC040904.1', 'ENSG00000267640_AC016582.3', 'ENSG00000267649_AC010327.4', 'ENSG00000267655_AC125437.1', 'ENSG00000267658_AC099811.3', 'ENSG00000267666_AC004156.1', 'ENSG00000267672_AC010632.2', 'ENSG00000267673_FDX2', 'ENSG00000267679_EIF5AP2', 'ENSG00000267680_ZNF224', 'ENSG00000267698_AC002116.2', 'ENSG00000267702_AP005131.6', 'ENSG00000267707_AC015961.2', 'ENSG00000267710_EDDM13', 'ENSG00000267717_SRSF10P1', 'ENSG00000267724_AC012254.3', 'ENSG00000267740_AC024592.3', 'ENSG00000267742_SINHCAFP2', 'ENSG00000267745_AC060766.7', 'ENSG00000267749_AC092068.3', 'ENSG00000267751_AC009005.1', 'ENSG00000267757_EML2-AS1', 'ENSG00000267758_AC099811.4', 'ENSG00000267766_AC022726.1', 'ENSG00000267769_AC011498.6', 'ENSG00000267780_AC021594.2', 'ENSG00000267787_AC027097.2', 'ENSG00000267793_AC009977.1', 'ENSG00000267796_LIN37', 'ENSG00000267801_AC087289.5', 'ENSG00000267809_NDUFV2P1', 'ENSG00000267811_AP001160.2', 'ENSG00000267834_AL592211.1', 'ENSG00000267838_AC245884.8', 'ENSG00000267855_NDUFA7', 'ENSG00000267858_MZF1-AS1', 'ENSG00000267871_ZNF460-AS1', 'ENSG00000267882_AL031666.2', 'ENSG00000267886_AC074135.1', 'ENSG00000267892_AC022144.1', 'ENSG00000267904_AC024075.1', 'ENSG00000267920_SNX6P1', 'ENSG00000267939_AC008946.1', 'ENSG00000267940_AC022762.2', 'ENSG00000267959_MIR3188', 'ENSG00000267980_AC007292.1', 'ENSG00000268001_CARD8-AS1', 'ENSG00000268006_PTOV1-AS1', 'ENSG00000268027_AC243960.1', 'ENSG00000268030_AC005253.1', 'ENSG00000268034_AC243960.2', 'ENSG00000268043_NBPF12', 'ENSG00000268051_AC008395.1', 'ENSG00000268061_NAPA-AS1', 'ENSG00000268066_FMR1-AS1', 'ENSG00000268069_AC004466.1', 'ENSG00000268081_AC123912.1', 'ENSG00000268087_AC008764.2', 'ENSG00000268093_AC022154.1', 'ENSG00000268105_AC124856.1', 'ENSG00000268117_VN1R84P', 'ENSG00000268119_AC010615.2', 'ENSG00000268129_AC026304.1', 'ENSG00000268154_RF00017', 'ENSG00000268186_ZNF114-AS1', 'ENSG00000268189_AC005785.1', 'ENSG00000268201_AC020915.2', 'ENSG00000268204_AC008763.1', 'ENSG00000268205_AC005261.1', 'ENSG00000268218_AC137932.3', 'ENSG00000268220_AC008040.5', 'ENSG00000268230_AC012313.3', 'ENSG00000268240_AC123912.2', 'ENSG00000268292_AC006547.3', 'ENSG00000268350_FAM156A', 'ENSG00000268357_VN1R81P', 'ENSG00000268362_AC092279.1', 'ENSG00000268364_SMC5-AS1', 'ENSG00000268375_AC010325.2', 'ENSG00000268379_AC025588.1', 'ENSG00000268400_AC008763.2', 'ENSG00000268403_AC132192.2', 'ENSG00000268412_TRMT112P6', 'ENSG00000268423_AC093503.1', 'ENSG00000268433_MTDHP3', 'ENSG00000268438_BNIP3P27', 'ENSG00000268469_BNIP3P38', 'ENSG00000268471_MIR4453HG', 'ENSG00000268472_AP002884.4', 'ENSG00000268509_AC026202.3', 'ENSG00000268516_AC020915.3', 'ENSG00000268521_VN1R83P', 'ENSG00000268536_AC005523.1', 'ENSG00000268555_AC123912.4', 'ENSG00000268565_AC005339.1', 'ENSG00000268568_AC007228.1', 'ENSG00000268573_AC011815.1', 'ENSG00000268575_AL031282.2', 'ENSG00000268583_AC011466.1', 'ENSG00000268584_AC073389.1', 'ENSG00000268592_RAET1E-AS1', 'ENSG00000268603_AC053503.5', 'ENSG00000268621_IGFL2-AS1', 'ENSG00000268649_AL132655.2', 'ENSG00000268650_AC005759.1', 'ENSG00000268658_LINC00664', 'ENSG00000268659_AL589863.1', 'ENSG00000268673_AC004597.1', 'ENSG00000268678_AC005261.2', 'ENSG00000268683_AC020910.1', 'ENSG00000268713_AC005261.3', 'ENSG00000268734_AC245128.3', 'ENSG00000268744_AC008758.4', 'ENSG00000268746_AC010519.1', 'ENSG00000268751_SCGB1B2P', 'ENSG00000268758_ADGRE4P', 'ENSG00000268798_AC027307.3', 'ENSG00000268804_LINC02132', 'ENSG00000268810_AC007193.1', 'ENSG00000268812_AC004264.1', 'ENSG00000268836_Z69706.1', 'ENSG00000268854_AC020909.3', 'ENSG00000268858_AL118506.1', 'ENSG00000268889_AC008750.7', 'ENSG00000268895_A1BG-AS1', 'ENSG00000268912_AC012313.5', 'ENSG00000268945_AC010422.2', 'ENSG00000268947_AC002128.1', 'ENSG00000268975_MIA-RAB4B', 'ENSG00000268996_MAN1B1-DT', 'ENSG00000269001_AC092070.2', 'ENSG00000269028_MTRNR2L12', 'ENSG00000269038_AP001462.1', 'ENSG00000269044_AC024075.2', 'ENSG00000269054_AC012313.6', 'ENSG00000269069_AC007842.1', 'ENSG00000269106_AC012313.7', 'ENSG00000269148_AC092301.1', 'ENSG00000269172_AC011443.1', 'ENSG00000269176_AP001160.3', 'ENSG00000269190_FBXO17', 'ENSG00000269194_AC006942.1', 'ENSG00000269210_AC019171.1', 'ENSG00000269220_LINC00528', 'ENSG00000269226_TMSB15B', 'ENSG00000269235_ZNF350-AS1', 'ENSG00000269242_AC010422.3', 'ENSG00000269243_AC008894.2', 'ENSG00000269246_AC011445.2', 'ENSG00000269292_AC093503.2', 'ENSG00000269293_ZSCAN16-AS1', 'ENSG00000269296_AC005614.1', 'ENSG00000269313_MAGIX', 'ENSG00000269318_AC007292.2', 'ENSG00000269335_IKBKG', 'ENSG00000269343_ZNF587B', 'ENSG00000269374_AC011497.2', 'ENSG00000269378_AC022149.1', 'ENSG00000269386_RAB11B-AS1', 'ENSG00000269388_AC018755.3', 'ENSG00000269397_AC011503.2', 'ENSG00000269399_AC008764.6', 'ENSG00000269404_SPIB', 'ENSG00000269416_LINC01224', 'ENSG00000269421_ZNF92P3', 'ENSG00000269439_AC010618.3', 'ENSG00000269446_AC006967.3', 'ENSG00000269463_AP001160.4', 'ENSG00000269473_AC012313.8', 'ENSG00000269481_AC010319.4', 'ENSG00000269482_Z69720.1', 'ENSG00000269489_AL589765.6', 'ENSG00000269514_AC024257.3', 'ENSG00000269556_TMEM185A', 'ENSG00000269559_AC093677.2', 'ENSG00000269578_AC008764.7', 'ENSG00000269600_AC016629.2', 'ENSG00000269604_AC005523.2', 'ENSG00000269609_RPARP-AS1', 'ENSG00000269646_AC010487.2', 'ENSG00000269688_AC008982.2', 'ENSG00000269696_AC005498.3', 'ENSG00000269713_NBPF9', 'ENSG00000269720_CCDC194', 'ENSG00000269737_AL691432.1', 'ENSG00000269743_SLC25A53', 'ENSG00000269793_ZIM2-AS1', 'ENSG00000269800_PLEKHA3P1', 'ENSG00000269807_AC007292.3', 'ENSG00000269813_AC010336.6', 'ENSG00000269814_AC008403.3', 'ENSG00000269815_AC010463.3', 'ENSG00000269821_KCNQ1OT1', 'ENSG00000269825_AC022150.4', 'ENSG00000269834_ZNF528-AS1', 'ENSG00000269837_IPO5P1', 'ENSG00000269845_AC092364.2', 'ENSG00000269846_AL136172.1', 'ENSG00000269858_EGLN2', 'ENSG00000269867_AC010326.3', 'ENSG00000269873_AC245884.10', 'ENSG00000269886_AC022382.1', 'ENSG00000269887_AL391001.1', 'ENSG00000269889_AC078802.1', 'ENSG00000269890_AL353593.1', 'ENSG00000269892_AC125494.2', 'ENSG00000269893_SNHG8', 'ENSG00000269894_AC018809.1', 'ENSG00000269896_AL513477.1', 'ENSG00000269899_AC025857.2', 'ENSG00000269900_RMRP', 'ENSG00000269902_AC234772.2', 'ENSG00000269903_AC025165.4', 'ENSG00000269906_AL606834.1', 'ENSG00000269907_AL158827.2', 'ENSG00000269910_AL049840.2', 'ENSG00000269911_FAM226B', 'ENSG00000269918_AF131215.6', 'ENSG00000269921_AC068620.1', 'ENSG00000269924_AC024451.4', 'ENSG00000269925_Z98884.2', 'ENSG00000269927_AC004817.3', 'ENSG00000269929_AL158152.1', 'ENSG00000269930_AC091057.3', 'ENSG00000269933_AL031429.2', 'ENSG00000269937_AC093525.7', 'ENSG00000269938_AC068790.2', 'ENSG00000269939_PCF11-AS1', 'ENSG00000269940_AL049840.3', 'ENSG00000269947_AC135178.5', 'ENSG00000269949_AC069307.1', 'ENSG00000269951_AC090181.2', 'ENSG00000269958_AL049840.4', 'ENSG00000269961_AC010359.1', 'ENSG00000269964_MEI4', 'ENSG00000269967_AL136115.2', 'ENSG00000269968_AC006064.4', 'ENSG00000269970_AL162424.1', 'ENSG00000269971_AL020997.2', 'ENSG00000269973_AC010969.2', 'ENSG00000269974_AC091057.4', 'ENSG00000269982_AC018809.2', 'ENSG00000269983_AC146944.4', 'ENSG00000269984_AC078795.1', 'ENSG00000269987_AC004542.2', 'ENSG00000269997_AC068790.3', 'ENSG00000270006_AC010531.6', 'ENSG00000270012_AC232271.1', 'ENSG00000270015_AC087481.3', 'ENSG00000270016_AC026150.3', 'ENSG00000270019_AC110769.2', 'ENSG00000270020_AC009108.3', 'ENSG00000270021_AC026691.1', 'ENSG00000270022_Z93241.1', 'ENSG00000270031_AL020997.3', 'ENSG00000270039_AC025165.5', 'ENSG00000270040_AL356055.1', 'ENSG00000270049_AC009061.2', 'ENSG00000270055_AC127502.2', 'ENSG00000270061_AC068790.5', 'ENSG00000270062_AL606834.2', 'ENSG00000270069_MIR222HG', 'ENSG00000270072_AC090559.2', 'ENSG00000270074_AC087203.3', 'ENSG00000270076_AF131215.7', 'ENSG00000270077_AP003117.1', 'ENSG00000270083_AL021878.2', 'ENSG00000270084_GAS5-AS1', 'ENSG00000270091_AC015726.1', 'ENSG00000270095_AC068790.6', 'ENSG00000270096_AC078795.2', 'ENSG00000270100_AC012065.4', 'ENSG00000270108_AL049840.5', 'ENSG00000270115_AL513327.3', 'ENSG00000270116_AP001429.1', 'ENSG00000270117_AP000769.2', 'ENSG00000270123_VTRNA2-1', 'ENSG00000270124_AC092127.2', 'ENSG00000270127_AC027020.2', 'ENSG00000270130_AC068790.7', 'ENSG00000270135_AC078795.3', 'ENSG00000270137_AF230666.2', 'ENSG00000270140_AC005520.3', 'ENSG00000270141_TERC', 'ENSG00000270147_AC068620.2', 'ENSG00000270157_AC004918.3', 'ENSG00000270164_LINC01480', 'ENSG00000270165_AC010530.1', 'ENSG00000270170_NCBP2-AS2', 'ENSG00000270175_AC023509.4', 'ENSG00000270177_AC104109.2', 'ENSG00000270179_AP002840.2', 'ENSG00000270182_AC004080.4', 'ENSG00000270190_AC068491.3', 'ENSG00000270194_AC097359.2', 'ENSG00000270195_AC016773.1', 'ENSG00000270231_NBPF8', 'ENSG00000270264_NDUFB8P2', 'ENSG00000270276_HIST2H4B', 'ENSG00000270277_AC009948.3', 'ENSG00000270332_SMC2-AS1', 'ENSG00000270335_AC093159.2', 'ENSG00000270344_POC1B-AS1', 'ENSG00000270361_AL451085.1', 'ENSG00000270362_HMGN3-AS1', 'ENSG00000270381_AL138963.2', 'ENSG00000270392_PFN1P2', 'ENSG00000270405_AC104692.2', 'ENSG00000270409_AC090950.1', 'ENSG00000270419_CAHM', 'ENSG00000270426_AC099343.2', 'ENSG00000270427_NRBF2P5', 'ENSG00000270433_H3F3AP2', 'ENSG00000270479_BNIP3P37', 'ENSG00000270482_AC026367.1', 'ENSG00000270504_AL391422.4', 'ENSG00000270528_AC021171.1', 'ENSG00000270533_CR382285.1', 'ENSG00000270557_AC013731.1', 'ENSG00000270558_AC025449.1', 'ENSG00000270562_AC097634.1', 'ENSG00000270574_AC010680.2', 'ENSG00000270589_AL158163.1', 'ENSG00000270604_HCG17', 'ENSG00000270605_AL353622.1', 'ENSG00000270629_NBPF14', 'ENSG00000270638_AL023806.1', 'ENSG00000270640_AC104695.3', 'ENSG00000270641_TSIX', 'ENSG00000270647_TAF15', 'ENSG00000270659_AC079610.2', 'ENSG00000270673_YTHDF3-AS1', 'ENSG00000270681_AC095055.1', 'ENSG00000270690_AC105129.3', 'ENSG00000270696_AC005034.3', 'ENSG00000270704_AC124312.4', 'ENSG00000270720_AC104785.1', 'ENSG00000270722_RF00003', 'ENSG00000270728_AL035413.2', 'ENSG00000270742_AC096947.1', 'ENSG00000270755_AL136141.1', 'ENSG00000270761_AL355353.1', 'ENSG00000270781_AC091133.5', 'ENSG00000270792_AL050403.2', 'ENSG00000270800_RPS10-NUDT3', 'ENSG00000270802_AC005776.1', 'ENSG00000270804_AC010326.4', 'ENSG00000270806_C17orf50', 'ENSG00000270813_NANOGNBP3', 'ENSG00000270816_LINC00221', 'ENSG00000270820_AC016727.1', 'ENSG00000270823_AC007938.2', 'ENSG00000270832_AC092120.2', 'ENSG00000270863_DDX55P1', 'ENSG00000270871_AC015849.3', 'ENSG00000270878_AL136038.4', 'ENSG00000270885_RASL10B', 'ENSG00000270894_AC015849.4', 'ENSG00000270933_AC010719.1', 'ENSG00000270951_AL121917.2', 'ENSG00000270953_AC007938.3', 'ENSG00000270956_AC009948.4', 'ENSG00000270959_LPP-AS2', 'ENSG00000270964_AC016355.1', 'ENSG00000270972_AC136475.9', 'ENSG00000270975_MAGOH3P', 'ENSG00000270977_AC015849.5', 'ENSG00000270986_HMGB1P51', 'ENSG00000270988_AC019257.2', 'ENSG00000270996_AC005034.4', 'ENSG00000271009_AC116667.1', 'ENSG00000271011_AC010680.3', 'ENSG00000271029_AC135178.6', 'ENSG00000271040_AL390955.2', 'ENSG00000271092_TMEM56-RWDD3', 'ENSG00000271100_AP000753.3', 'ENSG00000271109_AC008555.5', 'ENSG00000271119_AC026412.3', 'ENSG00000271122_AC018647.2', 'ENSG00000271127_AP000526.1', 'ENSG00000271133_AC004130.1', 'ENSG00000271141_AC010680.4', 'ENSG00000271147_ARMCX5-GPRASP2', 'ENSG00000271151_AC016737.1', 'ENSG00000271161_BOLA2P2', 'ENSG00000271200_AC099791.2', 'ENSG00000271204_AC016831.4', 'ENSG00000271228_AL121655.1', 'ENSG00000271265_AL355297.3', 'ENSG00000271270_TMCC1-AS1', 'ENSG00000271278_ELOCP33', 'ENSG00000271303_SRXN1', 'ENSG00000271327_AC010201.2', 'ENSG00000271335_AL117336.3', 'ENSG00000271344_AC018638.6', 'ENSG00000271347_AC124312.5', 'ENSG00000271361_HTATSF1P2', 'ENSG00000271380_AL451085.2', 'ENSG00000271383_NBPF19', 'ENSG00000271387_AL445228.2', 'ENSG00000271392_AC006237.1', 'ENSG00000271425_NBPF10', 'ENSG00000271427_AL358072.1', 'ENSG00000271437_AL356423.1', 'ENSG00000271447_MMP28', 'ENSG00000271452_AC005034.5', 'ENSG00000271500_AC005183.1', 'ENSG00000271503_CCL5', 'ENSG00000271533_Z83843.1', 'ENSG00000271550_BNIP3P11', 'ENSG00000271551_AL355297.4', 'ENSG00000271553_AC018638.7', 'ENSG00000271555_AC113139.1', 'ENSG00000271576_AL359504.2', 'ENSG00000271581_AL671883.2', 'ENSG00000271584_LINC02550', 'ENSG00000271590_AC108463.3', 'ENSG00000271598_AC008739.3', 'ENSG00000271601_LIX1L', 'ENSG00000271605_MILR1', 'ENSG00000271614_ATP2B1-AS1', 'ENSG00000271626_AC104763.2', 'ENSG00000271631_AL139041.1', 'ENSG00000271639_AC019072.1', 'ENSG00000271643_AC112220.2', 'ENSG00000271646_AC099343.3', 'ENSG00000271670_AC010998.2', 'ENSG00000271699_SNX29P2', 'ENSG00000271714_AC010501.1', 'ENSG00000271725_AC103858.1', 'ENSG00000271736_AL138900.3', 'ENSG00000271737_AC008608.2', 'ENSG00000271741_AC114490.2', 'ENSG00000271746_AL031848.2', 'ENSG00000271751_AP003392.5', 'ENSG00000271754_AL355802.2', 'ENSG00000271757_AP002360.2', 'ENSG00000271761_AL021368.1', 'ENSG00000271771_AC139792.1', 'ENSG00000271780_AL118558.3', 'ENSG00000271781_AC026740.1', 'ENSG00000271784_AL031055.1', 'ENSG00000271788_AC008875.1', 'ENSG00000271789_AL080317.2', 'ENSG00000271793_AL589666.1', 'ENSG00000271795_AC011337.1', 'ENSG00000271797_AC008494.3', 'ENSG00000271806_AL590822.2', 'ENSG00000271815_AC008897.3', 'ENSG00000271816_BMS1P4', 'ENSG00000271818_RN7SKP4', 'ENSG00000271828_AC008937.3', 'ENSG00000271833_AL445222.1', 'ENSG00000271843_AC012557.1', 'ENSG00000271849_AC012603.1', 'ENSG00000271851_AC087501.4', 'ENSG00000271853_AL162258.1', 'ENSG00000271855_AC073195.1', 'ENSG00000271856_LINC01215', 'ENSG00000271857_AL096865.1', 'ENSG00000271858_CYB561D2', 'ENSG00000271862_AC104118.1', 'ENSG00000271869_AC026979.2', 'ENSG00000271870_AC024060.1', 'ENSG00000271882_AP001330.5', 'ENSG00000271888_AL136162.1', 'ENSG00000271895_AL109811.3', 'ENSG00000271913_AL035530.2', 'ENSG00000271914_AL139286.2', 'ENSG00000271917_AL357568.2', 'ENSG00000271918_AC034236.2', 'ENSG00000271931_AL353135.1', 'ENSG00000271933_AL603756.1', 'ENSG00000271936_AC012073.1', 'ENSG00000271937_AC104187.1', 'ENSG00000271938_AC103724.4', 'ENSG00000271947_AC017076.1', 'ENSG00000271952_LINC01954', 'ENSG00000271959_AC100803.3', 'ENSG00000271963_AC026786.2', 'ENSG00000271964_AC090948.1', 'ENSG00000271966_AC021321.1', 'ENSG00000271967_AL583856.2', 'ENSG00000271971_AC120053.1', 'ENSG00000271976_AC012467.2', 'ENSG00000271978_AL359643.2', 'ENSG00000271980_AC012640.4', 'ENSG00000271983_AC023302.1', 'ENSG00000271986_RN7SL827P', 'ENSG00000271989_AL139424.1', 'ENSG00000271991_AC013400.1', 'ENSG00000271992_AL354872.2', 'ENSG00000271993_AC126118.1', 'ENSG00000272002_AC010904.2', 'ENSG00000272004_FO704657.1', 'ENSG00000272008_AL139274.2', 'ENSG00000272009_AL121944.1', 'ENSG00000272010_AC100814.1', 'ENSG00000272017_AL137784.2', 'ENSG00000272023_AC010240.3', 'ENSG00000272024_AC064807.4', 'ENSG00000272030_AL162258.2', 'ENSG00000272031_ANKRD34A', 'ENSG00000272033_AL136984.1', 'ENSG00000272037_AP002907.1', 'ENSG00000272040_AC010245.2', 'ENSG00000272043_AC016405.2', 'ENSG00000272047_GTF2H5', 'ENSG00000272049_AC091965.4', 'ENSG00000272054_AC007390.2', 'ENSG00000272056_AC013472.3', 'ENSG00000272057_AC016575.1', 'ENSG00000272068_AL365181.2', 'ENSG00000272072_AC004492.1', 'ENSG00000272076_AC090186.1', 'ENSG00000272077_AC124045.1', 'ENSG00000272079_AC004233.3', 'ENSG00000272081_AC008972.2', 'ENSG00000272084_AL137127.1', 'ENSG00000272086_AC025181.2', 'ENSG00000272092_AC087623.2', 'ENSG00000272097_AL024498.1', 'ENSG00000272102_AL133406.3', 'ENSG00000272103_AC026741.1', 'ENSG00000272106_AL691432.2', 'ENSG00000272112_AC011374.2', 'ENSG00000272114_AL136131.3', 'ENSG00000272115_AC233992.3', 'ENSG00000272128_AP006545.2', 'ENSG00000272129_AL359715.3', 'ENSG00000272130_AC091946.2', 'ENSG00000272137_AL451064.1', 'ENSG00000272140_AC022400.5', 'ENSG00000272142_LYRM4-AS1', 'ENSG00000272143_FGF14-AS2', 'ENSG00000272144_AC025171.4', 'ENSG00000272145_NFYC-AS1', 'ENSG00000272146_ARF4-AS1', 'ENSG00000272148_AC013403.2', 'ENSG00000272153_AL365330.1', 'ENSG00000272155_AC055822.1', 'ENSG00000272156_AC008280.3', 'ENSG00000272159_AC087623.3', 'ENSG00000272168_CASC15', 'ENSG00000272170_AL355385.1', 'ENSG00000272172_AC138696.2', 'ENSG00000272173_U47924.2', 'ENSG00000272181_AC012557.2', 'ENSG00000272182_AC135507.1', 'ENSG00000272186_AP003392.6', 'ENSG00000272189_AL024508.2', 'ENSG00000272192_AC100812.1', 'ENSG00000272195_AL356512.1', 'ENSG00000272196_HIST2H2AA4', 'ENSG00000272202_AC097358.2', 'ENSG00000272205_AL451050.2', 'ENSG00000272209_AL023583.1', 'ENSG00000272211_AC114760.2', 'ENSG00000272217_AL645940.1', 'ENSG00000272219_AC005072.1', 'ENSG00000272221_AL645933.2', 'ENSG00000272223_AL136304.1', 'ENSG00000272234_AC008945.1', 'ENSG00000272236_AL645939.4', 'ENSG00000272247_AC080013.5', 'ENSG00000272248_AL138831.2', 'ENSG00000272255_AC113361.1', 'ENSG00000272256_AC044849.1', 'ENSG00000272263_AC034198.2', 'ENSG00000272267_AC021242.3', 'ENSG00000272269_AL138724.1', 'ENSG00000272273_IER3-AS1', 'ENSG00000272275_AC092687.3', 'ENSG00000272277_AL031963.3', 'ENSG00000272288_AL451165.2', 'ENSG00000272301_AP002360.3', 'ENSG00000272305_AC096887.1', 'ENSG00000272308_AC104113.1', 'ENSG00000272316_AL021368.2', 'ENSG00000272323_AC026801.2', 'ENSG00000272325_NUDT3', 'ENSG00000272330_AC002044.1', 'ENSG00000272333_KMT2B', 'ENSG00000272334_AC011816.2', 'ENSG00000272335_AC093297.2', 'ENSG00000272338_AC067838.1', 'ENSG00000272341_AL137003.2', 'ENSG00000272342_LINC01115', 'ENSG00000272343_AC107952.2', 'ENSG00000272345_AL031775.1', 'ENSG00000272354_AC092354.1', 'ENSG00000272356_AL080317.3', 'ENSG00000272359_RNU4-89P', 'ENSG00000272361_AC005014.2', 'ENSG00000272366_AL158211.3', 'ENSG00000272368_AC074032.1', 'ENSG00000272369_AC008035.1', 'ENSG00000272370_AC092354.2', 'ENSG00000272374_Z97832.2', 'ENSG00000272375_AC026979.3', 'ENSG00000272379_AL008729.2', 'ENSG00000272382_AC025171.5', 'ENSG00000272383_AC006270.3', 'ENSG00000272384_AC016405.3', 'ENSG00000272391_POM121C', 'ENSG00000272395_IFNL4', 'ENSG00000272398_CD24', 'ENSG00000272402_AL031775.2', 'ENSG00000272405_AL365181.3', 'ENSG00000272412_RN7SL778P', 'ENSG00000272416_AC025175.1', 'ENSG00000272417_AC034229.4', 'ENSG00000272419_LINC01145', 'ENSG00000272420_AL513477.2', 'ENSG00000272425_AC009902.3', 'ENSG00000272426_BX284668.6', 'ENSG00000272432_AL031432.3', 'ENSG00000272434_AC137630.3', 'ENSG00000272440_AC080013.6', 'ENSG00000272444_AL118558.4', 'ENSG00000272447_AL135925.1', 'ENSG00000272449_AL139246.5', 'ENSG00000272455_AL391244.3', 'ENSG00000272456_AC087045.2', 'ENSG00000272459_AC139795.3', 'ENSG00000272461_AP005328.2', 'ENSG00000272462_U91328.1', 'ENSG00000272463_AL357054.4', 'ENSG00000272465_AL031768.1', 'ENSG00000272468_AL021807.1', 'ENSG00000272469_RN7SL760P', 'ENSG00000272476_AL024507.2', 'ENSG00000272477_AC144521.1', 'ENSG00000272489_AL132656.3', 'ENSG00000272498_AC090948.2', 'ENSG00000272501_AL662844.4', 'ENSG00000272502_AC104958.2', 'ENSG00000272505_AC104964.3', 'ENSG00000272506_AL357078.3', 'ENSG00000272509_AC087752.4', 'ENSG00000272518_AC036214.2', 'ENSG00000272523_LINC01023', 'ENSG00000272525_AC099522.2', 'ENSG00000272529_AC090948.3', 'ENSG00000272540_AL662797.1', 'ENSG00000272541_AL021368.3', 'ENSG00000272556_GTF2IP13', 'ENSG00000272562_AL512343.2', 'ENSG00000272563_AC016745.2', 'ENSG00000272567_AC109347.2', 'ENSG00000272568_AC005162.3', 'ENSG00000272572_AL138762.1', 'ENSG00000272574_AL596325.2', 'ENSG00000272576_AC027271.1', 'ENSG00000272578_AP000347.1', 'ENSG00000272583_AL592494.3', 'ENSG00000272588_AC139887.4', 'ENSG00000272599_AC016394.1', 'ENSG00000272602_ZNF595', 'ENSG00000272604_AC073073.2', 'ENSG00000272606_AC015982.1', 'ENSG00000272625_AP000919.4', 'ENSG00000272627_AC016542.1', 'ENSG00000272630_AL731563.3', 'ENSG00000272631_AC067750.1', 'ENSG00000272638_AC006027.1', 'ENSG00000272644_AC097468.3', 'ENSG00000272645_GTF2IP20', 'ENSG00000272646_AC079766.1', 'ENSG00000272650_AC110792.3', 'ENSG00000272654_AL358472.2', 'ENSG00000272655_POLR2J4', 'ENSG00000272656_AC024933.1', 'ENSG00000272661_AC021097.1', 'ENSG00000272662_AC073352.1', 'ENSG00000272663_AC093635.1', 'ENSG00000272666_U62317.1', 'ENSG00000272667_AC012306.2', 'ENSG00000272668_AL590560.1', 'ENSG00000272669_AL021707.6', 'ENSG00000272677_AC124016.1', 'ENSG00000272678_AC112503.1', 'ENSG00000272686_AC006333.2', 'ENSG00000272688_AP005329.3', 'ENSG00000272689_AC004832.4', 'ENSG00000272690_LINC02018', 'ENSG00000272692_AC010997.3', 'ENSG00000272693_AC073107.1', 'ENSG00000272696_AL359091.4', 'ENSG00000272701_MESTIT1', 'ENSG00000272702_AC010913.1', 'ENSG00000272707_AC046143.2', 'ENSG00000272711_AC019069.1', 'ENSG00000272716_AL121658.1', 'ENSG00000272717_AC112236.2', 'ENSG00000272719_AC006483.2', 'ENSG00000272720_AL022322.1', 'ENSG00000272721_AC131235.3', 'ENSG00000272733_AP000345.2', 'ENSG00000272734_ADIRF-AS1', 'ENSG00000272735_AC007881.3', 'ENSG00000272742_AC135457.1', 'ENSG00000272744_AC107214.2', 'ENSG00000272746_AP005131.7', 'ENSG00000272750_AL592148.3', 'ENSG00000272752_STAG3L5P-PVRIG2P-PILRB', 'ENSG00000272754_AL133245.1', 'ENSG00000272755_AC245297.2', 'ENSG00000272758_AC083798.2', 'ENSG00000272764_AL596094.1', 'ENSG00000272767_JMJD1C-AS1', 'ENSG00000272768_AC004854.2', 'ENSG00000272769_AC097532.2', 'ENSG00000272777_AC019131.2', 'ENSG00000272779_AC245060.4', 'ENSG00000272787_AC253536.6', 'ENSG00000272791_AC073389.3', 'ENSG00000272795_AC126283.1', 'ENSG00000272797_AC092954.1', 'ENSG00000272798_AL008721.1', 'ENSG00000272799_AC006238.1', 'ENSG00000272800_AC021851.1', 'ENSG00000272807_AC007038.2', 'ENSG00000272808_AC015712.6', 'ENSG00000272810_U91328.3', 'ENSG00000272812_AC004908.2', 'ENSG00000272817_AL359198.1', 'ENSG00000272821_U62317.2', 'ENSG00000272829_AC002470.1', 'ENSG00000272831_AC027644.3', 'ENSG00000272836_AL022328.1', 'ENSG00000272841_AL139393.2', 'ENSG00000272842_AL391834.1', 'ENSG00000272843_AC211476.2', 'ENSG00000272844_AC074044.1', 'ENSG00000272849_AC084018.1', 'ENSG00000272851_AC096772.1', 'ENSG00000272853_AC069544.1', 'ENSG00000272854_AC004839.1', 'ENSG00000272858_Z93930.3', 'ENSG00000272861_AC012360.2', 'ENSG00000272862_AC106052.1', 'ENSG00000272864_AC135803.1', 'ENSG00000272870_AC097534.2', 'ENSG00000272871_AL159169.2', 'ENSG00000272874_AL034548.1', 'ENSG00000272885_AC092574.1', 'ENSG00000272886_DCP1A', 'ENSG00000272888_LINC01578', 'ENSG00000272892_AL133551.1', 'ENSG00000272894_AC004982.2', 'ENSG00000272899_ATP6V1FNB', 'ENSG00000272902_TBC1D8-AS1', 'ENSG00000272905_AC018648.1', 'ENSG00000272906_AL353708.3', 'ENSG00000272908_AC006033.2', 'ENSG00000272909_AL122035.2', 'ENSG00000272913_AC009237.14', 'ENSG00000272914_AL359532.1', 'ENSG00000272918_AC005070.3', 'ENSG00000272927_AC107464.3', 'ENSG00000272931_AC099568.2', 'ENSG00000272933_AL391121.1', 'ENSG00000272936_AC096586.1', 'ENSG00000272941_AC083862.2', 'ENSG00000272942_AL022324.3', 'ENSG00000272948_AP001412.1', 'ENSG00000272953_AC092171.4', 'ENSG00000272954_AP000553.2', 'ENSG00000272963_OR7A19P', 'ENSG00000272966_AC064836.2', 'ENSG00000272967_AC073352.2', 'ENSG00000272969_AC024243.1', 'ENSG00000272973_AP000350.5', 'ENSG00000272977_AL008721.2', 'ENSG00000272979_AC093388.1', 'ENSG00000272980_Z94721.2', 'ENSG00000272983_AL117339.4', 'ENSG00000272986_AC009570.1', 'ENSG00000272989_LINC02012', 'ENSG00000272990_AC084036.1', 'ENSG00000272991_AF129408.1', 'ENSG00000272994_AC012360.3', 'ENSG00000273000_AP000347.2', 'ENSG00000273001_AL731533.2', 'ENSG00000273002_AL355388.2', 'ENSG00000273004_AL078644.1', 'ENSG00000273007_AC021205.3', 'ENSG00000273008_AC010864.1', 'ENSG00000273010_AL360270.3', 'ENSG00000273011_AC092681.3', 'ENSG00000273014_AC018645.2', 'ENSG00000273015_AC008124.1', 'ENSG00000273017_AP000240.1', 'ENSG00000273018_FAM106A', 'ENSG00000273026_AL358472.3', 'ENSG00000273027_AL844908.2', 'ENSG00000273033_LINC02035', 'ENSG00000273035_AC007684.1', 'ENSG00000273038_AL365203.2', 'ENSG00000273045_C2orf15', 'ENSG00000273055_AC005046.1', 'ENSG00000273058_AL359921.2', 'ENSG00000273059_AC239803.2', 'ENSG00000273061_CDC37L1-DT', 'ENSG00000273062_AL449106.1', 'ENSG00000273063_AC007250.1', 'ENSG00000273064_AC017083.1', 'ENSG00000273066_AL355987.4', 'ENSG00000273073_AC073869.3', 'ENSG00000273076_AL021707.7', 'ENSG00000273080_AC009309.1', 'ENSG00000273084_AC092171.5', 'ENSG00000273090_AC007378.1', 'ENSG00000273091_AP000255.1', 'ENSG00000273096_AL021707.8', 'ENSG00000273102_AP000569.1', 'ENSG00000273106_AC019129.2', 'ENSG00000273107_AL512598.1', 'ENSG00000273108_AL121929.2', 'ENSG00000273117_AC144652.1', 'ENSG00000273125_LINC01990', 'ENSG00000273129_PACERR', 'ENSG00000273136_NBPF26', 'ENSG00000273137_AL022328.2', 'ENSG00000273139_AC007663.3', 'ENSG00000273141_AP001269.4', 'ENSG00000273142_AC073335.2', 'ENSG00000273143_AL355512.1', 'ENSG00000273145_BX537318.1', 'ENSG00000273148_AL035563.1', 'ENSG00000273149_AL138963.3', 'ENSG00000273151_AC073957.3', 'ENSG00000273153_AC067747.1', 'ENSG00000273155_AC092587.1', 'ENSG00000273156_AC124016.2', 'ENSG00000273162_AL133215.2', 'ENSG00000273165_AL121652.1', 'ENSG00000273173_SNURF', 'ENSG00000273174_AC108673.2', 'ENSG00000273175_BX323046.1', 'ENSG00000273179_AC092535.4', 'ENSG00000273181_AC131235.4', 'ENSG00000273183_AC093726.2', 'ENSG00000273186_AL359091.5', 'ENSG00000273188_AL022328.3', 'ENSG00000273192_AL671710.1', 'ENSG00000273199_AP000692.2', 'ENSG00000273203_AC006946.2', 'ENSG00000273204_AC104506.1', 'ENSG00000273209_AC069148.1', 'ENSG00000273210_AP001437.1', 'ENSG00000273211_AC137630.4', 'ENSG00000273216_AC002059.1', 'ENSG00000273219_AC091736.1', 'ENSG00000273221_AL355816.2', 'ENSG00000273226_AL391834.2', 'ENSG00000273230_AC102953.2', 'ENSG00000273232_AC090912.3', 'ENSG00000273233_AC097724.1', 'ENSG00000273240_AC013468.1', 'ENSG00000273243_Z82243.1', 'ENSG00000273247_AC097376.2', 'ENSG00000273248_AC010997.4', 'ENSG00000273249_BX649632.1', 'ENSG00000273253_AL022328.4', 'ENSG00000273254_AF129075.2', 'ENSG00000273257_AC069200.1', 'ENSG00000273261_AC092953.2', 'ENSG00000273262_AL121928.1', 'ENSG00000273265_CNNM3-DT', 'ENSG00000273270_AC090114.2', 'ENSG00000273271_AP000254.1', 'ENSG00000273272_U62317.4', 'ENSG00000273275_AC017083.2', 'ENSG00000273284_AP001033.2', 'ENSG00000273289_AL121672.3', 'ENSG00000273295_AP000350.6', 'ENSG00000273297_AC009275.1', 'ENSG00000273300_AC000068.3', 'ENSG00000273302_AC016747.3', 'ENSG00000273305_AC009237.15', 'ENSG00000273306_AC018690.1', 'ENSG00000273311_DGCR11', 'ENSG00000273314_AC005229.4', 'ENSG00000273319_AC058791.1', 'ENSG00000273320_AC007032.1', 'ENSG00000273321_AC023983.2', 'ENSG00000273329_AC078846.1', 'ENSG00000273333_AL662884.1', 'ENSG00000273335_AP005432.2', 'ENSG00000273338_AC103591.3', 'ENSG00000273343_AC007663.4', 'ENSG00000273344_PAXIP1-AS1', 'ENSG00000273345_AC104109.4', 'ENSG00000273353_AL008718.3', 'ENSG00000273355_AP000894.4', 'ENSG00000273356_LINC02019', 'ENSG00000273361_AC021016.3', 'ENSG00000273363_AL353801.3', 'ENSG00000273366_Z83851.2', 'ENSG00000273367_AL355472.4', 'ENSG00000273373_AL355488.1', 'ENSG00000273374_AC069222.1', 'ENSG00000273375_AC055764.2', 'ENSG00000273381_AL158071.4', 'ENSG00000273382_AL356488.3', 'ENSG00000273384_AL137796.1', 'ENSG00000273387_AC005005.3', 'ENSG00000273391_AC083880.1', 'ENSG00000273394_AC128687.2', 'ENSG00000273399_AL159169.3', 'ENSG00000273402_AC004908.3', 'ENSG00000273406_AC245008.1', 'ENSG00000273416_AL732292.2', 'ENSG00000273424_AL008582.1', 'ENSG00000273428_AC004832.6', 'ENSG00000273432_AC004951.4', 'ENSG00000273437_AC108673.3', 'ENSG00000273447_AC004067.1', 'ENSG00000273448_AC006480.2', 'ENSG00000273449_AC093788.1', 'ENSG00000273451_AL031666.3', 'ENSG00000273455_AC072039.2', 'ENSG00000273456_AC064836.3', 'ENSG00000273466_AC012510.1', 'ENSG00000273472_AC096733.2', 'ENSG00000273485_AL139339.2', 'ENSG00000273486_AC096992.2', 'ENSG00000273489_AC008264.2', 'ENSG00000273521_AL162274.1', 'ENSG00000273524_RF00017', 'ENSG00000273559_CWC25', 'ENSG00000273565_AL691403.2', 'ENSG00000273568_AC131009.3', 'ENSG00000273576_AC009283.1', 'ENSG00000273590_SMIM11B', 'ENSG00000273604_EPOP', 'ENSG00000273611_ZNHIT3', 'ENSG00000273619_AL121832.2', 'ENSG00000273654_AC020904.2', 'ENSG00000273669_AC015819.1', 'ENSG00000273674_AC021752.1', 'ENSG00000273675_AL118556.2', 'ENSG00000273680_AC009318.2', 'ENSG00000273682_AC109583.2', 'ENSG00000273687_AC004223.3', 'ENSG00000273691_AC087284.1', 'ENSG00000273702_AC091271.1', 'ENSG00000273703_HIST1H2BM', 'ENSG00000273710_RF00017', 'ENSG00000273711_AC005520.5', 'ENSG00000273723_AL139089.1', 'ENSG00000273727_RF00003', 'ENSG00000273729_AC007686.3', 'ENSG00000273747_AC022558.3', 'ENSG00000273749_CYFIP1', 'ENSG00000273759_AL117379.1', 'ENSG00000273763_AC007318.2', 'ENSG00000273783_AL136040.1', 'ENSG00000273784_AL137058.2', 'ENSG00000273786_AC020658.4', 'ENSG00000273791_AC007204.1', 'ENSG00000273797_AL133445.2', 'ENSG00000273802_HIST1H2BG', 'ENSG00000273812_BX640514.2', 'ENSG00000273816_AC005695.3', 'ENSG00000273820_USP27X', 'ENSG00000273821_AL096828.3', 'ENSG00000273837_AC018755.4', 'ENSG00000273841_TAF9', 'ENSG00000273855_AC020658.5', 'ENSG00000273888_FRMD6-AS1', 'ENSG00000273891_AL731566.1', 'ENSG00000273893_AL133520.1', 'ENSG00000273899_NOL12', 'ENSG00000273951_AL031667.3', 'ENSG00000273965_AC243654.1', 'ENSG00000273973_AC025162.2', 'ENSG00000273979_RF00017', 'ENSG00000273983_HIST1H3G', 'ENSG00000273987_AC121761.2', 'ENSG00000273989_AC022079.1', 'ENSG00000273998_AL049794.1', 'ENSG00000274001_AL512506.1', 'ENSG00000274008_RF00017', 'ENSG00000274010_AC006011.2', 'ENSG00000274011_RF00017', 'ENSG00000274012_RN7SL2', 'ENSG00000274015_AL136038.5', 'ENSG00000274020_LINC01138', 'ENSG00000274021_AC024909.2', 'ENSG00000274026_FAM27E3', 'ENSG00000274038_AC007014.2', 'ENSG00000274068_AL449266.1', 'ENSG00000274070_CASTOR2', 'ENSG00000274092_AC106739.1', 'ENSG00000274093_AC009032.1', 'ENSG00000274104_AC020910.4', 'ENSG00000274105_AC084824.3', 'ENSG00000274135_RF00017', 'ENSG00000274173_AL035661.1', 'ENSG00000274180_NATD1', 'ENSG00000274184_AC011815.2', 'ENSG00000274191_AC026333.4', 'ENSG00000274210_RF00003', 'ENSG00000274211_SOCS7', 'ENSG00000274213_AC015912.3', 'ENSG00000274215_AC106028.4', 'ENSG00000274220_AC009163.6', 'ENSG00000274225_AP001065.1', 'ENSG00000274227_AC073575.2', 'ENSG00000274253_AC138649.1', 'ENSG00000274259_SYNGAP1-AS1', 'ENSG00000274265_AC245297.3', 'ENSG00000274267_HIST1H3B', 'ENSG00000274270_AL137060.1', 'ENSG00000274272_AC069281.2', 'ENSG00000274275_AC009831.3', 'ENSG00000274290_HIST1H2BE', 'ENSG00000274292_AC084018.2', 'ENSG00000274303_RF00100', 'ENSG00000274307_AC023449.2', 'ENSG00000274315_AC009318.3', 'ENSG00000274317_LINC02334', 'ENSG00000274330_AL160191.3', 'ENSG00000274333_CU633967.1', 'ENSG00000274340_AC032011.1', 'ENSG00000274341_AC005899.6', 'ENSG00000274349_ZNF658', 'ENSG00000274364_AL110115.1', 'ENSG00000274367_AC004233.4', 'ENSG00000274383_AC103691.1', 'ENSG00000274400_AC015967.1', 'ENSG00000274421_AL162390.1', 'ENSG00000274422_AC245060.5', 'ENSG00000274425_AC114271.1', 'ENSG00000274428_RF00003', 'ENSG00000274460_AC092119.2', 'ENSG00000274471_AC242376.2', 'ENSG00000274487_AC244154.1', 'ENSG00000274512_TBC1D3L', 'ENSG00000274514_RF00017', 'ENSG00000274515_AC105020.5', 'ENSG00000274523_RCC1L', 'ENSG00000274528_AC090970.2', 'ENSG00000274536_AL034397.3', 'ENSG00000274554_AC083806.2', 'ENSG00000274561_AC005332.3', 'ENSG00000274565_AC080038.1', 'ENSG00000274591_AC025031.3', 'ENSG00000274598_AC087893.1', 'ENSG00000274602_PI4KAP1', 'ENSG00000274605_AL355338.1', 'ENSG00000274618_HIST1H4F', 'ENSG00000274641_HIST1H2BO', 'ENSG00000274653_AC106782.6', 'ENSG00000274667_AC090517.2', 'ENSG00000274677_AC040169.3', 'ENSG00000274678_AC106886.3', 'ENSG00000274712_AC005332.4', 'ENSG00000274736_CCL23', 'ENSG00000274737_AC004466.2', 'ENSG00000274750_HIST1H3E', 'ENSG00000274751_AC120498.9', 'ENSG00000274756_AC243732.1', 'ENSG00000274767_AC243829.1', 'ENSG00000274769_AC016747.4', 'ENSG00000274776_AC090241.3', 'ENSG00000274799_RF00017', 'ENSG00000274818_AC004825.2', 'ENSG00000274825_AL023803.2', 'ENSG00000274828_AC068473.5', 'ENSG00000274840_AC132807.2', 'ENSG00000274841_RF00017', 'ENSG00000274845_RF02271', 'ENSG00000274849_AC023043.4', 'ENSG00000274859_AC131238.1', 'ENSG00000274885_AC087241.4', 'ENSG00000274897_PANO1', 'ENSG00000274898_AC001226.1', 'ENSG00000274904_AC093512.1', 'ENSG00000274922_AL139384.1', 'ENSG00000274925_ZKSCAN2-DT', 'ENSG00000274929_AL157813.1', 'ENSG00000274943_AC079684.1', 'ENSG00000274963_RN7SL600P', 'ENSG00000274964_AC026356.1', 'ENSG00000274967_RF00019', 'ENSG00000274979_AC020656.2', 'ENSG00000274987_AC092794.1', 'ENSG00000274995_AC013564.1', 'ENSG00000274997_HIST1H2AH', 'ENSG00000275004_ZNF280B', 'ENSG00000275014_RN7SL166P', 'ENSG00000275023_MLLT6', 'ENSG00000275029_HMGB1P24', 'ENSG00000275052_PPP4R3B', 'ENSG00000275055_AC011468.5', 'ENSG00000275056_AC020663.3', 'ENSG00000275066_SYNRG', 'ENSG00000275070_RF00017', 'ENSG00000275074_NUDT18', 'ENSG00000275084_SNORD91B', 'ENSG00000275091_AC022098.4', 'ENSG00000275092_AL031710.2', 'ENSG00000275097_AC024940.5', 'ENSG00000275111_ZNF2', 'ENSG00000275120_AC048382.5', 'ENSG00000275126_HIST1H4L', 'ENSG00000275131_AC241952.1', 'ENSG00000275132_RN7SL663P', 'ENSG00000275139_AL133492.1', 'ENSG00000275160_AL354718.1', 'ENSG00000275180_AC048341.2', 'ENSG00000275183_LENG9', 'ENSG00000275185_AC130324.3', 'ENSG00000275191_AC007497.1', 'ENSG00000275198_AL512791.2', 'ENSG00000275202_AL161421.1', 'ENSG00000275221_HIST1H2AK', 'ENSG00000275236_AC009120.5', 'ENSG00000275263_AC135048.3', 'ENSG00000275265_AC127002.1', 'ENSG00000275278_AC012150.2', 'ENSG00000275291_RF00003', 'ENSG00000275318_AL136981.2', 'ENSG00000275329_AL138781.2', 'ENSG00000275342_PRAG1', 'ENSG00000275343_AC010999.1', 'ENSG00000275355_RF00017', 'ENSG00000275363_AC100757.3', 'ENSG00000275371_AC012645.4', 'ENSG00000275379_HIST1H3I', 'ENSG00000275381_AC019206.1', 'ENSG00000275383_AC126773.4', 'ENSG00000275393_AC018695.6', 'ENSG00000275401_AL391095.3', 'ENSG00000275409_AC026367.2', 'ENSG00000275413_AC002553.2', 'ENSG00000275417_AC068726.1', 'ENSG00000275426_AC253576.2', 'ENSG00000275437_AL121832.3', 'ENSG00000275441_AC020765.2', 'ENSG00000275445_AC092119.3', 'ENSG00000275450_AL845472.1', 'ENSG00000275454_AC105020.6', 'ENSG00000275457_AL117332.1', 'ENSG00000275464_FP565260.1', 'ENSG00000275476_AC009318.4', 'ENSG00000275479_AC087741.2', 'ENSG00000275481_AC025031.4', 'ENSG00000275484_AP003419.3', 'ENSG00000275485_AL512652.1', 'ENSG00000275488_AC023509.5', 'ENSG00000275491_LINC01730', 'ENSG00000275494_AC133552.5', 'ENSG00000275506_AC010378.1', 'ENSG00000275512_AC007998.4', 'ENSG00000275532_AC006449.2', 'ENSG00000275538_RNVU1-19', 'ENSG00000275549_STPG3-AS1', 'ENSG00000275560_AC008115.3', 'ENSG00000275569_AL355073.2', 'ENSG00000275576_AL049539.1', 'ENSG00000275580_AC022306.2', 'ENSG00000275582_AL031670.1', 'ENSG00000275586_RF00017', 'ENSG00000275591_XKR5', 'ENSG00000275601_AC011330.2', 'ENSG00000275607_AC135507.2', 'ENSG00000275613_AC243830.1', 'ENSG00000275630_AC004816.2', 'ENSG00000275632_AL035461.2', 'ENSG00000275638_AC011939.3', 'ENSG00000275645_AC068338.3', 'ENSG00000275672_AC025580.3', 'ENSG00000275678_AL133320.1', 'ENSG00000275700_AATF', 'ENSG00000275703_U47924.3', 'ENSG00000275709_AC090527.3', 'ENSG00000275713_HIST1H2BH', 'ENSG00000275714_HIST1H3A', 'ENSG00000275719_AC008622.2', 'ENSG00000275720_AC243830.2', 'ENSG00000275734_AC010538.1', 'ENSG00000275740_AC091959.3', 'ENSG00000275759_AC026367.3', 'ENSG00000275763_C18orf65', 'ENSG00000275764_AC092747.4', 'ENSG00000275765_AC091982.3', 'ENSG00000275769_AC068792.1', 'ENSG00000275799_AP001059.2', 'ENSG00000275807_AC145285.6', 'ENSG00000275832_ARHGAP23', 'ENSG00000275835_TUBGCP5', 'ENSG00000275839_AC243654.2', 'ENSG00000275846_AL513548.3', 'ENSG00000275853_RF00017', 'ENSG00000275854_AC084824.4', 'ENSG00000275857_AC009133.4', 'ENSG00000275880_AL139385.1', 'ENSG00000275881_RF00017', 'ENSG00000275888_AC132872.3', 'ENSG00000275895_U2AF1L5', 'ENSG00000275896_PRSS2', 'ENSG00000275897_AC021491.4', 'ENSG00000275902_AC139530.3', 'ENSG00000275910_AC138932.5', 'ENSG00000275927_AC009152.1', 'ENSG00000275936_AC004263.1', 'ENSG00000275964_AL355001.2', 'ENSG00000275966_AC110285.6', 'ENSG00000275997_AC016292.2', 'ENSG00000276007_AC079414.3', 'ENSG00000276023_DUSP14', 'ENSG00000276026_AL031665.2', 'ENSG00000276030_AC073534.1', 'ENSG00000276043_UHRF1', 'ENSG00000276045_ORAI1', 'ENSG00000276058_STMN1P1', 'ENSG00000276071_AC074138.1', 'ENSG00000276075_AC027682.6', 'ENSG00000276085_CCL3L1', 'ENSG00000276092_AC040896.1', 'ENSG00000276101_AC027601.4', 'ENSG00000276107_AC037198.1', 'ENSG00000276115_AC026356.2', 'ENSG00000276116_FUT8-AS1', 'ENSG00000276131_AC009118.2', 'ENSG00000276136_AC016957.2', 'ENSG00000276141_WHAMMP3', 'ENSG00000276148_AC084824.5', 'ENSG00000276166_AC092118.2', 'ENSG00000276170_AC244153.1', 'ENSG00000276174_AC087683.2', 'ENSG00000276180_HIST1H4I', 'ENSG00000276182_AL163051.2', 'ENSG00000276188_AC069234.4', 'ENSG00000276203_ANKRD20A3', 'ENSG00000276213_RF00017', 'ENSG00000276216_AC245014.3', 'ENSG00000276223_AL118522.1', 'ENSG00000276231_PIK3R6', 'ENSG00000276234_TADA2A', 'ENSG00000276248_AL442125.1', 'ENSG00000276250_AC127024.6', 'ENSG00000276259_AC009118.3', 'ENSG00000276272_AC024884.2', 'ENSG00000276278_AC048382.6', 'ENSG00000276291_FRG1HP', 'ENSG00000276293_PIP4K2B', 'ENSG00000276334_AL133243.2', 'ENSG00000276337_AC105429.1', 'ENSG00000276368_HIST1H2AJ', 'ENSG00000276380_UBE2NL', 'ENSG00000276384_AC016876.3', 'ENSG00000276390_AC004241.3', 'ENSG00000276408_AC025287.2', 'ENSG00000276410_HIST1H2BB', 'ENSG00000276436_AL136301.1', 'ENSG00000276445_AC005393.1', 'ENSG00000276449_AC004076.2', 'ENSG00000276476_LINC00540', 'ENSG00000276488_AC008735.4', 'ENSG00000276494_RF01948', 'ENSG00000276500_BMS1P14', 'ENSG00000276505_AP000892.2', 'ENSG00000276509_AC239799.2', 'ENSG00000276517_AL133243.3', 'ENSG00000276523_AC025287.3', 'ENSG00000276524_AC010999.2', 'ENSG00000276529_AP001505.1', 'ENSG00000276533_AC018926.2', 'ENSG00000276550_HERC2P2', 'ENSG00000276564_AC130650.2', 'ENSG00000276570_AC010327.5', 'ENSG00000276571_AC002550.2', 'ENSG00000276573_AL442067.1', 'ENSG00000276593_AC022306.3', 'ENSG00000276600_RAB7B', 'ENSG00000276603_AL109614.1', 'ENSG00000276644_DACH1', 'ENSG00000276645_RF00017', 'ENSG00000276649_AL117335.1', 'ENSG00000276651_AC007950.2', 'ENSG00000276672_AL161891.1', 'ENSG00000276698_AL136295.6', 'ENSG00000276702_AC010809.3', 'ENSG00000276718_AC005840.4', 'ENSG00000276724_AC123768.4', 'ENSG00000276727_AC137834.2', 'ENSG00000276728_AC142472.1', 'ENSG00000276742_AL731566.2', 'ENSG00000276744_AC105137.2', 'ENSG00000276757_RN7SL192P', 'ENSG00000276791_AC092117.1', 'ENSG00000276805_AL133216.2', 'ENSG00000276809_AL138955.1', 'ENSG00000276810_AC244093.2', 'ENSG00000276814_AC004801.6', 'ENSG00000276840_PMS2P10', 'ENSG00000276846_AC016590.3', 'ENSG00000276853_AC026124.2', 'ENSG00000276855_AC015922.3', 'ENSG00000276900_AC023157.3', 'ENSG00000276903_HIST1H2AL', 'ENSG00000276916_AL442125.2', 'ENSG00000276931_AC009041.4', 'ENSG00000276934_AC009704.2', 'ENSG00000276957_AL158063.1', 'ENSG00000276966_HIST1H4E', 'ENSG00000276968_AL158196.1', 'ENSG00000276988_RF00017', 'ENSG00000277007_AC096642.1', 'ENSG00000277020_AL590096.1', 'ENSG00000277022_AL031663.3', 'ENSG00000277039_RF00017', 'ENSG00000277050_AL122125.1', 'ENSG00000277053_GTF2IP1', 'ENSG00000277072_STAG3L2', 'ENSG00000277075_HIST1H2AE', 'ENSG00000277112_ANKRD20A21P', 'ENSG00000277118_RF00017', 'ENSG00000277130_AC073569.3', 'ENSG00000277142_LINC00235', 'ENSG00000277147_LINC00869', 'ENSG00000277149_TYW1B', 'ENSG00000277150_F8A3', 'ENSG00000277151_AL138820.1', 'ENSG00000277157_HIST1H4D', 'ENSG00000277159_AL139384.2', 'ENSG00000277161_PIGW', 'ENSG00000277170_AC012676.3', 'ENSG00000277200_AC005696.4', 'ENSG00000277203_F8A1', 'ENSG00000277218_AL139123.1', 'ENSG00000277224_HIST1H2BF', 'ENSG00000277232_GTSE1-DT', 'ENSG00000277233_RF00017', 'ENSG00000277245_AC084782.3', 'ENSG00000277246_AL157762.1', 'ENSG00000277250_RF00017', 'ENSG00000277258_PCGF2', 'ENSG00000277283_AC004812.2', 'ENSG00000277287_AL109976.1', 'ENSG00000277301_AL034550.2', 'ENSG00000277324_AC093462.1', 'ENSG00000277342_AC048344.4', 'ENSG00000277368_AL138966.2', 'ENSG00000277369_AC010654.1', 'ENSG00000277371_RF00017', 'ENSG00000277383_AC010331.1', 'ENSG00000277386_AL138999.1', 'ENSG00000277396_RF00017', 'ENSG00000277399_GPR179', 'ENSG00000277406_SEC22B4P', 'ENSG00000277423_AC069234.5', 'ENSG00000277440_AC012676.4', 'ENSG00000277443_MARCKS', 'ENSG00000277447_AP005431.1', 'ENSG00000277449_CEBPB-AS1', 'ENSG00000277450_AC002094.4', 'ENSG00000277452_RN7SL473P', 'ENSG00000277453_AC010271.2', 'ENSG00000277462_ZNF670', 'ENSG00000277463_AC080038.2', 'ENSG00000277476_AC005332.5', 'ENSG00000277491_AC087392.5', 'ENSG00000277511_AC116407.2', 'ENSG00000277526_AC245123.1', 'ENSG00000277534_AC007996.1', 'ENSG00000277558_AL109923.1', 'ENSG00000277561_GOLGA8IP', 'ENSG00000277566_AC089999.2', 'ENSG00000277568_RF02271', 'ENSG00000277581_AL023803.3', 'ENSG00000277589_AC244093.4', 'ENSG00000277595_AC007546.1', 'ENSG00000277597_AC130343.2', 'ENSG00000277602_AC005363.2', 'ENSG00000277610_RNVU1-4', 'ENSG00000277632_CCL3', 'ENSG00000277634_RF00019', 'ENSG00000277639_AC007906.2', 'ENSG00000277654_AC087633.2', 'ENSG00000277662_AL354696.1', 'ENSG00000277675_AC211486.5', 'ENSG00000277687_AL139407.1', 'ENSG00000277715_AC079174.2', 'ENSG00000277728_AC097641.2', 'ENSG00000277734_TRAC', 'ENSG00000277744_AC011462.4', 'ENSG00000277763_AL138995.1', 'ENSG00000277767_AL442128.2', 'ENSG00000277775_HIST1H3F', 'ENSG00000277782_AC068870.2', 'ENSG00000277791_PSMB3', 'ENSG00000277794_RF00017', 'ENSG00000277806_AC006213.4', 'ENSG00000277825_AC020917.3', 'ENSG00000277840_AC026368.1', 'ENSG00000277879_AL391988.1', 'ENSG00000277895_AC135279.3', 'ENSG00000277900_RF00017', 'ENSG00000277901_AL390037.1', 'ENSG00000277911_AC243773.2', 'ENSG00000277918_RF00003', 'ENSG00000277938_AL035252.3', 'ENSG00000277945_AC107308.1', 'ENSG00000277950_RF00017', 'ENSG00000277954_AC092376.2', 'ENSG00000277958_RF00017', 'ENSG00000277959_AL162274.2', 'ENSG00000277969_AC006449.6', 'ENSG00000277972_CISD3', 'ENSG00000277978_AC010542.5', 'ENSG00000277988_FAM30B', 'ENSG00000277991_FP236241.1', 'ENSG00000277998_AC107075.1', 'ENSG00000277999_AC009093.6', 'ENSG00000278000_AC139100.2', 'ENSG00000278002_AL627171.1', 'ENSG00000278017_AC064801.1', 'ENSG00000278022_AC118658.1', 'ENSG00000278023_RDM1', 'ENSG00000278053_DDX52', 'ENSG00000278075_AC114341.1', 'ENSG00000278080_SPDYE15P', 'ENSG00000278090_LUNAR1', 'ENSG00000278095_AC022509.4', 'ENSG00000278099_RF00003', 'ENSG00000278107_AC027575.2', 'ENSG00000278126_AC139768.1', 'ENSG00000278129_ZNF8', 'ENSG00000278133_AC135050.6', 'ENSG00000278156_TSC22D1-AS1', 'ENSG00000278158_AP001059.3', 'ENSG00000278175_GLIDR', 'ENSG00000278177_AL354811.1', 'ENSG00000278192_AL118505.1', 'ENSG00000278200_LINC01971', 'ENSG00000278212_AC134878.2', 'ENSG00000278231_AL133342.1', 'ENSG00000278238_AL359513.1', 'ENSG00000278259_MYO19', 'ENSG00000278272_HIST1H3C', 'ENSG00000278276_AL110115.2', 'ENSG00000278291_AL161772.1', 'ENSG00000278311_GGNBP2', 'ENSG00000278318_ZNF229', 'ENSG00000278330_AC018529.2', 'ENSG00000278341_AC138028.6', 'ENSG00000278356_AC005911.1', 'ENSG00000278376_AP004609.3', 'ENSG00000278383_AL031673.1', 'ENSG00000278390_AL354696.2', 'ENSG00000278396_AL122023.1', 'ENSG00000278416_PMS2P2', 'ENSG00000278434_AC023830.3', 'ENSG00000278463_HIST1H2AB', 'ENSG00000278472_AC009268.2', 'ENSG00000278492_AC006213.5', 'ENSG00000278514_AC068831.6', 'ENSG00000278520_MIR7851', 'ENSG00000278530_CHMP1B2P', 'ENSG00000278535_DHRS11', 'ENSG00000278540_ACACA', 'ENSG00000278576_AL162171.3', 'ENSG00000278588_HIST1H2BI', 'ENSG00000278590_RN7SL113P', 'ENSG00000278600_AC015871.3', 'ENSG00000278601_AL158163.2', 'ENSG00000278611_ZNF426-DT', 'ENSG00000278615_C11orf98', 'ENSG00000278619_MRM1', 'ENSG00000278627_AC005962.1', 'ENSG00000278637_HIST1H4A', 'ENSG00000278662_GOLGA6L10', 'ENSG00000278668_AC005899.7', 'ENSG00000278677_HIST1H2AM', 'ENSG00000278700_RF00017', 'ENSG00000278702_RF00017', 'ENSG00000278703_AC100847.1', 'ENSG00000278705_HIST1H4B', 'ENSG00000278709_NKILA', 'ENSG00000278727_AC000403.1', 'ENSG00000278730_AC005332.6', 'ENSG00000278732_RF00017', 'ENSG00000278733_AC022079.2', 'ENSG00000278740_AC005332.7', 'ENSG00000278743_AC087239.1', 'ENSG00000278765_AC004477.3', 'ENSG00000278768_BACE1-AS', 'ENSG00000278771_RN7SL3', 'ENSG00000278784_AL136295.7', 'ENSG00000278791_MIR6723', 'ENSG00000278794_RF00017', 'ENSG00000278811_LINC00624', 'ENSG00000278818_RF00017', 'ENSG00000278828_HIST1H3H', 'ENSG00000278829_AC099811.5', 'ENSG00000278834_AC073508.3', 'ENSG00000278842_AC008147.4', 'ENSG00000278845_MRPL45', 'ENSG00000278847_AC006157.1', 'ENSG00000278861_AC117503.2', 'ENSG00000278864_AC055811.4', 'ENSG00000278865_AC012629.3', 'ENSG00000278867_AC090616.6', 'ENSG00000278869_BX539320.1', 'ENSG00000278873_PRO1804', 'ENSG00000278876_AC145207.9', 'ENSG00000278879_AP000560.1', 'ENSG00000278886_AC087821.1', 'ENSG00000278896_AC025031.5', 'ENSG00000278897_AC020951.1', 'ENSG00000278899_AL358852.1', 'ENSG00000278900_AC139792.2', 'ENSG00000278903_CU633906.2', 'ENSG00000278909_AC007608.4', 'ENSG00000278916_CEP83-DT', 'ENSG00000278917_AC006213.6', 'ENSG00000278920_AC005005.4', 'ENSG00000278922_AC002310.6', 'ENSG00000278931_CR381670.1', 'ENSG00000278932_CR381653.1', 'ENSG00000278948_AL031587.5', 'ENSG00000278949_AC127070.4', 'ENSG00000278950_AC138907.8', 'ENSG00000278952_AP003068.4', 'ENSG00000278954_AC130686.1', 'ENSG00000278963_AC005921.3', 'ENSG00000278965_AC122713.2', 'ENSG00000278969_AC026310.3', 'ENSG00000278970_HEIH', 'ENSG00000278971_AC091305.1', 'ENSG00000278972_AC015920.1', 'ENSG00000278974_AC093909.6', 'ENSG00000278979_AC007598.3', 'ENSG00000278983_AC048380.2', 'ENSG00000278985_AC092718.7', 'ENSG00000278986_AC091060.1', 'ENSG00000278987_AL031009.1', 'ENSG00000278989_AP001148.1', 'ENSG00000278991_AC090181.3', 'ENSG00000278993_AC002350.1', 'ENSG00000278997_AL662907.1', 'ENSG00000278999_AC008985.1', 'ENSG00000279019_AC009090.4', 'ENSG00000279020_C18orf15', 'ENSG00000279021_AC092139.2', 'ENSG00000279022_AL359715.4', 'ENSG00000279026_AC005225.4', 'ENSG00000279030_AC007336.3', 'ENSG00000279031_AC004232.3', 'ENSG00000279033_AC090984.1', 'ENSG00000279035_AC022211.4', 'ENSG00000279039_AC011447.6', 'ENSG00000279048_AC080080.1', 'ENSG00000279050_PWAR1', 'ENSG00000279057_AC141586.4', 'ENSG00000279059_AC007485.2', 'ENSG00000279064_FP236315.1', 'ENSG00000279066_HEXDC-IT1', 'ENSG00000279069_AC015813.5', 'ENSG00000279070_AC073263.2', 'ENSG00000279074_AC007998.5', 'ENSG00000279077_AC023090.2', 'ENSG00000279078_SND1-IT1', 'ENSG00000279080_AL022322.2', 'ENSG00000279085_AL022323.3', 'ENSG00000279086_AC073130.3', 'ENSG00000279088_AC022400.7', 'ENSG00000279089_AC005839.1', 'ENSG00000279092_AC025678.3', 'ENSG00000279095_AC243964.3', 'ENSG00000279098_AC097460.3', 'ENSG00000279103_AC138470.1', 'ENSG00000279106_AC009093.7', 'ENSG00000279107_AC138951.2', 'ENSG00000279108_AC008537.3', 'ENSG00000279110_AL022323.4', 'ENSG00000279114_Z99129.3', 'ENSG00000279117_AP001972.5', 'ENSG00000279118_AC093535.2', 'ENSG00000279119_AC006449.7', 'ENSG00000279122_AC020763.2', 'ENSG00000279129_AC046158.3', 'ENSG00000279133_AC018628.1', 'ENSG00000279135_AL512652.2', 'ENSG00000279138_AP002847.1', 'ENSG00000279140_AL590326.1', 'ENSG00000279145_AC011912.1', 'ENSG00000279147_AC112504.1', 'ENSG00000279148_AC126474.1', 'ENSG00000279159_AC003681.1', 'ENSG00000279162_AC141586.5', 'ENSG00000279166_AC009951.1', 'ENSG00000279168_AC105052.4', 'ENSG00000279175_AL033543.1', 'ENSG00000279176_AC079316.2', 'ENSG00000279179_AL662907.2', 'ENSG00000279191_AC068491.4', 'ENSG00000279192_PWAR5', 'ENSG00000279196_AC135048.4', 'ENSG00000279198_AC008894.3', 'ENSG00000279199_AC068669.1', 'ENSG00000279202_AC130448.1', 'ENSG00000279203_AC005785.2', 'ENSG00000279206_AC004943.3', 'ENSG00000279207_AC015813.6', 'ENSG00000279212_AL390961.3', 'ENSG00000279217_Z95114.1', 'ENSG00000279220_GPR1-AS', 'ENSG00000279227_AC009303.4', 'ENSG00000279233_AC122688.3', 'ENSG00000279236_AC064801.2', 'ENSG00000279246_AP003108.3', 'ENSG00000279249_AC007614.1', 'ENSG00000279250_AC022919.1', 'ENSG00000279253_AL121753.2', 'ENSG00000279259_AC087741.3', 'ENSG00000279265_AC000123.2', 'ENSG00000279267_AL078621.3', 'ENSG00000279277_AC012020.1', 'ENSG00000279278_AC245060.6', 'ENSG00000279281_AC015883.1', 'ENSG00000279283_AC131009.4', 'ENSG00000279288_AC073346.2', 'ENSG00000279296_PRAL', 'ENSG00000279306_AL139288.1', 'ENSG00000279307_AC012065.5', 'ENSG00000279317_AC006994.2', 'ENSG00000279320_AC069528.2', 'ENSG00000279328_AC073439.1', 'ENSG00000279329_AC020910.5', 'ENSG00000279330_AJ003147.3', 'ENSG00000279331_RBM12B-AS1', 'ENSG00000279333_AC096636.1', 'ENSG00000279342_AP000866.6', 'ENSG00000279344_AC007342.6', 'ENSG00000279345_Z98885.3', 'ENSG00000279347_AC021945.1', 'ENSG00000279348_AC012513.3', 'ENSG00000279354_AC090373.1', 'ENSG00000279355_AGPAT4-IT1', 'ENSG00000279356_AC007610.4', 'ENSG00000279357_AC007224.2', 'ENSG00000279360_AC007546.2', 'ENSG00000279361_AC079331.1', 'ENSG00000279364_AC114546.3', 'ENSG00000279369_AC046185.3', 'ENSG00000279370_AC004777.1', 'ENSG00000279377_AC003973.3', 'ENSG00000279382_AC018665.1', 'ENSG00000279386_AC021106.3', 'ENSG00000279394_AC015871.4', 'ENSG00000279400_AC008957.3', 'ENSG00000279406_AL359183.1', 'ENSG00000279407_AC007191.1', 'ENSG00000279409_AC020658.6', 'ENSG00000279410_AC099494.1', 'ENSG00000279412_AC020763.3', 'ENSG00000279415_AC099494.2', 'ENSG00000279416_AC099689.1', 'ENSG00000279425_AC092279.2', 'ENSG00000279428_AC087164.2', 'ENSG00000279432_AC015799.1', 'ENSG00000279433_AC018529.3', 'ENSG00000279434_AL049776.1', 'ENSG00000279439_AC105114.1', 'ENSG00000279443_AL513497.1', 'ENSG00000279453_Z99129.4', 'ENSG00000279456_AL353763.1', 'ENSG00000279462_AC093028.1', 'ENSG00000279464_AC096720.2', 'ENSG00000279467_AP000350.7', 'ENSG00000279474_AC125437.2', 'ENSG00000279476_AC092139.3', 'ENSG00000279481_AC104791.2', 'ENSG00000279485_AC016734.2', 'ENSG00000279488_AC004623.1', 'ENSG00000279491_AP003733.4', 'ENSG00000279494_AL117328.2', 'ENSG00000279495_AL928654.4', 'ENSG00000279499_AL157770.1', 'ENSG00000279500_AC108704.2', 'ENSG00000279502_AC016542.2', 'ENSG00000279513_AL157902.2', 'ENSG00000279518_AC083843.3', 'ENSG00000279519_AC007382.1', 'ENSG00000279528_AC115618.3', 'ENSG00000279529_AC008764.8', 'ENSG00000279539_AC006486.2', 'ENSG00000279541_AC005261.5', 'ENSG00000279544_AL133243.4', 'ENSG00000279549_AP000437.1', 'ENSG00000279554_AC130448.2', 'ENSG00000279555_AC091181.1', 'ENSG00000279557_AC010435.1', 'ENSG00000279561_AL845472.2', 'ENSG00000279568_AC093525.9', 'ENSG00000279569_AC020763.4', 'ENSG00000279570_AC099804.1', 'ENSG00000279571_AL162426.1', 'ENSG00000279573_AC134407.2', 'ENSG00000279583_AC009086.3', 'ENSG00000279584_AC005593.1', 'ENSG00000279589_AC079416.2', 'ENSG00000279591_AC002044.2', 'ENSG00000279594_AL049780.3', 'ENSG00000279598_AC009948.5', 'ENSG00000279601_AC005052.1', 'ENSG00000279602_AC109326.1', 'ENSG00000279605_AC067930.6', 'ENSG00000279608_AL353795.3', 'ENSG00000279613_AC124283.5', 'ENSG00000279616_AL096817.1', 'ENSG00000279617_AC005796.1', 'ENSG00000279620_AC099494.3', 'ENSG00000279621_AC020978.8', 'ENSG00000279623_AL359697.1', 'ENSG00000279631_AL158211.5', 'ENSG00000279632_AP003108.4', 'ENSG00000279638_AC008873.1', 'ENSG00000279641_AC120057.3', 'ENSG00000279649_AC020978.9', 'ENSG00000279652_Z82217.1', 'ENSG00000279653_AC004678.2', 'ENSG00000279656_AL132780.4', 'ENSG00000279659_AL451064.2', 'ENSG00000279662_AC131649.2', 'ENSG00000279665_AC012100.3', 'ENSG00000279670_AL359922.3', 'ENSG00000279673_AC092919.2', 'ENSG00000279689_AC022400.8', 'ENSG00000279691_AC113410.3', 'ENSG00000279692_AC110285.7', 'ENSG00000279696_AP001273.1', 'ENSG00000279700_AC131212.2', 'ENSG00000279706_AL353608.4', 'ENSG00000279716_AC006128.1', 'ENSG00000279717_AC005336.3', 'ENSG00000279720_CR392039.4', 'ENSG00000279721_AC018737.2', 'ENSG00000279722_AC007342.7', 'ENSG00000279730_SETD8P1', 'ENSG00000279733_AP001642.1', 'ENSG00000279738_AL022311.1', 'ENSG00000279742_AP000974.1', 'ENSG00000279744_AC132938.5', 'ENSG00000279753_AC011558.1', 'ENSG00000279759_AC118344.2', 'ENSG00000279765_AC013394.1', 'ENSG00000279766_AC067931.1', 'ENSG00000279785_AC008267.7', 'ENSG00000279786_AC105235.1', 'ENSG00000279789_AC120114.3', 'ENSG00000279791_AC018892.3', 'ENSG00000279792_AC015909.5', 'ENSG00000279794_AC024580.1', 'ENSG00000279796_AL133384.2', 'ENSG00000279799_AC006077.2', 'ENSG00000279800_BCLAF1P2', 'ENSG00000279801_AC111170.3', 'ENSG00000279803_AC009090.5', 'ENSG00000279811_AC093330.2', 'ENSG00000279814_AC010997.5', 'ENSG00000279819_AL390318.1', 'ENSG00000279821_AC145098.2', 'ENSG00000279822_AC016397.2', 'ENSG00000279827_AC136469.2', 'ENSG00000279833_AL031846.2', 'ENSG00000279836_AP002967.1', 'ENSG00000279838_AL356273.3', 'ENSG00000279839_AL512383.1', 'ENSG00000279841_AC092135.3', 'ENSG00000279845_AC097372.3', 'ENSG00000279860_AC008568.2', 'ENSG00000279862_AC092964.1', 'ENSG00000279863_AC069547.1', 'ENSG00000279865_AC006511.3', 'ENSG00000279873_LINC01126', 'ENSG00000279879_AC091152.4', 'ENSG00000279880_AC134407.3', 'ENSG00000279885_AP005060.1', 'ENSG00000279887_AC046158.4', 'ENSG00000279891_FLJ42393', 'ENSG00000279900_AP001767.4', 'ENSG00000279901_AC092117.2', 'ENSG00000279912_AC068448.1', 'ENSG00000279917_AC079331.2', 'ENSG00000279923_AC022417.1', 'ENSG00000279933_AL031595.1', 'ENSG00000279943_FLJ38576', 'ENSG00000279948_AC008895.1', 'ENSG00000279951_AL138688.2', 'ENSG00000279953_AC117503.3', 'ENSG00000279957_AC110769.3', 'ENSG00000279962_AP001525.1', 'ENSG00000279968_GVQW2', 'ENSG00000279977_AC008764.10', 'ENSG00000279981_AC018445.4', 'ENSG00000279982_AL162274.3', 'ENSG00000279991_AC009044.1', 'ENSG00000279995_AC139792.3', 'ENSG00000279996_AC004491.1', 'ENSG00000279997_AC133485.6', 'ENSG00000280007_AC008079.1', 'ENSG00000280010_AP001350.2', 'ENSG00000280011_AL031595.2', 'ENSG00000280018_CU634019.6', 'ENSG00000280022_AC126544.1', 'ENSG00000280023_AD000813.1', 'ENSG00000280028_AC007431.3', 'ENSG00000280033_AC116407.4', 'ENSG00000280035_AC011676.5', 'ENSG00000280036_AC020661.4', 'ENSG00000280038_DNM1P41', 'ENSG00000280046_AC104581.4', 'ENSG00000280047_AC091825.1', 'ENSG00000280063_AC012676.5', 'ENSG00000280064_AC130304.1', 'ENSG00000280067_AC023818.1', 'ENSG00000280069_AC127024.8', 'ENSG00000280071_GATD3B', 'ENSG00000280077_AL353763.2', 'ENSG00000280078_AC016526.3', 'ENSG00000280079_AC011447.7', 'ENSG00000280080_TBC1D22A-AS1', 'ENSG00000280087_AC011481.3', 'ENSG00000280088_AC126474.2', 'ENSG00000280099_AL603750.1', 'ENSG00000280103_AC007792.1', 'ENSG00000280106_AC008555.8', 'ENSG00000280107_AL022393.1', 'ENSG00000280109_PLAC4', 'ENSG00000280115_AC136603.1', 'ENSG00000280119_AC093642.2', 'ENSG00000280120_AC073857.1', 'ENSG00000280121_AC010335.3', 'ENSG00000280122_AC016168.3', 'ENSG00000280123_AC023632.6', 'ENSG00000280128_AL662795.2', 'ENSG00000280129_AL132780.5', 'ENSG00000280132_AC026471.6', 'ENSG00000280138_AC027290.2', 'ENSG00000280143_AP000892.3', 'ENSG00000280145_CU638689.4', 'ENSG00000280149_AC004877.2', 'ENSG00000280157_AL359510.2', 'ENSG00000280160_AC135050.7', 'ENSG00000280161_AC022413.1', 'ENSG00000280163_AC040160.2', 'ENSG00000280164_CU638689.5', 'ENSG00000280166_AC016542.3', 'ENSG00000280167_AP000943.4', 'ENSG00000280173_AC104447.1', 'ENSG00000280177_AC004408.2', 'ENSG00000280181_AC025262.2', 'ENSG00000280184_AL023806.3', 'ENSG00000280187_AC022107.1', 'ENSG00000280190_AC027279.4', 'ENSG00000280193_AC132219.2', 'ENSG00000280194_AD000864.1', 'ENSG00000280195_AC245140.2', 'ENSG00000280202_AC005831.1', 'ENSG00000280205_AC009716.2', 'ENSG00000280206_AC026401.3', 'ENSG00000280207_AC106795.5', 'ENSG00000280211_AC106886.4', 'ENSG00000280213_UCKL1-AS1', 'ENSG00000280214_AC027682.7', 'ENSG00000280216_AL022326.2', 'ENSG00000280227_AC079416.3', 'ENSG00000280228_AC079753.1', 'ENSG00000280231_AL031719.2', 'ENSG00000280237_MIR4697HG', 'ENSG00000280239_AC011498.7', 'ENSG00000280242_AL450226.2', 'ENSG00000280244_AL512356.3', 'ENSG00000280247_AC005578.1', 'ENSG00000280248_AC124319.4', 'ENSG00000280255_AC004947.2', 'ENSG00000280269_AP000577.1', 'ENSG00000280273_AF131216.4', 'ENSG00000280274_AC009145.4', 'ENSG00000280285_AC108215.1', 'ENSG00000280287_AC131212.3', 'ENSG00000280295_AC099811.6', 'ENSG00000280303_ERICD', 'ENSG00000280311_AC131212.4', 'ENSG00000280321_AC129502.1', 'ENSG00000280325_AC074183.2', 'ENSG00000280326_AC067931.2', 'ENSG00000280327_Z97633.1', 'ENSG00000280332_AC020917.4', 'ENSG00000280334_AC009084.2', 'ENSG00000280347_AC000123.3', 'ENSG00000280351_AC127496.7', 'ENSG00000280353_AC011466.4', 'ENSG00000280355_AL132656.4', 'ENSG00000280365_AC021766.1', 'ENSG00000280367_AP002364.1', 'ENSG00000280372_CR382285.2', 'ENSG00000280374_AC019080.5', 'ENSG00000280378_AL353898.3', 'ENSG00000280381_AC026362.2', 'ENSG00000280383_Z95331.1', 'ENSG00000280384_FP325332.1', 'ENSG00000280385_AP000648.3', 'ENSG00000280388_AC006330.1', 'ENSG00000280392_AC007496.3', 'ENSG00000280399_AC022497.1', 'ENSG00000280401_AC022532.1', 'ENSG00000280402_AC093525.10', 'ENSG00000280407_AC132872.4', 'ENSG00000280416_AC009084.3', 'ENSG00000280417_AC096887.2', 'ENSG00000280420_AC005355.2', 'ENSG00000280426_AC084876.2', 'ENSG00000280433_FP565260.6', 'ENSG00000280434_AL031595.3', 'ENSG00000280474_AL356481.2', 'ENSG00000280543_ASAP1-IT2', 'ENSG00000280594_BTG3-AS1', 'ENSG00000280604_AJ239328.1', 'ENSG00000280620_SCAANT1', 'ENSG00000280634_THRIL', 'ENSG00000280639_LINC02204', 'ENSG00000280649_AC245100.8', 'ENSG00000280665_AL513210.1', 'ENSG00000280670_CCDC163', 'ENSG00000280721_LINC01943', 'ENSG00000280734_LINC01232', 'ENSG00000280739_EIF1B-AS1', 'ENSG00000280767_AL732314.4', 'ENSG00000280789_PAGR1', 'ENSG00000280798_LINC00294', 'ENSG00000280828_AC090114.3', 'ENSG00000280832_GSEC', 'ENSG00000280927_CTBP1-AS', 'ENSG00000280969_RPS4Y2', 'ENSG00000280987_MATR3', 'ENSG00000281005_LINC00921', 'ENSG00000281026_N4BP2L2-IT2', 'ENSG00000281100_AC105749.1', 'ENSG00000281103_TRG-AS1', 'ENSG00000281106_TMEM272', 'ENSG00000281128_PTENP1-AS', 'ENSG00000281183_NPTN-IT1', 'ENSG00000281189_GHET1', 'ENSG00000281195_AC007878.1', 'ENSG00000281207_SLFNL1-AS1', 'ENSG00000281332_LINC00997', 'ENSG00000281344_HELLPAR', 'ENSG00000281357_ARRDC3-AS1', 'ENSG00000281358_RASSF1-AS1', 'ENSG00000281371_INE2', 'ENSG00000281376_ABALON', 'ENSG00000281392_LINC00506', 'ENSG00000281398_SNHG4', 'ENSG00000281404_LINC01176', 'ENSG00000281468_AC006504.8', 'ENSG00000281469_AC019226.1', 'ENSG00000281501_SEPSECS-AS1', 'ENSG00000281560_LSINCT5', 'ENSG00000281649_EBLN3P', 'ENSG00000281691_RBM5-AS1', 'ENSG00000281706_LINC01012', 'ENSG00000281731_AC110079.2', 'ENSG00000281772_AC019226.2', 'ENSG00000281849_AL732314.6', 'ENSG00000281852_LINC00891', 'ENSG00000281903_LINC02246', 'ENSG00000281912_LINC01144', 'ENSG00000281920_AC007389.5', 'ENSG00000282021_AC100810.3', 'ENSG00000282034_AC106886.5', 'ENSG00000282080_AC006511.4', 'ENSG00000282100_HSP90AB4P', 'ENSG00000282164_PEG13', 'ENSG00000282199_AC007993.3', 'ENSG00000282308_DPRXP3', 'ENSG00000282317_AL451007.2', 'ENSG00000282386_AL358472.4', 'ENSG00000282393_AC016588.2', 'ENSG00000282458_WASH5P', 'ENSG00000282508_LINC01002', 'ENSG00000282542_AC008993.1', 'ENSG00000282608_ADORA3', 'ENSG00000282742_AC093323.3', 'ENSG00000282787_AL157888.1', 'ENSG00000282826_FRG1CP', 'ENSG00000282828_AC009971.1', 'ENSG00000282851_BISPR', 'ENSG00000282855_AC093591.3', 'ENSG00000282870_FRG1DP', 'ENSG00000282876_Z98752.4', 'ENSG00000282885_AL627171.2', 'ENSG00000282915_AC091769.2', 'ENSG00000282933_RHOXF1P3', 'ENSG00000282936_AC004706.3', 'ENSG00000282951_AC008537.4', 'ENSG00000282961_PRNCR1', 'ENSG00000282968_AGGF1P10', 'ENSG00000282977_PCBP2-OT1', 'ENSG00000282978_AC110994.2', 'ENSG00000282988_AL031777.3', 'ENSG00000282995_FRG1EP', 'ENSG00000283029_AL139099.4', 'ENSG00000283041_AC008038.1', 'ENSG00000283045_AC103703.1', 'ENSG00000283050_GTF2IP12', 'ENSG00000283064_AL353759.1', 'ENSG00000283078_AL137077.2', 'ENSG00000283103_AC010642.2', 'ENSG00000283108_AC011451.3', 'ENSG00000283122_HYMAI', 'ENSG00000283125_AC022726.2', 'ENSG00000283128_AC009403.2', 'ENSG00000283154_IQCJ-SCHIP1', 'ENSG00000283156_AC068620.3', 'ENSG00000283196_AC006453.2', 'ENSG00000283208_AC001226.2', 'ENSG00000283236_AC074141.1', 'ENSG00000283240_AC007529.2', 'ENSG00000283288_SMIM33', 'ENSG00000283294_AP005212.4', 'ENSG00000283312_AC017104.4', 'ENSG00000283341_AC068205.2', 'ENSG00000283355_AC074194.2', 'ENSG00000283375_AC087521.4', 'ENSG00000283384_AL138694.1', 'ENSG00000283389_RF00017', 'ENSG00000283390_AC068631.3', 'ENSG00000283399_AC004381.2', 'ENSG00000283415_AC087280.2', 'ENSG00000283444_GPR141BP', 'ENSG00000283453_AC244258.1', 'ENSG00000283458_AC011139.1', 'ENSG00000283486_FAM95C', 'ENSG00000283573_AL157371.2', 'ENSG00000283632_EXOC3L2', 'ENSG00000283633_AP000547.3', 'ENSG00000283635_AC012485.3', 'ENSG00000283638_AC002407.1', 'ENSG00000283662_AC138904.3', 'ENSG00000283674_AC068587.4', 'ENSG00000283696_AL592295.4', 'ENSG00000283757_AL031686.1', 'ENSG00000283761_AC118553.2', 'ENSG00000283782_AC116366.3', 'ENSG00000283787_PRR33', 'ENSG00000283839_AC096667.1', 'ENSG00000283886_BX664615.2', 'ENSG00000283897_AC011416.3', 'ENSG00000283907_AD000090.1', 'ENSG00000283913_BMS1P21', 'ENSG00000283930_AL117339.5', 'ENSG00000283938_MIR3917', 'ENSG00000283959_AP002851.1', 'ENSG00000283994_AC092652.3', 'ENSG00000284024_HSPA14', 'ENSG00000284052_AC006460.2', 'ENSG00000284060_AC002472.2', 'ENSG00000284070_AP000356.2', 'ENSG00000284128_AP000356.3', 'ENSG00000284186_MIR3615', 'ENSG00000284188_AL451007.3', 'ENSG00000284237_AL356275.1', 'ENSG00000284292_AC004922.1', 'ENSG00000284308_C2orf81', 'ENSG00000284325_MIR3655', 'ENSG00000284428_IPO5P1', 'ENSG00000284431_AL022238.4', 'ENSG00000284523_AC004834.1', 'ENSG00000284526_AC015802.6', 'ENSG00000284543_LINC01226', 'ENSG00000284602_AL031432.4', 'ENSG00000284606_AC105233.5', 'ENSG00000284607_AL121936.1', 'ENSG00000284610_AC107918.4', 'ENSG00000284620_AF228730.5', 'ENSG00000284624_AC092902.5', 'ENSG00000284634_AC092821.3', 'ENSG00000284642_AL139424.2', 'ENSG00000284644_AC074386.1', 'ENSG00000284648_AC097493.4', 'ENSG00000284649_AC009093.8', 'ENSG00000284669_AC092053.3', 'ENSG00000284681_AC007240.1', 'ENSG00000284691_AC073111.5', 'ENSG00000284693_AL928921.2', 'ENSG00000284703_AL805961.2', 'ENSG00000284707_AC079781.5', 'ENSG00000284716_AL034417.3', 'ENSG00000284719_AL033527.5', 'ENSG00000284726_AL109936.6', 'ENSG00000284727_AC116562.4', 'ENSG00000284734_AC099063.4', 'ENSG00000284735_AL139424.3', 'ENSG00000284738_AL358472.5', 'ENSG00000284740_AL645728.2', 'ENSG00000284744_AL591163.1', 'ENSG00000284747_AL034417.4', 'ENSG00000284753_EEF1AKMT4', 'ENSG00000284770_TBCE', 'ENSG00000284773_AC114490.3', 'ENSG00000284828_AC012020.2', 'ENSG00000284830_AL049557.2', 'ENSG00000284874_SEPT5-GP1BB', 'ENSG00000284879_AC133644.3', 'ENSG00000284882_AL359762.1', 'ENSG00000284902_AC074008.1', 'ENSG00000284906_AC091057.6', 'ENSG00000284930_AC005280.2', 'ENSG00000284946_AC068831.7', 'ENSG00000284948_AC107959.4', 'ENSG00000284952_AC104472.3', 'ENSG00000284954_AL662884.3', 'ENSG00000284959_AC007262.2', 'ENSG00000284968_AC093827.4', 'ENSG00000284976_BX255925.3', 'ENSG00000284977_AL160272.1', 'ENSG00000285043_AC093512.2', 'ENSG00000285053_TBCE', 'ENSG00000285077_ARHGAP11B', 'ENSG00000285081_AC004593.2', 'ENSG00000285091_AL109840.2', 'ENSG00000285103_AL451123.1', 'ENSG00000285106_AC016831.6', 'ENSG00000285108_AC103718.1', 'ENSG00000285122_AC083829.2', 'ENSG00000285177_AL357556.4', 'ENSG00000285184_AC244033.2', 'ENSG00000285219_AL591485.1', 'ENSG00000285230_RALY-AS1', 'ENSG00000285231_OOSP3', 'ENSG00000285258_ATXN7', 'ENSG00000285278_TFAP2A-AS2', 'ENSG00000285280_AL390957.1', 'ENSG00000285331_AC090517.5', 'ENSG00000285354_AC010745.5', 'ENSG00000285399_AC104162.2', 'ENSG00000285410_GABPB1-IT1', 'ENSG00000285417_BX571818.1', 'ENSG00000285427_SOD2-OT1', 'ENSG00000285437_POLR2J3', 'ENSG00000285444_AL162377.3', 'ENSG00000285454_AC111006.1', 'ENSG00000285458_AC093827.5', 'ENSG00000285467_AL136419.1', 'ENSG00000285517_LINC00941', 'ENSG00000285531_Z83840.1', 'ENSG00000285533_AP001362.2', 'ENSG00000285535_AC021683.5', 'ENSG00000285542_AC013717.1', 'ENSG00000285545_AC124798.2', 'ENSG00000285554_AC242988.2', 'ENSG00000285560_AC103739.3', 'ENSG00000285571_AL513548.4', 'ENSG00000285589_AC010422.8', 'ENSG00000285595_AC105114.2', 'ENSG00000285596_AC017116.2', 'ENSG00000285600_AC023593.1', 'ENSG00000285608_AL161665.1', 'ENSG00000285612_AC004974.1', 'ENSG00000285624_AC025062.3', 'ENSG00000285627_AC005343.1', 'ENSG00000285630_AL590068.3', 'ENSG00000285632_AC084024.4', 'ENSG00000285642_AL139330.1', 'ENSG00000285646_AL021155.2', 'ENSG00000285651_AL450992.3', 'ENSG00000285658_AC127035.1', 'ENSG00000285661_AC127520.1', 'ENSG00000285663_AC108457.1', 'ENSG00000285667_AC012291.2', 'ENSG00000285669_AC026979.4', 'ENSG00000285676_AL158212.5', 'ENSG00000285679_AC097626.1', 'ENSG00000285688_AL161443.1', 'ENSG00000285693_AP002381.2', 'ENSG00000285696_AP002433.2', 'ENSG00000285702_AC211476.5', 'ENSG00000285708_AC097634.4', 'ENSG00000285713_AC098588.1', 'ENSG00000285719_AL356275.2', 'ENSG00000285721_AL031281.2', 'ENSG00000285725_AC004967.2', 'ENSG00000285728_AC098484.4', 'ENSG00000285730_Z94721.3', 'ENSG00000285737_AL138920.1', 'ENSG00000285744_AC083837.2', 'ENSG00000285747_AL133485.2', 'ENSG00000285751_AC021723.2', 'ENSG00000285752_AL031281.3', 'ENSG00000285756_BX890604.2', 'ENSG00000285761_AL645939.5', 'ENSG00000285789_AL162717.1', 'ENSG00000285793_AC125232.2', 'ENSG00000285796_AL162458.1', 'ENSG00000285799_AL645929.2', 'ENSG00000285803_AL442003.1', 'ENSG00000285824_AL353796.2', 'ENSG00000285825_AP003501.3', 'ENSG00000285827_AP001267.5', 'ENSG00000285830_AL109628.2', 'ENSG00000285844_FO393414.3', 'ENSG00000285850_AC009152.3', 'ENSG00000285851_AL359762.3', 'ENSG00000285852_AL353147.1', 'ENSG00000285856_AL353704.1', 'ENSG00000285857_AC016727.3', 'ENSG00000285864_AP000593.4', 'ENSG00000285870_AC012673.1', 'ENSG00000285872_AC007240.2', 'ENSG00000285875_AL035446.2', 'ENSG00000285877_AC007448.4', 'ENSG00000285881_AC105206.3', 'ENSG00000285884_AL022345.4', 'ENSG00000285886_AC211476.6', 'ENSG00000285887_AL009176.1', 'ENSG00000285901_AC008012.1', 'ENSG00000285906_AC083855.2', 'ENSG00000285908_AC080128.2', 'ENSG00000285923_AL160171.1', 'ENSG00000285933_AP003498.2', 'ENSG00000285938_AC072022.2', 'ENSG00000285943_AC112128.1', 'ENSG00000285948_AC123768.5', 'ENSG00000285967_NIPBL-DT', 'ENSG00000285972_CERNA2', 'ENSG00000285974_AC026624.1', 'ENSG00000285976_AL135905.2', 'ENSG00000285979_AC009090.6', 'ENSG00000285991_AL355312.5', 'ENSG00000285994_AL731559.1']


# In[7]:


preprocessor = Preprocess()
cite_train_x = preprocessor.fit_transform(pd.read_hdf(FP_CITE_TRAIN_INPUTS)[cnam_cite_start].values)
cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values
print(cite_train_y.shape)


# In[8]:


params = {
     'learning_rate': 0.1, 
     'metric': 'mae', 
     "seed": 42,
    'reg_alpha': 0.0014, 
    'reg_lambda': 0.2, 
    'colsample_bytree': 0.8, 
    'subsample': 0.5, 
    'max_depth': 10, 
    'num_leaves': 722, 
    'min_child_samples': 83, 
    }

model = MultiOutputRegressor(lgb.LGBMRegressor(**params, n_estimators=400))

model.fit(cite_train_x, cite_train_y)

y_va_pred = model.predict(cite_train_x)
mse = mean_squared_error(cite_train_y, y_va_pred)
print(mse)
del cite_train_x, cite_train_y
gc.collect()


# In[9]:


cite_test_x = preprocessor.transform(pd.read_hdf(FP_CITE_TEST_INPUTS)[cnam_cite_start].values)
test_pred = model.predict(cite_test_x)
del cite_test_x
test_pred.shape


# # Submission
# 
# We save the CITEseq predictions so that they can be merged with the Multiome predictions in the [Multiome quickstart notebook](https://www.kaggle.com/ambrosm/msci-multiome-quickstart).
# 
# The CITEseq test predictions produced by the ridge regressor have 48663 rows (i.e., cells) and 140 columns (i.e. proteins). 48663 * 140 = 6812820.
# 

# In[10]:


with open('citeseq_pred.pickle', 'wb') as f: pickle.dump(test_pred, f) # float32 array of shape (48663, 140)


# The final submission will have 65744180 rows, of which the first 6812820 are for the CITEseq predictions and the remaining 58931360 for the Multiome predictions. 
# 
# We now read the Multiome predictions and merge the CITEseq predictions into them:

# In[11]:


with open("../input/msci-multiome-quickstart/partial_submission_multi.pickle", 'rb') as f: submission = pickle.load(f)
submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
assert not submission.isna().any()
submission = submission.round(6) # reduce the size of the csv
submission.to_csv('submission.csv')
submission

