#!/usr/bin/env python
# coding: utf-8

# # Multiome Quickstart
# 
# This notebook shows how to cross-validate a baseline model and create a submission for the Multiome part of the *Multimodal Single-Cell Integration* competition without running out of memory.
# 
# It does not show the EDA - see the separate notebook [MSCI EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/ambrosm/msci-eda-which-makes-sense).
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
# The Multiome test predictions have 55935 rows and 23418 columns. 55935 \* 23418 = 1’309’885’830 predictions. We'll only submit 4.5 % of these predictions. According to the data description, this subset was created by sampling 30 % of the Multiome rows, and for each row, 15 % of the columns (i.e., 16780 rows and 3512 columns per row). Consequently, when reading the test data, we can immediately drop 70 % of the rows and keep only the remaining 16780.
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




