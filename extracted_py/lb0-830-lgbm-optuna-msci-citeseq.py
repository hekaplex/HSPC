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

