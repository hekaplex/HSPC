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




