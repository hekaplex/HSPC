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

