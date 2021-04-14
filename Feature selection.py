#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sb
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("/Users/nataliechang/Desktop/Eluvio_DS_Challenge.csv")


# In[3]:


data.head()


# In[4]:


data = data.drop(columns=['down_votes','category'])


# In[5]:


class_mapping = {label: idx for idx, label in enumerate(np.unique(data['over_18']))}
data['over_18'] = data['over_18'].map(class_mapping)
class_mapping2 = {label: idx for idx, label in enumerate(np.unique(data['author']))}
data['author'] = data['author'].map(class_mapping2)

data['date_created'] = data['date_created'].map(lambda x: x.replace('-',''))


# In[6]:


data


# # Feature Selection

# In[12]:


data.drop(columns=['up_votes','title'])


# In[10]:


X = np.array(data.drop(columns=['up_votes','title']))
thre = np.quantile(data['up_votes'], 0.8)
y = [1 if i > thre else 0 for i in data['up_votes']]

model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=500, learning_rate=0.001, 
                                    max_depth=800, feature_fraction=0.8, subsample=0.2,
                                    is_unbalance=True)
model.fit(X,y)
lgb.plot_importance(model, max_num_features=5,figsize=(12,9))
plt.title("Featurertances")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




