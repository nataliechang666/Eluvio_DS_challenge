#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# # Recommedation Filter

# In[2]:


data = pd.read_csv("/Users/nataliechang/Desktop/Eluvio_DS_Challenge.csv")


# In[3]:


data = data.drop(columns=['time_created','down_votes','category'])
data['date_created'] = pd.to_datetime(data['date_created'], format='%Y-%m-%d')
data.head()


# In[4]:


#ex: input 'Japan resumes refuelling mission'


# In[5]:


idx = data[data['title']=='Jump-start economy: Give health care to all '].index.values[0]


# In[6]:


#First, find the time earlier than current time for news i


# In[7]:


#data['date_created'][idx]


# In[8]:


d2 = data[data['date_created']<=data['date_created'][idx]]


# In[9]:


#Second, define whether this news is over 18
#If false, choose all news which are also under 18 (df[df['over_18']==df['over_18'][idx]])
#If True, choose all news (df = df)


# In[10]:


data['over_18'][idx]


# In[11]:


d3 = d2[d2['over_18']==d2['over_18'][idx]]


# In[12]:


#Third, find whether same author, then choose top 1 up_votes news


# In[13]:


d3


# In[14]:


d3['author']==d3['author'][idx]


# In[15]:


d4 = d3[d3['author']==d3['author'][idx]]


# In[16]:


d4 = d4.drop([idx]).sort_values('up_votes',ascending=False)


# In[17]:


d4.reset_index(drop=True, inplace=True)


# In[51]:


d4['title'][0]


# In[17]:


#Function for recommendation


# In[22]:


def recomm_news(title):
    idx = data[data['title'] == title].index.values[0]
    d2 = data[data['date_created']<=data['date_created'][idx]]
    if data['over_18'][idx]==False:
        d3 = d2[d2['over_18']==d2['over_18'][idx]]
    else:
        d3 = d2
    d4 = d3[d3['author']==d3['author'][idx]]
    if len(d4)==1:
        d4 = d3
        d4 = d4.drop([idx]).sort_values('up_votes',ascending=False)
        d4.reset_index(drop=True, inplace=True)
    else:
        d4 = d4.drop([idx]).sort_values('up_votes',ascending=False)
        d4.reset_index(drop=True, inplace=True)
    return d4['title'][0]


# In[23]:


recomm_news('Jump-start economy: Give health care to all ')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




