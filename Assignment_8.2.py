#!/usr/bin/env python
# coding: utf-8

# Read the Dataset

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


Adult=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")


# In[5]:



Adult.head()


# In[6]:


Adult.columns=['age','workclass','fnlwgt','education','education_num',
               'marital_status',
               'occupation','relationship','race','sex','capital_gain',
               'capital_loss','hours_per_week','native_country','Amount']


# In[7]:


Adult.head()


# In[3]:


from pandas import DataFrame, Series
import sqlite3 
from pandasql import sqldf as sql
import sys


# In[4]:


import sqlalchemy


# In[6]:


from sqlalchemy import create_engine


# In[13]:


engine = create_engine('sqlite:///:memory:',echo=True)


# In[15]:


from sqlalchemy.ext.declarative import declarative_base


# In[16]:


Base = declarative_base()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




