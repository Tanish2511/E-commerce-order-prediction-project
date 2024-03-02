#!/usr/bin/env python
# coding: utf-8

# # E-commerce Order Prediction Project
# 
# ## Overview
# This project aims to predict order patterns in an e-commerce setting using machine learning techniques. The dataset used for analysis is the Amazon sales dataset, which contains information about various orders placed on the Amazon platform.
# 
# ## Dataset
# The dataset comprises sales data from Amazon, including attributes such as order ID, date, status, fulfillment method, sales channel, product details, quantity, currency, amount, shipping details, and more. It provides valuable insights into customer purchasing behavior and order fulfillment processes.
# 
# 
# ## Project Phase - 1
# 
# 1. Data Loading and Exploration: 
# In this phase, we'll load the dataset, explore its structure, and perform initial data analysis to understand the underlying patterns and relationships.

# In[ ]:





# # Import Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[2]:


os.chdir("C:\\Users\\kural\\OneDrive\\Desktop\\E-commerce dataset")


# In[3]:


df = pd.read_csv("Amazon Sale Report.csv")
df.head()


# In[4]:


df.shape


# In[5]:


# Removing unwanted columns from the table

df = df.drop(labels = ['index' , 'Order ID', 'Unnamed: 22', 'ship-postal-code', 'promotion-ids'], axis = 1)
df.head()


# In[6]:


df.describe().T


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[10]:


df = df.dropna()


# In[11]:


df.isnull().sum()


# In[26]:


df.shape


# In[12]:


# finding the unique values in different columns


# In[13]:


unique_values1 = df['Status'].unique()
unique_values1


# In[14]:


unique_values2 = df['Fulfilment'].unique()
unique_values2


# In[15]:


unique_values3 = df['Sales Channel '].unique()
unique_values3


# In[16]:


unique_values4 = df['ship-service-level'].unique()
unique_values4


# In[17]:


unique_values5 = df['Category'].unique()
unique_values5


# In[18]:


unique_values6 = df['Size'].unique()
unique_values6


# In[19]:


unique_values7 = df['Courier Status'].unique()
unique_values7


# In[20]:


unique_values8 = df['ship-city'].unique()
unique_values8


# In[21]:


unique_values9 = df['ship-state'].unique()
unique_values9


# In[22]:


unique_values10 = df['ship-country'].unique()
unique_values10


# In[23]:


# converting 'Date' datatype from object to Datetime.


# In[24]:


df['Date'] = pd.to_datetime(df['Date']) 


# In[25]:


df['month'] = df['Date'].dt.month
df['month'].unique()


# In[ ]:





# ### Summary
# In this phase, we loaded the Amazon sales dataset and conducted preliminary exploration to understand its structure and contents. We inspected various attributes such as order ID, date, status, fulfillment method, sales channel, product details, quantity, currency, amount, and shipping information.
# 
# ### Key Insights
# - Data Overview : The dataset contains 32395 rows and 20 columns.
# - Attribute Types : We identified 16 categorical and numerical 2 attributes in the dataset.
# - Missing Values : I droped all null values.
# - Statistical Summary : We computed basic statistics such as mean, median, and standard deviation for numerical attributes.
# 
# ### Next Steps
# Based on our initial exploration, we plan to delve deeper into the data through exploratory data analysis (EDA). The EDA phase will involve:
# - Further analysis of individual features and their relationships.
# - Identification of potential outliers or anomalies.
# - Feature engineering to create new variables that may enhance model performance.
# - Additional visualizations to gain insights into customer behavior and order patterns.
# 
# Stay tuned for updates in the next phase of our project!

# In[ ]:




