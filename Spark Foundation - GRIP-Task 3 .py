#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Symrun Lopes 


# In[ ]:


## Spark Foundation - GRIP-Task 3


# In[ ]:


## As a business manager, try to find out the weak areas where you can work to make more profit.


# In[ ]:


# Import the data tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
def ignore_warn(*args, **kwargs):
    pass
warning_warn = ignore_warn          # ignore warning(from sklearn and seaborn)


# In[3]:


import os
print(os.listdir("."))              # for accessing folder and  file easily


# In[7]:


#import the data
df = pd.read_csv("Desktop\SampleSuperstore.csv")


# In[8]:


df.head()


# In[9]:


##Check the duplicates rows in DataFrame
df.duplicated().sum()


# In[10]:


# Drop the duplicates rows
df.drop_duplicates(inplace=True)


# In[11]:


df.duplicated().sum()


# In[12]:


# Information about data set
df.info()


# In[ ]:


## There are 9994 rows and total 13 columns and there is missing values. We also notice that we have 5 numerical columns and the rest 8 columns are categorical columns


# In[13]:


# Descr
df.describe()


# In[1]:


## Exploratory Data Analysis(EDA)


# In[14]:


# Let's plot some heatmap to find co-relational among the features
corrmat = df.corr()
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(corrmat, square=True)


# In[ ]:


## Each square in a heatmap shows how much any two feature correlates (is there some kind of relationship between them). Does the increase in one feature result in the increase of the other? (Positive correlation) or does the increase in one result in the decrease of the other? (Negative correlation).

# The vertical range running from 0.0 to 1.0 shows you the relative size of the correlation between any two features, where hotter values are higher and means more correlation.
# The diagonal almost-whitish portion of the matrix shows that all features are highly correlated to themselves.


# In[15]:


df.columns


# In[16]:


plt.style.use('dark_background')
dataset = df.drop(['Postal Code'],axis=1)
df['Ship Mode'].value_counts()


# In[17]:


sns.pairplot(df,hue='Ship Mode')


# In[18]:


cat_col = ['Ship Mode','Segment','Country','City','State','Region','Category','Sub-Category']
for col in cat_col:
    sns.set()
    cols = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Postal Code',
       'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount',
       'Profit']
plt.figure()    
sns.pairplot(df[cols], height=3.0, hue='Region')
plt.show()


# In[19]:


cat_col = ['Ship Mode','Segment','Country','City','State','Region','Category','Sub-Category']
for col in cat_col:
    sns.set()
    cols = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Postal Code',
       'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount',
       'Profit']
plt.figure()    
sns.pairplot(df[cols], height=3.0, hue='Sub-Category')
plt.show()


# In[ ]:


## The pairplot shows that the co-relation between the features(Features must be numerical type).


# In[20]:


sns.barplot(x='Region',y='Sales',data=df)


# In[ ]:


## The above plot shows that the central region has minimum number of sales and east & south region have maximum no of sales.


# In[21]:


sns.barplot(x='Region',y='Quantity',data=df)


# In[22]:


sns.barplot(x='Region',y='Profit',data=df)


# In[ ]:


## The above plot gives us the which region contains how much profit. West region contain most profit and central region contain min.


# In[23]:


sns.barplot(x='Ship Mode',y='Quantity',data=df)


# In[ ]:


## The above diagram shows that which ship mode contain maximum number of quantity.


# In[24]:


sns.barplot(x='Ship Mode',y='Profit',data=df)


# In[ ]:


## Shows that which ship mode has how much profit. Above plot clearly shows that first class contain maximum profit and standard class has minimum profit.


# In[25]:


sns.countplot(x='Ship Mode',data=df)


# In[ ]:


## Above plot shows that standard class,ship mode has mximum items.


# In[26]:


sns.jointplot(x='Sales',y='Profit',data=df,kind='reg')


# In[ ]:


## Salles and profit are positively corelated 


# In[27]:


plt.figure(figsize=(12,10))
df['Sub-Category'].value_counts().plot.pie(autopct='light')
plt.show()


# In[28]:


plt.figure(figsize=(15,12))
sns.countplot(x='State',data=df,palette='rocket',order=df['State'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


# In[29]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# In[30]:


plt.figure(figsize=(10,10))
dataset['Region'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()


# In[31]:


fig,ax=plt.subplots(figsize=(8,8))
ax.scatter(df['Sales'],df['Profit'])
ax.set_xlabel('Sales')
ax.set_xlabel('Profit')
plt.show()


# In[33]:


sns.lineplot(x='Discount',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# In[ ]:


## Discount and profit are negatively corelated 


# In[34]:


df.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['yellow','blue'],figsize=(8,5))
plt.ylabel('Profit/loss and sales')
plt.show()


# In[35]:


plt.figure(figsize=(10,8))
plt.title('Segment wise Sales in each region')
sns.barplot(x='Region',y='Sales',data=df,hue='Segment',order=df['Region'].value_counts().index,palette='rocket')
plt.xlabel('Region',fontsize=15)
plt.show()


# In[36]:


df.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['red','skyblue'],figsize=(8,5))
plt.ylabel('Profit/loss and sales')
plt.show()


# In[37]:


ps = df.groupby('State')[['Profit','Sales']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['red','blue'],figsize=(12,7))
plt.title('Profit/loss and Sales across states')
plt.xlabel('State')
plt.ylabel('Profit/loss and sales')
plt.show()


# In[38]:


top_states = df['State'].value_counts().nlargest(10)
top_states


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




