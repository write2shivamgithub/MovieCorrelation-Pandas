#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[66]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# ## Read in the data 

# In[67]:


df = pd.read_csv(r'D:\Data+Science@Consoleflare\Pandas\Correlation In Python\movies.csv')


# ## Let's Look at the data

# In[68]:


df.head()


# ## Let's see if there is any missing data

# In[69]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{}-{}%'.format(col,pct_missing))


# ## Data types for columns

# In[70]:


df.dtypes


# ## Changing data types for columns

# In[71]:


df['budget'] = df['budget'].fillna(df['budget'].mean())  # Replace NaN values with mean budget(P1)
df['budget'] = df['budget'].astype('int64')

df['gross'] = df['gross'].fillna(df['gross'].mean())  # Replace NaN values with mean gross(P2)
df['gross'] = df['gross'].astype('int64')
df.head()


# ## Create correct year column

# In[72]:


df['correctyear'] = df['released'].str.extract(r'(\d{4})')   #(P3)
#Uses the .str.extract() method to extract the year from the 'date' column.
#expression r'(\d{4})' captures a sequence of four digits representing the year.
df.head()


# In[73]:


df = df.sort_values(by = ['gross'],inplace = False,ascending=False)
pd.set_option('display.max_rows',None)
df


# ### Scatter plot with budget vs gross

# In[74]:


plt.scatter(x=df['budget'],y=df['gross'])
plt.title('Budget Vs Gross Earning')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earning')
plt.show()


# ## Plot budget VS gross using seaborn

# In[75]:


sns.regplot(x='budget',y='gross',data=df,scatter_kws={'color':'black'},line_kws={'color':'purple'})


# ## Looking at correlation

# In[76]:


df.corr(method='pearson')
#Only works on numerical data
#types of correlation:- pearson,kendall,spearman


# In[77]:


# High correlation between budegt and gross


# In[78]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matric for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[ ]:





# ## Converting Non-Numeric Into Numeric Movie Features

# In[79]:


df_numerized = df
for col_name in df_numerized.columns:
    if (df_numerized[col_name].dtype == 'object'):  #line checks if the data type of the column is 'object', indicating it contains string values.
        df_numerized[col_name] = df_numerized[col_name].astype('category')  # line converts the column to the 'category' data type, which is a categorical data type in pandas.
        df_numerized[col_name] = df_numerized[col_name].cat.codes  #line assigns the category codes to the column, replacing the original string values with numerical codes.
df_numerized.head()


# In[80]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation Matric for Non-Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[81]:


# votes and budget have highest correlation to gross earnings

