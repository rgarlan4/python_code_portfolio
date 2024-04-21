#!/usr/bin/env python
# coding: utf-8

# # Annotated follow-along guide: EDA using basic data functions with Python

# This notebook contains the code used in the following instructional video: [EDA using basic data functions with Python](https://www.coursera.org/learn/go-beyond-the-numbers-translate-data-into-insight/lecture/4k4Vg/eda-using-basic-data-functions-with-python).

# ## Introduction

# Throughout this notebook, we will implement discovering skills on a dataset. Before getting started, watch the associated instructional video and complete the in-video question. All of the code we will be implementing and related instructions are contained in this notebook.

# ## Overview
# 
# In this notebook, we will use pandas to examine 2018 lightning strike data collected by the National Oceanic and Atmospheric Administration (NOAA). Then, we will calculate the total number of strikes for each month and plot this information on a bar graph.

# ## Import packages and libraries
# 
# Before getting started, we will need to import all the required libraries and extensions. Throughout the course, we will be using pandas, numpy, and datetime for operations, and matplotlib, pyplot, and seaborn for plotting.

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


# In[2]:


# Read in the 2018 lightning strike dataset.
df = pd.read_csv('eda_using_basic_data_functions_in_python_dataset1.csv')


# In[3]:


# Inspect the first 10 rows.
df.head(10)


# Notice that the data is structured as one row per day along with the geometric location of the strike. 

# A quick way to determine how many rows and columns of data there are in total is to use `df.shape`. The information will be output as: ([rows], [columns]).

# In[4]:


df.shape


# The total number of rows is 3,401,012, and there are three columns. 

# In[5]:


# Get more information about the data, including data types of each column
df.info()


# ### Convert the date column to datetime

# `info` will provide the total number of rows (3,401,012) and columns (3). It will also state the names and data types of each column, as well as the size of the dataframe in memory. 
# 
# In this case, notice that the `date` column is an 'object' type rather than a 'date' type. Objects are strings. When dates are encoded as strings, they cannot be manipulated as easily. Converting string dates to datetime will enable you to work with them much more easily.
# 
# Let's convert to datetime using the pandas function `to_datetime()`.

# In[6]:


# Convert date column to datetime
df['date']= pd.to_datetime(df['date'])


# ### Calculate the days with the most strikes

# As part of discovering, you want to get an idea of the highest data points. For this dataset, we can calculate the top 10 days of 2018 with the most number of lightning strikes using the `groupby()`, `sum()`, and `sort_values()` functions from pandas.
# 
# When using `groupby()` on the date column, the function combines all rows with the same date into a single row. 
# 
# Then, using `sum()` performs a sum calculation on all other summable columns. In this case, we are summing all the lightning strikes that happened on each day. Notice that the `center_point_geom` column is not included in the output. That's because, as a string object, this column is not summable. 
# 
# Finally, `sort_values()` returns the results in descending order of total strikes for each day in the data.

# In[7]:


# Calculate days with most lightning strikes.
df.groupby(['date']).sum().sort_values('number_of_strikes', ascending=False).head(10) 


# A common mistake among data professionals is using `count()` instead of `sum()`, and vice versa. In this case, `count()` would return the number of occurrences of each date in the dataset, which is not the desired result.

# ### Extract the month data

# Next, we will extract the month data from the `date` column and add that extracted month data into a new column called `month`. `dt.month` extracts just the month information (as a numeric value) from the date. This is why converting the `date` column to datetime is very useful. 

# In[8]:


# Create a new `month` column
df['month'] = df['date'].dt.month
df.head()


# ### Calculate the number of strikes per month

# Now, we will sort our values by most strikes per month. Use `groupby()`, `sum()` and `sort_values()` from pandas again.

# In[9]:


# Calculate total number of strikes per month
df.groupby(['month']).sum().sort_values('number_of_strikes', ascending=False).head(12)


# ### Convert the month number to text 

# To help read the data more easily, let's convert the month number to text using the datetime function `dt.month_name()` and add this as a new column in the dataframe. `str.slice` will omit the text after the first three letters. 

# In[10]:


# Create a new `month_txt` column.
df['month_txt'] = df['date'].dt.month_name().str.slice(stop=3)
df.head()


# ### Create a new dataframe

# The objective is to plot the total number of strikes per month as a bar graph. To help with the plotting, we will create a new dataframe called `df_by_month`. This will allow us to easily access the month, month text, and total number of strikes for each month. 

# In[11]:


# Create a new helper dataframe for plotting.
df_by_month = df.groupby(['month','month_txt']).sum().sort_values('month', ascending=True).head(12).reset_index()
df_by_month


# ### Make a bar chart

# Now, let's make a bar chart. Pyplot's `plt.bar()` function takes positional arguments of `x` and `height`, representing the data used for the x- and y- axes, respectively. The x-axis will represent months, and the y-axis will represent strike count.

# In[12]:


plt.bar(x=df_by_month['month_txt'],height= df_by_month['number_of_strikes'], label="Number of strikes")
plt.plot()

plt.xlabel("Months(2018)")
plt.ylabel("Number of lightning strikes")
plt.title("Number of lightning strikes in 2018 by months")
plt.legend()
plt.show()


# ## Conclusion
# 
# If you have successfully completed the material above, congratulations! You now have some of the fundamental elements of data discovery that you can apply to your own datasets. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




