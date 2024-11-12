#!/usr/bin/env python
# coding: utf-8

# ## Importing Data

# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# ## Loading Data to Data Frame

# In[51]:


df = pd.read_csv("data.csv")
df.head(5)


# In[53]:


df.tail(5)


# In[55]:


df.describe()


# ## Checking the Data Type

# In[57]:


df.dtypes


# ## Dropping the Data that doesn't work for this case

# In[59]:


df = df.drop(['Engine Fuel Type','Market Category','Vehicle Style','Popularity',
              'Number of Doors','Vehicle Size'], axis = 1)
df.head(5)


# ## Renaming the Data/Columns

# In[61]:


df = df.rename(columns={"Engine HP":"HP", "Engine Cylinders":"Cylinders",
                        "Transmission Type":"Transmission", 
                        "Driven_Wheels":"Drive Mode", "highway MPG":"MPG-H",
                        "city mpg":"MPG-C", "MSRP":"Price"})
df.head(5)


# ## Dropping Duplicate Rows

# In[63]:


df.shape


# In[65]:


duplicate_rows_df = df[df.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_df.shape)


# In[67]:


df.count()


# In[69]:


df = df.drop_duplicates()
df.head(5)


# In[71]:


df.count()


# ## Dropping the Missing or Null Values

# In[73]:


print(df.isnull().sum())


# In[75]:


df = df.dropna()
df.count()


# In[77]:


print(df.isnull().sum())


# ## Detecting Outliers - IQR score technique

# In[79]:


sns.boxplot(x=df['Price'])


# In[81]:


sns.boxplot(x=df['HP'])


# In[83]:


sns.boxplot(x=df['Cylinders'])


# In[85]:


## Need to add numeric_only = True. Credit to Jason.
Q1 = df.quantile(0.25,numeric_only=True)
Q3 = df.quantile(0.75,numeric_only=True)
IQR = Q3 - Q1
print (IQR)


# In[91]:


## df = df[~((df < (Q1 -1.5*IQR)) | (df >(Q3 + 1.5 * IQR))).any(axis=1)]
## df.shape

# It doesn't work. It has to create a mask to filter the DataFrame so that it is only numeric values (Jason)
# Reindex Q1, Q3 and IQR to match the columns in numeric_df
Q1 = Q1.reindex(numeric_df.columns)
Q3 = Q3.reindex(numeric_df.columns)
IQR = IQR.reindex(numeric_df.columns)

#Create the mask using the reindexed values and align it with df's index
mask = ~((numeric_df < (Q1 -1.5*IQR)) | (numeric_df > (Q3 + 1.5 *IQR))).any(axis=1)
mask = mask.reindex(df.index,fill_value=False)

#Apply the mask to the original df
df = df[mask]
df.shape
#Credits to Jason. Thank you!


# ## Plot Different Features against One Another (Scatter), against frequency(histogram)

# ## Histogram

# In[93]:


df.Make.value_counts().nlargest(40).plot(kind="bar",figsize=(10,5))
plt.title("Number of cars by make")
plt.ylabel('Number of Cars')
plt.xlable('Make')                               


# ## Heat Maps

# In[3]:


plt.figure(figsize=(10,5))
c = df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# ## Scatterplot

# In[42]:


fig,ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['Price'])
ax.set_xlabel('HP')
ax.set_ylabel('Price')
plt.show()


# ## 3D Table

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

#defining a user-defined function to plot the three-dimensional scatter plot
def three_dimensional_scatter(x1,y1,z1,x2,y2,z2):
    #creating figure object in the plot
    fig = plt.figure(figsize = (9,9))
    #created three-dimensional workspace 
    ax = fig.add_subplot(111, projection='3d')
    #plotting the first three-dimensional scatter plot using the parameters 
    ax.scatter3D(x1, y1, z1, color = "green")
    #plotting the second three-dimensional scatter plot using the parameters
    ax.scatter3D(x2, y2, z2, color = "red")
    #defining title to the plot
    plt.title("Creating three-dimensional scatter plot using matplotlib and numpy")
    #defining legends to the plot
    plt.legend(["first","second"])
    #displaying the three-dimensional scatter plot
    plt.show()
    
#creating the main() function
def main():
    #creating data points for the z-axis
    z1 = np.arange(0,150,1)
    #creating data points for the x-axis
    x1 = np.random.randint(8000, size =(150))
    #creating data points for the y axis
    y1 = np.random.randint(800, size =(150))
    z2 = np.arange(0,150,1)
    #creating data points for the x-axis
    x2 = np.random.randint(8000, size =(150))
    #creating data points for the y axis
    y2 = np.random.randint(800, size =(150))
    #calling the main() function
    three_dimensional_scatter(x1,y1,z1,x2,y2,z2)

#declaring the main() function as the driving code of the program.
if __name__ == "__main__":
    main()


# In[ ]:




