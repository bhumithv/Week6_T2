#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


# In[17]:


data = pd.read_csv("C:\\Users\\bhumi\\OneDrive\\Desktop\\Week6\\Super-Store-Sales-main\\Super-Store-Sales-main\\train.csv")


# In[15]:


data.head()


# # Creating a copy of the original dataset

# In[18]:


data = data.copy()


# # Getting first 5 values of the dataset

# In[19]:


data.head()


# # Getting last 5 values of the dataset

# In[20]:


data.tail()


# # Getting all the columns of the dataset

# In[21]:


data.columns


# # Making the names of the columns of the dataset to lowercase and replacing the spaces with underscore

# In[22]:


data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(' ', '_')


# # Checking for null values in the dataset

# In[23]:


data.isna().sum()


# # Getting information about the dataset

# In[24]:


data.info(verbose=True)    


# # Getting shape of the dataset

# In[25]:


data.shape


# # Data cleaning

# # Getting the names of the state with empty postal code

# In[26]:


states_with_empty_postal = data.loc[data['postal_code'].isnull(), 'state'].unique()

print("States with empty postal codes:", states_with_empty_postal)


# # Filling the postal_code for Vermont with '05401' (looked up on the internet)

# In[27]:


data['postal_code'].fillna(05401.0, inplace=True)

data['postal_code'] = data['postal_code'].astype(int)


# # Checking for null values in the dataset

# In[28]:


data.isna().sum()


# # Getting a random sample from the dataset

# In[29]:


data.sample()


# # Converting the date columns to 'datetime' datatype

# In[30]:


date_columns = ['order_date', 'ship_date']

data[date_columns] = data[date_columns].apply(lambda col: pd.to_datetime(col, format='%d/%m/%Y', errors='coerce'))

data.dtypes


# # Getting description of the dataset

# In[31]:


data['sales'].describe()


# # Getting the unique values present in ship_mode column

# In[32]:


data['ship_mode'].unique()


# # Mapping the ship_mode values as

# In[33]:


ship_mode_mapping = {'Standard Class': 0, 'First Class': 1, 'Second Class': 2, 'Same Day': 4}

data['ship_mode'] = data['ship_mode'].replace(ship_mode_mapping)

data['ship_mode'] = data['ship_mode'].astype(int)


# # Getting the Quantity of the items sold from order_id

# In[35]:


order_quantity_df = data.groupby('order_id')['row_id'].count().reset_index()
order_quantity_df.rename(columns={'row_id': 'quantity'}, inplace=True)

# Merging the order_quantity_df with the original DataFrame to include 'quantity' information
merged_df = pd.merge(data, order_quantity_df, on = 'order_id')

# Now, 'quantity' column represents the number of products in each order
print(merged_df[['order_id', 'quantity']])


# # Data Visualization

# In[38]:


plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.hist(data['sales'], bins=20, edgecolor='black', alpha=0.7, color='blue')
plt.title('Histogram for Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
plt.hist(data['sub-category'], bins=20, edgecolor='black', alpha=0.7, color='green')
plt.title('Histogram for Sub-category')
plt.xlabel('Sub- Category')
plt.xticks(rotation=90)
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# # Visualizing a time-series plot for sales over the past 4 year

# In[39]:


data['order_year'] = data['order_date'].dt.year

# Group by 'order_year' and calculate the total sales for each year
yearly_sales = data.groupby('order_year')['sales'].sum().reset_index()

# Plotting a time series plot for total sales per year
plt.figure(figsize=(12, 6))
plt.plot(yearly_sales['order_year'], yearly_sales['sales'], marker='o', linestyle='-', color='b')
plt.title('Total Sales Over Time')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(yearly_sales['order_year'].astype(int))
plt.show()


# # Performing seasonal decomposition

# In[40]:


yearly_sales['order_year'] = pd.to_datetime(yearly_sales['order_year'], format='%Y')
yearly_sales.set_index('order_year', inplace=True)

# Perform seasonal decomposition
result = seasonal_decompose(yearly_sales['sales'], model='multiplicative')

# Plot the decomposed components
plt.figure(figsize=(12, 8))

# Original time series
plt.subplot(4, 1, 1)
plt.plot(yearly_sales['sales'], label='Original')
plt.legend()
plt.title('Original Time Series')

# Trend component
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()
plt.title('Trend Component')

# Seasonal component
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()
plt.title('Seasonal Component')

# Residual component
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residuals')
plt.legend()
plt.title('Residual Component')

plt.tight_layout()
plt.show()


# # Visualizing the Box Plot of Sales

# In[41]:


plt.figure(figsize=(8, 6))
yearly_sales.boxplot(column='sales')
plt.title('Box Plot of Sales')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()


# # Visualizing the Sales by State and City via sunburst chart

# In[42]:


fig = px.sunburst(
    data,
    path=['state', 'city'],
    values='sales',
    title='Sunburst Chart of Sales by State and City'
)

fig.show()


# # Visualizing

# In[43]:


# Setting 'order_date' as the index
data.set_index('order_date', inplace=True)

# Resampling data for different time periods
monthly_sales = data['sales'].resample('M').sum()
quarterly_sales = data['sales'].resample('Q').sum()
yearly_sales = data['sales'].resample('Y').sum()

# Plotting line charts for sales variation over time
plt.figure(figsize=(12, 6))

# Monthly sales
plt.subplot(3, 1, 1)
monthly_sales.plot(marker='o')
plt.title('Monthly Sales Variation')
plt.xlabel('Date')
plt.ylabel('Sales')

# Quarterly sales
plt.subplot(3, 1, 2)
quarterly_sales.plot(marker='o')
plt.title('Quarterly Sales Variation')
plt.xlabel('Date')
plt.ylabel('Sales')

# Yearly sales
plt.subplot(3, 1, 3)
yearly_sales.plot(marker='o')
plt.title('Yearly Sales Variation')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()

# Resetting the index to bring 'order_date' back as a column
data.reset_index(inplace=True)


# # Visualizing Correlation Matrix

# In[44]:


numeric_variables = ['sales', 'postal_code', 'ship_mode']

correlation_matrix = data[numeric_variables].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# # Visualizing Scatter Plot between ship_mode and Sales

# In[49]:


numeric_variable_to_explore = 'ship_mode'

# Plotting scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[numeric_variable_to_explore], y=data['sales'])
plt.title(f'Scatter Plot between {numeric_variable_to_explore} and Sales')
plt.xlabel(numeric_variable_to_explore)
plt.ylabel('Sales')
plt.xticks(data['ship_mode'].astype(int))
plt.show()


# # Visualizing Pair Plots for Numeric Variables

# In[50]:


numeric_variables_for_pairplot = ['sales', 'ship_mode']

sns.pairplot(data[numeric_variables_for_pairplot])
plt.suptitle('Pair Plots for Numeric Variables', y=1.02)
plt.show()


# # Visualizing stacked bar chart for sales across different regions and categories

# In[52]:


region_category_sales_df = data[['region', 'category', 'sales']]

pivot_df = region_category_sales_df.pivot_table(index='region', columns='category', values='sales', aggfunc='sum', fill_value=0)

plt.figure(figsize=(12, 8))
pivot_df.plot(kind='bar', stacked=True)
plt.title('Sales Comparison Across Regions and Categories')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.xticks(rotation=90, fontsize=8)
plt.legend(title='Category', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# # Visualizing bar chart for total sales across different customer segments

# In[54]:


agg_customer_segment_sales = data.groupby('category')['sales'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='sales', data=agg_customer_segment_sales)
plt.title('Total Sales Comparison Across Customer Segments')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()


# # Visualizing scatter plot for the relationship between sales and quantity sold

# In[55]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='quantity', y='sales', data=merged_df)
plt.title('Sales vs. Quantity Sold')
plt.xlabel('Quantity Sold')
plt.ylabel('Sales')
plt.show()


# # Visualizing bar chart for quantity sold across different customer segments

# In[57]:


plt.figure(figsize=(10, 6))
sns.barplot(x=data['sub-category'], y=merged_df['quantity'])
plt.title('Quantity Sold Across Customer Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Quantity Sold')
plt.ylim(0, 5)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Visualizing bar chart for product-wise sales

# In[59]:


plt.figure(figsize=(12, 6))
sns.barplot(x='sub-category', y='sales', data=data)
plt.title('Product-wise Sales Analysis')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()


# # Visualizing grouped bar chart for region-wise product sales

# In[60]:


# Creating a pivot table to get region-wise product sales
pivot_df = pd.pivot_table(data, values='sales', index='sub-category', columns='region', aggfunc=np.sum, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
pivot_df.plot(kind='bar', ax=ax, width=0.8)
plt.title('Region-wise Product Sales')
plt.xlabel('Sub-category')
plt.ylabel('Sales')
plt.legend(title='Region', bbox_to_anchor=(1, 1))
plt.xticks(rotation=90)
plt.show()


# 
