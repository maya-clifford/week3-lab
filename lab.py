# %% [markdown]
# ## Part 1 

# For the college completion dataset, one potential question
# it could answer is how to predict what colleges are in the 
# highest percentile for students who graduate in the typical 
# amount of time. 
# 
# For the campus recruitment dataset, it can answer the 
# question of if performance in school relates to your salary 
# after graduation by trying to predict salary based on the
# other features. 

# %% [markdown]
# ## Part 2 

# %% 
# import pacakges 
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %% [markdown]
# ### College Completion Dataset 
# A generic question that this dataset could address is predicting 
# if a college is in a high percentile for the proportion of students
# who graduate in four years and what factors play into that. 
# 
# An independent business metric for this problem is if colleges are 
# able to increase how many students graduate on time by changing the 
# factors that are most infulential in the model. 


# %% 
# College Completion Dataset 
# Read in the data 
url = ("https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv")
completion = pd.read_csv(url)

# Check for null values and data types 
completion.info()

# %%
# The dataset has an X in the hbcu and flagship columns when true, so 
# Make the column have 1 replace the X and 0 replace null
xcols = ['hbcu', 'flagship']
completion[xcols] = completion[xcols].notna().astype(int)
# Check if this was successful 
completion.head(6)
# The hbcu and flagship columns were successfully changed to numeric 

# %%
# Remove columns with less than 1,000 null values 
# In this case, since the dataset has almost 3,800 rows, we're considering
# less than 1,000 non-null values to mean that the data wouldn't be useful
# This will only be removing the nickames and vsa enroll/grad columns 
# which wouldn't be that useful in answering our question anyway. 

# Create a list of the columns that have less than 1,000 values and drop
# the columns in the list 
to_drop = completion.columns[completion.count() < 1000]
completion = completion.drop(columns=to_drop)

# Look at the info of the smaller dataset to see what datatypes to change
completion.info()

# %%
# By investigating the data using Data Wrangler, we can see that the columns 
# that could be reasonably categorical are level and control (have 2 and 3 
# distinct values respectively). 

# Make a list with these two columns and convert them to categorical variables
categorical = ['level', 'control']
completion[categorical] = completion[categorical].astype('category')
completion.dtypes

# index, unitid, chronname, site, and similar are all unique identifiers so we can drop 
# them. 
identifiers = ['index', 'unitid', 'chronname', 'site', 'similar']
completion = completion.drop(columns=identifiers)

# state, city, basic, and counted_pct all have too many values to be used as 
# a categorical variable and there aren't any dominating values so we'll make 
# a seperate data frame without them. Longitude and latitude are also unnecessary 
# for our investigation so we'll remove them from the new data frame as well. 
too_big = ['state', 'city', 'basic', 'counted_pct', 'long_x', 'lat_y']
completion_cleaned = completion.drop(columns=too_big)

# %% [markdown]
# ### Scaling the data using min max scaler 

# %% 
# make a list of all of the columns that are float or integer 
numeric_cols = list(completion_cleaned.select_dtypes('number'))
# convert those columns into the min max scale using MinMaxScaler()
completion_cleaned[numeric_cols] = MinMaxScaler().fit_transform(completion_cleaned[numeric_cols])
# view the data to ensure it was done right 
completion_cleaned.head()

# %% [markdown]
# ### One-hot encoding factor variables 

# %% 
# The two categorical variables we have are level and control. 
# We want to perform one-hot encoding on them to turn them into 
# numeric data. 

# To do this, first select all columns that are the category datatype
category_list = list(completion_cleaned.select_dtypes('category'))

# Use get_dummies method in Pandas to perform one-hot encoding 
completion_encoded = pd.get_dummies(completion_cleaned, columns=category_list)

# Check the info of the new dataframe to ensure it worked correctly 
completion_encoded.info()

# %% [markdown]
# ### Calculate the prevalence of our target variable 

# In this case, our target variable is numeric as it's a percentile, but 
# we can split it into colleges that are in the top quartile and those that 
# aren't. 

# %% 
# First visualize the distribution of the grad_100_percentile column 
print(completion_encoded.boxplot(column='grad_100_percentile', vert=False, grid=False))
      
# Also look at the summary statistics of the column to obtain what 
# number is the 75th percentile 
print(completion_encoded.grad_100_percentile.describe())
# The upper quartile is at 0.73, so we'll want to split the column there

# %%
# We can see from the boxplot and summary statistics that the data has a 
# min of 0, a max of 1, and an upper quartile of 0.73 

# Now we want to make a binary target variable, grad_100_percentile_f
# A value of 1 in this column indicates a signfigantly above average 
# college when it comes to on time graduation rate, and a 0 means everything 
# else. 
completion_encoded['grad_100_percentile_f'] = pd.cut(completion_encoded.grad_100_percentile, 
                                                    bins=[0, 0.73, 1],
                                                    labels=[0,1])

# verify the new column 
completion_encoded.info()
 
# %% 
# calculate the prevalence 
prevalence = (completion_encoded.grad_100_percentile_f.value_counts()[1] / len(completion_encoded.grad_100_percentile_f))

print(f'Prevalence: {prevalence:.2%}')

# %% [markdown]
# ### Partitioning the dataset into train, tune, and test 

# We can drop the grad_100_percentile, grad_100_value, grad_150_percentile, and grad_150_value
# columns because grad_100_percentile is our target variable and the other 3 are directly tied 
# to that, meaning that it wouldn't be useful information for a college to have to increase their 
# percentile nationally of students who graduate on time. 

# make a list of the columns we want to drop 
cols = ['grad_100_value', 'grad_100_percentile', 'grad_150_value', 'grad_150_percentile']

# Drop these columns 
completion_clean = completion_encoded.drop(cols, axis=1)

# Also want to drop any rows with NaN in grad_100_percentile_f 
completion_clean = completion_clean.dropna(subset=['grad_100_percentile_f'])
completion_clean.head()

# %%
# We want to split our dataset into a train (about 70% of the data), tune (about 15% of the data), 
# and test (about 15% of the data) set. 
# 70% of 3228 (the number of rows in the dataset) is 2,269.6. 
# Because we can't get .6 of a row and we want the tune and test sets be the same size, we'll make 
# the train set be 2,260 entries so that the tune and test sets have 484 entries each. 

# First split the training data from the rest, stratifying by the grad_100_percentile_f column so the
# proportions are preserved  
train, test = train_test_split(completion_clean, 
                               train_size = 2260, 
                               stratify=completion_clean.grad_100_percentile_f
                               )

# %%
# verify the split sizes 
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# %%
# then split the test set into tune and test 
tune, test = train_test_split(test, train_size=.5, stratify=test.grad_100_percentile_f)

# %%
