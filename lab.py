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

# %%
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
# verify the prevalence of the train, tune, and test sets
prevalence_train = (train.grad_100_percentile_f.value_counts()[1] / len(train.grad_100_percentile_f))
prevalence_tune = (tune.grad_100_percentile_f.value_counts()[1] / len(tune.grad_100_percentile_f))
prevalence_test = (test.grad_100_percentile_f.value_counts()[1] / len(test.grad_100_percentile_f))
print(f'Prevalence Training set: {prevalence_train:.2f}')
print(f'Prevalence Tuning set: {prevalence_tune:.2f}')
print(f'Prevalence Testing set: {prevalence_test:.2f}')

# %% [markdown]
# ### Job Placement Dataset
# 
# An independent business metric for this problem would be if a school's
# average student salary after graduation increased after using this data. 


# %% 
# Job Placement Dataset 
# Read in the data 
url = ("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
job = pd.read_csv(url)

# Check for null values and data types 
job.info()


# %%
# Change rows that have a null value for salary to 0 because salary is only 
# missing if the student wasn't placed in a job.

# Use fillna to make all null values in the salary column 0. 
job['salary'] = job['salary'].fillna(0)
# verify that this worked 
job.info()


# %%
# By investigating the data using Data Wrangler, we can see that all string 
# columns only have 2 or 3 unique values, meaning they can be converted to 
# categorical with no other changes needed to the data

# Make a list of all string columns and convert them to categorical variables
category = list(job.select_dtypes('str'))
job[category] = job[category].astype('category')
# check the new data types 
job.dtypes

# %%
# sl_no is a unique identifier so we can drop it. We can also drop the status 
# column because it's now encoded in the salary column, since a salary of 0 
# indicates that a student wasn't placed. 

# make a list of columns to drop
to_drop = ['sl_no', 'status']
job = job.drop(columns=to_drop)

# %% [markdown]
# ### Scaling the data using min max scaler 

# %% 
# make a list of all of the columns that are float or integer 
numeric_cols = list(job.select_dtypes('number'))
# convert those columns into the min max scale using MinMaxScaler()
job[numeric_cols] = MinMaxScaler().fit_transform(job[numeric_cols])
# view the data to ensure it was done right 
job.head()

# %% [markdown]
# ### One-hot encoding factor variables 

# %% 
# The two categorical variables we have are level and control. 
# We want to perform one-hot encoding on them to turn them into 
# numeric data. 

# To do this, first select all columns that are the category datatype
category_list = list(job.select_dtypes('category'))

# Use get_dummies method in Pandas to perform one-hot encoding 
job_encoded = pd.get_dummies(job, columns=category_list)

# Check the info of the new dataframe to ensure it worked correctly 
job_encoded.info()

# %% [markdown]
# ### Calculate the prevalence of our target variable 

# In this case, our target variable is numeric as it's a salary, but 
# we can split it into high and low salaries 

# %% 
# First visualize the distribution of the salary column 
print(job_encoded.boxplot(column='salary', vert=False, grid=False))
      
# Also look at the summary statistics of the column to obtain what 
# number is the upper quartile
print(job_encoded.salary.describe())
# The upper quartile is at 0.135, so we'll want to split the column there

# %%
# We can see from the boxplot and summary statistics that the data has a 
# min of 0, a max of 1, and an upper quartile of 0.3 

# Now we want to make a binary target variable, salary_f
# A value of 1 in this column indicates a signfigantly above average 
# salary after graduation, and a 0 means everything else. 
job_encoded['salary_f'] = pd.cut(job_encoded.salary, 
                                                    bins=[-0.01, 0.300532, 1.01],
                                                    labels=[0,1])

# verify the new column 
job_encoded.info()
 
# %% 
# calculate the prevalence 
prevalence = (job_encoded.salary_f.value_counts()[1] / len(job_encoded.salary_f))

print(f'Prevalence: {prevalence:.2%}')

# %% [markdown]
# ### Partitioning the dataset into train, tune, and test 

# %% 
# also drop salary column since this is our target variable 
target = ['salary']
job_clean = job_encoded.drop(columns=target)

# %%
# We want to split our dataset into a train (about 70% of the data), tune (about 15% of the data), 
# and test (about 15% of the data) set. 
# 70% of 215 (the number of rows in the dataset) is 150.5. 
# Because we can't get .5 of a row and we want the tune and test sets be the same size, we'll make 
# the train set be 151 entries so that the tune and test sets have 32 entries each. 

# First split the training data from the rest, stratifying by the salary_f column so the
# proportions are preserved  
train, test = train_test_split(job_clean, 
                               train_size = 151, 
                               stratify=job_clean.salary_f
                               )

# %%
# verify the split sizes 
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# %%
# then split the test set into tune and test 
tune, test = train_test_split(test, train_size=.5, stratify=test.salary_f)

# %%
# verify the prevalence of the train, tune, and test sets
prevalence_train = (train.salary_f.value_counts()[1] / len(train.salary_f))
prevalence_tune = (tune.salary_f.value_counts()[1] / len(tune.salary_f))
prevalence_test = (test.salary_f.value_counts()[1] / len(test.salary_f))
print(f'Prevalence Training set: {prevalence_train:.2f}')
print(f'Prevalence Tuning set: {prevalence_tune:.2f}')
print(f'Prevalence Testing set: {prevalence_test:.2f}')

# %% [markdown]
# ## Part 3 
# 
# For the college completion data set, my instincts tell me that it should do 
# decently well answering my question. There's a lot of data points and features 
# that could be useful, so I think that there should be enough data for the model 
# to work with. One thing I'm not sure how to deal with in the model is that many 
# of the features have a column for both the value and the percentile for a given 
# thing (such as endowment value and percentile). I think that this could impact 
# the model because it means that those features are essentially being counted 
# twice, which could be problematic. I'm not sure, however, which would be better 
# to keep because the percentile shows how the college's performance in that category
# compares to colleges that might not be in the dataset, but the value may be more 
# informative with differences in say the endowment between colleges. I think that 
# the included features should allow it to answer my question pretty well because they 
# cover a lot of the data outside of academics that relate to colleges. The model 
# being able to predict if a college will have a high normal-time graduation rate 
# could give colleges an idea of what they can change and how that would impact the 
# amount of students that graduate on time. There are also fixed factors that a college
# can't change, like if it's a HBCU or if it's a 2-year or 4-year college, but these are 
# still important to include in the model because they impact the graduation rates. 
#
# I think that the job dataset might do not as well answering my question. It only has 
# a little over 200 data points, which means that any anomalies in the data will be 
# more impactful on the model. Also, another potential issue is that a few of the 
# columns have to do with the percentages of how the student scored at various points 
# in their schooling. This could lead to some variablility in the scores depending on 
# how the tests are administered and graded. If the students came from different schools
# and had different graders, there's a chance that some tests were graded differently 
# from others, which would change the percentage that the student scored. It does seem 
# like the data included would be very indicative of how a student's salary was affected
# by their performance in school, but it also doesn't include other factors that could 
# impact their salary, like if they had connections in the company that they ended up
# working for. Overall, I think that the data can help make a model to predict salary, but
# it'll be important to remember that there are outside factors that could also impact 
# salary after graduation. 
