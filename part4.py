# %% [markdown]
# ## Part 4 - Functions to clean and partition data 

# %% 
# import pacakges 
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %% 
# read in the job dataset to test the functions 
url = ("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
job = pd.read_csv(url)

# %%
# function that removes columns that have less than n non-null values 
def drop_null_columns(df, n):
    # make a list of the columns that have less than n non-null values
    to_drop = df.columns[df.count() < n]
    # drop the columns in the list 
    df = df.drop(columns=to_drop)
    # return the new smaller dataframe 
    return df

# %% 
# check that the function works using the job data
job = drop_null_columns(job, 200)

# %%
# make a function that converts a list of columns in a data frame to categorial 
# variables 

def make_categorical(df, list=None): 
    # input the data frame and a list of columns you want to make categorical 
    # if you simply want to convert all object columns, you don't need to input
    # anything as that's the default value for list 
    if list is None: 
        list = df.select_dtypes('object').columns
    # turn the list columns of the data frame into categorical variables 
    df[list] = df[list].astype('category')
    # return the data frame 
    return df 

# %% 
# check on job data frame 
job = make_categorical(job)
job.dtypes

# %%
# make a function that drops an inputted list of columns 
def drop_cols(df, cols): 
    # function takes a data frame and a list of columns that you want to drop
    # drop the list of columns from the data frame 
    df = df.drop(columns=cols)
    # return the data frame 
    return df 

# %% 
# check with job df 
job_clean = drop_cols(job, ['gender'])
job_clean.head()


# %% 
# make a function that does min max scaling on all numeric columns 
def min_max_scaling(df, n_cols=None): 
    # function takes a data frame and optionally a list of columns to perform scaling on
    # if you want to scale all numeric columns, a list of them isn't necessary as the
    # function makes that automatically
    if n_cols is None: 
        n_cols = list(df.select_dtypes('number'))
    # function performs min max scaling on those columns 
    df[n_cols] = MinMaxScaler().fit_transform(df[n_cols])
    # return the data frame
    return df

# %% 
# test on jobs df 
job = min_max_scaling(job)
job.head()

# %% 
# make a function that performs one-hot encoding on all categorical variables or a set 
# list of columns
def one_hot_encoding(df, cols=None): 
    # To do this, first select all columns that are the category datatype
    if cols is None: 
        cols = list(df.select_dtypes('category'))
    # Use get_dummies method in Pandas to perform one-hot encoding 
    df_encoded = pd.get_dummies(df, columns=cols)
    # return the data frame 
    return df_encoded 

# %% 
# check on job df 
job_encoded = one_hot_encoding(job)
job_encoded.dtypes

# %% 
# make a function to make the target variable binary and find the prevalence
def find_prevalence(df, target_var):
    # function takes a data frame and the target varaible (a column in the df)
    # it uses .describe() on the column to find the upper quartile, min, and max to 
    # figure out where to cut the column
    upper_q = df[f'{target_var}'].describe()['75%']
    min = df[f'{target_var}'].describe()['min']
    max = df[f'{target_var}'].describe()['max']
    # make a new column in the data frame called target_var_f that cuts the target_var
    # column into a bin with values below the upper quartile, labeling them with 0 and 
    # a bin with values above the upper quartile, labeling them with 1
    df[f'{target_var}_f'] = pd.cut(df[f'{target_var}'], 
                                   bins=[min-0.01, upper_q, max+0.01], 
                                   labels=[0,1])
    prevalence = (df[f'{target_var}_f'].value_counts()[1] / len(df[f'{target_var}_f']))
    # print the prevalence
    print(f'Prevalence: {prevalence:.2f}')
    # return the data frame and prevalance 
    return df, prevalence 

# %% 
# test on job df 
job_new, prevalence = find_prevalence(job_encoded, 'degree_p')
job_new.head()

# %%
# make a function to drop any rows with null values in certian column(s)
def drop_null_row(df, cols): 
    # use dropna on the subset of the list of columns inputted to drop all rows
    # from the data frame that have a null value in any of the columns 
    df = df.dropna(subset=cols)
    return df 


# %%
# make a function to partition the data into train and test sets 
def train_tune_and_test(df, train_size, target_var_f): 
    # make the train and test sets by 
    train, test = train_test_split(df, 
                                   train_size=train_size, 
                                   stratify=df[f'{target_var_f}'])
    tune, test = train_test_split(test, 
                                  train_size=.5, 
                                  stratify=test[f'{target_var_f}'])
    return train, tune, test 

# %% 
# test on job df 
train, tune, test  = train_tune_and_test(job_new, 150, target_var_f='degree_p_f')