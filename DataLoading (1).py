# Databricks notebook source
# MAGIC %md
# MAGIC # Terrorism Around The World
# MAGIC [Kaggle Link](https://www.kaggle.com/code/ash316/terrorism-around-the-world/notebook)

# COMMAND ----------

'''importing the dataset'''
import os

database_FILE_path = '/dbfs/FileStore/Global_Terrorism/globalterrorismdb_0718dist.csv'
database_SPARK_path = 'dbfs:/FileStore/Global_Terrorism/globalterrorismdb_0718dist.csv'

if not os.path.exists(database_FILE_path):
    print(f'{database_FILE_path}: does not exist!')

print(os.path.isfile(database_FILE_path))

# COMMAND ----------

# Read a CSV file into a DataFrame, with headers and schema inference enabled

raw_df = (spark.read
    .option('header', 'true')  # Specify that the CSV file has a header row
    .csv(database_SPARK_path, inferSchema=True)  # Read the CSV file at the specified path and infer the schema
)


# COMMAND ----------

''' Persisting as Table '''

# Define the name of the permanent table
permanent_tab = 'globalterrorism_complete'

# Write the DataFrame to the permanent table in Parquet format
(raw_df.write
    .format("parquet")  # Specify the format to write as Parquet
    .saveAsTable(permanent_tab)  # Save the DataFrame as a table with the specified name
)


# COMMAND ----------

# Print the schema of the DataFrame
raw_df.printSchema()  # This line prints the schema of the DataFrame to the console

# COMMAND ----------

'''Loading the dataset'''''

# Load the CSV file into a DataFrame using Spark's CSV reader
# Set the "mode" option to "PERMISSIVE" to handle corrupt records gracefully
raw_df = (spark.read
   .format("csv")
   .option("mode", "PERMISSIVE")
   .load(database_SPARK_path))


# COMMAND ----------

import math
from dateutil.parser import parse
from pyspark.sql.types import *


def isNumericEntry(x:str):
    '''
    Parameters:
        x (str): Row value of CSV column in dataset.
        
    Use:
        Verifies whether the string is a value that can be converted into a number.
        
    Returns: Boolean
    '''
    try:
        r = float(x)
    except ValueError:
        return False
    
    return True

def isDateEntry(x, fuzzy = False):
    '''
    Parameters:
        x (str): Row value of CSV column in dataset.
        
    Use:
        Verifies whether the string can be parsed into a date.
        
    Returns: Boolean
    '''
    
    try:
        parse(x, fuzzy = fuzzy)
        return True
    except:
        return False
    
def containsDecimals(x:str):
    '''
    Parameters:
        x (str): Row value of CSV column in dataset.
        
    Use:
        Verifies whether the number in the string contains a decimal.
        
    Returns: Boolean
    '''
    
    if '.' in x or ',' in x:
        return True
    return False

def isBoolean(x:str):
    '''
    Parameters:
        x (str): Row value of CSV column in dataset.
        
    Use:
        Verifies whether the string is a value that can be converted into a Boolean.
        
    Returns: Boolean
    '''
    
    if x == 'True' or x == 'False':
        return True
    return False

def resolveColumnType(x:str):
    
    '''
    Parameters:
        x (str): Row value of CSV column in dataset.
        
    Use:
        Function is meant to automatically detect which of the generic values the data should fall under.
        These types can either be pyspark.sql.types: StringType (Default), BooleanType, DateType, IntegerType, FloatType.\
        
    Returns: StringType (Default) or BooleanType or DateType or IntegerType or FloatType
    '''
    
    #Check if the entry is numeric (Can be converted to a value)
    if isNumericEntry(x):
        if containsDecimals(x):
            return FloatType()
        else:
            return IntegerType()
        
    #Check if the entry is a date
    if isDateEntry(x):
        return DateType()
    
    #Check if the entry is a boolean variable
    if isBoolean(x):
        return BooleanType()

    #Default return type
    return StringType()

def recastColumnTypes(raw_df):
    
    '''
    
    Parameters:
        raw_df (spark.dataframe.Dataframe): Spark Dataframe for which we want to automatically recast column types
        
    Use:
        This is an -EXTREMELY- generic way of converting the data to the appropiate type during loading.
        Use it if you only have CSV data saved as strings.
        
    Returns: spark.dataframe.Dataframe
    '''
    # Extract the column names from the first row of the DataFrame
    column_names = raw_df.collect()[0]

    # Extract the data rows from the DataFrame
    row_list = raw_df.collect()[1:]

    # Create a new DataFrame with the extracted column names and rows
    clean_dataframe = spark.createDataFrame(data = row_list, schema = column_names)

    # Iterate over each column name
    for name in column_names:
        # Iterate over each entry in the column and find a non-null value to determine the appropriate type
        for entry in clean_dataframe.select(name).collect():
            if entry[0] is not None:
                # Recast the column to the resolved type
                clean_dataframe = clean_dataframe.withColumn(name, col(name).cast(resolveColumnType(entry[0])))
                break

    return clean_dataframe
    

# COMMAND ----------

'''Cleaning the dataset'''

from pyspark.sql.functions import col

clean_dataframe = recastColumnTypes(raw_df)

# Print the schema and verify whether the fields are correct.
clean_dataframe.printSchema()

# COMMAND ----------

''' Persisting as Table '''

# Specify the name for the permanent table
permanent_tab = 'globalterrorism'

# Write the clean DataFrame to a Parquet file and save it as a table
(
    clean_dataframe.write  # Write operation on the DataFrame
    .format("parquet")  # Specify the file format as Parquet
    .saveAsTable(permanent_tab)  # Save the DataFrame as a table with the given name
)


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify that the table exists.
# MAGIC DESCRIBE TABLE globalterrorism

# COMMAND ----------

