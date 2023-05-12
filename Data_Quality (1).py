# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality
# MAGIC <br>**Clarity** >> Check if the metadata is understandable without the need for interpretation. Rename columns if they need.
# MAGIC <br>**Accuracy** >> check the data if it is coherent with the actual data
# MAGIC <br>**Completeness** >> check the empty and missing values
# MAGIC <br>**Validity** >> check the data if it conforms with the syntax of its definition
# MAGIC <br>**Uniqueness** >> check the duplicate records
# MAGIC <br>**Consistency** >> check the consistency between the associated data in multiple columns.
# MAGIC <br>**Timeliness** >> check the time lapse between the creation of the data and its availability appropriate.
# MAGIC <br>**Traceability** >> check if the source of the data is being reached, along with any transformations it may have gone through
# MAGIC <br>**Availability** >> Facilitate access to source data by the user

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

def column_details(regex, df):
  # We will focus on each column in detail
  # Uniqe Values, DTYPE, NUNIQUE, NULL_RATE
  global columns
  columns=[col for col in df.columns if re.search(regex, col)]

  print('Unique Values of the Features:\nfeature: DTYPE, NUNIQUE, NULL_RATE\n')
  for i in df[columns]:
      print(f'{i}: {df[i].dtype}, {df[i].nunique()}, %{round(df[i].isna().sum()/len(df[i])*100,2)}\n{pd.Series(df[i].unique()).sort_values().values}\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Dataset

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.pandas as ps
import re

import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")

%matplotlib inline

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

# COMMAND ----------

# MAGIC %fs ls dbfs:/user/hive/warehouse

# COMMAND ----------

from pyspark.sql import SparkSession

# create Spark session
spark = SparkSession \
    .builder \
    .appName('globalterrorism') \
    .getOrCreate()

# read table as Spark dataframe
df = spark.read.table("globalterrorism")
display(df)

# COMMAND ----------

# check dtypes of the columns
df.dtypes

# COMMAND ----------

# check the shape of the dataset
(df.count(), len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clarity
# MAGIC Check if the metadata is understandable without the need for interpretation. Rename columns if they need.

# COMMAND ----------

# Select core columns
core_columns=['iyear','imonth','iday','extended','country_txt','region_txt','city','success','suicide','attacktype1_txt','targtype1_txt','gname','weaptype1_txt','nkill']

# Rename the column names
df = df[core_columns] \
        .withColumnRenamed("iyear", "Year") \
        .withColumnRenamed("imonth", "Month") \
        .withColumnRenamed("iday", "Day") \
        .withColumnRenamed("extended", "Extended") \
        .withColumnRenamed("country_txt", "Country") \
        .withColumnRenamed("region_txt", "Region") \
        .withColumnRenamed("city", "City") \
        .withColumnRenamed("success", "Success") \
        .withColumnRenamed("suicide", "Suicide") \
        .withColumnRenamed("attacktype1_txt", "Attack_Type") \
        .withColumnRenamed("targtype1_txt", "Target_Type") \
        .withColumnRenamed("gname", "Attack_Group") \
        .withColumnRenamed("weaptype1_txt", "Weapon") \
        .withColumnRenamed("nkill", "Number_of_Killed")

# COMMAND ----------

# MAGIC %md
# MAGIC **Year** This field contains the year in which the incident occurred.
# MAGIC
# MAGIC **Month** This field contains the number of the month in which the incident occurred.
# MAGIC
# MAGIC **Day** This field contains the numeric day of the month on which the incident occurred.
# MAGIC
# MAGIC **Extended** 1 = "Yes" The duration of an incident extended more than 24 hours. 0 = "No" The duration of an incident extended less
# MAGIC
# MAGIC **Country** This field identifies the country or location where the incident occurred.
# MAGIC
# MAGIC **Region** This field identifies the region in which the incident occurred.
# MAGIC
# MAGIC **City** Name of the city, village, or town in which the incident occurred
# MAGIC
# MAGIC **Success** Success of a terrorist strike
# MAGIC
# MAGIC **Suicide** 1 = "Yes" The incident was a suicide attack. 0 = "No" There is no indication that the incident was a suicide
# MAGIC
# MAGIC **Attack_Type** The general method of attack and broad class of tactics used.
# MAGIC
# MAGIC **Target_Type** The general type of target/victim
# MAGIC
# MAGIC **Target_Sub_Type** The more specific target category
# MAGIC
# MAGIC **Attack_Group** The name of the group that carried out the attack
# MAGIC
# MAGIC **Weapon** General type of weapon used in the incident
# MAGIC
# MAGIC **Number_of_Killed** The number of total confirmed fatalities for the incident

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Accuracy
# MAGIC Check the data if it is coherent with the actual data

# COMMAND ----------

df.describe().display()

# COMMAND ----------

# 1. Are the values within the expected range?
df.describe().toPandas().set_index('summary').T

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:**
# MAGIC - Number_of_Killed has a negative value!
# MAGIC - Success and Suicide have values other than 0 or 1!
# MAGIC - Month and Day columns shouldn't have the value of 0!

# COMMAND ----------

# /*2) Accuracy should run on complete records only, if a client is missing some risk drivers then it is not going to help -- If Cmd_4 contains any data then this will be where the values are problematic, otherwise everything can be stored in Accuracy_2*/

from pyspark.sql.functions import col
globalterrorism_accuracy_problem_risk_drivers = df.filter((col('Number_of_Killed') < 0) | (col('Success') > 1) | (col('Suicide') > 1) | (col('Month') == 0) | (col('Day') == 0))
globalterrorism_accuracy_problem_risk_drivers.display()

# COMMAND ----------

(globalterrorism_accuracy_problem_risk_drivers.write
    .format("parquet")  # Specify the format to write as Parquet
    .mode("overwrite")  
    .saveAsTable("globalterrorism_accuracy_problem_risk_drivers")  # Save the DataFrame as a table with the specified name
)

# COMMAND ----------

df = df.filter((col('Number_of_Killed') >= 0) & (col('Success') <= 1) & (col('Suicide') <= 1) & (col('Month') > 0) & (col('Day') > 0))
df.display()

# COMMAND ----------

# Check the numerical columns if they are within expected range.
selected_cols = ['Year', 'Month', 'Day', 'Extended', 'Success', 'Suicide', 'Number_of_Killed']
summary_stats = df.select(selected_cols).describe()

# Show the summary statistics
summary_stats.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Completeness
# MAGIC Check the empty and missing values

# COMMAND ----------

# check null values
from pyspark.sql.functions import col,isnan, when, count

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

from pyspark.sql.functions import col, when

# Fill missing values in 'City' column with 'Unknown'
df = df.withColumn('City', when(col('City').isNull(), 'Unknown').otherwise(col('City')))

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validity
# MAGIC Check the data if it conforms with the syntax of its definition

# COMMAND ----------

# Check the categorical columns if they have the same written format in terms of upper/lower case.
selected_cols = ['Country', 'Region', 'City', 'Attack_Type']
column_details(regex='', df=df.toPandas()[selected_cols])

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation**
# MAGIC - Some of the city names are written in capital letters and some are in lowercase. All must be written in the same format.

# COMMAND ----------

# Capitalize the first letter of each word, lower the rest

from pyspark.sql.functions import col, initcap

selected_cols = ['Country', 'Region', 'City', 'Attack_Type']
for col_name in selected_cols:
    df = df.withColumn(col_name, initcap(col(col_name)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Uniqueness
# MAGIC Check the duplicate records

# COMMAND ----------

# Check for the duplications
df.toPandas().duplicated().sum()

# COMMAND ----------

# Drop duplicated rows.
df = spark.createDataFrame(df.toPandas().drop_duplicates())

# COMMAND ----------

# Check for the duplications
df.toPandas().duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consistency
# MAGIC Check the consistency between the associated data in multiple columns.

# COMMAND ----------

# Check the number of days of the months
from pyspark.sql.functions import max
df.groupby("Month").agg(max("Day").alias("max_day")).sort("Month").display()

# COMMAND ----------

# Check the unique values of columns, dtypes of columns and missing value rates of columns
column_details(regex='', df=df.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Timeliness
# MAGIC Check the time lapse between the creation of the data and its availability appropriate.

# COMMAND ----------

# Generate Datetime and Month_of_Year columns.
from pyspark.sql.functions import concat, lit, to_date, to_timestamp

df = df.withColumn("Datetime", to_date(concat("Year", lit("-"), "Month", lit("-"), "Day")))
df = df.withColumn('Month_of_Year', F.date_format(F.to_date('Datetime', 'yyyy-MM-dd'), 'yyyyMM'))

# COMMAND ----------

# Check the time series if it has a gap.

plt.figure(figsize=(30,5))
ax = df.toPandas()['Datetime'].value_counts().sort_index().plot()
plt.title('Number Of Terrorist Activities by Datatime');

# COMMAND ----------

# Check the data for the year of 1993
df_temp = df.toPandas()
df_temp[(df_temp['Year']==1993)].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation**
# MAGIC - There is no data for 1993.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Traceability
# MAGIC Check if the source of the data is being reached, along with any transformations it may have gone through

# COMMAND ----------

# Save the updated version of the dataframe to the delta lake
(df.write
    .format("parquet")  # Specify the format to write as Parquet
    .mode("overwrite")  
    .saveAsTable("globalterrorism_live")  # Save the DataFrame as a table with the specified name
)

# COMMAND ----------

# Check if the source of the data is being reached 
df = spark.read.table("globalterrorism_live")
display(df)

# COMMAND ----------

# check the shape of the dataset
(df.count(), len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Availability
# MAGIC Facilitate access to source data by the user

# COMMAND ----------

# Connect your Azure Databricks workspace to an Azure Data Lake Storage (ADLS) account where the source data resides. 
# This integration allows users to access the data stored in ADLS directly from Databricks.