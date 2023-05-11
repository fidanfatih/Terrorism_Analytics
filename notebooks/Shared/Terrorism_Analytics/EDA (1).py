# Databricks notebook source
# MAGIC %md
# MAGIC ## Terrorism Around The World (EDA)
# MAGIC [Kaggle Link](https://www.kaggle.com/code/ash316/terrorism-around-the-world/notebook)
# MAGIC
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
# MAGIC ### Functions

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
# MAGIC ### Read dataset

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

from pyspark.sql import SparkSession

# create Spark session
spark = SparkSession \
    .builder \
    .appName('globalterrorism') \
    .getOrCreate()

# read table as Spark dataframe
df = spark.read.table("globalterrorism_live")
display(df)

# COMMAND ----------

# check the shape of the dataset
(df.count(), len(df.columns))

# COMMAND ----------

column_details(regex='', df=df.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate Analysis

# COMMAND ----------

pandas_df=df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist Activities by Year

# COMMAND ----------

plt.figure(figsize=(30,8))
ax=sns.countplot(x=pandas_df['Year'])
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('Number Of Terrorist Activities By Year');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** The terrorist attack peaked from 2012 with year 2014 as the highest number of terrorist attacks followed by 2015 and 2016

# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist Activities by Cities

# COMMAND ----------

plt.figure(figsize=(25,5))
ax=sns.countplot(x=pandas_df['City'],order=pandas_df['City'].value_counts().index[1:11],palette='plasma')
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('Number Of Terrorist Activities in Top 10 Cities');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Most of number of terrorist activities was observed in the city of Baghdad and Karachi

# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist Activities by Countries

# COMMAND ----------

plt.figure(figsize=(25,5))
ax=sns.countplot(x = pandas_df['Country'],order=pandas_df['Country'].value_counts().index[:10],palette='plasma')
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('Number Of Terrorist Activities in Top 10 Countries');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Iraq,Pakistan and Afghanistan were the major countries for the terrorist activities.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist Activities by Region

# COMMAND ----------

plt.figure(figsize=(25,5))
ax=sns.countplot(x = pandas_df['Region'],order=pandas_df['Region'].value_counts().index[:10],palette='plasma')
ax.bar_label(ax.containers[0],label_type='edge')
ax.set_xticklabels(plt.xticks()[1], size = 10)
plt.title('Number Of Terrorist Activities in Top 10 Region');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** The terrorist attacks in the region Middle East,North Africa and South Asia are more frequent.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Top 10 Terrorist Groups

# COMMAND ----------

plt.figure(figsize=(25,8))
ax=sns.countplot(x=pandas_df['Attack_Group'],order=pandas_df['Attack_Group'].value_counts().index[1:11],palette='plasma')
ax.bar_label(ax.containers[0],label_type='edge')
ax.set_xticklabels(plt.xticks(rotation=90)[1], size = 10)
plt.title('Top 10 Terrorist Groups');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** The terrorist attacks were mostly caused by attack groups known as Taliban and Islamic State of Iraq and the Levant (ISIL)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Popular Attack Types

# COMMAND ----------

ax=pd.DataFrame(pandas_df['Attack_Type'].value_counts(normalize=True)*100).plot.bar(figsize=(30,8))
ax.bar_label(ax.containers[0],label_type='edge', fmt='%1.1f%%')
ax.set_xticklabels(plt.xticks(rotation=0)[1], size = 10)
plt.title('Popular Attack Types');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Bombing/Explosion,Armed Assualt and Assasinations makes up for more than 85% of attack methods.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Popular Targets

# COMMAND ----------

ax=pd.DataFrame(pandas_df['Target_Type'].value_counts(normalize=True)*100).plot.bar(figsize=(30,8))
ax.bar_label(ax.containers[0],label_type='edge', fmt='%1.1f%%')
ax.set_xticklabels(plt.xticks(rotation=80)[1], size = 10)
plt.title('Popular Targets');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Most common types of targets includes Private citizen and Property,Military, Police and Government with more than 65% of total targets.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Popular Weapons

# COMMAND ----------

ax=pd.DataFrame(pandas_df['Weapon'].value_counts(normalize=True)*100)[:5].plot.bar(figsize=(30,8))
ax.bar_label(ax.containers[0],label_type='edge', fmt='%1.1f%%')
ax.set_xticklabels(plt.xticks(rotation=80)[1], size = 10)
plt.title('Popular Weapons');

# COMMAND ----------

# MAGIC %md
# MAGIC **Observation:** Explosives and Firearms makes for more than 85% of the weapons used in terrorist activites.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Success Rate in Terrorist Attacks

# COMMAND ----------

plt.figure(figsize=(10,6))
ax = sns.countplot(x='Success', data= pandas_df)
labels(ax, pandas_df)
plt.title('Success Rate in Terrorist Attacks');

# COMMAND ----------

# MAGIC %md
# MAGIC #### Suicide Terrorist Operations

# COMMAND ----------

plt.figure(figsize=(10,6))
ax = sns.countplot(x='Suicide', data= pandas_df)
labels(ax, pandas_df)
plt.title('Suicide Terrorist Operation');

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bivariate Analysis

# COMMAND ----------

pandas_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Region Vs. Year

# COMMAND ----------

df_temp = pandas_df.groupby(['Year','Region'])["Region"].count().to_frame().rename(columns={'Region': 'Count'}).reset_index()

plt.figure(figsize=(30,10))
sns.lineplot(x="Year", y="Count", hue ='Region', data = df_temp)
plt.title('Region Vs. Year', fontsize = 20)
plt.xticks(rotation=0);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Region Vs. Weapon

# COMMAND ----------

df_temp = pandas_df.groupby(['Region','Weapon'])["Region"].count().to_frame().rename(columns={'Region': 'Count'}).reset_index()
df_temp = pd.pivot_table(df_temp, values=['Count'], index=['Region'], columns=['Weapon'],aggfunc = np.sum).fillna(0).reindex(pandas_df.groupby("Region")["Weapon"].count().sort_values().index)
df_temp.plot(kind='barh', stacked=True, figsize=(30,15))
plt.title('Region Vs. Weapon', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Attack_Group Vs. Weapon

# COMMAND ----------

pandas_df=df.toPandas()
top_10_attack_group = pandas_df["Attack_Group"].value_counts()[1:11].index
pandas_df.loc[:,'Attack_Group'] = pandas_df.loc[:,'Attack_Group'].apply(lambda x:x if x in top_10_attack_group else "Other")

df_temp = pandas_df.groupby(['Attack_Group','Weapon'])["Attack_Group"].count().to_frame().rename(columns={'Attack_Group': 'Count'}).reset_index()
df_temp = pd.pivot_table(df_temp, values=['Count'], index=['Attack_Group'], columns=['Weapon'],aggfunc = np.sum).fillna(0).reindex(pandas_df.groupby("Attack_Group")["Weapon"].count().sort_values().index)[:-1]
df_temp.plot(kind='barh', stacked=True, figsize=(30,15))
plt.title('Attack_Group Vs. Weapon', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Attack_Type Vs. Number_of_Killed

# COMMAND ----------

pandas_df=df.toPandas()
ax = pandas_df.groupby('Attack_Type')['Number_of_Killed'].sum().drop('Unknown').sort_values(ascending=False).plot.bar(figsize=(30,5),color='r')
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('People killed per Attack type', fontsize = 20);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Attack_Group Vs. Number_of_Killed

# COMMAND ----------

ax = pandas_df.groupby('Attack_Group')['Number_of_Killed'].sum().drop('Unknown').sort_values(ascending=False)[:20].plot.bar(figsize=(30,5),color='r')
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('People killed per Attack_Group', fontsize = 20);

# COMMAND ----------

import plotly.express as px

df_temp = pandas_df.groupby('Attack_Group')['Number_of_Killed'].sum().drop('Unknown').reset_index().sort_values('Number_of_Killed', ascending=False)[:20]
fig = px.treemap(df_temp, path=['Attack_Group'], values='Number_of_Killed')
fig.update_traces(textinfo='label+value')  # Add labels and values to the treemap
fig.update_layout(title='People killed per Attack_Group',font=dict(size=20))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Region Vs. Number_of_Killed

# COMMAND ----------

ax = pandas_df.groupby('Region')['Number_of_Killed'].sum().sort_values(ascending=False)[:20].plot.bar(figsize=(30,5),color='r')
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('People killed per Region', fontsize = 20);

# COMMAND ----------

import plotly.express as px

df_temp = pandas_df.groupby('Region')['Number_of_Killed'].sum().reset_index().sort_values('Number_of_Killed', ascending=False)[:20]
fig = px.treemap(df_temp, path=['Region'], values='Number_of_Killed')
fig.update_traces(textinfo='label+value')  # Add labels and values to the treemap
fig.update_layout(title='People killed per Region',font=dict(size=20))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Country Vs. Number_of_Killed

# COMMAND ----------

ax = pandas_df.groupby('Country')['Number_of_Killed'].sum().sort_values(ascending=False)[:20].plot.bar(figsize=(30,8),color='r')
ax.bar_label(ax.containers[0],label_type='edge')
plt.title('People killed per Country', fontsize = 20);

# COMMAND ----------

import plotly.express as px

df_temp = pandas_df.groupby('Country')['Number_of_Killed'].sum().reset_index().sort_values('Number_of_Killed', ascending=False)[:20]
fig = px.treemap(df_temp, path=['Country'], values='Number_of_Killed')
fig.update_traces(textinfo='label+value')  # Add labels and values to the treemap
fig.update_layout(title='People killed per Country',font=dict(size=20))
fig.show()


# COMMAND ----------

from pyspark.sql import functions as F

print('Total Number of lives lost in terrorist attacks:')
df.select(F.sum('Number_of_Killed')).first()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multivariate Analysis

# COMMAND ----------

pandas_df=df.toPandas()
sns.heatmap(pandas_df.corr(),annot=True)
plt.rcParams['figure.figsize']=(25,12);

# COMMAND ----------

top_5_attack_group = pandas_df["Attack_Group"].value_counts()[:7].index
pandas_df.loc[:,'Attack_Group'] = pandas_df.loc[:,'Attack_Group'].apply(lambda x:x if x in top_5_attack_group else "Other")

sns.pairplot(pandas_df[["Year","Number_of_Killed","Attack_Group"]], hue="Attack_Group", height=6)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Maps

# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist_Attacts by Country

# COMMAND ----------

import plotly.express as px
import pandas as pd
pandas_df=df.toPandas()
df_temp = pandas_df.groupby('Country')['Country'].count().to_frame().rename(columns={"Country": "Number_of_Terrorist_Attacts"}).reset_index()

# Create the choropleth map using Plotly Express
fig = px.choropleth(df_temp, locations='Country', locationmode='country names', color='Number_of_Terrorist_Attacts',
                    title='Terrorist_Attacts by Country', color_continuous_scale='Reds')

# Set the size of the map
fig.update_layout(
    autosize=False,
    width=1500,
    height=1000
)

# Show the plot
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist_Attacts by Coordinates

# COMMAND ----------

import plotly.graph_objects as go
import pandas as pd

pandas_df = spark.read.table("globalterrorism").select(col("latitude"), col("longitude"),col("gname")).toPandas()

# Create the Plotly Scattermapbox plot
fig = go.Figure(data=go.Scattermapbox(
    lat=pandas_df['latitude'],
    lon=pandas_df['longitude'],
    mode='markers',
    marker=dict(
        size=3,
        color='red',
        opacity=0.5
    )
))

# Set the mapbox layout
fig.update_layout(
    mapbox=dict(
        style='carto-positron',
        center=dict(lat=20, lon=0),
        zoom=1
    ),
    width=1500,
    height=1000,
    title='Terrorist_Attacts by Coordinates'
)

# Show the plot
fig.show()



# COMMAND ----------

# MAGIC %md
# MAGIC #### Terrorist_Attacts by Attack_Group

# COMMAND ----------

import plotly.express as px
import pandas as pd

pandas_df = spark.read.table("globalterrorism").select(col("latitude"), col("longitude"),col("gname")).toPandas().dropna().rename(columns={"gname": "Attack_Group"})
top_20_attack_group = pandas_df["Attack_Group"].value_counts()[:20].index
pandas_df.loc[:,'Attack_Group'] = pandas_df.loc[:,'Attack_Group'].apply(lambda x:x if x in top_20_attack_group else "Other")

# Plot the map
fig = px.scatter_mapbox(pandas_df, lat='latitude', lon='longitude', hover_name='Attack_Group', color='Attack_Group',
                        mapbox_style="carto-positron", zoom=1,
                        opacity=0.5)

# Set the figure size
fig.update_layout(width=2000, height=1000,title='Terrorist_Attacts by Attack_Group')

# Show the map
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary
# MAGIC - The terrorist attack peaked from 2012 with year 2014 as the highest number of terrorist attacks followed by 2015 and 2016
# MAGIC - Most of number of terrorist activities was observed in the city of Baghdad and Karachi
# MAGIC - Iraq,Pakistan and Afghanistan were the major countries for the terrorist activities.
# MAGIC - The terrorist attacks in the region Middle East,North Africa and South Asia are more frequent.
# MAGIC - The terrorist attacks were mostly caused by attack groups known as Taliban and Islamic State of Iraq and the Levant (ISIL)
# MAGIC - Bombing/Explosion,Armed Assualt and Assasinations makes up for more than 85% of attack methods.
# MAGIC - Most common types of targets includes Private citizen and Property,Military, Police and Government with more than 65% of total targets.
# MAGIC - Explosives and Firearms makes for more than 85% of the weapons used in terrorist activites.