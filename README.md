# Predict Next-day rain in Austrailia
## About the data set
This dataset contains comprehensive weather measurements collected from 49 different meteorological centers. It includes key parameters such as temperature, evaporation, wind speed, wind direction, humidity, atmospheric pressure, rainfall, and an indicator for whether it rained that day (RainToday).
## Problem Statement
In this analysis, we aim to train a binary classification model using logistic regression to predict whether it will rain tomorrow in Australia. The model is developed based on various weather features, such as temperature, humidity, wind speed, and atmospheric pressure, to enhance predictive accuracy.
## Project Overview
This Project involves:
+ Loading and Exploring the data set
+ Performing exploratory data analysis (EDA)
+ Split data into Train, Validation, and Test Sets
+ Handle Missing Values
+ Scale Numerical Features
+ Encode categorical Features
+ Train Logistic regression model
+ Evaluate Model Performance
+ Evaluate model on Train, Validation and Test Sets

## Steps

1.Importing necessary libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

```
2. Load data
```
raw_df= pd.read_csv("weatherAUS.csv")
```
3. Initial exploration
```
raw_df.head()
raw_df.info()
raw_df.nunique()
```
4. Perform EDA
```
px.histogram(raw_df, x='Location', title='Location vs Rainy Days', color='RainToday')
px.histogram(raw_df, x='Temp3pm', title= 'Temporature at 3 pm vs Rain tommorow', color= 'RainTomorrow')
px.histogram(raw_df, x= 'RainToday', title= 'Rain Today vs Rain Tommorow', color= 'RainTomorrow' )
px.scatter(raw_df.sample(2000),
              title= 'Min temp vs max temp',
              x= 'MinTemp',
              y= 'MaxTemp',
              color='RainToday')
px.scatter(raw_df.sample(2000),
          title= 'Temp 3 pm vs Humidity 3 pm',
          x= 'Temp3pm',
          y= 'Humidity3pm',
          color= 'RainTomorrow')
px.scatter(raw_df.sample(2000),
              title= 'Min temp vs max temp',
              x= 'MinTemp',
              y= 'MaxTemp',
              color='RainTomorrow')

```
5. Split data into Train, Validation, and Test Sets
```
from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(raw_df, test_size= 0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

print('train_df.shape :', train_df.shape)
print ('train_val_df.shape :', train_val_df.shape)
print ('test_df.shape :', test_df.shape)

plt.title('No of rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year);

year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
train_val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

print('train_df.shape :', train_df.shape)
print ('train_val_df.shape :', train_val_df.shape)
print ('test_df.shape :', test_df.shape)
```
6. Define Input & Target Columns
```
input_cols = list(train_df.columns)[1:-1] 
target_col = 'RainTomorrow'
print(input_cols)

train_inputs = train_df[input_cols].copy()
train_target = train_df[target_col].copy()

val_inputs = train_val_df[input_cols].copy()
val_target = train_val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_target = test_df[target_col].copy()
```
7. Identify Numerical & Categorical features
