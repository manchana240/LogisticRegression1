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
```
numerical_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
```
8. Handle Missing Values
```
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'mean')

raw_df[numerical_cols].isna().sum()
train_inputs[numerical_cols].isna().sum()
imputer.fit(raw_df[numerical_cols])
list(imputer.statistics_)
train_inputs[numerical_cols] = imputer.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = imputer.transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = imputer.transform(test_inputs[numerical_cols])
```
9. Scale Numerical Features
```
raw_df[numerical_cols].describe()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(raw_df[numerical_cols])

train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])
```
10. Encode categorical Features
```
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoder.fit(raw_df[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols].fillna('unknown'))
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols].fillna('unknown'))
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols].fillna('unknown'))
pd.set_option('display.max_columns', None)
```
11. Save processed data to disk
```
!pip install pyarrow

train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')

pd.DataFrame(train_target).to_parquet('train_targets.parquet')
pd.DataFrame(val_target).to_parquet('val_targets.parquet')
pd.DataFrame(test_target).to_parquet('test_targets.parquet')

train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')
train_target = pd.read_parquet('train_targets.parquet')[target_col]
val_target = pd.read_parquet('val_targets.parquet')[target_col]
test_target = pd.read_parquet('test_targets.parquet')[target_col]
```
12. Train Logistic regression model
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'liblinear')
model.fit(train_inputs[numerical_cols + encoded_cols], train_target)
print([numerical_cols + encoded_cols])
print(model.coef_.tolist())
print(model.intercept_)
```
