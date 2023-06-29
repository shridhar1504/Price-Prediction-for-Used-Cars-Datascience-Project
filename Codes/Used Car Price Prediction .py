#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction 

# ***
# _**Importing the required libraries & packages**_
# 

# In[1]:


import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle 
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\Desktop\\DS Project')
df = pd.read_csv('dataset.csv')


# ## Data Cleaning

# _**Viewing the dataset for identifying the unwanted columns**_

# In[3]:


df.head()


# _**Dropping the `Unnamed: 0`, `Location` and `New_Price` column since it is not much important for the prediction**_

# In[4]:


df.drop(['Unnamed: 0','Location','New_Price'],axis = 1,inplace = True)


# _**Checking for the null values in the dataset**_

# In[5]:


df.isna().sum()


# _**There are several cars in the dataset, some of them with a count higher than 1. Sometimes the resale value of a car also depends on manufacturer of the car.So, here extracting the manufacturer name from this column and adding it to the dataset**_

# In[6]:


Manfacturer = df['Name'].str.split(" ",expand = True)
df['Manfacturer'] = Manfacturer[0]


# _**The `Year` column from the dataset has no significance on its own so calculating the years of cars used till now and adding it to the dataset.**_

# In[7]:


curr_time = datetime.datetime.now()
df['Years Used'] = df['Year'].apply(lambda x : curr_time.year - x)


# _**Dropping the `Name` and `Year` column since the needed data is extracted from it and added to the dataset in seperate columns as `Manfacturer` and `Years Used`**_

# In[8]:


df.drop(['Name','Year'],axis = 1,inplace = True)


# _**The `Mileage` column defines the mileage of the car which affects the price of the car during the sales.So that extracting the numerical values from the `Mileage` column and since it has missing values, filling out the missing values with the mean & modifying the column with the values**_

# In[9]:


Mileage = df['Mileage'].str.split(" ",expand = True)
df['Mileage'] = pd.to_numeric(Mileage[0],errors = 'coerce')
df['Mileage'].fillna(df['Mileage'].astype('float').mean(),inplace = True)


# _**The `Engine` column has the CC of the car which affects the price of the car during the sales.So that removing CC and  extracting the numerical values from the `Engine` column and since it has missing values, filling out the missing values with the mean & modifying the column with the values**_

# In[10]:


Engine = df['Engine'].str.split(" ",expand = True)
df['Engine'] = pd.to_numeric(Engine[0],errors = 'coerce')
df['Engine'].fillna(df['Engine'].astype('float').mean(),inplace = True)


# _**The `Power` column has the bhp of the car which affects the price of the car during the sales.So that removing bhp and  extracting the numerical values from the `Power` column and since it has missing values, filling out the missing values with the mean & modifying the column with the values**_

# In[11]:


Power = df['Power'].str.split(" ",expand = True)
df['Power'] = pd.to_numeric(Power[0],errors = 'coerce')
df['Power'].fillna(df['Power'].astype('float').mean(),inplace = True)


# _**Since the `Seats` column has some missing values, filling out the missing values with the mean values of the same column**_

# In[12]:


df['Seats'].fillna(df['Seats'].astype('float').mean(),inplace = True)


# _**After filling out the missing value with the appropriate values, checking for the null values in the dataset**_

# In[13]:


df.isna().sum()


# ## Data Visualization

# _**Plotting the Bar Graph with count of cars based on the `Manfacturer` and confirm that there are no null values and identify all unique values from the `Manfacturer` and saving the PNG File**_

# In[14]:


plt.rcParams['figure.figsize'] = 20,10
Cars = df['Manfacturer'].value_counts()
plot = sns.barplot(x = Cars.index,y = Cars.values,data = df)
plt.xticks(rotation = 90)
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Count of Car based on Manfacturer')
plt.xlabel('Manfacturer')
plt.ylabel('Count of Cars')
plt.savefig('Count of Cars.png')
plt.show()


# _**Getting the Correlation Values from all the numeric columns from the dataset using Seaborn Heatmap & saving the PNG File**_

# In[15]:


sns.heatmap(df.corr(),cmap = sns.cubehelix_palette(as_cmap = True),annot = True, cbar = True,square = True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# _**Assigning the dependent and independent variable**_

# In[16]:


x = df.drop(['Price'],axis = 1)
y = df['Price']


# ## Data Preprocessing

# _**Splitting the dependent variable & independent variable into training and test dataset using train test split**_

# In[17]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 45)


# _**Creating the dummy columns for all the categorical columns such as `Manfacturer`, `Fuel_Type`,
# `Transmission`, `Owner_Type` in the training independent dataset and getting the dimensions of the training independent dataset for cross-check**_

# In[18]:


x_train = pd.get_dummies(x_train,columns = ['Manfacturer', 'Fuel_Type', 'Transmission', 'Owner_Type'],drop_first = True)
print(x_train.shape)


# _**Similarly, creating the dummy columns for all the categorical columns such as `Manfacturer`, `Fuel_Type`,
# `Transmission`, `Owner_Type` in the test independent dataset and getting the dimensions of the test independent dataset for cross-check**_

# In[19]:


x_test = pd.get_dummies(x_test,columns = ['Manfacturer', 'Fuel_Type', 'Transmission', 'Owner_Type'],drop_first = True)
print(x_test.shape)


# _**By the dimensional checking its so evident that the dummy column creation is different in training and test independent data. So that filling in all the missing columns of test independent data with 0**_

# In[20]:


miss_col = set(x_train.columns) - set(x_test.columns)
for col in miss_col:
    x_test[col] = 0
x_test = x_test[x_train.columns]


# _**Now after processing, the dimensions of the column of both training and test independent data are same**_

# In[21]:


print(x_train.shape)
print(x_test.shape)


# _**Standardizing the independent training variable and independent test variable of the dataset**_

# In[22]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# ## Model Fitting

# _**Defining the Function for the ML algorithms using GridSearchCV Algorithm and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format, Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset**_

# In[23]:


def FitModel(x,y,algo_name,algorithm,GridSearchParams,cv):
    np.random.seed(10)
    grid = GridSearchCV(estimator = algorithm, param_grid = GridSearchParams, cv = cv,
                       scoring = 'r2', verbose = 0,n_jobs = -1)
    grid_result = grid.fit(x_train,y_train)
    pred = grid_result.predict(x_test)
    best_params = grid_result.best_params_
    pickle.dump(grid_result,open(algo_name,'wb'))
    print('Algorithm Name : ',algo_name,'\n')
    print('Best Params : ',best_params,'\n')
    print('Percentage of R2 Score : {} %'.format(100 * r2_score(y_test,pred)),'\n')
    print('Mean Absolute Error : ',mean_absolute_error(y_test,pred),'\n')
    print('Root Mean Squared Error : ',sqrt(mean_squared_error(y_test,pred)),'\n')


# _**Running the function with empty parameters since the Linear Regression model doesn't need any special parameters and fitting the Linear Regression Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Linear Regression.**_

# In[24]:


param = {}
FitModel(x,y,'Linear Regression',LinearRegression(),param,cv = 10)


# _**Running the function with empty parameters since the Lasso model doesn't need any special parameters and fitting the Lasso Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Lasso.**_

# In[25]:


FitModel(x,y,'Lasso',Lasso(),param,cv = 10)


# _**Running the function with empty parameters since the Ridge model doesn't need any special parameters and fitting the Ridge Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Ridge.**_

# In[26]:


FitModel(x,y,'Ridge',Ridge(),param,cv = 10)


# _**Running the function with some appropriate parameters and fitting the Random Forest Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Random Forest.**_

# In[27]:


params = {'n_estimators' : [44,109,314],
          'random_state' : [45]}
FitModel(x,y,'Random Forest',RandomForestRegressor(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the Extra Trees Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Extra Tree.**_

# In[28]:


FitModel(x,y,'Extra Tree',ExtraTreesRegressor(),params, cv = 10)


# _**Running the function with some appropriate parameters and fitting the XG Boost Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name XG Boost.**_

# In[29]:


FitModel(x,y,'XG Boost',XGBRegressor(),params,cv = 10)


# _**Running the function with some appropriate parameters and fitting the Cat Boost Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Cat Boost.**_

# In[30]:


params = {'verbose' : [0]}
FitModel(x,y,'Cat Boost',CatBoostRegressor(),params, cv = 10)


# _**Running the function with empty parameters since the Light GBM model doesn't need any special parameters and fitting the Light GBM Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Light GBM.**_

# In[31]:


FitModel(x,y,'Light GBM',LGBMRegressor(),param,cv = 10)


# ## Boosting the Model

# _**Defining the Function for the ML algorithms using GridSearchCV Algorithm and boosting the model using AdaBoostRegressor Algorithm  and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, R2 Score in percentage format, Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset**_

# In[32]:


def BoostModel(x,y,algo_name,algorithm,GridSearchParams,cv):
    np.random.seed(10)
    grid = GridSearchCV(estimator = algorithm, param_grid = GridSearchParams, cv = cv,
                       scoring = 'r2', verbose = 0,n_jobs = -1)
    grid_result = grid.fit(x_train,y_train)
    AB = AdaBoostRegressor(base_estimator = grid_result,learning_rate = 1)
    boostmodel = AB.fit(x_train,y_train)
    pred = boostmodel.predict(x_test)
    pickle.dump(boostmodel,open(algo_name,'wb'))
    print('Algorithm Name : ',algo_name,'\n')
    print('Percentage of R2 Score : {} %'.format(100 * r2_score(y_test,pred)),'\n')
    print('Mean Absolute Error : ',mean_absolute_error(y_test,pred),'\n')
    print('Root Mean Squared Error : ',sqrt(mean_squared_error(y_test,pred)),'\n')


# _**Running the function with empty parameters since the Linear Regression model doesn't need any special parameters and boosting the Linear Regression Algorithm and getting the Algorithm Name, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name  Boosted Linear Regression.**_

# In[33]:


param = {}
BoostModel(x,y,'Boosted Linear Regression',LinearRegression(),param,cv = 10)


# _**Running the function with empty parameters since the Lasso model doesn't need any special parameters and boosting the Lasso Algorithm and getting the Algorithm Name, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name  Boosted Lasso.**_

# In[34]:


BoostModel(x,y,'Boosted Lasso',Lasso(),param,cv = 10)


# _**Running the function with empty parameters since the Ridge model doesn't need any special parameters and boosting the Ridge Algorithm and getting the Algorithm Name, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name  Boosted Ridge.**_

# In[35]:


BoostModel(x,y,'Boosted Ridge',Ridge(),param,cv = 10)


# _**Running the function with some appropriate parameters and boosting the Random Forest Regressor Algorithm and getting the Algorithm Name, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Boosted Random Forest.**_

# In[36]:


params = {'n_estimators' : [44,109,314],
          'random_state' : [45]}
BoostModel(x,y,'Boosted Random Forest',RandomForestRegressor(),params,cv = 10)


# _**Running the function with some appropriate parameters and boosting the Extra Trees Regressor Algorithm and getting the Algorithm Name, R2 Score in percentage format ,Mean Absolute error and Root Mean Squared error between the predicted values and dependent test dataset and also the pickle file with the name Boosted Extra Tree.**_

# In[37]:


BoostModel(x,y,'Boosted Extra Tree',ExtraTreesRegressor(),params, cv = 10)


# For, further predictions use the model with the highest r2 score and load the pickle file with the format as
# **<span style = "color:red"> pickle.load(open(algo_name,'rb'))   </span>**
