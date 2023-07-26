#!/usr/bin/env python
# coding: utf-8

# https://medium.com/geekculture/xgboost-versus-random-forest-898e42870f30

# https://towardsdatascience.com/time-series-forecasting-with-machine-learning-b3072a5b44ba

# https://lightgbm.readthedocs.io/en/latest/Parameters.html

# https://analyticsindiamag.com/deep-learning-xgboost-or-both-what-works-best-for-tabular-data/ and https://arxiv.org/abs/2207.08815

# Gradient Boosting 
# - Boosting: Boosting is an ensemble learning technique where multiple weak learners (usually decision trees) are combined to create a strong learner. The weak learners are trained sequentially, with each one focusing on the mistakes of its predecessors. 
# - Boosting happens to be iterative learning which means the model will predict something initially and self analyses its mistakes as a predictive toiler and give more weightage to the data points in which it made a wrong prediction in the next iteration
# -  Gradient Boosting is a specific variant of boosting that uses gradients  (the gradient of the loss function) to optimize the model's performance
# - Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function (different types of loss functions can be used)

# The LightGBM Regressor:
# - ML model that can predict numerical values (regression) based on input features. It works by creating a decision tree ensemble in a way that optimizes training speed and memory usage. It leverages gradient-based techniques to efficiently handle large datasets, making it suitable for time series forecasting tasks like predicting energy production or other continuous numeric variables.
# - uses a different approach to tree building known as the Gradient-based One-Side Sampling (GOSS) algorithm. It also uses a histogram-based algorithm to bucket feature values, which can lead to faster training times compared to XGBoost. 
# -  designed to take advantage of parallelism on both CPUs and GPUs, making it potentially faster in certain scenarios, especially on GPU hardware.

# In[3]:


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import weibull_min
from itertools import product

import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor, LGBMClassifier, Booster

get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


file_path_1 = ""
df= pd.read_csv(file_path_1, sep='\t')


# In[35]:


df


# In[9]:


df['DATE'] = pd.to_datetime(df['DATE'])
#set datetime as index
df = df.set_index(df.DATE)
#drop datetime column
df.drop('DATE', axis=1, inplace=True)

#create hour, day and month variables from datetime index
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month


# In[11]:


#drop casual and registered columns
df.drop(['WF', 'WT','Index'], axis=1, inplace=True)


# The horizon parameter determines the number of time steps into the future for which the model makes predictions.

# Which means the model is making predictions for the next 24 time steps (or 240 minutes/4 hours in this case, considering the data has a 10-minute interval).

# The code is not predicting future values; it is predicting ActivePower values for the last part of the dataset (the testing set) using the trained model. The model was trained on past data to predict the corresponding ActivePower values for the testing period. The purpose is to evaluate the model's performance on unseen data to assess how well it can generalize.

# In[13]:


def train_time_series_with_folds(df, horizon=24):
    X = df.drop('ActivePower', axis=1)
    y = df['ActivePower']
    
    #take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    
    #create, train and do inference of the model
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    #calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)    
    
    print(f'Mean Absolute Error (MAE): {mae}')
    
    #calculate MSE
    mse = np.round(mean_squared_error(y_test, predictions), 3)
    print(f'Mean Squared Error (MSE): {mse}')
    
    #calculate MAPE
    mape = np.round(np.mean(np.abs((y_test - predictions) / y_test)) * 100, 3)
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
    
    #plot reality vs prediction for the last week of the dataset
    fig = plt.figure(figsize=(16,8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Hour', fontsize=16)
    plt.ylabel('AP', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()
    
    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()


# In[14]:


train_time_series_with_folds(df)


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def hyperparameter_tuning_with_random_search(df, num_iterations, param_search_space, horizon=24):
    X = df.drop('ActivePower', axis=1)
    y = df['ActivePower']
    
    # take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    
    # Initialize a list to store the results of each iteration
    results = []
    
    for i in range(num_iterations):
        # Sample random hyperparameters from the search space
        params = {param: np.random.choice(values) for param, values in param_search_space.items()}
        
        # Create and train the model with the sampled hyperparameters
        model = LGBMRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate evaluation metrics (MAE, MSE, MAPE)
        mae = np.round(mean_absolute_error(y_test, predictions), 3)
        mse = np.round(mean_squared_error(y_test, predictions), 3)
        mape = np.round(np.mean(np.abs((y_test - predictions) / y_test)) * 100, 3)
        
        # Store the results of the iteration in the 'results' list
        results.append({'Iteration': i+1, 'Hyperparameters': params, 'MAE': mae, 'MSE': mse, 'MAPE': mape})
        
        # Print the trial number and evaluation metrics before each plot
        print(f"\nTrial {i+1} - MAE: {mae}, MSE: {mse}, MAPE: {mape}%")
        
        # Plot reality vs prediction for the last week of the dataset
        fig = plt.figure(figsize=(16, 8))
        plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
        plt.plot(y_test, color='red', label='Real')
        plt.plot(pd.Series(predictions, index=y_test.index), color='green', label='Prediction')
        plt.xlabel('Hour', fontsize=16)
        plt.ylabel('Number of Shared Bikes', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.show()

    # Sort the results based on the MSE values in ascending order
    sorted_results_mse = sorted(results, key=lambda x: x['MSE'])
    
    # Print the best 5 trials with lower MSE values
    print("\nBest 5 trials based on lower MSE values:")
    for i in range(min(5, num_iterations)):
        print(f"Iteration {sorted_results_mse[i]['Iteration']} - MSE: {sorted_results_mse[i]['MSE']}")
        print(f"Hyperparameters: {sorted_results_mse[i]['Hyperparameters']}")
        print("----------------------------------------------------")

    # Sort the results based on the MAPE values in ascending order
    sorted_results_mape = sorted(results, key=lambda x: x['MAPE'])
    
    # Print the best 5 trials with lower MAPE values
    print("\nBest 5 trials based on lower MAPE values:")
    for i in range(min(5, num_iterations)):
        print(f"Iteration {sorted_results_mape[i]['Iteration']} - MAPE: {sorted_results_mape[i]['MAPE']}%")
        print(f"Hyperparameters: {sorted_results_mape[i]['Hyperparameters']}")
        print("----------------------------------------------------")
    

# Define the parameter search space
param_search_space = {
    'learning_rate': [0.01, 0.001,0.1],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20],
    'num_leaves': [20, 30, 40, 50],
    'min_child_samples': [10, 20, 30, 40],
    'reg_alpha': [0.0, 0.1, 0.5, 1.0]
}

# Assuming you have already loaded and preprocessed the data in 'df'
num_iterations = 1000
hyperparameter_tuning_with_random_search(df, num_iterations, param_search_space)


# Best 5 trials based on lower MSE values:
# Iteration 763 - MSE: 101901.974
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 200, 'max_depth': 15, 'num_leaves': 50, 'min_child_samples': 10, 'reg_alpha': 0.0}
# ----------------------------------------------------
# Iteration 46 - MSE: 101902.036
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 200, 'max_depth': 15, 'num_leaves': 50, 'min_child_samples': 10, 'reg_alpha': 0.1}
# ----------------------------------------------------
# Iteration 465 - MSE: 101902.036
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 200, 'max_depth': 15, 'num_leaves': 50, 'min_child_samples': 10, 'reg_alpha': 0.1}
# ----------------------------------------------------
# Iteration 429 - MSE: 101903.157
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 200, 'max_depth': 15, 'num_leaves': 50, 'min_child_samples': 10, 'reg_alpha': 0.5}
# ----------------------------------------------------
# Iteration 638 - MSE: 101908.211
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 200, 'max_depth': 10, 'num_leaves': 50, 'min_child_samples': 10, 'reg_alpha': 1.0}
# ----------------------------------------------------
# 
# Best 5 trials based on lower MAPE values:
# Iteration 24 - MAPE: 37.656%
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 50, 'max_depth': 20, 'num_leaves': 50, 'min_child_samples': 30, 'reg_alpha': 1.0}
# ----------------------------------------------------
# Iteration 25 - MAPE: 37.656%
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 50, 'max_depth': 10, 'num_leaves': 50, 'min_child_samples': 30, 'reg_alpha': 0.1}
# ----------------------------------------------------
# Iteration 61 - MAPE: 37.656%
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 50, 'max_depth': 20, 'num_leaves': 50, 'min_child_samples': 30, 'reg_alpha': 0.5}
# ----------------------------------------------------
# Iteration 195 - MAPE: 37.656%
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 50, 'max_depth': 10, 'num_leaves': 50, 'min_child_samples': 30, 'reg_alpha': 0.0}
# ----------------------------------------------------
# Iteration 292 - MAPE: 37.656%
# Hyperparameters: {'learning_rate': 0.001, 'n_estimators': 50, 'max_depth': 10, 'num_leaves': 50, 'min_child_samples': 30, 'reg_alpha': 0.1}
# ----------------------------------------------------

# # Creating Lag features and rolling statistics

# Lag features involve using past values of the target variable (in this case, ActivePower and WindSpeed) as new features, while rolling statistics compute statistics (e.g., mean, standard deviation) over a window of past values. Both lag features and rolling statistics can capture temporal patterns and dependencies in time-series data

# These lines create new columns ActivePower_lag1 and WindSpeed_lag1 in the DataFrame df, where the values of ActivePower and WindSpeed are shifted by 6 time steps (rows) backward. This means that for each row, the corresponding value from 6 time steps ago is placed in the new column. This creates lag features that can capture the historical behavior of the variables, which can be useful for time series forecasting.

# These lines create new columns ActivePower_mean_rolling2, ActivePower_std_rolling2, WindSpeed_mean_rolling2, and WindSpeed_std_rolling2 in the DataFrame df. The rolling statistics are computed over a rolling window of 2 time steps (rows). For each row, the mean and standard deviation of the ActivePower and WindSpeed values within the last 2 time steps are calculated and stored in the corresponding new columns. Rolling statistics can help capture short-term trends and fluctuations in the data, which can also be relevant for time series forecasting.

# In[100]:


import pandas as pd

file_path_1 = "/Users/E708126/OneDrive - EDP/jupyter_notebooks/Synthetic Data/Data/Arganil/Original/arganil_turbine_2.txt"
df= pd.read_csv(file_path_1, sep='\t')

# Convert 'DATE' column to datetime type and set it as the index
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Create lag features for ActivePower and WindSpeed with a lag of 6
df['ActivePower_lag1'] = df['ActivePower'].shift(6)
df['WindSpeed_lag1'] = df['WindSpeed'].shift(6)

# Create rolling statistics (mean and standard deviation) for ActivePower and WindSpeed over a window of 2 time steps
df['ActivePower_mean_rolling2'] = df['ActivePower'].rolling(window=2).mean()
df['ActivePower_std_rolling2'] = df['ActivePower'].rolling(window=2).std()
df['WindSpeed_mean_rolling2'] = df['WindSpeed'].rolling(window=2).mean()
df['WindSpeed_std_rolling2'] = df['WindSpeed'].rolling(window=2).std()

df.head()


# For the lag features (df['ActivePower_lag1'] and df['WindSpeed_lag1']), if lag is of 144 time steps,  corresponds to a 1440-minute lag in the data (assuming each time step represents 10 minutes). This means we are using the value of ActivePower and WindSpeed from 1440 minutes (24 hours) ago as features to predict the current value.

# To find the best lag value, you can try different lag values (e.g., 12, 24, 48, etc.) and evaluate the model's performance with each choice. 

# default parameters of XGBoost:
# - num_leaves: 31
# - max_depth: -1 (no limit)
# - learning_rate: 0.1
# - n_estimators: 100
# - min_child_samples: 20
# - reg_alpha: 0.0

# In[101]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_time_series_with_folds(df, horizon=30):
    df.drop(['WF', 'WT', 'Index'], axis=1, inplace=True)
    #df['DATE'] = pd.to_datetime(df['DATE'])
    #set datetime as index
    #df = df.set_index(df.DATE)
    #drop datetime column
    #df.drop('DATE', axis=1, inplace=True)

    #create hour, day and month variables from datetime index
    #df['hour'] = df.index.hour
    #df['day'] = df.index.day
    #df['month'] = df.index.month
    X = df.drop('ActivePower', axis=1)
    y = df['ActivePower']

    #take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    print("Training data start date:", X_train.index[0])
    print("Training data end date:", X_train.index[-1])

    print("Test data start date:", X_test.index[0])
    print("Test data end date:", X_test.index[-1])
    
    #create, train and do inference of the model
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    #calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    print(f'Mean Absolute Error (MAE): {mae}')

    #calculate MSE
    mse = np.round(mean_squared_error(y_test, predictions), 3)
    print(f'Mean Squared Error (MSE): {mse}')

    #calculate MAPE
    mape = np.round(np.mean(np.abs((y_test - predictions) / y_test)) * 100, 3)
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

    #plot reality vs prediction for the last week of the dataset
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    ax.plot(y_test.index, y_test, color='red', label='Real')
    ax.plot(y_test.index, predictions, color='green', label='Prediction')
    plt.xlabel('Date and Time', fontsize=16)
    plt.ylabel('AP', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.show()

    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()


# Call the training function with the updated DataFrame

#file_path_1 = ""
#df= pd.read_csv(file_path_1, sep='\t')
#df.drop(['WF', 'WT','Index'], axis=1, inplace=True)
train_time_series_with_folds(df)


# 6, 2 por agora é a melhor

# # Hyperparameter tuning of lags and windows

# In[82]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_time_series_with_folds(df, horizon=48):
    #df.drop(['WF', 'WT','Index'], axis=1, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    #set datetime as index
    df = df.set_index(df.DATE)
    #drop datetime column
    df.drop('DATE', axis=1, inplace=True)

    #create hour, day and month variables from datetime index
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    X = df.drop('ActivePower', axis=1)
    y = df['ActivePower']

    # Take last part of the dataset for validation
    X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # Create, train and do inference of the model
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    print(f'Mean Absolute Error (MAE): {mae}')

    # Calculate MSE
    mse = np.round(mean_squared_error(y_test, predictions), 3)
    print(f'Mean Squared Error (MSE): {mse}')

    # Calculate MAPE
    mape = np.round(np.mean(np.abs((y_test - predictions) / y_test)) * 100, 3)
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

    # Plot reality vs prediction for the last week of the dataset
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    ax.plot(y_test.index, y_test, color='red', label='Real')
    ax.plot(y_test.index, predictions, color='green', label='Prediction')
    plt.xlabel('Date and Time', fontsize=16)
    plt.ylabel('Number of Shared Bikes', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.show()

    # Create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()
    return mae, mse, mape

def hyperparameter_tuning(df, lag_values=[1, 6, 12, 24, 48], window_values=[2, 3, 5, 10]):
    results = []

    for lag in lag_values:
        for window in window_values:
            df['ActivePower_lag'] = df['ActivePower'].shift(lag)
            df['WindSpeed_lag'] = df['WindSpeed'].shift(lag)
            df['WindSpeed_mean_rolling'] = df['WindSpeed'].rolling(window=window).mean()
            df['WindSpeed_std_rolling'] = df['WindSpeed'].rolling(window=window).std()

            print(f"Lag: {lag}, Window: {window}")
            mae, mse, mape = train_time_series_with_folds(df)

            results.append({
                'Lag': lag,
                'Window': window,
                'MAE': mae,
                'MSE': mse,
                'MAPE': mape
            })

    # Sort the results based on MSE in ascending order (lower MSE is better)
    results.sort(key=lambda x: x['MSE'])

    # Print the best three iterations
    print("\nBest 3 Iterations:")
    for i, result in enumerate(results[:3], 1):
        print(f"Iteration {i}: Lag={result['Lag']}, Window={result['Window']}, "
              f"MAE={result['MAE']}, MSE={result['MSE']}, MAPE={result['MAPE']}")

    return results 
# Assuming you have already loaded and preprocessed the data in 'df'
# Call the hyperparameter_tuning function with the DataFrame





# In[83]:


file_path_1 = ""
df= pd.read_csv(file_path_1, sep='\t')
df.drop(['WF', 'WT','Index'], axis=1, inplace=True)
hyperparameter_tuning(df)


# Best 3 Iterations:
# - Iteration 1: Lag=48, Window=10, MAE=27.337, MSE=1004.36, MAPE=1.364
# - Iteration 2: Lag=12, Window=2, MAE=25.293, MSE=1005.601, MAPE=1.271
# - Iteration 3: Lag=48, Window=3, MAE=27.78, MSE=1133.359, MAPE=1.397

# In[ ]:




