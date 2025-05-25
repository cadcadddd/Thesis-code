from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import random
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.ticker as ticker

#retrieve data
data_PL = pd.read_csv(''r'D:\TemporaryDirectory\testPL.csv', index_col=0, encoding='utf-8')

#Set the number of iterations and record model evaluation metrics
n_iterations = 80
train_rmse_list = []
test_rmse_list = []
train_r2_list = []
test_r2_list = []

# Splitting the data
data_train, data_test = train_test_split(data_PL, test_size=0.2, random_state=999)

# Extract feature data and target variables
X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
y_test = data_test.iloc[:, -1]


# Noise parameters (set appropriate mean and standard deviation based on data characteristics)
noise_params1 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.17},  
    'l_f': {'mean': 0, 'std_dev': 0.1},  
    'V_f': {'mean': 0, 'std_dev':0.002267},  
    "f_c'": {'mean': 0, 'std_dev': 0.146}, 
    'd': {'mean': 0, 'std_dev': 0.046667},
    'c/d': {'mean': 0, 'std_dev': 0.015},  
    'l/d': {'mean': 0, 'std_dev': 0.017}}  

noise_params5 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.17*5},
    'l_f': {'mean': 0, 'std_dev': 0.1*5},  
    'V_f': {'mean': 0, 'std_dev':0.002267*5},  
    "f_c'": {'mean': 0, 'std_dev': 0.146*5},  
    'd': {'mean': 0, 'std_dev': 0.046667*5}, 
    'c/d': {'mean': 0, 'std_dev': 0.015*5},  
    'l/d': {'mean': 0, 'std_dev': 0.017*5}}  

noise_params15 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.17*15},  
    'l_f': {'mean': 0, 'std_dev': 0.1*15}, 
    'V_f': {'mean': 0, 'std_dev':0.002267*15},  
    "f_c'": {'mean': 0, 'std_dev': 0.146*15},  
    'd': {'mean': 0, 'std_dev': 0.046667*15},  
    'c/d': {'mean': 0, 'std_dev': 0.015*15}, 
    'l/d': {'mean': 0, 'std_dev': 0.017*15}}  


noise_params10 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.17*10},  
    'l_f': {'mean': 0, 'std_dev': 0.1*10},  
    'V_f': {'mean': 0, 'std_dev':0.002267*10}, 
    "f_c'": {'mean': 0, 'std_dev': 0.146*10},  
    'd': {'mean': 0, 'std_dev': 0.046667*10},  
    'c/d': {'mean': 0, 'std_dev': 0.015*10},  
    'l/d': {'mean': 0, 'std_dev': 0.017*10}}  




# run iteratively
for i in range(n_iterations):
    # Copy X_test
    X_test = data_test.iloc[:, :-1].copy()
    # Add noise to input features only
    for feature in X_test.columns:
        mean = noise_params1[feature]['mean']        #Selection of noise distribution
        std_dev = noise_params1[feature]['std_dev']        #Selection of noise distribution
        noise = np.random.normal(mean, std_dev, X_test[feature].shape)

        # Add noise only to elements that are not 0
        X_test[feature] = X_test[feature].astype('float64')
        X_test.loc[X_test[feature] != 0, feature] += noise[X_test[feature] != 0]

    # Model training
    # Interchangeable models                #Selection of ML models
    XGBoost_model = xgb.XGBRegressor(
        max_depth=5,
        colsample_bytree=0.875,
        colsample_bylevel=0.4,
        colsample_bynode=0.152,
        n_estimators=180,
        learning_rate=0.400,
        subsample=0.5,
        random_state=999
    )
    GBDT_model = GradientBoostingRegressor(
        learning_rate=0.475,
        n_estimators=300,
        max_depth=3,
        subsample=0.918,
        min_samples_split=9,
        min_samples_leaf=1,
        max_features=0.367,
        random_state=999
    )
    # Fitting the final model
    final_model.fit(X_train, y_train)

    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)

    # Calculate metrics for training and test sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Storage indicators
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)

# Output result
for i in range(n_iterations):
    print(f"Iteration {i + 1}:")
    print(f"R^2 (Train): {train_r2_list[i]}")
    print(f"RMSE (Train): {train_rmse_list[i]}")
    print(f"R^2 (Test): {test_r2_list[i]}")
    print(f"RMSE (Test): {test_rmse_list[i]}\n")

# Calculate and print the average test set RMSE
average_test_rmse = np.mean(test_rmse_list)
max_test_rmse = np.max(test_rmse_list)
min_test_rmse = np.min(test_rmse_list)

print(f"Average RMSE (Test): {average_test_rmse:.4f}")
print(f"Maximum RMSE  (Test): {max_test_rmse:.4f}")
print(f"Minimum RMSE (Test): {min_test_rmse:.4f}\n")
# Setting the graphic size
plt.figure(figsize=(10, 6))
# Setting font global properties
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14

# Setting the axis title
plt.xlabel('Actual Values', fontsize=16, fontweight='normal')
plt.ylabel('Predicted Values', fontsize=16, fontstyle='normal')
plt.title('', fontsize=18)

plt.subplot(1, 1, 1)
plt.plot(range(1, n_iterations + 1), test_rmse_list, label='test RMSE', marker='o')
plt.xlabel('Iteration')
plt.ylabel('RMSE')

# Setting the x-axis range
plt.xlim(0, n_iterations)

# Setting the x-axis large scale
plt.xticks(range(1, n_iterations + 1, 5))

# Setting the small scale
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
# Hide labels for small scales
ax.tick_params(axis='x', which='minor', labelbottom=False)

# Set the legend position to the upper left corner
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()
plt.show()
