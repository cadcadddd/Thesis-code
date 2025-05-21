import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import random
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap

#retrieve data
data_PL = pd.read_csv('testPL.csv',index_col=0,encoding='utf-8')
data_BT = pd.read_csv('testBT.csv',index_col=0,encoding='utf-8')

index_PL=data_PL.index
#Correlation coefficients between variables
correlation_matrix = data_PL.corr()
print("correlation matrix：",correlation_matrix)
col=data_PL.columns
# Heat mapping
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


#Get training set and testing set(pullout set)
data_train, data_test= train_test_split(data_PL,test_size=0.2, random_state=999)
X_background=data_PL.iloc[:,0:-1]
X_train=data_train.iloc[:,0:-1]
X_test =data_test.iloc[:,0:-1]
feature_train=data_train.iloc[:,0:-1].columns
feature_test=data_test.iloc[:,0:-1].columns
y_train=data_train.iloc[:,-1]
y_test=data_test.iloc[:,-1]

#Loading the Beam Test (BT) dataset
X_BT=data_BT.iloc[:,0:-1]
Y_BT=data_BT.iloc[:,-1]
feature_BT =data_BT.iloc[:,0:-1].columns


# Create a Root Mean Square Error (RMSE) scorer
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
scoring = make_scorer(rmse,greater_is_better=False)


n = 7  # Number of parameters
p = 300  # Number of Population
m = 10  # Maximum value for random parameters
mr = 0.7  # Mutation rate
epochs = 150  # Number of generations

def randomGeneration(NumberOfRows, NumberOfQueens, m):
    generation_list = []
    for i in range(NumberOfRows):
        gene = [
            random.randint(1, m),    # n_estimators
            random.uniform(1, m),    # learning_rate
            random.randint(1, m),    # max_depth
            random.uniform(1, m),    #subsample
            random.uniform(1, m),    #colsample_bytree
            random.uniform(1, m),    #colsample_bylevel
            random.uniform(1, m)     #colsample_bynode
        ]
        generation_list.append(gene)
    return generation_list

def cross_over(generation_list, p, n):
    if p % 2 != 0:
        p -= 1

    for i in range(0, p, 2):
        child1 = generation_list[i][:n // 2] + generation_list[i + 1][n // 2:n]
        child2 = generation_list[i + 1][:n // 2] + generation_list[i][n // 2:n]
        generation_list.append(child1)
        generation_list.append(child2)

    return generation_list

def mutation(generation_list, p, n, m, mr):
    chosen_ones = list(range(p, len(generation_list)))
    random.shuffle(chosen_ones)
    chosen_ones = chosen_ones[:int(p * mr)]

    for i in chosen_ones:
        cell = random.randint(0, n - 1)
        val = random.randint(1, m)
        generation_list[i][cell] = val
    return generation_list

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def fitness(population_list):
    fitness_results = []
    for individual in population_list:
        try:
            model = xgb.XGBRegressor(
                n_estimators=individual[0] * 30,
                learning_rate=individual[1] / 10,
                max_depth=individual[2] + 1,
                subsample=individual[3] / 10,
                colsample_bytree=individual[4] / 10,
                colsample_bylevel=individual[5] / 10,
                colsample_bynode=individual[6] / 10,
                random_state=999
            )

            model.fit(X_train, y_train)                       # fit a model
            y_pred = model.predict(X_test)

            rmse_value = rmse(y_test, y_pred)                 # Calculating RMSE
            fitness_results.append({
                'params': individual,                           # Save parameters
                'rmse': rmse_value                              # Save RMSE
            })
        except Exception as e:
            print(f'Error evaluating parameters {individual}: {e}')
            fitness_results.append({'params': individual, 'rmse': None})

    return fitness_results

# primary cycle
population = randomGeneration(p, n, m)
best_params = None
no_improvement_count = 0  # Continuous unimproved counting

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}: Population size = {len(population)}')
    population = cross_over(population, p, n)
    population = mutation(population, p, n, m, mr)

    fitness_results = fitness(population)
    valid_results = [result for result in fitness_results if result['rmse'] is not None]

    if valid_results:
        valid_results.sort(key=lambda x: x['rmse'])

        # 更新种群
        population = [result['params'] for result in valid_results[:p]]

        # 保存最佳超参数
        if best_params is None or valid_results[0]['rmse'] < best_params['rmse']:
            best_params = {
                'params': valid_results[0]['params'],
                'rmse': valid_results[0]['rmse']
            }
            # Print the current optimal parameters
            current_best = best_params['params']
            print(
                f"Best parameters after epoch {epoch + 1}: n_estimators={current_best[0]}, learning_rate={current_best[1]},subsample={current_best[3]},colsample_bytree={current_best[4]},colsample_LEVEL={current_best[5]},colsample_bynode={current_best[6]}, max_depth={current_best[2]}")
            print(f"Best RMSE: {best_params['rmse']}")
            no_improvement_count = 0  # eset No Improvement Count
        else:
            no_improvement_count += 1
            print("No improvement found in this epoch.")
            print("Maximum number of non-improvements：",no_improvement_count)
    else:
        print("No valid results found, continuing to next epoch.")

# Check if valid_results is empty
if best_params:
    print('Best hyperparameters found:')
    print(best_params['params'])
    print(f'Best RMSE: {best_params["rmse"]}')

    # Training the final model with optimal parameters
    final_model = xgb.XGBRegressor(
        max_depth=best_params['params'][2] + 1,
        colsample_bytree=best_params['params'][4]/10,
        colsample_bylevel=best_params['params'][5]/10,
        colsample_bynode=best_params['params'][6]/10,
        n_estimators=best_params['params'][0] * 30,
        learning_rate=best_params['params'][1] / 10,
        subsample = best_params['params'][3] / 10,
        random_state=999
    )

    # Fitting the final model
    final_model.fit(X_train, y_train)

else:
    print("No valid parameters found after optimization.")

# Prediction using the best model
y_test_pred = final_model.predict(X_test)
y_train_pred = final_model.predict(X_train)
Y_BT_pred    =final_model.predict(X_BT)


#Define RMSE
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

# 计算训练集和测试集的RRMSE
print ("XGBoostmodel evaluation--PL training set：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("XGBoostmodel evaluation--PL testing set：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("XGBoost evaluation--Beam bending set：")
print ('R^2:',r2_score(Y_BT,Y_BT_pred))
print ('MSE',mean_squared_error(Y_BT,Y_BT_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_BT,Y_BT_pred)))
print ('MAE',mean_absolute_error(Y_BT,Y_BT_pred))


joblib.dump(final_model, "XGBoost_model.pkl")


#SHAP
# Setting font global properties
plt.rcParams['font.family'] = 'serif'  # Font Type
plt.rcParams['font.serif'] = ['Times New Roman']  # Specify a specific font
plt.rcParams['font.size'] = 14  # Font size
# Analysing with SHAP
explainer = shap.TreeExplainer(final_model, X_train)
shap_values = explainer.shap_values(X_train)

shap_values_array = shap_values # NumPy array for extracting SHAP values

avg_shap_values = np.abs(shap_values_array).mean(axis=0) # Calculate the average of the SHAP values for each factor

shap.summary_plot(shap_values, X_train, plot_type="dot", show=True) # Visual interpretation (scatterplot)

sorted_idx = np.argsort(avg_shap_values)[::1]  # Sorting features by SHAP mean size

sorted_features = np.array(X_train.columns)[sorted_idx] # Rearrange feature names and SHAP values based on sorted indexes
sorted_shap_values = avg_shap_values[sorted_idx]


# Visualisation of the importance of features in descending order of SHAP means
plt.figure(figsize=(12, 6))
bars = plt.barh(sorted_features, sorted_shap_values, color='skyblue')
plt.xlabel('Average SHAP Value')
plt.ylabel('Features')
plt.title('Average SHAP Values for Each Feature')
plt.show()
