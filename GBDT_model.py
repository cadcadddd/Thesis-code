from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score
import shap
import matplotlib.pyplot as plt
import random
import seaborn as sns

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



#Divide the training set and testing set(PULLOUT SET)
data_train, data_test= train_test_split(data_PL,test_size=0.2, random_state=999)
X_train=data_train.iloc[:,0:-1]
X_test =data_test.iloc[:,0:-1]
feature_train=data_train.iloc[:,0:-1].columns
feature_test=data_test.iloc[:,0:-1].columns
feature_BT =data_BT.iloc[:,0:-1].columns
y_train=data_train.iloc[:,-1]
y_test=data_test.iloc[:,-1]

#Read beam bending test set
X_BT=data_BT.iloc[:,0:-1]
Y_BT=data_BT.iloc[:,-1]


n = 7  # Number of parameters (like hyperparameters)
p = 300  # Number of Population
m = 10  # Maximum value for random parameters
mr = 0.7  # Mutation rate
epochs = 150  # Number of generations

def randomGeneration(NumberOfRows, NumberOfQueens, m):
    generation_list = []
    for i in range(NumberOfRows):
        gene = [
            random.uniform(1, m),    # learning_rate
            random.randint(1, m),    # n_estimators
            random.randint(1, m),    # max_depth
            random.uniform(1, m),    #subsample
            random.randint(1, m),    #min_samples_split
            random.randint(1, m),    #min_samples_leaf
            random.uniform(1, m),    #max_features
        ]
        generation_list.append(gene)
    return generation_list

def cross_over(generation_list, p, n):
    if p % 2 != 0:
        p -= 1  # If odd, reduce by one individual

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
            model = GradientBoostingRegressor(
                learning_rate=individual[0] /10,
                n_estimators=individual[1] *30,
                max_depth=individual[2] + 1,
                subsample=individual[3] / 10,
                min_samples_split=individual[4]+1,
                min_samples_leaf=individual[5],
                max_features=individual[6] / 10,
                random_state=999
            )

            model.fit(X_train, y_train)                       # fit the model
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

# 主循环
population = randomGeneration(p, n, m)
best_params = None

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}: Population size = {len(population)}')
    population = cross_over(population, p, n)
    population = mutation(population, p, n, m, mr)

    fitness_results = fitness(population)
    valid_results = [result for result in fitness_results if result['rmse'] is not None]

    if valid_results:
        valid_results.sort(key=lambda x: x['rmse'])

        # Renewal of stocks
        population = [result['params'] for result in valid_results[:p]]

        # Saving the optimal hyperparameters
        if best_params is None or valid_results[0]['rmse'] < best_params['rmse']:
            best_params = {
                'params': valid_results[0]['params'],
                'rmse': valid_results[0]['rmse']
            }
            # Print the current optimal parameters
        current_best = best_params['params']
        print(
                f"Best parameters after epoch {epoch + 1}: learning_rate={current_best[0]}, n_estimators={current_best[1]}, max_depth={current_best[2]}")
        print(f"Best RMSE: {best_params['rmse']}")

    else:
        print("No valid results found, stopping the algorithm.")
        break

# Check if valid_results is empty
if best_params:
    print('Best hyperparameters found:')
    print(best_params['params'])
    print(f'Best RMSE: {best_params["rmse"]}')

    # Training the final model with optimal parameters
    final_model = GradientBoostingRegressor(
        learning_rate=best_params['params'][0] /10,
        n_estimators=best_params['params'][1]* 30,
        max_depth=best_params['params'][2],
        subsample=best_params['params'][3]/10,
        min_samples_split=best_params['params'][4] +1,
        min_samples_leaf=best_params['params'][5],
        max_features = best_params['params'][6] / 10,
        random_state = 999
    )

    # 拟合最终模型
    final_model.fit(X_train, y_train)

else:
    print("No valid parameters found after optimization.")

# Prediction using the best model
y_test_pred = final_model.predict(X_test)
y_train_pred = final_model.predict(X_train)
Y_BT_pred    =final_model.predict(X_BT)


# Calculation of assessment indicators
print ("GBDT evaluation--PL training set：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("GBDT evaluation--PL testing set：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("GBDT evaluation--Beam bending set：")
print ('R^2:',r2_score(Y_BT,Y_BT_pred))
print ('MSE',mean_squared_error(Y_BT,Y_BT_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_BT,Y_BT_pred)))
print ('MAE',mean_absolute_error(Y_BT,Y_BT_pred))

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

