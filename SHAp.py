import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor
# 读取数据
data_PL = pd.read_csv(r'D:\TemporaryDirectory\testPL.csv', index_col=0, encoding='utf-8')

data_train, data_test = train_test_split(data_PL, test_size=0.2, random_state=999)

# 提取特征数据和目标变量
X_train = data_train.iloc[:, :-1].copy()
y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, :-1].copy()  # 使用 copy() 创建副本以防止修改原始数据
y_test = data_test.iloc[:, -1]
#获取训练集和验证集
feature_data_train=data_train.iloc[:,0:-1].columns
feature_data_PL=data_PL.iloc[:,0:-1].columns

# 模型训练
final_model = xgb.XGBRegressor(
    max_depth=5,
    colsample_bytree=0.875,
    colsample_bylevel=0.4,
    colsample_bynode=0.152,
    n_estimators=180,
    learning_rate=0.400,
    subsample=0.5,
    random_state=999
)


final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)
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


