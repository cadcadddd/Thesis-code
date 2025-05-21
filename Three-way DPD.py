import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

#Reading a dataset and dividing it
data_PL = pd.read_csv(r'D:\TemporaryDirectory\testPL.csv', index_col=0, encoding='utf-8')
data_train, data_test = train_test_split(data_PL, test_size=0.2, random_state=999)

# Extract feature data and target variables
X_train = data_train.iloc[:, :-1].copy()
y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, :-1].copy()  
y_test = data_test.iloc[:, -1]


feature_data_train=data_train.iloc[:,0:-1].columns

# model training(PDP mapping using the best model)
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

def get_varying_features(fixed_feature, features):
    for fixed in fixed_feature:
        # Acquisition of unfixed features
        varying_features = [f for f in features if f != fixed]

        # Ensure that only two unfixed features are selected
        if len(varying_features) >= 2:
            return varying_features[0], varying_features[1]

    raise ValueError("There are not enough unfixed features to choose from.")

def plot_3d_pdp(model, X, varying_features,fixed_feature,fixed_value=None, grid_resolution=50):#grid_resolution determines the grid size
    #Default fixed value
    if fixed_value is None:
        fixed_value = X[fixed_feature].mean()

    #Creating a Grid
    feature_1_vals = np.linspace(X[varying_features[0]].min(), X[varying_features[0]].max(), grid_resolution)
    feature_2_vals = np.linspace(X[varying_features[1]].min(), X[varying_features[1]].max(), grid_resolution)
    #Generate gridGenerate grid（50×50）
    grid_1, grid_2 = np.meshgrid(feature_1_vals, feature_2_vals)
    #spreading grid（2500×2）
    grid_points = np.c_[grid_1.ravel(), grid_2.ravel()]
    # Create a blank matrix with the same dimensions as X（2500×8）
    X_grid = np.zeros((grid_points.shape[0], X.shape[1]))

    # Iterate over fixed features, filling in the mean first
    for feature in fixed_feature:
        if feature in X_test.columns:
            fixed_value = X_test[feature].mean()  
            X_grid[:, X_test.columns.get_loc(feature)] = fixed_value  
        else:
            print(f"Features '{feature}' is not in X_test.columns, check spelling or formatting.")

    #Characteristics of filling changes
    X_grid[:, X.columns.get_loc(varying_features[0])] = grid_points[:, 0]
    X_grid[:, X.columns.get_loc(varying_features[1])] = grid_points[:, 1]

    # anticipate
    preds = model.predict(X_grid).reshape(grid_1.shape)

    # Drawing 3D images
    fig = plt.figure(figsize=(14, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # Contour mapping
    surface = ax.plot_surface(grid_1, grid_2, preds, cmap='viridis', alpha=0.8)

    # Adjusting the size of the axes scale numbers
    ax.tick_params(axis='x', labelsize=14, pad=0)
    ax.tick_params(axis='y', labelsize=14, pad=0)
    ax.tick_params(axis='z', labelsize=14, pad=0)
    # Add a colour heat bar
    cbar =fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=14)
    # Setting axis labels and titles
    ax.set_xlabel(varying_features[0], fontsize=20)
    ax.set_ylabel(varying_features[1], fontsize=20)

    plt.savefig("3D one.pdf", format='pdf', bbox_inches='tight')
    plt.show()

#Drawings
features = data_train.iloc[:,0:-1].columns
fixed_feature=['d', 'l/d', 'V_f','l_f' , "f_c'"] #Fixed features
varying_feature=['c/d', 'l_f/d_f']
plot_3d_pdp(final_model, X_test, varying_features=varying_feature,fixed_feature=fixed_feature, grid_resolution=25)

fixed_feature1=[ "f_c'", 'l_f/d_f', 'V_f','l_f' , 'l/d']
varying_feature1=['d','c/d']
plot_3d_pdp(final_model, X_test, varying_features=varying_feature1,fixed_feature=fixed_feature1, grid_resolution=25)

fixed_feature2=['c/d', 'l/d', 'V_f','l_f' ,"f_c'" ]
varying_feature2=['d', 'l_f/d_f']
plot_3d_pdp(final_model, X_test, varying_features=varying_feature2,fixed_feature=fixed_feature2, grid_resolution=25)

fixed_feature3=['d', 'l_f/d_f', 'V_f','l_f' , 'l/d']
varying_feature3=['c/d', "f_c'"]
plot_3d_pdp(final_model, X_test, varying_features=varying_feature3,fixed_feature=fixed_feature3, grid_resolution=25)

fixed_feature4=['c/d', 'l_f/d_f', 'V_f','l_f' , 'l/d']
varying_feature4=['d', "f_c'"]
plot_3d_pdp(final_model, X_test, varying_features=varying_feature4,fixed_feature=fixed_feature4, grid_resolution=25)

fixed_feature5=['c/d','d' , 'V_f',"l_f",'l/d' ]
varying_feature5=['l_f/d_f', "f_c'"]
plot_3d_pdp(final_model, X_test, varying_features=varying_feature5,fixed_feature=fixed_feature5, grid_resolution=25)

# Assessment of indicators
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("XGBoost Model Evaluation--PL Training Set：")
print('R^2:', train_r2)
print('RMSE:', train_rmse)

print("XGBoost Todel Tvaluation--PL Testing Tet：")
print('R^2:', test_r2)
print('RMSE:', test_rmse)

