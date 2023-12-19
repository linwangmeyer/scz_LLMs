import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score, recall_score, f1_score)
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ------------------------------------------
# load data
# ------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_all.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
'''filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2), 
                         ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'n_sentence', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_avg = filtered_df.groupby('ID')[['PatientCat','PANSS Pos','TLI_DISORG','n_sentence','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()
'''

filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID', 'PatientCat','TLI_DISORG', 'stim', 'n_sentence', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_avg = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','n_sentence','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()

# ------------------------------------------
# check and visualize data
# ------------------------------------------
# check data
df_avg.isna().sum()
scz = (df_avg['PatientCat']==1).sum()
hc = (df_avg['PatientCat']==2).sum()
print(f'number of patients: {scz}')
print(f'number of healthy controls: {hc}')

# visualize correlation matrix
df_roi = df_avg[['entropyApproximate','TLI_DISORG','n_1','n_2','n_3','n_4','n_5']]
correlation_matrix = df_roi.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Pairwise Correlation Heatmap")
plt.show()

# visualize group difference: scatter plot
sns.scatterplot(data=df_roi, x='PatientCat', y='entropyApproximate')
slope, intercept = np.polyfit(df_roi['PatientCat'], df_roi['entropyApproximate'], 1)
regression_line = slope * df_roi['PatientCat'] + intercept
plt.plot(df_roi['PatientCat'], regression_line, color='red', label='Linear Regression')


# visualize group difference: box plot
plt.figure()
df_roi.boxplot(column='entropyApproximate', by='PatientCat')
plt.show()


# ------------------------------------------
# linear regression
# ------------------------------------------
# get critical variables
df_avg = filtered_df.groupby('ID')[['TLI_DISORG','n_sentence','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()

df_X = df_avg[['entropyApproximate','n_1','n_2','n_3']]
df_y = df_avg['TLI_DISORG']

# Split pre-processed data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# Train linear regression on the transformed training set
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

# Make predictions on the transformed testing set
y_pred = linreg.predict(X_test_scaled)

# Evaluate the performance of the linear regression model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')


# ------------------------------------------
# linear regression: baseline
# ------------------------------------------
df_X = df_avg[['n_sentence','n_1','n_2','n_3']]
df_y = df_avg['TLI_DISORG']

# Split pre-processed data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# Train linear regression on the transformed training set
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

# Make predictions on the transformed testing set
y_pred = linreg.predict(X_test_scaled)

# Evaluate the performance of the linear regression model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')




# ------------------------------------------
# cross validation
# ------------------------------------------
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(linreg, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f'Cross-validated Mean Squared Error: {cv_mse}')


# ------------------------------------------
# regularlized regression: baseline
# ------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_X = df_avg[['entropyApproximate','n_1','n_2','n_3']]
df_y = df_avg['TLI_DISORG']

# Split pre-processed data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# Define the parameter grid for Ridge regression
param_grid = {'alpha': [0.1, 1, 10, 100]}

# Create Ridge regression model
ridge = Ridge()

# Perform grid search with cross-validation
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Train Ridge regression with the best hyperparameters
best_ridge = Ridge(alpha=best_alpha)
best_ridge.fit(X_train_scaled, y_train)

# Make predictions on the transformed testing set
y_pred = best_ridge.predict(scaler.transform(X_test.values))

# Evaluate the performance of the Ridge regression model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')

# Print the best hyperparameters
print(f'Best Alpha: {best_alpha}')

