
## --------------------------------------------------------------------
# Data modeling: continous TLI or PANSS measures
## --------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score, recall_score, f1_score,mean_squared_error, r2_score, mean_absolute_error)
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV

#-------------------------------------------
# Some functions
#--------------------------------------------
def model_evaluate(y_test,y_pred,show_metric=True, best_params=None):
    # evaluate model performance
    accuracy = accuracy_score(y_test, np.round(y_pred))
    cf_matrix = confusion_matrix(y_test, np.round(y_pred))
    precision = precision_score(y_test, np.round(y_pred))
    recall = recall_score(y_test, np.round(y_pred))
    f1 = f1_score(y_test, np.round(y_pred))    
    
    if show_metric:
        print("Model performance:")
        print("Accuracy:", accuracy)
        print('------------------------')
        print("Confusion Matrix:\n", cf_matrix)
        print('------------------------')
        print("Precision:", precision)
        print('------------------------')
        print("Recall:", recall)
        print('------------------------')
        print("F1 Score:", f1)
        
        if best_params:
            print('------------------------')
            print("Best parameters:", best_params)
    
    return accuracy, cf_matrix, precision, recall, f1



# check assumptions for linear regression
def residual_analysis(y_test, y_pred):
    """
    Perform residual analysis for Lasso regression model.

    Parameters:
    y_test (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    None
    """
    # Calculate residuals
    residuals = y_test - y_pred

    # Residual plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True, color='blue')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

    # QQ plot to check normality of residuals
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')



def preprocess_data(df, include_gender=True):
    # Drop unnecessary columns
    columns_to_drop = ['ID', 'PatientCat']
    
    if not include_gender:
        columns_to_drop.append('Gender')
    
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Include Gender as a dummy variable if specified
    if include_gender:
        df['Gender'] = df['Gender'].map({1.0: 'M', 2.0: 'F'})
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    return df




def get_train_test_data(df, sel_predictors, target_variables):
    # Drop rows with any missing data
    df = df.dropna(subset=sel_predictors + target_variables)
    
    if 'Gender_M' in df.columns:
        X = df[sel_predictors + ['Gender_M']].copy()
    else:
        X = df[sel_predictors]
    
    y = df[target_variables]  # Target variable(s)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def normalize_data(X_train,X_test):
    # Normalize data only for the numerical variables
    if 'Gender_M' in df.columns:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.iloc[:, :-1])
        X_test_scaled = scaler.transform(X_test.iloc[:, :-1])

        X_train_scaled = np.concatenate([X_train_scaled, X_train.iloc[:, -1].values.reshape(-1, 1)], axis=1)
        X_test_scaled = np.concatenate([X_test_scaled, X_test.iloc[:, -1].values.reshape(-1, 1)], axis=1)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled



#-------------------------------------------
# Get data ready
#--------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.join(os.path.dirname(current_directory),'stimuli','Relabeld','analysis')
outputfile_label = ['1min', 'spontaneous', 'concatenated']
outputfile = outputfile_label[1]
fname = os.path.join(parent_folder,'clean_all_' + outputfile + '.csv')
df = pd.read_csv(fname)

# Get data with or without Gender as a predictor
df_sel = preprocess_data(df.copy(), include_gender=True)
df_sel.drop(columns=['PANSS_Neg', 'PANSS_Pos'], inplace=True)

# Get variables of interest
sel_predictors = df_sel.drop(columns=['TLI_IMPOV', 'TLI_DISORG']).columns.tolist()
target_variables = ['TLI_IMPOV','TLI_DISORG']   
df_sel = df_sel[sel_predictors+target_variables].dropna()

# Get training and testing data
X = df_sel[sel_predictors]
target_variables = ['TLI_IMPOV','TLI_DISORG']   
y = df_sel[target_variables]  # Target variable(s)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize training data
X_train_scaled, X_test_scaled = normalize_data(X_train,X_test)

#----------------------------------------------------------
# Use Lasso regression to filter out irrelevant features
#----------------------------------------------------------
lasso_cv_impov = LassoCV(alphas=np.logspace(-6, 1, 100), cv=5)
lasso_cv_disorg = LassoCV(alphas=np.logspace(-6, 1, 100), cv=5)

# Train LassoCV models
lasso_cv_impov.fit(X_train_scaled, y_train['TLI_IMPOV'])
lasso_cv_disorg.fit(X_train_scaled, y_train['TLI_DISORG'])

# Print best parameters and best scores
print("Best alpha for IMPOV:", lasso_cv_impov.alpha_)
print("Best Mean Absolute Error (IMPOV):", lasso_cv_impov.score(X_test_scaled, y_test['TLI_IMPOV']))
print("Best alpha for DISORG:", lasso_cv_disorg.alpha_)
print("Best Mean Absolute Error (DISORG):", lasso_cv_disorg.score(X_test_scaled, y_test['TLI_DISORG']))

# Retrieve best models
best_lasso_impov = lasso_cv_impov
best_lasso_disorg = lasso_cv_disorg

# Predict on test set using best models
y_pred_impov = best_lasso_impov.predict(X_test_scaled)
y_pred_disorg = best_lasso_disorg.predict(X_test_scaled)

# Evaluate model performance
mae_impov = mean_absolute_error(y_test['TLI_IMPOV'], y_pred_impov)
mae_disorg = mean_absolute_error(y_test['TLI_DISORG'], y_pred_disorg)
r2_impov = r2_score(y_test['TLI_IMPOV'], y_pred_impov)
r2_disorg = r2_score(y_test['TLI_DISORG'], y_pred_disorg)

print(f"Mean Absolute Error (IMPOV): {mae_impov}")
print(f"Mean Absolute Error (DISORG): {mae_disorg}")
print(f"R2 (IMPOV): {r2_impov}")
print(f"R2 (DISORG): {r2_disorg}")

# Test for assumptions
residual_analysis(y_test['TLI_IMPOV'], y_pred_impov)
residual_analysis(y_test['TLI_DISORG'], y_pred_disorg)

# Retrieve coefficients and feature names
coefficients_impov = best_lasso_impov.coef_
coefficients_disorg = best_lasso_disorg.coef_
feature_names = X_train.columns
abs_coef_impov = np.abs(coefficients_impov)
abs_coef_disorg = np.abs(coefficients_disorg)
sorted_indices_impov = np.argsort(abs_coef_impov)[::-1]
sorted_indices_disorg = np.argsort(abs_coef_disorg)[::-1]
sorted_feature_names_impov = feature_names[sorted_indices_impov]
sorted_feature_names_disorg = feature_names[sorted_indices_disorg]
sorted_coef_impov = coefficients_impov[sorted_indices_impov]
sorted_coef_disorg = coefficients_disorg[sorted_indices_disorg]

# Plot feature importance for TLI_IMPOV
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names_impov, sorted_coef_impov)
plt.xlabel('Coefficient Magnitude')
plt.title('Feature Importance for TLI_IMPOV')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.savefig(os.path.join(parent_folder,'plots','ML_01_Lasso_Predictbeta.png'), format='png')
plt.show()

# Plot feature importance for TLI_DISORG
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names_disorg, sorted_coef_disorg)
plt.xlabel('Coefficient Magnitude')
plt.title('Feature Importance for TLI_DISORG')
plt.gca().invert_yaxis()
plt.savefig(os.path.join(parent_folder,'plots','ML_02_Lasso_PredictDISORG_beta.png'), format='png')
plt.show()


#----------------------------------------------
# random forest: based on the above selected features
#----------------------------------------------
# run the model
param_grid = {
    'n_estimators': [30, 50, 80],
    'max_depth': [None, 5, 8],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [1, 2, 4]
}
# Grid Search for TLI_IMPOV model
grid_search_impov = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search_impov.fit(X_train_scaled, y_train['TLI_IMPOV'])

# Grid Search for TLI_DISORG model
grid_search_disorg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search_disorg.fit(X_train_scaled, y_train['TLI_DISORG'])

# Get the best models from Grid Search
best_model_impov = grid_search_impov.best_estimator_
best_model_disorg = grid_search_disorg.best_estimator_

# Evaluate TLI_IMPOV predictions
improv_predictions = best_model_impov.predict(X_test_scaled)
mse_impov = mean_absolute_error(y_test['TLI_IMPOV'], improv_predictions)
r2_impov = r2_score(y_test['TLI_IMPOV'], improv_predictions)
print("TLI_IMPOV Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mse_impov:.2f}")
print(f"R-squared (R2): {r2_impov:.2f}")

# Evaluate TLI_DISORG predictions
disorg_predictions = best_model_disorg.predict(X_test_scaled)
mse_disorg = mean_absolute_error(y_test['TLI_DISORG'], disorg_predictions)
r2_disorg = r2_score(y_test['TLI_DISORG'], disorg_predictions)
print("\nTLI_DISORG Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mse_disorg:.2f}")
print(f"R-squared (R2): {r2_disorg:.2f}")

# Feature selection based on their importance
feature_importance = best_model_impov.feature_importances_
feature_importance_dict = dict(zip(X_train.columns, feature_importance))
sorted_features_impov = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_features_impov

feature_importance = best_model_disorg.feature_importances_
feature_importance_dict = dict(zip(X_train.columns, feature_importance))
sorted_features_disorg = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_features_disorg



#---------------------------------------
# linear regression
#---------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

X = df_sel[['entropyApproximate','num_repetition', 'type_token_ratio',
       'average_word_frequency', 'self_corrections']]
y_impov = df_sel['TLI_IMPOV']  # Target variable TLI_IMPOV
y_disorg = df_sel['TLI_DISORG']  # Target variable TLI_DISORG

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = LinearRegression()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Calculate cross-validated scores
scores = cross_val_score(model, X_scaled, y_impov, cv=kf, scoring='r2')

# Fit the model on the entire dataset to get coefficients
model.fit(X_scaled, y_impov)

# Get feature importance (coefficients)
importance = model.coef_

# Print the results
print(f'Mean cross-validated R^2 score: {np.mean(scores)}')
print('Feature importances (coefficients):')
for col, coef in zip(X.columns, importance):
    print(f'{col}: {coef}')
    
    

#---------------------------------------
# Ridge regression with cross-validation
#---------------------------------------
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Initialize Ridge Regression model with standard scaling
model_impov = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Calculate cross-validated scores
scores = cross_val_score(model_impov, X, y_impov, cv=kf, scoring='r2')

# Print cross-validated scores
print(f'Cross-validated R^2 scores: {scores}')
print(f'Mean cross-validated R^2 score: {np.mean(scores)}')

# Train the model on the entire training data
model_impov.fit(X, y_impov)

# Get coefficients (importance) and corresponding feature names
coefficients = model_impov.named_steps['ridge'].coef_
feature_names = X.columns

# Pair feature names with coefficients
feature_importance = dict(zip(feature_names, coefficients))

# Sort features based on absolute coefficient values
sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

# Print sorted feature importance
print("Feature importances (sorted by absolute value):")
for feature, importance in sorted_features:
    print(f"Feature: {feature}, Coefficient: {importance:.4f}")

