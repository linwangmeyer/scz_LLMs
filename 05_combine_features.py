## --------------------------------------------------------------------
# Check final effects
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
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score, recall_score, f1_score)
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

## --------------------------------------------------------------------
# Combine all measures with subject info
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
outputfile_label = ['1min', 'spontaneous', 'concatenated']
for k in range(3):
    outputfile = outputfile_label[k]

    fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
    df_var = pd.read_csv(fname_var)

    fname = os.path.join(parent_folder,'topic_measures_' + outputfile + '.csv')
    df_topic = pd.read_csv(fname)
    
    fname = os.path.join(parent_folder,'word2vec_' + outputfile + '.csv')
    df_w2v = pd.read_csv(fname)
    
    fname = os.path.join(parent_folder,'similarity_measures_' + outputfile + '.csv')
    df_sensim = pd.read_csv(fname)
    
    fname = os.path.join(parent_folder,'SenSimilarity_backwards_measures_' + outputfile + '.csv')
    df_sensim_backward = pd.read_csv(fname)
    
    fname = os.path.join(parent_folder,'syntax_measures_' + outputfile + '.csv')
    df_syntax = pd.read_csv(fname)
    
    fname = os.path.join(parent_folder,'lexical_measures_' + outputfile + '.csv')
    df_lexical = pd.read_csv(fname)
    
    # Merge DataFrames on a common key (e.g., 'ID')
    merged_df = df_var.merge(df_topic, on='ID', how='outer') \
                  .merge(df_w2v, on=['ID', 'stim'], how='left') \
                  .merge(df_sensim, on=['ID', 'stim'], how='left') \
                  .merge(df_sensim_backward, on=['ID', 'stim'], how='left') \
                  .merge(df_syntax, on=['ID', 'stim'], how='left') \
                  .merge(df_lexical, on=['ID', 'stim'], how='left')
                  
    filtered_df = merged_df.dropna(subset = ['stim'], how='all')
    filtered_df.drop(columns=['nsen','nword','subord_index'], inplace=True)
    
    fname_all = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv')
    filtered_df.to_csv(fname_all,index=False)

    df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]
    fname_goi = os.path.join(parent_folder,'TOPSY_TwoGroups_' + outputfile + '.csv')
    df_goi.to_csv(fname_goi,index=False)


# ------------------------------------------------------------------------
# intitial visualization
df_avg = filtered_df.drop(columns='stim').groupby('ID').mean().reset_index()

columns_include = ['ID','AgeScan1', 'PatientCat', 'Gender', 'PANSS Neg', 'PANSS Pos', 'Trails-B', 'DSST_Writen', 'DSST_Oral', 'Category Fluency (animals)', 'SOFAS',
 'TLI_IMPOV', 'TLI_DISORG','entropyApproximate','n_1', 'n_2', 'n_3', 'n_4', 'n_5','num_all_words', 'num_content_words', 'num_repetition',
 'consec_mean', 'consec_std','s0_mean','n_segment', 'senN_4', 'senN_3', 'senN_2', 'senN_1', 'N_fillers',
 'N_immediate_repetation', 'false_starts', 'self_corrections', 'length_utter', 'clause_density', 'dependency_distance',
 'content_function_ratio','type_token_ratio','average_word_frequency']

columns_include = ['ID','AgeScan1', 'PatientCat', 'Gender', 
 'TLI_IMPOV', 'TLI_DISORG','entropyApproximate','n_1', 'n_2', 'n_3', 'n_4', 'n_5','num_all_words', 'num_content_words', 'num_repetition',
 'consec_mean', 'consec_std','s0_mean','n_segment', 'senN_4', 'senN_3', 'senN_2', 'senN_1', 'N_fillers',
 'N_immediate_repetation', 'false_starts', 'self_corrections', 'length_utter', 'clause_density', 'dependency_distance',
 'content_function_ratio','type_token_ratio','average_word_frequency']

df_sel = df_avg[columns_include]
df_sel.dropna(inplace=True)

corr = df_sel.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# ------------------------------------------------------------------------
# combine features based on domain knowledge and correlation patterns
# Compute mean values
df_sel['mean_w2v'] = df_sel[['n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean(axis=1)
df_sel['mean_sensim'] = df_sel[['senN_1', 'senN_2', 'senN_3', 'senN_4']].mean(axis=1)
df_sel.drop(columns=['n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'senN_1', 'senN_2', 'senN_3', 'senN_4'], inplace=True)

# visulize transform data
corr = df_sel.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

df_cat = df_sel.groupby('PatientCat').mean().reset_index()

# visualize by patient category
plt.figure(figsize=(16, 12))
for i, col in enumerate(df_sel.columns[3:]):
    plt.subplot(5, 5, i + 1)
    sns.boxplot(x='PatientCat', y=col, data=df_sel)
    plt.title(col)
    plt.xlabel('Patient Category')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# visualization of outliers
df_sel.plot(kind='box', subplots=True, layout=(4,5), figsize=(20,10), sharex=False, sharey=False)


## --------------------------------------------------------------------
# Visualize and test effects
## --------------------------------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_directory)
outputfile_label = ['1min', 'spontaneous', 'concatenated']
for k in range(3):
    outputfile = outputfile_label[k]
    fname_all = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv')
    filtered_df = pd.read_csv(fname_all,index=False)
    
    filtered_df.dropna(inplace=True)
    r, p_value = pearsonr(df_sel['TLI_IMPOV'], df_sel['clause_density'])
    print(f'correlation between TLI and language estimation:'
        f'\ncorrelation {r},'
        f'\np value: {p_value}')
    sns.scatterplot(data=df_sel, x='TLI_IMPOV', y='clause_density')
    slope, intercept = np.polyfit(df_sel['TLI_IMPOV'], df_sel['clause_density'], 1)
    regression_line = slope * df_sel['TLI_IMPOV'] + intercept
    plt.plot(df_sel['TLI_IMPOV'], regression_line, color='red', label='Linear Regression')
    #plt.savefig('BERT_scatter_patients.eps', format='eps', bbox_inches='tight')
    plt.show()


## --------------------------------------------------------------------
# Data modeling: continous TLI scores
## --------------------------------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_directory)
outputfile_label = ['1min', 'spontaneous', 'concatenated']
outputfile = outputfile_label[1]
fname = os.path.join(parent_folder,'stimuli','TOPSY_All_' + outputfile + '.csv')
df = pd.read_csv(fname)
df_avg = df.drop(columns='stim').groupby('ID').mean().reset_index()
df_avg['mean_w2v'] = df_avg[['n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean(axis=1)
df_avg['mean_sensim'] = df_avg[['senN_1', 'senN_2', 'senN_3', 'senN_4']].mean(axis=1)
df_avg.drop(columns=['n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'senN_1', 'senN_2', 'senN_3', 'senN_4'], inplace=True)

# including all variables
columns_include = ['ID','AgeScan1', 'PatientCat', 'Gender', 
 'TLI_IMPOV', 'TLI_DISORG','entropyApproximate',
 'mean_w2v',
 'num_all_words', 'num_content_words', 'num_repetition',
 'consec_mean', 'consec_std','s0_mean','n_segment', 
 'mean_sensim', 'N_fillers',
 'N_immediate_repetation', 'false_starts', 'self_corrections', 'length_utter', 'clause_density', 'dependency_distance',
 'content_function_ratio','type_token_ratio','average_word_frequency']

# only including targeted variables
columns_include = ['ID','TLI_IMPOV', 'TLI_DISORG','mean_w2v',
 's0_mean','n_segment', 'mean_sensim', 
 'dependency_distance','clause_density','N_fillers',
 'num_content_words','type_token_ratio','average_word_frequency']
df_sel = df_avg[columns_include]
df_sel.dropna(inplace=True)
df_sel = df_sel[(df_sel['average_word_frequency']>4.5)]

# including demongraphic variables
columns_include = ['ID','AgeScan1', 'PatientCat', 'Gender', 
 'TLI_IMPOV', 'TLI_DISORG','mean_w2v',
 's0_mean','n_segment', 'mean_sensim', 
 'dependency_distance','clause_density','N_fillers',
 'num_content_words','type_token_ratio','average_word_frequency']
df_sel = df_avg[columns_include]
df_sel.dropna(inplace=True)
df_sel = df_sel[(df_sel['average_word_frequency']>4.5)]
#df_sel = df_sel[(df_sel['average_word_frequency']>4.5) & (df_sel['AgeScan1']<35)]
df_sel['Gender'] = df_sel['Gender'].map({1.0: 'M', 2.0: 'F'})
df_sel = pd.get_dummies(df_sel, columns=['Gender'], drop_first=True)

# visulize pairwise correlation
corr = df_sel.drop(columns=['ID']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# visualize distribution
df_sel.iloc[:,3:-2].hist()

# visualization of outliers
df_sel.iloc[:,3:-2].plot(kind='box', subplots=True, layout=(4,5), figsize=(20,10), sharex=False, sharey=False)

# Calculating VIF for each feature
#predictors = df_sel.drop(columns=['ID', 'PatientCat', 'TLI_IMPOV', 'TLI_DISORG', 'Gender_M']).columns
predictors = df_sel.drop(columns=['ID', 'TLI_IMPOV', 'TLI_DISORG']).columns
X = df_sel[predictors]
X = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data

sel_predictors = vif_data[vif_data['VIF'] < 10]['Feature'].tolist()


#-------------------------------------------
# relationship between s0_mean and TIL_DISORG
#--------------------------------------------
#df_sel = df_sel.loc[(df_sel['PatientCat']== 1) | (df_sel['PatientCat']== 2)]
df_plot = df_sel.groupby('ID')[['PatientCat','TLI_DISORG', 's0_mean']].mean().reset_index()
sns.scatterplot(data=df_plot, x='TLI_DISORG', y='s0_mean', hue='PatientCat', palette=['blue', 'red', 'yellow', 'green'])
#plt.savefig('BERT_TLI_DISORG_Entropy.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_sel, x='TLI_DISORG', y='s0_mean')
slope, intercept = np.polyfit(df_sel['TLI_DISORG'], df_sel['s0_mean'], 1)
regression_line = slope * df_sel['TLI_DISORG'] + intercept
plt.plot(df_sel['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
plt.savefig('/Users/linwang/Partners HealthCare Dropbox/Lin Wang-Meyer/OngoingProjects/sczTopic/plots/s0mean_pos.eps', format='eps')
plt.show()

# correlation
df_sel[['TLI_DISORG','s0_mean']].corr()
r, p_value = pearsonr(df_sel['TLI_DISORG'], df_sel['s0_mean'])
print(f'correlation between TLI and similarity for s0_mean:'
f'\ncorrelation {r},p value: {p_value}')

# categorical difference: HC vs. FEP
# test assumption for two sample t-test
group1 = df_sel.loc[(df_sel['PatientCat']== 1),'s0_mean'].values
group2 = df_sel.loc[(df_sel['PatientCat']== 2),'s0_mean'].values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(group1, bins=10, kde=True, color='blue', label='Group 1')
plt.title('Histogram of Group 1')
plt.legend()
plt.subplot(1, 2, 2)
sns.histplot(group2, bins=10, kde=True, color='green', label='Group 2')
plt.title('Histogram of Group 2')
plt.legend()
plt.tight_layout()
plt.show()

# test for normality
stat1, p_value1 = stats.shapiro(group1)
stat2, p_value2 = stats.shapiro(group2)
print(f'Shapiro-Wilk test for Group 1: Statistics={stat1}, p-value={p_value1}')
print(f'Shapiro-Wilk test for Group 2: Statistics={stat2}, p-value={p_value2}')

# test for equal variance
# Perform Levene's test for homogeneity of variances
stat_levene, p_value_levene = stats.levene(group1, group2)
print(f'Levene\'s test: Statistics={stat_levene}, p-value={p_value_levene}')

#t_statistic, p_value = stats.ttest_ind(group1, group2)
#Perform Mann-Whitney U test / Wilcoxon rank-sum (non-parametric test)
u_statistic, p_value = stats.mannwhitneyu(group1, group2)
print(f'word2vec similarity before word n and {col}:')
print(f'u-Statistic: {u_statistic}, P-Value: {p_value}')


#-------------------------------------------
# machine learning models
#--------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------------------------------------------------
# Get data for training and testing: data-driven features
#df_sel = df_sel.loc[(df_sel['PatientCat']== 1) | (df_sel['PatientCat']== 2)]
X =  df_sel[sel_predictors + ['Gender_M']].copy()
y_improv = df_sel['TLI_IMPOV']  # Target variable TLI_IMPOV
y_disorg = df_sel['TLI_DISORG']  # Target variable TLI_DISORG

# Split data into training and testing sets
X_train, X_test, y_train_improv, y_test_improv, y_train_disorg, y_test_disorg = \
    train_test_split(X, y_improv, y_disorg, test_size=0.2, random_state=42)

# Normalize data only for the numerial variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.iloc[:,:-1])
X_test_scaled = scaler.transform(X_test.iloc[:,:-1])

X_train_scaled = np.concatenate([X_train_scaled,X_train.iloc[:,-1].values.reshape(-1,1)],axis=1)
X_test_scaled = np.concatenate([X_test_scaled,X_test.iloc[:,-1].values.reshape(-1,1)],axis=1)

# ---------------------------------------------------------
# Get data for training and testing: hypothesis driven
X =  df_sel[sel_predictors]
y_improv = df_sel['TLI_IMPOV']  # Target variable TLI_IMPOV
y_disorg = df_sel['TLI_DISORG']  # Target variable TLI_DISORG

# Split data into training and testing sets
X_train, X_test, y_train_improv, y_test_improv, y_train_disorg, y_test_disorg = \
    train_test_split(X, y_improv, y_disorg, test_size=0.2, random_state=42)

# Normalize data only for the numerial variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#---------------
# Use Lasso regression to reduce irrelevant features
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Create Lasso regression models
lasso_improv = Lasso()
lasso_disorg = Lasso()
param_grid_improv = {'alpha': np.logspace(-4, 1, 100)}
param_grid_disorg = {'alpha': np.logspace(-4, 1, 100)}

grid_improv = GridSearchCV(estimator=lasso_improv, param_grid=param_grid_improv, scoring='r2', cv=5)
grid_improv.fit(X_train_scaled, y_train_improv)
grid_disorg = GridSearchCV(estimator=lasso_disorg, param_grid=param_grid_disorg, scoring='r2', cv=5)
grid_disorg.fit(X_train_scaled, y_train_disorg)

# Print best parameters and best scores
print("Best parameters for IMPOV:", grid_improv.best_params_)
print("Best R² score for IMPOV:", grid_improv.best_score_)
print("Best parameters for DISORG:", grid_disorg.best_params_)
print("Best R² score for DISORG:", grid_disorg.best_score_)

# Retrieve best models
best_lasso_improv = grid_improv.best_estimator_
best_lasso_disorg = grid_disorg.best_estimator_

# Predict on test set using best models
y_pred_improv = best_lasso_improv.predict(X_test_scaled)
y_pred_disorg = best_lasso_disorg.predict(X_test_scaled)

# check assumptions
residual_analysis(y_test_improv, y_pred_improv)
residual_analysis(y_test_disorg, y_pred_disorg)

# Evaluate model performance
mse_improv = mean_absolute_error(y_test_improv, y_pred_improv)
mse_disorg = mean_absolute_error(y_test_disorg, y_pred_disorg)
r2_improv = r2_score(y_test_improv, y_pred_improv)
r2_disorg = r2_score(y_test_disorg, y_pred_disorg)
print(f"Mean Absolute Error (IMPOV): {mse_improv}")
print(f"Mean Absolute Error (DISORG): {mse_disorg}")
print(f"R2 (IMPOV): {r2_improv}")
print(f"R2 (DISORG): {r2_disorg}")

# Retrieve coefficients and feature names
coefficients_improv = best_lasso_improv.coef_
coefficients_disorg = best_lasso_disorg.coef_
feature_names = X.columns
abs_coef_improv = np.abs(coefficients_improv)
abs_coef_disorg = np.abs(coefficients_disorg)
sorted_indices_improv = np.argsort(abs_coef_improv)[::-1]
sorted_indices_disorg = np.argsort(abs_coef_disorg)[::-1]
sorted_feature_names_improv = feature_names[sorted_indices_improv]
sorted_feature_names_disorg = feature_names[sorted_indices_disorg]
sorted_coef_improv = coefficients_improv[sorted_indices_improv]
sorted_coef_disorg = coefficients_disorg[sorted_indices_disorg]

# Plot feature importance for TLI_IMPOV
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names_improv, sorted_coef_improv)
plt.xlabel('Coefficient Magnitude')
plt.title('Feature Importance for TLI_IMPOV')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()

# Plot feature importance for TLI_DISORG
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names_disorg, sorted_coef_disorg)
plt.xlabel('Coefficient Magnitude')
plt.title('Feature Importance for TLI_DISORG')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


#------------------------
# random forest: based on the above selected features
features_improv = feature_names[abs_coef_improv > 0.01]
features_disorg = feature_names[abs_coef_disorg > 0.01]

X_improv = df_sel[features_improv]
y_improv = df_sel['TLI_IMPOV']

X_disorg = df_sel[features_disorg]
y_disorg = df_sel['TLI_DISORG']

# Split data into training and testing sets
X_train_improv, X_test_improv, y_train_improv, y_test_improv = \
    train_test_split(X_improv, y_improv, test_size=0.2, random_state=42)

X_train_disorg, X_test_disorg, y_train_disorg, y_test_disorg = \
    train_test_split(X_disorg, y_disorg, test_size=0.2, random_state=42)
    
# Normalize data
scaler = StandardScaler()
X_train_improv_scaled = scaler.fit_transform(X_train_improv)
X_test_improv_scaled = scaler.transform(X_test_improv)

X_train_disorg_scaled = scaler.fit_transform(X_train_disorg)
X_test_disorg_scaled = scaler.transform(X_test_disorg)

# run the model
param_grid = {
    'n_estimators': [30, 50, 80],
    'max_depth': [None, 5, 8],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [1, 2, 4]
}
# Grid Search for TLI_IMPOV model
grid_search_improv = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search_improv.fit(X_train_improv_scaled, y_train_improv)

# Grid Search for TLI_DISORG model
grid_search_disorg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search_disorg.fit(X_train_disorg_scaled, y_train_disorg)

# Get the best models from Grid Search
best_model_improv = grid_search_improv.best_estimator_
best_model_disorg = grid_search_disorg.best_estimator_

# Evaluate TLI_IMPOV predictions
improv_predictions = best_model_improv.predict(X_test_improv_scaled)
mse_improv = mean_absolute_error(y_test_improv, improv_predictions)
r2_improv = r2_score(y_test_improv, improv_predictions)
print("TLI_IMPOV Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mse_improv:.2f}")
print(f"R-squared (R2): {r2_improv:.2f}")

# Evaluate TLI_DISORG predictions
disorg_predictions = best_model_disorg.predict(X_test_disorg_scaled)
mse_disorg = mean_absolute_error(y_test_disorg, disorg_predictions)
r2_disorg = r2_score(y_test_disorg, disorg_predictions)
print("\nTLI_DISORG Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mse_disorg:.2f}")
print(f"R-squared (R2): {r2_disorg:.2f}")

# Feature selection based on their importance
feature_importance = best_model_improv.feature_importances_
feature_importance_dict = dict(zip(X_improv.columns, feature_importance))
sorted_features_improv = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_features_improv

feature_importance = best_model_disorg.feature_importances_
feature_importance_dict = dict(zip(X_disorg.columns, feature_importance))
sorted_features_disorg = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_features_disorg



#----------------------------
# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

X = df_sel[['mean_sensim', 'type_token_ratio',
       'average_word_frequency', 'mean_w2v', 's0_mean', 'N_fillers', 'dependency_distance']]
y_improv = df_sel['TLI_IMPOV']  # Target variable TLI_IMPOV
y_disorg = df_sel['TLI_DISORG']  # Target variable TLI_DISORG

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = LinearRegression()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Calculate cross-validated scores
scores = cross_val_score(model, X_scaled, y_improv, cv=kf, scoring='r2')

# Fit the model on the entire dataset to get coefficients
model.fit(X_scaled, y_improv)

# Get feature importance (coefficients)
importance = model.coef_

# Print the results
print(f'Mean cross-validated R^2 score: {np.mean(scores)}')
print('Feature importances (coefficients):')
for col, coef in zip(X.columns, importance):
    print(f'{col}: {coef}')
    
    

#----------------------------
# Ridge regression with cross-validation
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Initialize Ridge Regression model with standard scaling
model_improv = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Calculate cross-validated scores
scores = cross_val_score(model_improv, X, y_improv, cv=kf, scoring='r2')

# Print cross-validated scores
print(f'Cross-validated R^2 scores: {scores}')
print(f'Mean cross-validated R^2 score: {np.mean(scores)}')

# Train the model on the entire training data
model_improv.fit(X_train, y_train_improv)

# Get coefficients (importance) and corresponding feature names
coefficients = model_improv.named_steps['ridge'].coef_
feature_names = X.columns

# Pair feature names with coefficients
feature_importance = dict(zip(feature_names, coefficients))

# Sort features based on absolute coefficient values
sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

# Print sorted feature importance
print("Feature importances (sorted by absolute value):")
for feature, importance in sorted_features:
    print(f"Feature: {feature}, Coefficient: {importance:.4f}")




## --------------------------------------------------------------------
# Data modeling: categorical group labels
## --------------------------------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_directory)
outputfile_label = ['1min', 'spontaneous', 'concatenated']
outputfile = outputfile_label[1]
fname = os.path.join(parent_folder,'stimuli','TOPSY_All_' + outputfile + '.csv')
df = pd.read_csv(fname)
df_avg = df.drop(columns='stim').groupby('ID').mean().reset_index()
df_avg['mean_w2v'] = df_avg[['n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean(axis=1)
df_avg['mean_sensim'] = df_avg[['senN_1', 'senN_2', 'senN_3', 'senN_4']].mean(axis=1)
df_avg.drop(columns=['n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'senN_1', 'senN_2', 'senN_3', 'senN_4'], inplace=True)

# only including targeted variables
columns_include = ['ID', 'PatientCat', 'TLI_IMPOV', 'TLI_DISORG','mean_w2v',
 's0_mean','n_segment', 'mean_sensim', 
 'dependency_distance','clause_density','N_fillers',
 'num_content_words','type_token_ratio','average_word_frequency']
df_sel = df_avg[columns_include]
df_sel.dropna(inplace=True)
df_sel = df_sel[(df_sel['average_word_frequency']>4.5)]

# including demongraphic variables
columns_include = ['ID','AgeScan1', 'PatientCat', 'Gender', 
 'TLI_IMPOV', 'TLI_DISORG','mean_w2v',
 's0_mean','n_segment', 'mean_sensim', 
 'dependency_distance','clause_density','N_fillers',
 'num_content_words','type_token_ratio','average_word_frequency']
df_sel = df_avg[columns_include]
df_sel.dropna(inplace=True)
df_sel = df_sel[(df_sel['average_word_frequency']>4.5)]
#df_sel = df_sel[(df_sel['average_word_frequency']>4.5) & (df_sel['AgeScan1']<35)]
df_sel['Gender'] = df_sel['Gender'].map({1.0: 'M', 2.0: 'F'})
df_sel = pd.get_dummies(df_sel, columns=['Gender'], drop_first=True)

# visulize pairwise correlation
corr = df_sel.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


#----------------------------
# Random forest, over sampling to deal with imbalanced data

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

df_sel = df_sel[(df_sel['PatientCat']==1) | (df_sel['PatientCat']==2)]
df_sel['PatientCat'] = df_sel['PatientCat'].apply(lambda x: x - 1)

# Define features and target
X = df_sel.drop(columns = ['PatientCat','ID','TLI_IMPOV', 'TLI_DISORG'], axis=1)
y = df_sel['PatientCat']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the model with class weights
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#------------------
# Logistic regression to identify important variables
from sklearn.linear_model import LogisticRegression

# Split pre-processed data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# run regression
#logreg = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# examine the importance of predictors
coefficients = logreg.coef_[0]
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coefficients
})
importance_df['Absolute_Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Absolute_Coefficient', ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at the top
plt.show()

# Make predictions on the transformed testing set
y_pred = logreg.predict(X_test_scaled)

# evaluate model performance
model_evaluate(y_test,y_pred,show_metric=True)



# ------------------------------------------------------------------------------
# Use AUC (Area Under the ROC Curve) to identify the optiona threshold
# ------------------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Get predicted probabilities for positive class
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Get AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC score: {auc}')

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate F1 scores for each threshold
f1_scores = []
for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred))

# Find the threshold that maximizes the F1 score
optimal_threshold = thresholds[np.argmax(f1_scores)]
f_max = np.max(f1_scores)
print(f'At optional threshold {optimal_threshold}, the maximal F1-score is {f_max}')

# visualize the AUC: it the performance of the model across various threshold settings.
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# evaluate model performance
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
model_evaluate(y_test,y_pred_optimal,show_metric=True,best_params=optimal_threshold)


#------------------------------
# logistic regression, gridsearchCV
# get importance rank for the variables
#------------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y,random_state=42)

# Logistic Regression with Elastic Net
model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'l1_ratio': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],  # Mix of L1 and L2 regularization
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Regularization strength
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Model evaluation
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = grid_search.cv_results_['mean_test_score']
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())


# Check if coefficients are zero
coefficients = best_model.coef_[0]
if np.all(coefficients == 0):
    print("All coefficients are zero. This indicates too strong regularization.")
else:
    # Extracting and sorting coefficients
    feature_names = X.columns
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute Coefficient': np.abs(coefficients)
    })

    # Sorting features by absolute coefficient value
    coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)

    print("\nFeature Coefficients Sorted by Importance:")
    print(coef_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=coef_df)
    plt.title('Feature Importance (Elastic Net Coefficients)')
    plt.show()
    
    

# --------------------------------------------------
# pca on all variables: select the top n components
# --------------------------------------------------
# get data
df_avg = df.drop(columns='stim').groupby('ID').mean().reset_index()
df_avg.dropna(subset=['n_1'],inplace=True)
df_X = df_avg.drop(columns=['ID', 'AgeScan1', 'PatientCat', 'Gender', 'SES', 'PANSS Tot',
       'PANSS Neg', 'PANSS Pos', 'PANSS_p2', 'Trails-B', 'DSST_Writen',
       'DSST_Oral', 'Category Fluency (animals)', 'DUP_weeks', 'SOFAS',
       'TLI_IMPOV', 'TLI_DISORG', 'TLITOTA;'])
df_y = df_avg['PatientCat'].apply(lambda x: x-1)

# Normalize data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_X)

# Perform PCA
pca = PCA()
pca.fit(scaled_features)

# Get the components that explain >95% variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_explained_variance >= 0.9) + 1 #15 components

# Perform PCA with the chosen number of components
pca = PCA(n_components=num_components)
pca_features = pca.fit_transform(scaled_features)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(f'Number of component selected: {num_components} components')
print(f'Top three components explained variance: {explained_variance_ratio}')
print(f'Total explained variance: {cumulative_explained_variance[-1]}')

# visualize explained variance for the first two components
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df_y, cmap='viridis', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of Transformed Data')
plt.show()


# --------------------------------------------------
# xgboost
# --------------------------------------------------
import xgboost as xgb

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': len(df_sel['PatientCat'].unique()),  # number of classes
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}

num_round = 50  # Number of boosting rounds
model = xgb.train(params, dtrain, num_round)

y_pred = model.predict(dtest)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# --------------------------------------------------
# decision tree
# --------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

# Train and test
dt_classifier = DecisionTreeClassifier(max_depth=5)
dt_classifier.fit(X_train, y_train)

# Get feature importances
importances = dt_classifier.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance - Decision Tree')
plt.tight_layout()
plt.show()
