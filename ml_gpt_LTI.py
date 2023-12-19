import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID', 'PatientCat', 'TLI_DISORG', 'stim', 'n_sentence', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_avg = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','n_sentence','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()

# load data from Tori
df_gpt = pd.read_csv(parent_folder+'SczProd_gpt3_data_clean.csv')

# groupby subject
columns_to_keep = ['Subject'] + [f'cloze_{i}wContext_gpt3' for i in range(2, 51)] + ['Utterance_Length_Wrds']
df_selected = df_gpt[columns_to_keep]
df_gpt_avg = df_selected.groupby(['Subject']).mean()

# combine data
df_ml = df_avg.merge(df_gpt_avg,left_on='ID',right_on='Subject')
df_ml.dropna(inplace=True)

# ------------------------------------------
# check and visualize data
# ------------------------------------------
# check data
df_ml.isna().sum()
scz = (df_ml['PatientCat']==1).sum()
hc = (df_ml['PatientCat']==2).sum()
print(f'number of patients: {scz}')
print(f'number of healthy controls: {hc}')

# group effect
r, p_value = pearsonr(df_ml['PatientCat'], df_ml['cloze_50wContext_gpt3'])
print(f'correlation between Patient category and cloze_50wContext_gpt3:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')

# visualize correlation matrix
df_roi = df_ml
correlation_matrix = df_roi.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Pairwise Correlation Heatmap")
plt.show()

# visualize group difference: scatter plot
sns.scatterplot(data=df_roi, x='PatientCat', y='cloze_50wContext_gpt3')
slope, intercept = np.polyfit(df_roi['PatientCat'], df_roi['cloze_50wContext_gpt3'], 1)
regression_line = slope * df_roi['PatientCat'] + intercept
plt.plot(df_roi['PatientCat'], regression_line, color='red', label='Linear Regression')


# visualize group difference: box plot
plt.figure()
df_roi.boxplot(column='cloze_50wContext_gpt3', by='PatientCat')
plt.show()


# --------------------------------------------------
# pca on gpt-values: select the top n components
# --------------------------------------------------
# only use gpt-cloze data
columns_roi = [f'cloze_{i}wContext_gpt3' for i in range(2, 51)]
df_X = df_ml[columns_roi]
df_ml['PatientCat'] = df_ml['PatientCat'].map({1.0: 0, 2.0: 1})
df_y = df_ml['PatientCat']

# Normalize data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_X)

# Perform PCA
pca = PCA()
pca.fit(scaled_features)

# Get the components that explain >95% variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

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


# ------------------------------------------
# ------------------------------------------
########### Linear regression ##########
# ------------------------------------------
# ------------------------------------------

# ------------------------------------------
# linear regression on pca component
# ------------------------------------------
columns_roi = [f'cloze_{i}wContext_gpt3' for i in range(2, 51)]
df_X = df_ml[columns_roi]
df_y = df_ml['TLI_DISORG']

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA on the normalized data
num_components = 1
pca = PCA(n_components=num_components)
pca.fit(X_train_scaled)

# Transform the training and testing sets using PCA
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train linear regression on the transformed training set
linreg = LinearRegression()
linreg.fit(X_train_pca, y_train)

# Make predictions on the transformed testing set
y_pred = linreg.predict(X_test_pca)

# Evaluate the performance of the linear regression model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')


#----------------------
# cross validation
#----------------------
df_y = df_ml['TLI_DISORG']

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(scaled_features)

# Perform PCA on the normalized data
num_components = 1
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train_scaled)

# Create a linear regression model
model = LinearRegression()
cv_scores = cross_val_score(model, X_train_pca, df_y, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE scores to positive and calculate the mean
mse_scores = -cv_scores
average_mse = mse_scores.mean()

print(f'Cross-validated MSE scores: {mse_scores}')
print(f'Average MSE: {average_mse}')


# ------------------------------------------
# linear regression on baseline predictor
# ------------------------------------------
df_X = df_ml['Utterance_Length_Wrds']
df_y = df_ml['TLI_DISORG']

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values.reshape(-1,1))
X_test_scaled = scaler.transform(X_test.values.reshape(-1,1))

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
# ------------------------------------------
########### Classification ##########
# ------------------------------------------
# ------------------------------------------

# ------------------------------------------
# logistic regression
# ------------------------------------------
# Split pre-processed data
df_ml = df_avg.merge(df_gpt_avg,left_on='ID',right_on='Subject')
df_ml.dropna(inplace=True)
df_ml['PatientCat'] = df_ml['PatientCat'].map({1.0: 0, 2.0: 1})

columns_roi = [f'cloze_{i}wContext_gpt3' for i in range(2, 51)]
df_X = df_ml[columns_roi]

df_y = df_ml['PatientCat']
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, stratify=df_y, random_state=123)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA on the normalized data
num_components = 1
pca = PCA(n_components=num_components)
pca.fit(X_train_scaled)

# Transform the training and testing sets using PCA
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train logistic regression on the transformed training set
logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

# Make predictions on the transformed testing set
y_pred = logreg.predict(X_test_pca)

# evaluate model performance
accuracy, cf_matrix, precision, recall, f1 = model_evaluate(y_test,y_pred,show_metric=True)


# ------------------------------------------------------------------------------
# Use AUC (Area Under the ROC Curve) to identify the optiona threshold
# ------------------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_pca, y_train)

# Get predicted probabilities for positive class
y_pred_proba = logreg.predict_proba(X_test_pca)[:, 1]

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


# ------------------------------------------
# support vector machine
# ------------------------------------------
def model_svm(X_train,X_test,y_train,y_test,show_metric=True):
    '''Build and evaluate SVM model on normalized data'''
    
    param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 'scale'],
    'kernel': ['linear', 'rbf']
    }
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)
    
    # evaluate model performance
    accuracy, cf_matrix, precision, recall, f1 = model_evaluate(y_test,y_pred,show_metric=show_metric,best_params=best_params)
    
    return best_model, accuracy, cf_matrix, precision, recall, f1

best_model, accuracy, cf_matrix, precision, recall, f1 = model_svm(X_train_pca,X_test_pca,y_train,y_test)



# ------------------------------------------
# linear discriminant analysis
# ------------------------------------------
def model_lda(X_train, X_test, y_train, y_test, show_metric=True):
    '''Build and evaluate LDA model on the data'''
    
    param_grid = {}
    lda = LinearDiscriminantAnalysis(solver='svd')
    grid_search = GridSearchCV(lda, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)

    # evaluate model performance
    accuracy, cf_matrix, precision, recall, f1 = model_evaluate(y_test,y_pred,show_metric=show_metric,best_params=best_params)

    return best_model, accuracy, cf_matrix, precision, recall, f1

best_model, accuracy, cf_matrix, precision, recall, f1 = model_lda(X_train_pca,X_test_pca,y_train,y_test)



# ------------------------------------------
# naive bays
# ------------------------------------------
def model_naivebayes(X_train,X_test,y_train,y_test,show_metric=True):
    '''Build and evaluate Naive Bayes model on normalized data'''
    
    param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)

    # evaluate model performance
    accuracy, cf_matrix, precision, recall, f1 = model_evaluate(y_test,y_pred,show_metric=show_metric,best_params=best_params)
    
    return best_model, accuracy, cf_matrix, precision, recall, f1

best_model, accuracy, cf_matrix, precision, recall, f1 = model_naivebayes(X_train_pca,X_test_pca,y_train,y_test)