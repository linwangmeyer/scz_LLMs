import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

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
fname_var = os.path.join(parent_folder,'TOPSY_all_1min.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
'''filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2), 
                         ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'num_all_words', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_avg = filtered_df.groupby('ID')[['PatientCat','PANSS Pos','TLI_DISORG','num_all_words','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()
'''

filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID', 'PatientCat', 'stim', 'num_all_words', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_avg = filtered_df.groupby('ID')[['PatientCat','num_all_words','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()

# load data from Tori
df_gpt = pd.read_csv(parent_folder+'slopes_without_controls_forLin.csv')

# combine data
df_ml = df_avg.merge(df_gpt,left_on='ID',right_on='Subject')

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
r, p_value = pearsonr(df_ml['PatientCat'], df_ml['slopes_without_control_vars'])
print(f'correlation between Patient category and slopes_without_control_vars:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')

# visualize correlation matrix
df_roi = df_ml[['entropyApproximate','slopes_without_control_vars','PatientCat','n_1','n_2','n_3','n_4','n_5']]
correlation_matrix = df_roi.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Pairwise Correlation Heatmap")
plt.show()

# visualize group difference: scatter plot
sns.scatterplot(data=df_roi, x='PatientCat', y='slopes_without_control_vars')
slope, intercept = np.polyfit(df_roi['PatientCat'], df_roi['slopes_without_control_vars'], 1)
regression_line = slope * df_roi['PatientCat'] + intercept
plt.plot(df_roi['PatientCat'], regression_line, color='red', label='Linear Regression')


# visualize group difference: box plot
plt.figure()
df_roi.boxplot(column='slopes_without_control_vars', by='PatientCat')
plt.show()


# ------------------------------------------
# logistic regression
# ------------------------------------------
# get critical variables
df_ml = df_avg.merge(df_gpt,left_on='ID',right_on='Subject')
df_ml['PatientCat'] = df_ml['PatientCat'].map({1.0: 0, 2.0: 1})
df_X = df_ml[['slopes_without_control_vars','entropyApproximate','n_1','n_2','n_3','n_4','n_5']]
df_y = df_ml['PatientCat']

# Split pre-processed data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, stratify=df_y, random_state=123)
# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# run regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

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
model_evaluate(y_test,y_pred,show_metric=True,best_params=optimal_threshold)


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

best_model, accuracy, cf_matrix, precision, recall, f1 = model_svm(X_train_scaled,X_test_scaled,y_train,y_test)



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

best_model, accuracy, cf_matrix, precision, recall, f1 = model_lda(X_train_scaled,X_test_scaled,y_train,y_test)



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

best_model, accuracy, cf_matrix, precision, recall, f1 = model_naivebayes(X_train_scaled,X_test_scaled,y_train,y_test)