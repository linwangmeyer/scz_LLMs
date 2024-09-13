## --------------------------------------------------------------------
# Data modeling: categorical group labels
## --------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score, recall_score, f1_score)
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from psmpy import PsmPy
import scipy.stats as stats

## --------------------------------------------------------------------
# Some functions
## --------------------------------------------------------------------
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


# Get all critical metrics
def get_precision_recall_f1_g_measure(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = np.round(float(precision_score(y_test, y_pred)),4)
    recall = np.round(float(recall_score(y_test, y_pred)),4)
    f1 = np.round(float(f1_score(y_test, y_pred)),4)
    g_measure = np.round(float(np.sqrt(precision * recall)),4)
    return accuracy, precision, recall, f1, g_measure,confusion_matrix(y_test, y_pred)



def preprocess_data_cat(df, include_gender=True):
    # Drop unnecessary columns
    columns_to_drop = ['ID', 'TLI_IMPOV','TLI_DISORG','PANSS_Neg','PANSS_Pos']
    
    if not include_gender:
        columns_to_drop.append('Gender')
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # only include HC and FEP
    df = df[(df['PatientCat']==1.0) | (df['PatientCat']==2.0)]
    df['PatientCat'] = df['PatientCat'].map({1.0: 0, 2.0: 1})
    
    # Include Gender as a dummy variable if specified
    if include_gender:
        df['Gender'] = df['Gender'].map({1.0: 'M', 2.0: 'F'})
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    return df


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

# --------------------------------------------
# Get data with oversampling
#--------------------------------------------
# Get data with or without Gender as a predictor
df_sel = preprocess_data_cat(df.copy(), include_gender=True)

# Get variables of interest
sel_predictors = df_sel.drop(columns=['PatientCat']).columns.tolist()
#sel_predictors = ['s0_mean', 'mean_w2v', 'clause_density', 'content_function_ratio']
target_variables = ['PatientCat']   
df_sel = df_sel[sel_predictors+target_variables].dropna()

# Get training and testing datasets
X = df_sel[sel_predictors[1:-1]]
y = df_sel[target_variables]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize training data
X_train_scaled, X_test_scaled = normalize_data(X_train,X_test)

#comment out if you don't want to use SMOTE to deal with imbalanced data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#------------------------------------------------------------
# Random forest, over sampling to deal with imbalanced data
#------------------------------------------------------------
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

# get performance metrics
accuracy, precision, recall, f1, g_measure,cm = get_precision_recall_f1_g_measure(y_test, y_pred)
print(f"#accuracy: {accuracy:.2f}")
print(f"#precision: {precision:.2f}")
print(f"#recall: {recall:.2f}")
print(f"#f1: {f1:.2f}")
print(f"#G-measure: {g_measure:.2f}")
print(f'#Confusion Matrix: \n{cm}')
#accuracy: 0.85
#precision: 0.86
#recall: 0.92
#f1: 0.89
#G-measure: 0.89
#Confusion Matrix: 
#[[ 5  2]
# [ 1 12]]

# Check feature importances
importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest Classifier')
plt.gca().invert_yaxis()
plt.savefig(os.path.join(parent_folder,'plots','ML_03_RandomForest_PatientCat_beta.eps'), format='eps', dpi=300)
plt.show()

# Check the direction of the feature importances
features = importance_df['Feature'].to_list()

means_by_patientcat = df_sel[features].groupby(df_sel['PatientCat']).mean()
means_cat_0 = means_by_patientcat.loc[0]
means_cat_1 = means_by_patientcat.loc[1]
features_with_larger_mean_cat_1 = means_cat_1[means_cat_1 > means_cat_0].index.tolist()
features_with_larger_mean_cat_1

# Boxplot to visualize the difference
n_cols = 4
n_rows = (len(features) + n_cols - 1) // n_cols  # Calculate the number of rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
fig.tight_layout(pad=4.0)
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(x='PatientCat', y=col, data=df_sel, ax=axes[i])
    axes[i].set_title(f'{col}')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.show()

# --------------------------------------------------
# Use L1 regularized logisticto identify important variables
# --------------------------------------------------
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers that support L1 regularization
    'penalty': ['l1'],  # L1 regularization
    'max_iter': [100, 500, 1000]  # Number of iterations
}

logreg = LogisticRegression()
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Use the best estimator for predictions
best_logreg = grid_search.best_estimator_
y_pred = best_logreg.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# get performance metrics
accuracy, precision, recall, f1, g_measure,cm = get_precision_recall_f1_g_measure(y_test, y_pred)
print(f"#accuracy: {accuracy:.2f}")
print(f"#precision: {precision:.2f}")
print(f"#recall: {recall:.2f}")
print(f"#f1: {f1:.2f}")
print(f"#G-measure: {g_measure:.2f}")
print(f'#Confusion Matrix: \n{cm}')
#accuracy: 0.65
#precision: 0.75
#recall: 0.69
#f1: 0.72
#G-measure: 0.72
#Confusion Matrix: 
#[[4 3]
# [4 9]]

# Examine the importance of predictors
coefficients = best_logreg.coef_[0]
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
plt.gca().invert_yaxis() 
plt.savefig(os.path.join(parent_folder,'plots','ML_04_L1LogisticRegression_PatientCat_beta.png'), format='png')
plt.show()



#-----------------------------------------------------
# logistic regression Elastic Net, gridsearchCV
# get importance rank for the variables
#-----------------------------------------------------
model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)
param_grid = {
    'l1_ratio': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],  # Mix of L1 and L2 regularization
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]  # Regularization strength
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Model evaluation
y_pred = best_model.predict(X_test_scaled)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# get performance metrics
accuracy, precision, recall, f1, g_measure,cm = get_precision_recall_f1_g_measure(y_test, y_pred)
print(f"#accuracy: {accuracy:.2f}")
print(f"#precision: {precision:.2f}")
print(f"#recall: {recall:.2f}")
print(f"#f1: {f1:.2f}")
print(f"#G-measure: {g_measure:.2f}")
print(f'#Confusion Matrix: \n{cm}')
#accuracy: 0.75
#precision: 0.83
#recall: 0.77
#f1: 0.80
#G-measure: 0.80
#Confusion Matrix: 
#[[ 5  2]
# [ 3 10]]

# Check if coefficients are zero
coefficients = best_model.coef_[0]
if ~np.all(coefficients == 0):
    # Extracting and sorting coefficients
    feature_names = X_train.columns
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
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title('Feature Importance (Elastic Net Coefficients)')
    plt.savefig(os.path.join(parent_folder,'plots','ML_07_ElasticNet_PatientCat_beta.png'), format='png', bbox_inches='tight')
    plt.show()
else:
    print("All coefficients are zero. This indicates too strong regularization.")

# ------------------------------------------------------------------------------
# Naive Bayes
# ------------------------------------------------------------------------------
# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train,)
y_pred = gnb.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------------------
# SVM
# ------------------------------------------------------------------------------
# For linearly kernal
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)

# For RBF kernel
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42, gamma='scale')

# For polynomial kernel
svm_model = SVC(kernel='poly', degree=3, class_weight='balanced', random_state=42, gamma='scale')

svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Define the parameter grid for RBF kernel
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
# Define the parameter grid for Polynomial kernel
param_grid_poly = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4, 5],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['poly']
}

# Combine parameter grids into one list
param_grid = [param_grid_rbf, param_grid_poly]
grid = GridSearchCV(SVC(class_weight='balanced', random_state=42), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_scaled, y_train)
y_pred_grid = grid.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred_grid))


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

print(classification_report(y_test, y_pred_optimal))
print(confusion_matrix(y_test, y_pred_optimal))
    
    

# --------------------------------------------------
# pca on all variables: select the top n components
# --------------------------------------------------
# get data
df_X = df_sel.drop(columns=['PatientCat'])
df_y = y

# Normalize data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_X)

# Perform PCA
pca = PCA()
pca.fit(scaled_features)

# Get the components that explain >95% variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_explained_variance >= 0.9) + 1 #10 components

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
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df_y.squeeze(), cmap='viridis', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of Transformed Data')
plt.show()


# --------------------------------------------------
# xgboost
# --------------------------------------------------
import xgboost as xgb

# Initialize the XGBoost classifier
model = xgb.XGBClassifier(random_state=42)

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Use GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print a confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# decision tree
# --------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

# Train and test
dt_classifier = DecisionTreeClassifier(max_depth=5)
dt_classifier.fit(X_train_scaled, y_train)

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



# --------------------------------------------
# Get subset of data with propensity scores
#--------------------------------------------
df['PatientCat'] = df['PatientCat'].astype('category')
df.dropna(subset=['PatientCat','mean_sensim'], inplace=True)
df_sel = df[(df['PatientCat']==1.0) | (df['PatientCat']==2.0)]
df_sel['PatientCat'] = df_sel['PatientCat'].map({1.0: 0, 2.0: 1})
df_sel.drop(columns=['TLI_IMPOV', 'TLI_DISORG', 'PANSS_Neg', 'PANSS_Pos'], inplace=True)

# propensity score matching
df_sel['index'] = df_sel.index
psm = PsmPy(df_sel, treatment='PatientCat', indx='index', exclude = ['entropyApproximate', 'num_repetition', 'consec_mean', 's0_mean',
       'n_segment', 'N_fillers', 'false_starts', 'self_corrections',
       'clause_density', 'dependency_distance', 'content_function_ratio',
       'type_token_ratio', 'average_word_frequency', 'mean_w2v', 'mean_sensim'])
psm.logistic_ps(balance=True)
psm.knn_matched(matcher='propensity_score', replacement=False, caliper=None)
matched_df = psm.df_matched

# check number of data point for each Patient
print(matched_df.groupby('PatientCat').size())

# Get data from df_sel based on the index of matched_df
df_sel = df_sel.loc[matched_df['index']]
df_sel.drop(columns=['index','Age', 'Gender'], inplace=True, errors='ignore')

# statistically compare age and gender between two patientCat group
print(df_sel.groupby('PatientCat')[['Age', 'Gender']].describe())
t_statistic, p_value = stats.ttest_ind(df_sel.loc[df_sel['PatientCat']== 1,'Gender'].values,df_sel.loc[df_sel['PatientCat']== 0,'Gender'].values)
print(f'#One-sample t-test: Statistics={np.round(t_statistic,2)}, p-value={np.round(p_value,4)}')

# Get training and testing datasets
X = df_sel.drop(columns=['PatientCat'])
y = df_sel['PatientCat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize training data
X_train_scaled, X_test_scaled = normalize_data(X_train,X_test)
