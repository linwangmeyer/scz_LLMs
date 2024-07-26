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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

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

# Get data with or without Gender as a predictor
df_sel = preprocess_data_cat(df.copy(), include_gender=True)

# Get variables of interest
sel_predictors = df_sel.drop(columns=['PatientCat']).columns.tolist()
#sel_predictors = ['s0_mean', 'mean_w2v', 'clause_density', 'content_function_ratio']
target_variables = ['PatientCat']   
df_sel = df_sel[sel_predictors+target_variables].dropna()

# Get training and testing datasets
X = df_sel[sel_predictors]
y = df_sel[target_variables]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize training data
X_train_scaled, X_test_scaled = normalize_data(X_train,X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


#------------------------------------------------------------
# Random forest, over sampling to deal with imbalanced data
#------------------------------------------------------------
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

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
plt.savefig(os.path.join(parent_folder,'plots','ML_03_RandomForest_PatientCat_beta.png'), format='png')
plt.show()

# Perform cross-validation to evaluate accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")


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
grid_search.fit(X_train_resampled, y_train_resampled)
print(f"Best parameters: {grid_search.best_params_}")

# Use the best estimator for predictions
best_logreg = grid_search.best_estimator_
y_pred = best_logreg.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

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
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_model = grid_search.best_estimator_

# Model evaluation
y_pred = best_model.predict(X_test_scaled)
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
    sns.barplot(x='Absolute Coefficient', y='Feature', data=coef_df)
    plt.title('Feature Importance (Elastic Net Coefficients)')
    plt.savefig(os.path.join(parent_folder,'plots','ML_03_Classification_beta.png'), format='png', bbox_inches='tight')
    plt.show()
    



# ------------------------------------------------------------------------------
# Use AUC (Area Under the ROC Curve) to identify the optiona threshold
# ------------------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_resampled, y_train_resampled)

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
df_X = df_sel.drop(columns=['Gender_M', 'PatientCat'])
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

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

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
