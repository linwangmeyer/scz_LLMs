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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind

# ------------------------------------------
# load data
# ------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_all.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)

filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID', 'PatientCat', 'TLI_DISORG','stim', 'n_sentence', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_avg = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','n_sentence','entropyApproximate','n_1','n_2','n_3','n_4','n_5']].mean().reset_index()

# load data from Alban
df_md = pd.read_csv(parent_folder+'TOPSY_Database_csv.csv')

# combine data
df_cmb = df_avg.merge(df_md)

# --------------------------------------------------
# get data from patients with medication
df_med = df_cmb[df_cmb['TotalDefinedDailyDose@Scan'].notna()]

similarity_columns = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for col in similarity_columns:
    r, p_value = pearsonr(df_med['CGIS'], df_med[col])
    print(f'correlation between CGIS and similarity for {col}:'
        f'\ncorrelation {r},p value: {p_value}')
    
r, p_value = pearsonr(df_med['CGIS'],df_med['entropyApproximate'])
print(f'correlation between CGIS and w2v:'
    f'\ncorrelation {r},p value: {p_value}')


# --------------------------------------------------
# compare patients with vs. without medication
df_med = df_cmb[df_cmb['TotalDefinedDailyDose@Scan'].notna()]
df_nomed = df_cmb[(df_cmb['TotalDefinedDailyDose@Scan'].isna()) & (df_cmb['PatientCat']==2.0)]

df_med[['entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']].describe().mean()
df_nomed[['entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']].describe().mean()

columns_of_interest = ['entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for column in columns_of_interest:
    t_stat, p_value = ttest_ind(df_med[column].dropna(), df_nomed[column].dropna(), equal_var=False)
    print(f"#T-test for '{column}': t = {t_stat}, p-value = {p_value}")
#T-test for 'entropyApproximate': t = 0.8095885410584575, p-value = 0.4210471189140075
#T-test for 'n_1': t = -0.8594388215323234, p-value = 0.39317118848675225
#T-test for 'n_2': t = -1.0401403786508103, p-value = 0.30204655189372037
#T-test for 'n_3': t = -0.45723978699924644, p-value = 0.6489972295996397
#T-test for 'n_4': t = -1.4679116624429978, p-value = 0.1469607968475763
#T-test for 'n_5': t = -1.0572447184208569, p-value = 0.2942226399472683



# --------------------------------------------------
# relate to CGIS (Clinical Global Impressions Scale)

# group difference
t_stat, p_value = ttest_ind(df_cmb.loc[df_cmb['PatientCat']==1.0,'CGIS'].dropna(), df_cmb.loc[df_cmb['PatientCat']==2.0,'CGIS'].dropna(), equal_var=False)
print(f"#T-test for CGIS between patient groups': t = {t_stat}, p-value = {p_value}")

# correlate with LLM measures
df_select = df_cmb[df_cmb['CGIS'].notna()]
columns_of_interest = ['TLI_DISORG','entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for column in columns_of_interest:
    r, p_value = pearsonr(df_select['TLI_DISORG'],df_select[column])
    print(f'correlation between CGIS and {column}:'
        f'\ncorrelation {r},p value: {p_value}')


# --------------------------------------------------
# relate to YMRS1 (Young Mania Rating Scale)
YMRS_cols = [f'YMRS{i}' for i in range(1,12)]
df_YMRS = df_cmb.loc[df_cmb[YMRS_cols].notna(),YMRS_cols].mean(axis=1)
ymrs = df_cmb[YMRS_cols].sum(axis=1)

# correlate with LLM measures
columns_of_interest = ['TLI_DISORG','entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for column in columns_of_interest:
    r, p_value = pearsonr(ymrs,df_cmb[column])
    print(f'correlation between ymrs and {column}:'
        f'\ncorrelation {r},p value: {p_value}')


# --------------------------------------------------
# relate to BNSS (Brief Negative Symptom Scale)
BNSS_cols = [f'BNSS{i}' for i in range(1,14)]
bnss = df_cmb[BNSS_cols].sum(axis=1)

# correlate with LLM measures
columns_of_interest = ['TLI_DISORG','entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for column in columns_of_interest:
    r, p_value = pearsonr(bnss,df_cmb[column])
    print(f'correlation between bnss and {column}:'
        f'\ncorrelation {r},p value: {p_value}')
    

# --------------------------------------------------
# relate to CDS (Calgary Depression Scale for Schizophrenia)
CDS_cols = [f'CDS{i}' for i in range(1,10)]
cds = df_cmb[CDS_cols].sum(axis=1)

# correlate with LLM measures
columns_of_interest = ['TLI_DISORG','entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for column in columns_of_interest:
    r, p_value = pearsonr(cds,df_cmb[column])
    print(f'correlation between cds and {column}:'
        f'\ncorrelation {r},p value: {p_value}')

df_clinical = pd.DataFrame({'Mania': ymrs,
                       'NegSymp': bnss,
                       'DepressScz': cds})

# --------------------------------------------------
# combine all clinical and LLM measures
df_all = pd.concat([df_cmb[['ID', 'PatientCat', 'TLI_DISORG', 'n_sentence', 'entropyApproximate',
       'n_1', 'n_2', 'n_3', 'n_4', 'n_5',
       'TotalDefinedDailyDose@Scan','CGIS']],df_clinical],axis=1)
fname = os.path.join(parent_folder,'TOPSY_TwoGroups_alldata.csv')
df_all.to_csv(fname,index=False)



# --------------------------------------------------
# load all available data
fname = os.path.join(parent_folder,'TOPSY_TwoGroups_alldata.csv')
df_all.to_csv(fname,index=False)

correlation_matrix = df_all.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

