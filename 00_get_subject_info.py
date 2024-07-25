
# Get demoncratic info
import json
import pandas as pd
import numpy as np
import nltk
import re
import os
import string
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

## --------------------------------------------------------------------
# Load data
## --------------------------------------------------------------------
# Merge old and new subject variables
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.join(os.path.dirname(current_directory),'stimuli','variables')
fname1 = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df1 = pd.read_csv(fname1)
df1['DSST_total'] = df1[['DSST_Writen','DSST_Oral']].mean(axis=1)
columns = df1.columns.tolist()
dsst_total_idx = columns.index('DSST_total')
dsst_oral_idx = columns.index('DSST_Oral')
columns.insert(dsst_oral_idx + 1, columns.pop(dsst_total_idx))
df1 = df1[columns]

fname2 = os.path.join(parent_folder,'updaate topsy summary.xlsx')
df2 = pd.read_excel(fname2)
df2 = df2[df2.columns[:4]]

# merge two files
df = df1.merge(df2,how='left', left_on='ID', right_on='ID participant')
df.drop(columns=['ID participant',' category at baseline'],inplace=True)

#rename variables
new_var = ['ID', 'Age', 'PatientCat', 'Gender', 'SES', 'PANSS_Total','PANSS_Neg', 'PANSS_Pos', 'PANSS_p2',
           'Trails_B', 'DSST_Writen', 'DSST_Oral','DSST_Total','SemFluency','DUP_weeks','SOFAS',
           'TLI_IMPOV', 'TLI_DISORG', 'T', 'PatientLabel_Old', 'PatientLabel_New']
df.columns = new_var

#rename old labels
label_mapping = {
    1: 'HC',
    2: 'FEP',
    3: 'chronic',
    4: 'CHR'
}
df['PatientLabel_Old'] = df['PatientCat'].map(label_mapping)

# rename new labels
df['PatientLabel_New'].replace({
       'SZ': 'scz',
       'Sz': 'scz',
       'Schizoaffective': 'sczAffective',
       'schizoaffective': 'sczAffective',
       'Schizophreniform': 'sczForm'
       }, inplace=True)
df['PatientLabel_New'].unique()


df_FEP = df.loc[df['PatientCat']==2,['ID', 'Age', 'PatientCat','SOFAS','TLI_Total','PatientLabel_Old','PatientLabel_New']]
df_FEP.groupby(['PatientLabel_New']).count()

df.groupby(['PatientCat']).count() # 36 HC, 74 FEP, 16 chronic, 19 CHR

fname = os.path.join(parent_folder,'sub_variables.csv')
df.to_csv(fname,index=False)

# remove subjects without TLItotal score: 34 HC, 70 FEP
df[~df['TLI_Total'].isna()].groupby(['PatientCat']).count()
df_stat = df[~df['TLI_Total'].isna()]
mean_values = df_stat.groupby('PatientCat')[['Age', 'PatientCat', 'SES', 'PANSS_Total', 'PANSS_Neg',
       'PANSS_Pos', 'Trails_B', 'DSST_Total', 'SemFluency',
       'TLI_IMPOV', 'TLI_DISORG', 'TLI_Total','SOFAS']].mean().round(2)
std_values = df_stat.groupby('PatientCat')[['Age', 'PatientCat', 'SES', 'PANSS_Total', 'PANSS_Neg',
       'PANSS_Pos', 'Trails_B', 'DSST_Total', 'SemFluency',
       'TLI_IMPOV', 'TLI_DISORG', 'TLI_Total','SOFAS']].std().round(2)

# get the mean (std) values for the measures
mean_values.transpose()
std_values.transpose()

# get number of participants in each category
df_stat[['PatientCat','Gender']].groupby(['PatientCat','Gender']).size()
#PatientCat  Gender
#1.0         1.0       22
#            2.0       12
#2.0         1.0       56
#            2.0       14


## --------------------------------------------------------------------
# Statistical test on the variables between patients (2) vs. controls (1)
## --------------------------------------------------------------------
# List of variables of interest
variables_of_interest = ['Age', 'Gender', 'SES', 'PANSS_Total', 'PANSS_Neg',
       'PANSS_Pos', 'Trails_B', 'DSST_Total', 'SemFluency',
       'TLI_IMPOV', 'TLI_DISORG', 'TLI_Total','SOFAS']

# Perform two-sample t-test for each variable
stat_result = {}
for variable in variables_of_interest:
    # Split the DataFrame into two groups based on the 'PatientCat' column
    controls = df_stat[df_stat['PatientCat'] == 1.0][variable]
    patients = df_stat[df_stat['PatientCat'] == 2.0][variable]
    
    t_statistic, p_value = ttest_ind(controls, patients, nan_policy='omit')
    
    dof_controls = np.sum(~np.isnan(controls)) - 1
    dof_patients = np.sum(~np.isnan(patients)) - 1
    dof = dof_controls + dof_patients
    
    stat_result[variable] = [t_statistic, p_value, dof]

all_results = [{'variable': key,
                'Tval': value[0].round(2),
                'Pval': value[1].round(2),
                'dof': value[2]}
               for key, value in stat_result.items()]
df = pd.DataFrame(all_results)
df

### Test the distribution of the genders
# Define the observed counts: Controls (female, male); Patients (female, male)
observed_counts = [
    [12, 22],
    [14, 56]
]
chi2_stat, p_val, dof, expected_counts = chi2_contingency(observed_counts)
print("Chi-square statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected counts:")
print(expected_counts)