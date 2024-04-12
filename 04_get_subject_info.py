
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
# Compare between two groups
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
df['DSST_total'] = df[['DSST_Writen','DSST_Oral']].mean(axis=1)
include_var = ['ID', 'AgeScan1', 'PatientCat', 'Gender', 'SES', 'PANSS Tot',
       'PANSS Neg', 'PANSS Pos', 'Trails-B', 'DSST_total','Category Fluency (animals)',
       'TLI_IMPOV', 'TLI_DISORG', 'TLITOTA;', 'entropyApproximate',
       'n_sentence', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5',
       'num_all_words','num_content_words','num_repeated_words']
df_keep = df[include_var]

df_stimavg = df_keep.groupby('ID').mean()
df_stimavg.groupby(['PatientCat']).count() # 36 HC, 74 FEP

# remove subjects without TLItotal score: 34 HC, 70 FEP
df_stimavg[~df_stimavg['TLITOTA;'].isna()].groupby(['PatientCat']).count()
df_stat = df_stimavg[~df_stimavg['TLITOTA;'].isna()]
mean_values = df_stat.groupby('PatientCat')[['AgeScan1', 'PatientCat', 'SES', 'PANSS Tot', 'PANSS Neg',
       'PANSS Pos', 'Trails-B', 'DSST_total', 'Category Fluency (animals)',
       'TLI_IMPOV', 'TLI_DISORG', 'TLITOTA;',
       'n_sentence','num_all_words','num_content_words','num_repeated_words']].mean().round(2)
std_values = df_stat.groupby('PatientCat')[['AgeScan1', 'PatientCat', 'SES', 'PANSS Tot', 'PANSS Neg',
       'PANSS Pos', 'Trails-B', 'DSST_total', 'Category Fluency (animals)',
       'TLI_IMPOV', 'TLI_DISORG', 'TLITOTA;',
       'n_sentence','num_all_words','num_content_words','num_repeated_words']].std().round(2)

# get the mean (std) values for the measures
mean_values.transpose()
std_values.transpose()

# get number of participants in each category
df_stat[['PatientCat','Gender']].groupby(['PatientCat','Gender']).size()
#PatientCat  Gender
#1.0         1.0       22
#2.0       12
#2.0         1.0       56
#            2.0       14


## --------------------------------------------------------------------
# Statistical test on the variables between patients (2) vs. controls (1)
## --------------------------------------------------------------------
# List of variables of interest
variables_of_interest = ['AgeScan1', 'Gender', 'SES', 'PANSS Tot', 'PANSS Neg',
       'PANSS Pos', 'Trails-B', 'DSST_total', 'Category Fluency (animals)',
       'TLI_IMPOV', 'TLI_DISORG', 'TLITOTA;', 'entropyApproximate',
       'n_sentence', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'num_all_words',
       'num_content_words', 'num_repeated_words']

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