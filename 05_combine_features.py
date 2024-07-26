## --------------------------------------------------------------------
# Combine all variables and explore the data patterns
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


def check_vif(df):
    # Use VIF to check collinearity
    columns_to_drop = ['PatientCat', 'TLI_IMPOV', 'TLI_DISORG', 'Gender_M', 'PANSS_Neg', 'PANSS_Pos']
    X = df.drop(columns=columns_to_drop, errors='ignore').dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

## --------------------------------------------------------------------
# Combine all measures with subject info
## --------------------------------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.join(os.path.dirname(current_directory),'stimuli','Relabeld','analysis')
outputfile_label = ['1min', 'spontaneous', 'concatenated']
for k in range(3):
    outputfile = outputfile_label[k]

    fname_var = os.path.join(os.path.dirname(current_directory),'stimuli','variables','sub_variables.csv')
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

## --------------------------------------------------------------------
# EDA
## --------------------------------------------------------------------
outputfile = 'spontaneous'
fname = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv')
df_spon = pd.read_csv(fname)

outputfile = '1min'
fname = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv')
df_1min = pd.read_csv(fname)

outputfile = 'concatenated'
fname = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv')
df_concat = pd.read_csv(fname)

# average across three stimuli
df_avg = df_spon.drop(columns='stim').groupby('ID').mean().reset_index()

# ------------------------------------------------------------------------
# Check missing values
missing_percentage = df_avg.isnull().mean() * 100
plt.figure(figsize=(12, 8))
missing_percentage.plot(kind='bar')
plt.title('Percentage of Missing Values', fontsize=16)
plt.xlabel('Variables', fontsize=14)
plt.ylabel('Percentage of Missing Values (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.savefig(os.path.join(parent_folder,'plots','01_EDA_MissingValues.png'), format='png', bbox_inches='tight')
plt.show()

# Only include columns of interest
columns_include = ['ID','Age','PatientCat', 'Gender', 'PANSS_Neg', 'PANSS_Pos', 'TLI_IMPOV', 'TLI_DISORG',
 'entropyApproximate','n_1', 'n_2', 'n_3', 'n_4', 'n_5','num_all_words', 'num_content_words', 'num_repetition',
 'consec_mean','s0_mean','n_segment', 'senN_4', 'senN_3', 'senN_2', 'senN_1', 
 'N_fillers', 'N_immediate_repetation', 'false_starts', 'self_corrections', 'length_utter', 'clause_density', 'dependency_distance',
 'content_function_ratio','type_token_ratio','average_word_frequency']

df_sel = df_avg[columns_include]

# ------------------------------------------------------------------------
# check data distribution and identify outliers
plot_cols = df_sel.columns.tolist()[1:]
df_melted = df_sel[plot_cols].melt(var_name='Variable', value_name='Value')
g = sns.FacetGrid(df_melted, col='Variable', col_wrap=6, height=4, sharex=False, sharey=False)
g.map(sns.histplot, 'Value')  # Plotting violin plot for each variable
g.set_titles(col_template="{col_name}")
plt.savefig(os.path.join(parent_folder,'plots','02_EDA_DistributionOutlier.png'), format='png', bbox_inches='tight')
plt.show()

# remove rows with outlier values
df_sel = df_sel[(df_sel['average_word_frequency']> 4.5) 
                & (df_sel['N_fillers'] < 20) 
                & (df_sel['content_function_ratio'] <= 2.0)]

# remove variable with skewed distribution
df_sel.drop(columns=['N_immediate_repetation'], inplace=True)

# ------------------------------------------------------------------------
# visualize pairwise correlation for feature selection
corr = df_sel.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Pairwise correlation: All variables')
plt.savefig(os.path.join(parent_folder,'plots','03_EDA_PairwiseRawVars.png'), format='png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------
# combine features based on domain knowledge and correlation patterns
df_sel['mean_w2v'] = df_sel[['n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean(axis=1)
df_sel['mean_sensim'] = df_sel[['senN_1', 'senN_2', 'senN_3', 'senN_4']].mean(axis=1)
df_sel.drop(columns=['n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'senN_1', 'senN_2', 'senN_3', 'senN_4','ID'], errors='ignore', inplace=True)

# Check VIF
vif_data = check_vif(df_sel)
keep_features = vif_data[vif_data['VIF'] < 10]['Feature'].tolist()
df_sel = df_sel[keep_features + ['TLI_IMPOV', 'TLI_DISORG', 'PANSS_Neg', 'PANSS_Pos','PatientCat']]

# visulize data after selecting variables based on VIF
corr = df_sel.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Pairwise correlation: Selected variables')
plt.savefig(os.path.join(parent_folder,'plots','04_EDA_PairwiseNewVars.png'), format='png', bbox_inches='tight')
plt.show()

# df_sel.drop(columns=['num_repetition','entropyApproximate','false_starts','self_corrections','mean_sensim'], errors='ignore',inplace=True)

# ------------------------------------------------------------------------
# save cleaned data
outfile = 'spontaneous'
fname = os.path.join(parent_folder,'clean_all_' + outputfile + '.csv')
df_sel.to_csv(fname, index=False)

# ------------------------------------------------------------------------
# visualize linear relationship between variables
list_x = ['PANSS_Neg', 'PANSS_Pos', 'TLI_IMPOV', 'TLI_DISORG']
list_y = list(set(df_sel.columns.tolist()) - set(list_x) - set(['Gender','Age','PatientCat']))
num_x = len(list_x)
num_y = len(list_y)
fig, axes = plt.subplots(nrows=num_x, ncols=num_y, figsize=(30, 15), constrained_layout=True)
for i, x in enumerate(list_x):
    for j, y in enumerate(list_y):
        ax = axes[i, j]
        sns.scatterplot(data=df_sel, x=x, y=y, ax=ax)
        ax.set_title(f'{x} vs {y}', fontsize=8)
        ax.set_xlabel(x, fontsize=8)
        ax.set_ylabel(y, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder,'plots','05_EDA_pairplot_continuousVars.png'), format='png', bbox_inches='tight')
plt.show()


# ------------------------------------------------------------------------
# visualize by patient category
df_cat = df_sel.groupby('PatientCat').mean().reset_index()
plt.figure(figsize=(16, 12))
for i, col in enumerate(df_sel.columns[2:-1]):
    plt.subplot(5, 5, i + 1)
    sns.boxplot(x='PatientCat', y=col, data=df_sel)
    plt.title(col)
    plt.xlabel('Patient Category')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder,'plots','06_EDA_byPateintCategory.png'), format='png', bbox_inches='tight')
plt.show()


## --------------------------------------------------------------------
# Visualize and test effects
## --------------------------------------------------------------------
df_sel.dropna(subset=['TLI_IMPOV','dependency_distance'],inplace=True)
r, p_value = pearsonr(df_sel['TLI_IMPOV'], df_sel['dependency_distance'])
print(f'#correlation between TLI_IMPOV and dependency_distance:'
    f'\n#correlation {np.round(r,2)},p value: {np.round(p_value,4)}')
#correlation between TLI_IMPOV and dependency_distance:
#correlation -0.26,p value: 0.0022
sns.scatterplot(data=df_sel, x='TLI_IMPOV', y='dependency_distance')
slope, intercept = np.polyfit(df_sel['TLI_IMPOV'], df_sel['dependency_distance'], 1)
regression_line = slope * df_sel['TLI_IMPOV'] + intercept
plt.plot(df_sel['TLI_IMPOV'], regression_line, color='red', label='Linear Regression')
plt.savefig(os.path.join(parent_folder,'plots','scatter_IMPOV_Syntax.png'), format='png', bbox_inches='tight')
plt.show()


#---------------------------------------------------------------
# Statistical tests: relationship between s0_mean and TIL_DISORG
#----------------------------------------------------------------
df_plot = df_sel[['PatientCat','TLI_DISORG', 's0_mean']]
sns.scatterplot(data=df_plot, x='TLI_DISORG', y='s0_mean', hue='PatientCat', palette=['blue', 'red', 'yellow', 'green'])
plt.savefig(os.path.join(parent_folder,'plots','scatter_DISORG_s0mean_PatientCat.png'), format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_sel, x='TLI_DISORG', y='s0_mean')
slope, intercept = np.polyfit(df_sel['TLI_DISORG'], df_sel['s0_mean'], 1)
regression_line = slope * df_sel['TLI_DISORG'] + intercept
plt.plot(df_sel['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
plt.savefig(os.path.join(parent_folder,'plots','scatter_DISORG_s0mean.png'), format='png', bbox_inches='tight')
plt.show()

# correlation
df_sel[['TLI_DISORG','s0_mean']].corr()
r, p_value = pearsonr(df_sel['TLI_DISORG'], df_sel['s0_mean'])
print(f'#correlation between TLI_DISORG and s0_mean:'
f'\n#correlation {np.round(r,2)},p value: {np.round(p_value,4)}')
#correlation between TLI_DISORG and s0_mean:
#correlation -0.29,p value: 0.0006

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
print(f'#Shapiro-Wilk test for Group 1: Statistics={np.round(stat1,2)}, p-value={np.round(p_value,4)}')
print(f'#Shapiro-Wilk test for Group 2: Statistics={np.round(stat2,2)}, p-value={np.round(p_value2,4)}')
#Shapiro-Wilk test for Group 1: Statistics=0.97, p-value=0.0008
#Shapiro-Wilk test for Group 2: Statistics=0.99, p-value=0.6933

# test for equal variance
# Perform Levene's test for homogeneity of variances
stat_levene, p_value_levene = stats.levene(group1, group2)
print(f'#Levene\'s test: Statistics={np.round(stat_levene,2)}, p-value={np.round(p_value_levene,4)}')
#Levene's test: Statistics=6.23, p-value=0.0142

#t_statistic, p_value = stats.ttest_ind(group1, group2)
#Perform Mann-Whitney U test / Wilcoxon rank-sum (non-parametric test)
u_statistic, p_value = stats.mannwhitneyu(group1, group2)
print(f'#word2vec similarity before word n and {col}:')
print(f'#u-Statistic: {np.round(u_statistic,2)}, P-Value: {np.round(p_value,4)}')
#word2vec similarity before word n and mean_sensim:
#u-Statistic: 1599.0, P-Value: 0.0005

