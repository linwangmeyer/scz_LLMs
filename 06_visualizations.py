
import json
import pandas as pd
import numpy as np
import re
import os
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
sns.set(style="whitegrid")


## --------------------------------------------------------------------
# Load results of individual measure and put them together
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'

fname = os.path.join(parent_folder,'approximate_entropy.csv')
df_app = pd.read_csv(fname)

fname = os.path.join(parent_folder,'transform_entropy.csv')
df_transform = pd.read_csv(fname)

fname = os.path.join(parent_folder,'similarity_entropy.csv')
df_sim = pd.read_csv(fname)

fname = os.path.join(parent_folder,'n_sentence.csv')
df_nsen = pd.read_csv(fname)

fname = os.path.join(parent_folder,'word2vec.csv')
df_w2v = pd.read_csv(fname)

fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df_var = pd.read_csv(fname_var)

df_measures = df_app.merge(df_transform, on=['ID','stim']).merge(df_sim, on=['ID','stim']).merge(df_nsen, on=['ID','stim']).merge(df_w2v, on=['ID','stim'])

df_merge = df_var.merge(df_measures, on='ID',how='outer')
filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')

fname_all = os.path.join(parent_folder,'TOPSY_all.csv')
filtered_df.to_csv(fname_all,index=False)

df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]
fname_goi = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df_goi.to_csv(fname_goi,index=False)


## --------------------------------------------------------------------
# Load final results and plot
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df = pd.read_csv(fname)

## --------------------------------------------------------------------
# Visualize Entropy
## --------------------------------------------------------------------
plot_path = os.path.join('/'.join(parent_folder.split('/')[:-2]),'plots')

## --------------------------------------------------------------------
# Group difference: voilin plot for entropyApproximate
palette = [(1, 0, 0, 0.2), (0, 0, 1, 0.2)]  # Red and blue with 0.5 alpha
ax = sns.violinplot(x='PatientCat', y='entropyApproximate', data=df, palette=palette)
ax.set_xticks([0, 1]) 
ax.set_xticklabels(['Controls', 'Patients']) 
plt.xlabel('Patient Category')
plt.ylabel('Entropy Approximate')
plt.title('Entropy Approximate by Patient Category')
plot_path = os.path.join('/'.join(parent_folder.split('/')[:-2]),'plots')
plt.savefig(os.path.join(plot_path,'Entropy_PatientCat_violinplot'))
plt.show()


## --------------------------------------------------------------------
# Relationship between LTI and entropy: only within patient group
filtered_df = df.loc[df['PatientCat'] == 2,['PatientCat','TLI_DISORG','stim','entropyApproximate']]
filtered_df_keep = filtered_df.dropna(subset=['TLI_DISORG', 'entropyApproximate'])
r, p_value = pearsonr(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'])
print(f'correlation between TLI and Approximate Entropy estimation:'
      f'\nr = {r}, p= {p_value}')
#r =  0.37398886723554053, p = 2.432861039150624e-08

sns.scatterplot(data=filtered_df_keep, x='TLI_DISORG', y='entropyApproximate', color='red',size=100)
slope, intercept = np.polyfit(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'], 1)
regression_line = slope * filtered_df_keep['TLI_DISORG'] + intercept
plt.plot(filtered_df_keep['TLI_DISORG'], regression_line, color='black')
plt.savefig(os.path.join(plot_path,'Entropy_LTI_patient_scatterplot'))
plt.show()

# Relationship between LTI and entropy: both groups
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID','PatientCat','TLI_DISORG','stim','entropyApproximate']]
filtered_df_keep = filtered_df.dropna(subset=['TLI_DISORG', 'entropyApproximate'])
r, p_value = pearsonr(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'])
print(f'correlation between TLI and Approximate Entropy estimation:'
      f'\nr = {r}, p= {p_value}')
#r = 0.2915755596526078, p= 1.8039175648830606e-07

df_plot = filtered_df_keep.groupby('ID')[['PatientCat','TLI_DISORG','entropyApproximate']].mean().reset_index()
scatter = sns.scatterplot(data=df_plot, x='TLI_DISORG', y='entropyApproximate', hue='PatientCat', palette={1.0: 'blue', 2.0: 'red'})
scatter.legend(handles=scatter.legend_.legendHandles, labels=['Controls', 'Patients'], bbox_to_anchor=(1, 0), loc='lower right')
plt.xlabel('TLI_DISORG')
plt.ylabel('Entropy Approximate')
plt.title('correlation between TLI and Approximate Entropy estimation of two groups')
plt.legend(title='Category')
plt.savefig(os.path.join(plot_path,'Entropy_LTI_twoGroups_scatterplot'))
plt.show()


## --------------------------------------------------------------------
# Visualize word2vec similarity
## --------------------------------------------------------------------
columns_to_melt = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
melted_df = pd.melt(df[['PatientCat'] + columns_to_melt], id_vars=['PatientCat'], value_vars=columns_to_melt, var_name='wordpos', value_name='w2v')

# Volin plot
g = sns.FacetGrid(melted_df, col='wordpos', height=5, aspect=1.2, sharey=False)
g.map_dataframe(sns.violinplot, x='PatientCat', y='w2v', split=True, inner='quart', palette='Set1')
g.set_xticklabels(['Controls', 'Patients'])
g.set_axis_labels('Patient Category', 'w2v Value')
g.set_titles(col_template='{col_name} - wordpos')
g.fig.suptitle('Semantic similarity by Patient Category for Different word positions', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plot_path,'W2v_PatientCat_wordpos_violinplot'))
plt.show()


# Line plot with CI
plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_df, x='wordpos', y='w2v', hue='PatientCat',
             palette={1.0: 'blue', 2.0: 'red'})
plt.xlabel('Word Position')
plt.ylabel('w2v Values')
plt.title('semantic similarity by Word Position and PatientCat')
plt.legend(title='PatientCat')
plt.tight_layout()
plt.savefig(os.path.join(plot_path,'W2v_PatientCat_wordpos_lineplot'))
plt.show()

# values
melted_df.groupby(['PatientCat','wordpos'])['w2v'].agg(['mean','sem'])
