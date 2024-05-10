from utils import read_data_fromtxt, calculate_entropy_app, cal_entropy_weighted, calculate_entropy_similarity
import json
import math
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy import stats
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

## --------------------------------------------------------------------
# Load pre-trained models
## --------------------------------------------------------------------
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#topic_model = BERTopic.load("davanstrien/chat_topics")
# for a list of pre-trained topics, see: https://huggingface.co/models?library=bertopic&sort=downloads

## --------------------------------------------------------------------
# Read data
## --------------------------------------------------------------------
# Get list of folder and file names
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
child_folders = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
text_file_list = []
folder_file = {}
for child_folder in child_folders:
    child_folder_path = os.path.join(parent_folder, child_folder)
    text_files = [file for file in os.listdir(child_folder_path) if file.endswith(".txt")]
    folder_file[child_folder] = text_files

## --------------------------------------------------------------------
# Get data and conduct analysis:
file_list = []
sentence_count_list = []
word_count_list = []
window=30
topic_entropy_app = [] #the approxiamte distribution approach, moving window similarity
topic_entropy_trans = [] #Get dominant topic and its associated probability for each sentence
topic_sim = [] #dominant topic pairwise similarity
topic_entropy_sim = [] #find_topic based on similarity
top_n = 2376
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname)
        stim_all = [stim[key] for key in stim if key.startswith('P')] #select all responses from the patient
        stim_full = ' '.join(stim_all)

        # get file name
        file_list.append(filename.split('.')[0][6:])
        
        # approximate approach
        appdistr_topic, _ = topic_model.approximate_distribution(stim_full,window=window,use_embedding_model=True)
        entropy_val = calculate_entropy_app(appdistr_topic[0])
        topic_entropy_app.append(entropy_val)
        
        # dominant topic and then calculate entropy
        stim_sen = sent_tokenize(stim_full)
        domtopic, prob = topic_model.transform(stim_sen)
        entropy_transform = cal_entropy_weighted(domtopic,prob)
        topic_entropy_trans.append(entropy_transform)
        
        # get pairwise cosine similarity between identified topic embeddings normalized by number of sentences
        sim_matrix = cosine_similarity(np.array(topic_model.topic_embeddings_)[domtopic,:])
        labels = [topic_model.get_topic_info(label).Name.to_list()[0] for label in domtopic]
        blow_index = np.tril_indices(sim_matrix.shape[0], k=-1)
        mean_sim = np.mean(sim_matrix[blow_index])
        topic_sim.append(mean_sim)
        
        # Use find_topic to get similarity/probability of the whole input
        _, similarity = topic_model.find_topics(stim_full,top_n=top_n)
        entropy_val = calculate_entropy_similarity(similarity)
        topic_entropy_sim.append(entropy_val)
        
        # Get number of sentences, number of words information
        n_sentence = len(stim_sen)
        n_word = len(stim_full.split())
        
        sentence_count_list.append(n_sentence)
        word_count_list.append(n_word)

df = pd.DataFrame({
    'filename': file_list,
    'entropyApproximate': topic_entropy_app,
    'entropyTransform': topic_entropy_trans,
    'TransformSimilarity': topic_sim,
    'entropySimilarity': topic_entropy_sim,
    'nsen': sentence_count_list,
    'nword': word_count_list
})

df[['ID', 'stim']] = df['filename'].str.split('_', expand=True)
df.drop(columns=['filename'], inplace=True)

fname = os.path.join(parent_folder,'topic_measures_concatenated.csv')
df.to_csv(fname,index=False)


## --------------------------------------------------------------------
# Load results and statistical tests
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname = os.path.join(parent_folder,'topic_measures_concatenated.csv')
df = pd.read_csv(fname)
df['ID'] = df['ID'].astype('int64')

fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df_var = pd.read_csv(fname_var)

df_merge = df_var.merge(df,on='ID',how='outer')

filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')
fname_all = os.path.join(parent_folder,'TOPSY_all_concatenated.csv')
filtered_df.to_csv(fname_all,index=False)

df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]
fname_goi = os.path.join(parent_folder,'TOPSY_TwoGroups_concatenated.csv')
df_goi.to_csv(fname_goi,index=False)


## --------------------------------------------------------------------
# Load results and statistical tests
## --------------------------------------------------------------------

## --------------------------------------------------------------------
# Relationship between LTI and number of word
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups_concatenated.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
df['PatientCat'] = df['PatientCat'].astype('int')
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID','PatientCat','TLI_DISORG','TLI_IMPOV','nword','stim','entropyApproximate',]]
filtered_df.dropna(inplace=True)
r, p_value = pearsonr(filtered_df['TLI_DISORG'], filtered_df['entropyApproximate'])
print(f'correlation between TLI and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
sns.scatterplot(data=filtered_df, x='TLI_DISORG', y='entropyApproximate')
slope, intercept = np.polyfit(filtered_df['TLI_DISORG'], filtered_df['entropyApproximate'], 1)
regression_line = slope * filtered_df['TLI_DISORG'] + intercept
plt.plot(filtered_df['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
plt.savefig('BERT_scatter_patients.eps', format='eps', bbox_inches='tight')
plt.show()

# Scatter plot, color coding patient group
df_plot = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','nword','entropyApproximate']].mean().reset_index()
sns.scatterplot(data=df_plot, x='TLI_DISORG', y='entropyApproximate', hue='PatientCat', palette=['blue', 'red'])
plt.savefig('BERT_TLI_DISORG_Entropy.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_plot, x='TLI_DISORG', y='nword', hue='PatientCat', palette=['blue', 'red'])
plt.savefig('BERT_TLI_DISORG_nwords.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_plot, x='nword', y='entropyApproximate', hue='PatientCat', palette=['blue', 'red'])
plt.savefig('BERT_nwords_Entropy.png', format='png', bbox_inches='tight')
plt.show()


## --------------------------------------------------------------------
# Compare between two groups
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)

filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['PatientCat','TLI_DISORG','entropyTransform','TransformSimilarity','entropyApproximate','entropySimilarity','n_sentence']]
filtered_df.dropna(inplace=True)
df_ctrl = filtered_df[filtered_df['PatientCat']==1]
df_scz = filtered_df.loc[(df['PatientCat'] == 2) & (filtered_df['TLI_DISORG']>2)]

t_statistic, p_value = stats.ttest_ind(df_ctrl['entropyTransform'].values, df_scz['entropyTransform'].values)
u_statistic, p_value = stats.mannwhitneyu(df_ctrl['entropyTransform'].values, df_scz['entropyTransform'].values)


r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['entropyTransform'])
print(f'correlation between Patient category and Transform Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['entropyApproximate'])
print(f'correlation between Patient category and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['TransformSimilarity'])
print(f'correlation between Patient category and Approximate Topic Similarity estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['entropySimilarity'])
print(f'correlation between Patient category and Similarity-based Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
filtered_df.groupby('PatientCat').agg(['mean', 'std'])
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['n_sentence'])
print(f'correlation between Patient category and Number of sentence:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
filtered_df.groupby('PatientCat').agg(['mean', 'std'])
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['entropyApproximate'])
print(f'correlation between Patient category and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')

# Violin plot of entropy approximate by patient category
palette = [(1, 0, 0, 0.2), (0, 0, 1, 0.2)]  # Red and blue with 0.5 alpha
ax = sns.violinplot(x='PatientCat', y='entropyApproximate', data=filtered_df, palette=palette)
ax.set_xticks([0, 1])  # Set the ticks at positions 0 and 1
ax.set_xticklabels(['Controls', 'Patients'])  # Set custom tick labels
plt.xlabel('Patient Category')
plt.ylabel('Entropy Approximate')
plt.title('Entropy Approximate by Patient Category')
plt.show()

# Box plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
filtered_df.boxplot(column='entropyApproximate', by='PatientCat', ax=axes[0,0])
axes[0,0].set_title('Approximate Entropy')
axes[0,0].set_xlabel('Category')
axes[0,0].set_ylabel('Entropy')
axes[0,0].set_xticklabels(['Controls', 'Patients'])  # Set x-axis tick labels

filtered_df.boxplot(column='entropyTransform', by='PatientCat', ax=axes[0,1])
axes[0,1].set_title('Transform Entropy')
axes[0,1].set_xlabel('Category')
axes[0,1].set_ylabel('Entropy')
axes[0,1].set_xticklabels(['Controls', 'Patients'])  # Set x-axis tick labels

filtered_df.boxplot(column='entropySimilarity', by='PatientCat', ax=axes[1,0])
axes[1,0].set_title('Similarity Entropy')
axes[1,0].set_xlabel('Category')
axes[1,0].set_ylabel('Entropy')
axes[1,0].set_xticklabels(['Controls', 'Patients'])  # Set x-axis tick labels

filtered_df.boxplot(column='n_sentence', by='PatientCat', ax=axes[1,1])
axes[1,1].set_title('Num. Sentence')
axes[1,1].set_xlabel('Category')
axes[1,1].set_ylabel('Num. Sentence')
axes[1,1].set_xticklabels(['Controls', 'Patients'])  # Set x-axis tick labels

plt.tight_layout()
plot_path = os.path.join('/'.join(parent_folder.split('/')[:-2]),'plots')
plt.savefig(os.path.join(plot_path,'Participants_category'))
plt.show()

## --------------------------------------------------------------------
# Relationship between LTI and entropy
filtered_df = df.loc[df['PatientCat'] == 2,['PatientCat','TLI_DISORG','stim','entropyApproximate','entropySimilarity','entropyTransform','TransformSimilarity']]
#filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['PatientCat','stim','TLI_DISORG','entropyApproximate','entropySimilarity']]
filtered_df_keep = filtered_df.dropna(subset=['TLI_DISORG', 'entropyApproximate','entropySimilarity','entropyTransform','TransformSimilarity'])
r, p_value = pearsonr(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'])
print(f'correlation between TLI and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
sns.scatterplot(data=filtered_df_keep, x='TLI_DISORG', y='entropyApproximate')
slope, intercept = np.polyfit(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'], 1)
regression_line = slope * filtered_df_keep['TLI_DISORG'] + intercept
plt.plot(filtered_df_keep['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
plt.savefig('BERT_scatter_patients.eps', format='eps', bbox_inches='tight')
plt.show()

# Scatter plot, color coding patient group
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID','PatientCat','stim','TLI_DISORG','entropyTransform','entropyApproximate','entropySimilarity']]
df_plot = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','entropyTransform','entropyApproximate','entropySimilarity']].mean().reset_index()
sns.scatterplot(data=df_plot, x='TLI_DISORG', y='entropyApproximate', hue='PatientCat', palette='viridis')

# Scatter plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
axes[0,0].scatter(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'], label='Data')
axes[0,0].set_title('TLI_DISORG vs entropyApproximate')
axes[0,0].set_xlabel('TLI_DISORG')
axes[0,0].set_ylabel('entropyApproximate')
slope, intercept = np.polyfit(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'], 1)
regression_line = slope * filtered_df_keep['TLI_DISORG'] + intercept
axes[0,0].plot(filtered_df_keep['TLI_DISORG'], regression_line, color='red', label='Linear Regression')

axes[0,1].scatter(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyTransform'], label='Data')
axes[0,1].set_title('TLI_DISORG vs entropyTransform')
axes[0,1].set_xlabel('TLI_DISORG')
axes[0,1].set_ylabel('entropyTransform')
slope, intercept = np.polyfit(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyTransform'], 1)
regression_line = slope * filtered_df_keep['TLI_DISORG'] + intercept
axes[0,1].plot(filtered_df_keep['TLI_DISORG'], regression_line, color='red', label='Linear Regression')

axes[1,0].scatter(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropySimilarity'], label='Data')
axes[1,0].set_title('TLI_DISORG vs entropySimilarity')
axes[1,0].set_xlabel('TLI_DISORG')
axes[1,0].set_ylabel('entropySimilarity')
slope, intercept = np.polyfit(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropySimilarity'], 1)
regression_line = slope * filtered_df_keep['TLI_DISORG'] + intercept
axes[1,0].plot(filtered_df_keep['TLI_DISORG'], regression_line, color='red', label='Linear Regression')

axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
plt.tight_layout()
plot_path = os.path.join('/'.join(parent_folder.split('/')[:-2]),'plots')
plt.savefig(os.path.join(plot_path,'scatter_LTI_Entropy'))
plt.show()


## --------------------------------------------------------------------
# Relationship between panss and entropy
filtered_df = df.loc[df['PatientCat'] == 2,['PatientCat','PANSS Pos','stim','entropyApproximate','entropySimilarity','entropyTransform','TransformSimilarity']]
#filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['PatientCat','stim','PANSS Pos','entropyApproximate','entropySimilarity']]
filtered_df_keep = filtered_df.dropna(subset=['PANSS Pos', 'entropyApproximate','entropySimilarity','entropyTransform','TransformSimilarity'])
r, p_value = pearsonr(filtered_df_keep['PANSS Pos'], filtered_df_keep['entropyApproximate'])
print(f'correlation between TLI and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
sns.scatterplot(data=filtered_df_keep, x='PANSS Pos', y='entropyApproximate')

r, p_value = pearsonr(filtered_df_keep['PatientCat'], filtered_df_keep['entropyApproximate'])
print(f'correlation between Patient category and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')

## --------------------------------------------------------------------
# Regression analysis: modeling data
## --------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns

fname_var = os.path.join(parent_folder,'TOPSY_all.csv')
df = pd.read_csv(fname_var)
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['PatientCat','TLI_DISORG','entropyApproximate','entropySimilarity','n_sentence','stim']]
#filtered_df = df.loc[df['PatientCat'] == 2,['TLI_DISORG','entropyApproximate','entropySimilarity','n_sentence']]
df = filtered_df.dropna()

## --------------------------------------------------------------------
# visualize data
sns.pairplot(df)

correlation_matrix = df.loc[:,df.columns != 'stim'].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

## --------------------------------------------------------------------
# transform data
df['log_TLI'] = np.log(df['TLI_DISORG']+0.001)
df['sqrt_TLI'] = np.sqrt(df['TLI_DISORG'])
sns.histplot(df['sqrt_TLI'])
scaler = StandardScaler()
scaler.fit_transform(df[['entropySimilarity','TLI_DISORG']])

## --------------------------------------------------------------------
# model data
model = LinearRegression()
model.fit(df['TLI_DISORG'].to_numpy().reshape(-1,1), df['entropySimilarity'].to_numpy().reshape(-1,1))
residuals = df['entropyApproximate'].to_numpy().reshape(-1,1) - model.predict(df['TLI_DISORG'].to_numpy().reshape(-1,1))
plt.scatter(df['TLI_DISORG'].to_numpy().reshape(-1,1), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('TLI_DISORG')
plt.ylabel('Residuals')
plt.show()

## --------------------------------------------------------------------
# explained variance
r2 = r2_score(df['entropyApproximate'].to_numpy().reshape(-1,1), model.predict(df['TLI_DISORG'].to_numpy().reshape(-1,1)))
print(f"R-squared for TLI_DISORG: {r2}")


## --------------------------------------------------------------------
# Model multiple variables
X = df[['PatientCat','n_sentence','TLI_DISORG','stim']]  # Independent variables (var1 and var2)
y = df['entropyApproximate']   # Dependent variable (var_dependent)
X['stim'] = X['stim'].apply(lambda x: int(x[-1]))

import statsmodels.api as sm
# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())


'''
## --------------------------------------------------------------------
# Get topic distribution using approximate distribution
## --------------------------------------------------------------------
fname_scz=r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/TOPSY_001_Picture1.txt'
stim_scz = read_data_fromtxt(fname_scz)

fname_ctl=r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/TOPSY_041_Picture1.txt'
stim_ctl = read_data_fromtxt(fname_ctl)


window=20
appdistr_topic_scz, _ = topic_model.approximate_distribution(stim_scz,window=window,use_embedding_model=True)
entropy_scz = calculate_entropy_app(appdistr_topic_scz[0])

appdistr_topic_ctl, _ = topic_model.approximate_distribution(stim_ctl,window=window,use_embedding_model=True)
entropy_ctl = calculate_entropy_app(appdistr_topic_ctl[0])

print(f'entropy for scz: {entropy_scz}, entropy for control: {entropy_ctl}')


## --------------------------------------------------------------------
# Use moving window to get dominant topic and its associated probability
## --------------------------------------------------------------------
stim_rep = stim_scz.replace('...', ',')
sentences = stim_rep.split('.')
domtopic_scz, prob_scz = topic_model.transform(sentences[:-1])
entropy_scz = cal_entropy_weighted(domtopic_scz,prob_scz)

stim_rep = stim_ctl.replace('...', ',')
sentences = stim_rep.split('.')
domtopic_ctl, prob_ctl = topic_model.transform(sentences[:-1])
entropy_ctl = cal_entropy_weighted(domtopic_ctl,prob_ctl)

print(f'entropy for scz: {entropy_scz}, entropy for control: {entropy_ctl}')
'''
