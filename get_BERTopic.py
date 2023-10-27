import json
import math
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    

def read_data_fromtxt(fname):
    '''input: the full path of the txt file
    output: segemented sentences based on period marker from patients' speech''' 
    with open(fname,'r') as file:
        stim = file.read()
    stim = stim.replace('\xa0', '')
    stim = stim.replace('<','')
    stim = stim.replace('>','') 
    input_sections = [section.strip() for section in stim.split('* P') if section.strip() and not section.startswith('* E')]
    processed_sections = []
    for section in input_sections:
        lines = section.split('\n')
        lines = [line for line in lines if not line.startswith('* E')]
        for text in lines:
            text = re.sub(r'\.{3}', 'DOTDOTDOT', text)
            text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', 'DOTABBREVIATION', text)
            sentences = re.split(r'\.', text)
            sentences = [sentence.strip().replace('DOTDOTDOT', '...').replace('DOTABBREVIATION', '.') for sentence in sentences if sentence.strip()]
            processed_sections.append(sentences)
        sentence_list = [sentence for sublist in processed_sections for sentence in sublist]
    return sentence_list


def calculate_entropy_app(probabilities):
    """
    Calculate the entropy of a list of probability values.
    """
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy


def cal_entropy_weighted(topics,confidence):
    '''Calculate the entropy of identified topics, weighted by their confidence values'''
    topic_probs = {}
    for topic, prob in zip(topics, confidence):
        if topic in topic_probs:
            topic_probs[topic] += prob
        else:
            topic_probs[topic] = prob
    topic_probs_array = np.array(list(topic_probs.values()))
    normalized_probs = topic_probs_array / np.sum(topic_probs_array)
    entropy = -np.sum(normalized_probs * np.log2(normalized_probs))
    return entropy


def calculate_entropy_similarity(similarity_values):
    """
    Calculate the entropy of a list of similarity values.
    """
    probabilities = similarity_values / np.sum(similarity_values)
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

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
# Get data and conduct analysis: the approxiamte distribution approach

window=30
topic_entropy_app = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname)  
        stim_full = '. '.join(stim)      
        appdistr_topic, _ = topic_model.approximate_distribution(stim_full,window=window,use_embedding_model=True)
        entropy_val = calculate_entropy_app(appdistr_topic[0])
        topic_entropy_app[filename.split('.')[0][6:]] = entropy_val

df = pd.DataFrame(list(topic_entropy_app.items()), columns=['ID_stim', 'entropyApproximate'])
df[['ID', 'stim']] = df['ID_stim'].str.split('_', expand=True)
df.drop(columns=['ID_stim'], inplace=True)
df = df[['ID', 'stim', 'entropyApproximate']]
fname = os.path.join(parent_folder,'approximate_entropy.csv')
df.to_csv(fname,index=False)

# combine data and save them together with other variables
entropy_Values = df.groupby('ID')['entropyApproximate'].mean().reset_index()
entropy_Values['ID'] = entropy_Values['ID'].astype('int64')

fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df_var = pd.read_csv(fname_var)
df_merge = df_var.merge(entropy_Values,on='ID')
df_merge.to_csv(fname_var,index=False)


## --------------------------------------------------------------------
# Get dominant topic and its associated probability for each sentence
# Get topic distribution of full speech output
## --------------------------------------------------------------------
topic_entropy_trans = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname)
        domtopic, prob = topic_model.transform(stim)
        entropy_transform = cal_entropy_weighted(domtopic,prob)
        topic_entropy_trans[filename.split('.')[0][6:]] = entropy_transform
df = pd.DataFrame(list(topic_entropy_trans.items()), columns=['ID_stim', 'entropyTransform'])
df[['ID', 'stim']] = df['ID_stim'].str.split('_', expand=True)
df.drop(columns=['ID_stim'], inplace=True)
df = df[['ID', 'stim', 'entropyTransform']]
fname = os.path.join(parent_folder,'transform_entropy.csv')
df.to_csv(fname,index=False)

# combine data and save them together with other variables
entropy_Values = df.groupby('ID')['entropyTransform'].mean().reset_index()
entropy_Values['ID'] = entropy_Values['ID'].astype('int64')

fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df_var = pd.read_csv(fname_var)
df_merge = df_var.merge(entropy_Values,on='ID')
df_merge.to_csv(fname_var,index=False)


## --------------------------------------------------------------------
# Use find_topic to get similarity/probability of the whole input
## --------------------------------------------------------------------
topic_entropy_sim = {}
#top_n = 100
top_n = 2376
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname)  
        stim_full = '. '.join(stim)      
        _, similarity = topic_model.find_topics(stim_full,top_n=top_n)
        entropy_val = calculate_entropy_similarity(similarity)
        topic_entropy_sim[filename.split('.')[0][6:]] = entropy_val

df = pd.DataFrame(list(topic_entropy_sim.items()), columns=['ID_stim', 'entropySimilarity'])
df[['ID', 'stim']] = df['ID_stim'].str.split('_', expand=True)
df.drop(columns=['ID_stim'], inplace=True)
df = df[['ID', 'stim', 'entropySimilarity']]
fname = os.path.join(parent_folder,'similarity_entropy.csv')
df.to_csv(fname,index=False)

# combine data and save them together with other variables
entropy_Values = df.groupby('ID')['entropySimilarity'].mean().reset_index()
entropy_Values['ID'] = entropy_Values['ID'].astype('int64')

fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df_var = pd.read_csv(fname_var)
df_merge = df_var.merge(entropy_Values,on='ID')
df_merge.to_csv(fname_var,index=False)

## --------------------------------------------------------------------
# Get number of sentences, number of words information
## --------------------------------------------------------------------
topic_entropy_sim = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname)
        n_sentence = len(stim)
        topic_entropy_sim[filename.split('.')[0][6:]] = n_sentence

df = pd.DataFrame(list(topic_entropy_sim.items()), columns=['ID_stim', 'n_sentence'])
df[['ID', 'stim']] = df['ID_stim'].str.split('_', expand=True)
df.drop(columns=['ID_stim'], inplace=True)
df = df[['ID', 'stim', 'n_sentence']]
fname = os.path.join(parent_folder,'n_sentence.csv')
df.to_csv(fname,index=False)

# combine data and save them together with other variables
entropy_Values = df.groupby('ID')['n_sentence'].mean().reset_index()
entropy_Values['ID'] = entropy_Values['ID'].astype('int64')

fname_var = os.path.join(parent_folder,'TOPSY_subjectspec_variables.csv')
df_var = pd.read_csv(fname_var)
df_merge = df_var.merge(entropy_Values,on='ID')
df_merge.to_csv(fname_var,index=False)


## --------------------------------------------------------------------
# Load results and statistical tests
## --------------------------------------------------------------------
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
# Compare between two groups
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df = pd.read_csv(fname_var)
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['PatientCat','entropyTransform','entropyApproximate','entropySimilarity','n_sentence']]
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['entropyTransform'])
print(f'correlation between Patient category and Transform Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
r, p_value = pearsonr(filtered_df['PatientCat'], filtered_df['entropyApproximate'])
print(f'correlation between Patient category and Approximate Entropy estimation:'
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
filtered_df = df.loc[df['PatientCat'] == 2,['PatientCat','TLI_DISORG','stim','entropyApproximate','entropySimilarity','entropyTransform']]
#filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['PatientCat','stim','TLI_DISORG','entropyApproximate','entropySimilarity']]
filtered_df_keep = filtered_df.dropna(subset=['TLI_DISORG', 'entropyApproximate','entropySimilarity','entropyTransform'])
r, p_value = pearsonr(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'])
print(f'correlation between TLI and Approximate Entropy estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
sns.scatterplot(data=filtered_df_keep, x='TLI_DISORG', y='entropyApproximate')
slope, intercept = np.polyfit(filtered_df_keep['TLI_DISORG'], filtered_df_keep['entropyApproximate'], 1)
regression_line = slope * filtered_df_keep['TLI_DISORG'] + intercept
plt.plot(filtered_df_keep['TLI_DISORG'], regression_line, color='red', label='Linear Regression')

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
