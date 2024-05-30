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
from scipy.stats import zscore

from bert_utils import read_data_fromtxt,calculate_entropy_app
from bert_utils import get_speech_before_time_up



def segment_text(text, subset_size=20, overlap=5):
    "Segment the full text to overlapping segments"
    words = text.split()
    subsets = []
    for i in range(0, len(words), subset_size - overlap):
        subset = ' '.join(words[i:i+subset_size])
        subsets.append(subset)
    if len(subsets) > 1 and subsets[-1] in subsets[-2]:
        subsets.pop()  # Remove the last subset
    return subsets


def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence from p to q.
    """
    
    # Add a smooth constant in case there are zero values
    smoothing_constant = 0.01 * 1 / len(p)
    p_smooth = p + smoothing_constant
    q_smooth = q + smoothing_constant

    # Ensure the smoothed distributions still sum up to 1
    p_smooth /= np.sum(p_smooth)
    q_smooth /= np.sum(q_smooth)

    # Calculate KL divergence
    kl_div = np.sum(np.where(p_smooth != 0, p_smooth * np.log(p_smooth / q_smooth), 0))
    
    return kl_div


def get_kl_std(parent_folder, foldername, filename, mode, window):
    "Get the stimuli as strings; output: K-L divergence between consequative segments and calculate the mean and std values"
    
    fname = os.path.join(parent_folder, foldername, filename)
    stim = read_data_fromtxt(fname)
    
    if mode == 'spontaneous':
        stim_all = stim['P1']
    elif mode == 'before_time_up':
        stim_cmb = get_speech_before_time_up(stim)
        if len(stim_cmb[0]) > 1: #if there are more than 1 turn
            stim_all = ' '.join(stim_cmb)
        else:
            stim_all = stim_cmb
    elif mode == 'full_speech':
        stim_cmb = [stim[key] for key in stim if key.startswith('P')]
        stim_all = ' '.join(stim_cmb)
    else:
        raise ValueError('select one of the three modes: spontaneous, before_time_up, full_speech')
    
    # get file name
    file_list = filename.split('.')[0][6:]
    
    # get segments of texts
    subsets = segment_text(stim_all)
    for i, subset in enumerate(subsets):
        print(f"Subset {i+1}: {subset}")

    # get topic distribution 
    window=30
    appdistr_topic, _ = topic_model.approximate_distribution(subsets,window=window,min_similarity=0.1,use_embedding_model=True)

    # calculate k-l divergence 
    kl = []
    for i in range(appdistr_topic.shape[0]-1):
        p = appdistr_topic[i]
        q = appdistr_topic[i+1]
        kl.append(kl_divergence(p,q))

    # get std of kl values
    kl_std = np.std(kl)
    kl_mean = np.mean(kl)
    
    # Get number of segments, number of words information
    n_segment = len(kl)
    n_word = len(stim_all.split())
    
    return file_list, kl_std, kl_mean, n_segment, n_word



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
# Get data and conduct analysis
## --------------------------------------------------------------------
mode_label = ['before_time_up','spontaneous', 'full_speech']
outputfile_label = ['1min', 'spontaneous', 'concatenated']
k = 0
mode = mode_label[k]
outputfile = outputfile_label[k]

kl_measures = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        id_stim, kl_std, kl_mean, n_segment, n_word = get_kl_std(parent_folder, foldername, filename, mode, window=30)
        kl_measures[id_stim] = {'kl_std': kl_std,
                                    'kl_mean': kl_mean,
                                    'n_segment': n_segment,
                                    'nword': n_word}
result_data = [(id_stim, values['kl_std'], values['kl_mean'], values['n_segment'], values['nword']) for id_stim, values in kl_measures.items()]
columns = ['ID_stim', 'kl_std', 'kl_mean', 'n_segment', 'nword']
result_df = pd.DataFrame(result_data, columns=columns)
result_df[['ID', 'stim']] = result_df['ID_stim'].str.split('_', expand=True)
result_df.drop(columns=['ID_stim'], inplace=True)
result_df['ID'] = result_df['ID'].astype('int64')
fname = os.path.join(parent_folder,'KL_measures_' + outputfile + '.csv')
result_df.to_csv(fname, index=False)



## --------------------------------------------------------------------
# Combine with subject info
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
outputfile_label = ['1min', 'spontaneous', 'concatenated']
for k in range(3):
    outputfile = outputfile_label[k]
    fname = os.path.join(parent_folder,'KL_measures_' + outputfile + '.csv')
    df_kl = pd.read_csv(fname)

    fname_var = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv') #containing topic measures
    df_var = pd.read_csv(fname_var)

    df_merge = df_var.merge(df_kl, on=['ID','stim'],how='outer')
    filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')
    filtered_df.drop(columns=['nword'], inplace=True)

    fname_all = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv')
    filtered_df.to_csv(fname_all,index=False)

    df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]
    fname_goi = os.path.join(parent_folder,'TOPSY_TwoGroups_' + outputfile + '.csv')
    df_goi.to_csv(fname_goi,index=False)


## --------------------------------------------------------------------
# Visualize
## --------------------------------------------------------------------
# Compare between two groups
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups_spontaneous.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2), 
                         ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'num_all_words', 
                          'entropyApproximate', 'kl_std', 'kl_mean', 'n_segment']]
filtered_df.dropna(inplace=True)


# Violin plot of entropy approximate by patient category
palette = [(1, 0, 0, 0.2), (0, 0, 1, 0.2)]  # Red and blue with 0.5 alpha
ax = sns.violinplot(x='PatientCat', y='kl_std', data=filtered_df, palette=palette)
ax.set_xticks([0, 1])  # Set the ticks at positions 0 and 1
ax.set_xticklabels(['Controls', 'Patients'])  # Set custom tick labels
plt.xlabel('Patient Category')
plt.ylabel('K-L std')
plt.title('K-L std by Patient Category')
plt.show()


# Relationship between LTI and number of word
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups_1min.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
df['PatientCat'] = df['PatientCat'].astype('int')
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID','PatientCat','TLI_DISORG','TLI_IMPOV','num_all_words','stim','kl_std','kl_mean','n_segment']]
filtered_df.dropna(inplace=True)
r, p_value = pearsonr(filtered_df['TLI_DISORG'], filtered_df['kl_std'])
print(f'correlation between TLI and KL estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
sns.scatterplot(data=filtered_df, x='TLI_DISORG', y='kl_std')
slope, intercept = np.polyfit(filtered_df['TLI_DISORG'], filtered_df['kl_std'], 1)
regression_line = slope * filtered_df['TLI_DISORG'] + intercept
plt.plot(filtered_df['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
plt.show()


# Scatter plot, color coding patient group
df_plot = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','n_segment','kl_std']].mean().reset_index()
sns.scatterplot(data=df_plot, x='TLI_DISORG', y='kl_std', hue='PatientCat', palette=['blue', 'red'])
#plt.savefig('BERT_TLI_DISORG_Entropy.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_plot, x='TLI_DISORG', y='n_segment', hue='PatientCat', palette=['blue', 'red'])
#plt.savefig('BERT_TLI_DISORG_n_segments.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_plot, x='n_segment', y='kl_std', hue='PatientCat', palette=['blue', 'red'])
plt.savefig('BERT_n_segments_Entropy.png', format='png', bbox_inches='tight')
plt.show()
