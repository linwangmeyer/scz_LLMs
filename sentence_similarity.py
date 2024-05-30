import json
import math
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr
from scipy.stats import zscore
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize

from bert_utils import read_data_fromtxt,get_speech_before_time_up


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

# Read the official labels of the pictures
fname_apriori = '/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/official_labels.xlsx'
df = pd.read_excel(fname_apriori)
p1 = df[df['Picture ID']=='Picture 1']['Brief summary'].to_list()
p2 = df[df['Picture ID']=='Picture 2']['Brief summary'].to_list()
p3 = df[df['Picture ID']=='Picture 3']['Brief summary'].to_list()


def get_sentence_similarity(parent_folder, foldername, filename,mode):
    "Get the stimuli as strings; output: slope of the similarity time course"
    
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
    
    stim_sen = sent_tokenize(stim_all)
    if 'Picture1' in filename:
        stim_full = p1 + stim_sen
    elif 'Picture2' in filename:
        stim_full = p2 + stim_sen
    elif 'Picture3' in filename:
        stim_full = p3 + stim_sen
    
    embeddings = model.encode(stim_full) #number_sen * 384 embeddings
    
    # Task 1: Similarity between consecutive sentences
    consecutive_similarities = cosine_similarity(embeddings[1:], embeddings[:-1]).diagonal()

    # Task 2: Similarity between each sentence s1, s2, ..., s12 and s0
    s0_similarities = cosine_similarity(embeddings[1:], embeddings[0].reshape(1, -1)).flatten()

    # get summary measures
    consec_mean = np.mean(consecutive_similarities)
    consec_std = np.std(consecutive_similarities)
    consec_diff_std = np.std(np.diff(consecutive_similarities))
    
    s0_mean = np.mean(s0_similarities)
    s0_std = np.std(s0_similarities)
    s0_diff_std = np.std(np.diff(s0_similarities))
    
    # Get number of segments
    n_segment = len(stim_sen)
    file_list = filename.split('.')[0][6:]
    
    return file_list, consec_mean, consec_std, consec_diff_std, s0_mean, s0_std, s0_diff_std, n_segment



## --------------------------------------------------------------------
# Load pre-trained models
## --------------------------------------------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')


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
k = 2
mode = mode_label[k]
outputfile = outputfile_label[k]

similarity_measures = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        if 'Picture4' not in filename: #not analyzing picture4
            id_stim, consec_mean, consec_std, consec_diff_std, s0_mean, s0_std, s0_diff_std, n_segment = get_sentence_similarity(parent_folder, foldername, filename, mode)
            similarity_measures[id_stim] = {'consec_mean': consec_mean,
                                        'consec_std': consec_std,
                                        'consec_diff_std': consec_diff_std,
                                        's0_mean': s0_mean,
                                        's0_std': s0_std,
                                        's0_diff_std': s0_diff_std,
                                        'n_segment': n_segment
                                        }
result_data = []
for id_stim, values in similarity_measures.items():
    result_data.append((id_stim, values['consec_mean'], values['consec_std'], values['consec_diff_std'],values['s0_mean'], values['s0_std'], values['s0_diff_std'], values['n_segment']))
columns = ['ID_stim', 'consec_mean', 'consec_std', 'consec_diff_std', 's0_mean', 's0_std', 's0_diff_std', 'n_segment']
result_df = pd.DataFrame(result_data, columns=columns)
result_df[['ID', 'stim']] = result_df['ID_stim'].str.split('_', expand=True)
result_df.drop(columns=['ID_stim'], inplace=True)
result_df['ID'] = result_df['ID'].astype('int64')
fname = os.path.join(parent_folder,'similarity_measures_' + outputfile + '.csv')
result_df.to_csv(fname, index=False)



## --------------------------------------------------------------------
# Combine with subject info
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
outputfile_label = ['1min', 'spontaneous', 'concatenated']
for k in range(2,3):
    outputfile = outputfile_label[k]
    fname = os.path.join(parent_folder,'similarity_measures_' + outputfile + '.csv')
    df_stim = pd.read_csv(fname)

    fname_var = os.path.join(parent_folder,'TOPSY_all_' + outputfile + '.csv') #containing topic measures
    df_var = pd.read_csv(fname_var)
    df_var.loc[df_var['stim'] == 'Picture 1','stim']='Picture1'
    df_var = df_var.drop(df_var[df_var['stim'] == 'Picture4'].index)
    df_var.dropna(subset=['stim'], inplace=True)

    df_merge = df_var.merge(df_stim, on=['ID','stim'],how='outer')
    filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')

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
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups_concatenated.csv')
df = pd.read_csv(fname_var)
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2), 
                         ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'num_all_words', 
                          'entropyApproximate','consec_mean', 'consec_std','consec_diff_std',
                          's0_mean', 's0_std', 's0_diff_std', 'n_segment']]
filtered_df.dropna(inplace=True)


# Violin plot of entropy approximate by patient category
palette = [(1, 0, 0, 0.2), (0, 0, 1, 0.2)]  # Red and blue with 0.5 alpha
ax = sns.violinplot(x='PatientCat', y='consec_std', data=filtered_df, palette=palette)
ax.set_xticks([0, 1])  # Set the ticks at positions 0 and 1
ax.set_xticklabels(['Controls', 'Patients'])  # Set custom tick labels
plt.xlabel('Patient Category')
plt.ylabel('consec_std')
plt.title('consec_std by Patient Category')
plt.show()


# Relationship between LTI and number of word
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups_concatenated.csv')
df = pd.read_csv(fname_var)
df['PatientCat'] = df['PatientCat'].astype('int')
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2),['ID','PatientCat','TLI_DISORG','TLI_IMPOV','num_all_words','stim','entropyApproximate','consec_mean', 'consec_std','consec_diff_std','s0_mean', 's0_std', 's0_diff_std', 'n_segment']]
filtered_df.dropna(inplace=True)
r, p_value = pearsonr(filtered_df['TLI_DISORG'], filtered_df['s0_mean'])
print(f'correlation between TLI and Sentence Similarity estimation:'
      f'\ncorrelation {r},'
      f'\np value: {p_value}')
sns.scatterplot(data=filtered_df, x='TLI_DISORG', y='s0_mean')
slope, intercept = np.polyfit(filtered_df['TLI_DISORG'], filtered_df['s0_mean'], 1)
regression_line = slope * filtered_df['TLI_DISORG'] + intercept
plt.plot(filtered_df['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
plt.show()


# Scatter plot, color coding patient group
df_plot = filtered_df.groupby('ID')[['PatientCat','TLI_DISORG','n_segment','consec_std']].mean().reset_index()
sns.scatterplot(data=df_plot, x='TLI_DISORG', y='consec_std', hue='PatientCat', palette=['blue', 'red'])
#plt.savefig('BERT_TLI_DISORG_Entropy.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_plot, x='TLI_DISORG', y='n_segment', hue='PatientCat', palette=['blue', 'red'])
#plt.savefig('BERT_TLI_DISORG_n_segments.png', format='png', bbox_inches='tight')
plt.show()

sns.scatterplot(data=df_plot, x='n_segment', y='consec_std', hue='PatientCat', palette=['blue', 'red'])
plt.savefig('BERT_n_segments_Entropy.png', format='png', bbox_inches='tight')
plt.show()

#-----------------
# insights
#-----------------
# For the consequative sentence-level simialrity
# For the 1min and concatenated speech: more severe TLI_DISORG, greater consec_std --> Derailment: a sequence of completely unrelated thoughts

# For the on-target measure, i.e. simialrity to the picture label provided a priori
# For all types of speech samples: more severe TLI_DISORG, lower similarity to the target label --> tangentiality: off-the-point, oblique or irrelevant answers given to questions