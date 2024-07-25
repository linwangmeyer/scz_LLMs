# Get word2vec semantic similarity for speech output
from bert_utils import  process_file_w2v
from bert_utils import get_content_words,get_word2vec
from bert_utils import read_data_fromtxt,get_speech_before_time_up
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

## --------------------------------------------------------------------
# Read data
## --------------------------------------------------------------------
# Get list of folder and file names
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.join(os.path.dirname(current_directory),'stimuli','Relabeld')
text_files = [file for file in os.listdir(parent_folder) if file.endswith(".txt")]

## --------------------------------------------------------------------
# Get list of missing words, and included content words
## --------------------------------------------------------------------
missing_words_list = {}
content_words_list = {}
for filename in text_files:
    print(f'file: {filename}')
    fname = os.path.join(parent_folder, filename)
    stim = read_data_fromtxt(fname)
    
    #only first response
    #stim_all = stim['P1']
    
    #up to time out
    #stim_cmb = get_speech_before_time_up(stim)
    #if len(stim_cmb[0]) > 1: #if there are more than 1 turn
    #    stim_all = ' '.join(stim_cmb)
    #else:
    #    stim_all = stim_cmb
    
    #all speech
    stim_cmb = [stim[key] for key in stim if key.startswith('P')]
    stim_all = ' '.join(stim_cmb)
            
    content_words = get_content_words(stim_all)
    content_words_list[filename.split('.')[0][6:]] = content_words
    _,missing_words = get_word2vec(content_words)
    missing_words_list[filename.split('.')[0][6:]] = missing_words       
concatenated_values = [value for values in missing_words_list.values() for value in values]
all_content_words = [value for values in content_words_list.values() for value in values]
unique_words_all = set(all_content_words)
with open(os.path.join(parent_folder, 'analysis', 'removed_words_word2vec_spontaneous.txt'), 'w') as file:
    for word in concatenated_values:
        file.write(word  +'\n')
with open(os.path.join(parent_folder, 'analysis', 'unique_words_spontaneous.txt'), 'w') as file:
    for word in unique_words_all:
        file.write(word + '\n')



## --------------------------------------------------------------------
# Get word2vec similarity values
## --------------------------------------------------------------------
# speech of different conditions
mode_label = ['before_time_up','spontaneous', 'full_speech']
outputfile_label = ['1min', 'spontaneous', 'concatenated']

for k in range(3):
    mode = mode_label[k]
    outputfile = outputfile_label[k]

    word2vec_similarity = {}
    for filename in text_files:
        print(f'file: {filename}')
        id_stim, w2v_sim, num_all_words, num_content_words, num_repetition = process_file_w2v(parent_folder, filename, mode=mode)
        word2vec_similarity[id_stim] = {'w2v_sim': w2v_sim, 'num_all_words': num_all_words, 'num_content_words': num_content_words, 'num_repetition': num_repetition}

    result_data = [(id_stim, *similarity_values['w2v_sim'], similarity_values['num_all_words'], similarity_values['num_content_words'], similarity_values['num_repetition']) for id_stim, similarity_values in word2vec_similarity.items()]
    columns = ['ID_stim', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'num_all_words', 'num_content_words', 'num_repetition']
    result_df = pd.DataFrame(result_data, columns=columns)
    result_df[['ID', 'stim']] = result_df['ID_stim'].str.split('_', expand=True)
    result_df.drop(columns=['ID_stim'], inplace=True)
    result_df = result_df[['ID', 'stim', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5', 'num_all_words', 'num_content_words', 'num_repetition']]
    result_df['ID'] = result_df['ID'].astype('int64')
    fname = os.path.join(parent_folder, 'analysis', 'word2vec_' + outputfile + '.csv')
    result_df.to_csv(fname,index=False)


## --------------------------------------------------------------------
# Combine with subject info
## --------------------------------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.join(os.path.dirname(current_directory),'stimuli','Relabeld')
outputfile_label = ['1min', 'spontaneous', 'concatenated']
for k in range(3):
    outputfile = outputfile_label[k]
    fname = os.path.join(parent_folder,'analysis', 'word2vec_' + outputfile + '.csv')
    df_w2v = pd.read_csv(fname)
    
    fname_var = os.path.join(os.path.dirname(parent_folder),'variables','TOPSY_subjectspec_variables.csv')
    df_var = pd.read_csv(fname_var)

    df_merge = df_var.merge(df_w2v, on=['ID','stim'],how='outer')
    
    # all participants
    filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')
    filtered_df.drop(columns=['nword'], inplace=True)

    # only two groups
    df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]


## --------------------------------------------------------------------
# Visualize
## --------------------------------------------------------------------
# Compare between two groups

filtered_df = filtered_df.loc[(filtered_df['PatientCat'] == 1) | (filtered_df['PatientCat'] == 2), 
                         ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'n_segment', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
columns_to_melt = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
melted_df = pd.melt(filtered_df[['PatientCat'] + columns_to_melt], id_vars=['PatientCat'], value_vars=columns_to_melt, var_name='wordpos', value_name='w2v')

# Violin plots
g = sns.FacetGrid(melted_df, col='wordpos', height=5, aspect=1.2, sharey=False)
g.map_dataframe(sns.violinplot, x='PatientCat', y='w2v', split=True, inner='quart', palette='Set1')
g.set_xticklabels(['Controls', 'Patients'])
g.set_axis_labels('Patient Category', 'w2v Value')
g.set_titles(col_template='{col_name} - wordpos')
g.fig.suptitle('Semantic similarity by Patient Category for Different word positions', y=1.02)
plt.tight_layout()
plt.show()


# bar plot
df_check = filtered_df.groupby('ID')[['PatientCat','entropyApproximate','TLI_DISORG','n_segment', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean()
w2v = df_check.groupby('PatientCat')[['n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean()
w2v = w2v.reset_index()
w2v.drop('PatientCat', axis=1, inplace=True)

ax = w2v.plot(kind='bar', figsize=(10, 6))
ax.set_xlabel('group')
ax.set_ylabel('word2vec similarity')
ax.set_title('Word2Vec Similarity')
ax.set_xticklabels(['controls', 'patients'], rotation=45)  # Labels for the ticks with rotation
ax.set_xticklabels(['controls', 'patients'])  # Labels for the ticks
plt.tight_layout()
plot_path = os.path.join('/'.join(parent_folder.split('/')[:-2]),'plots')
plt.savefig(os.path.join(plot_path,'wor2vec_group'))
plt.show()


# box plot
columns_to_compare = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']

melted_df = pd.melt(df_check[['PatientCat','n_1', 'n_2', 'n_3', 'n_4', 'n_5']], id_vars=['PatientCat'], value_vars=['n_1', 'n_2', 'n_3', 'n_4', 'n_5'],
                    var_name='Similarity', value_name='Value')

plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
ax = sns.boxplot(data=melted_df, x='Similarity', y='Value', hue='PatientCat')
ax.set_xlabel('Similarity between n and preceding words')
ax.set_ylabel('word2vec similarity value')
ax.set_title('Similarity Comparison by Patient Category')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(title='Patient Category')
plt.tight_layout()
plot_path = os.path.join('/'.join(parent_folder.split('/')[:-2]),'plots')
plt.savefig(os.path.join(plot_path,'wor2vec_group_boxplot'))
plt.show()



## --------------------------------------------------------------------
# Statistical test
## --------------------------------------------------------------------
from scipy import stats
# for each position, test group differences
similarity_columns = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for col in similarity_columns:
    group_1 = df_check[df_check['PatientCat'] == 1][col]
    group_2 = df_check[df_check['PatientCat'] == 2][col]
    t_statistic, p_value = stats.ttest_ind(group_1.values, group_2.values)
    # Alternatively, perform Mann-Whitney U test (non-parametric test)
    # u_statistic, p_value = stats.mannwhitneyu(group_1, group_2)
    print(f'word2vec similarity before word n and {col}:')
    print(f'T-Statistic: {t_statistic}, P-Value: {p_value}')
    print('-------------------------')

# for health controls, test position difference
df_contrl = df_check[df_check['PatientCat'] == 1]
similarity_columns = [('n_1', 'n_2'), ('n_1', 'n_3'), ('n_1', 'n_4'), ('n_1', 'n_5')]
for col in similarity_columns:        
    group_1 = df_contrl[col[0]]
    group_2 = df_contrl[col[1]]
    t_statistic, p_value = stats.ttest_ind(group_1, group_2)
    print(f'controls: word2vec similarity before {col[0]} and {col[1]}:')
    print(f'T-Statistic: {t_statistic}, P-Value: {p_value}')
    print('-------------------------')

# for patients, test position difference
df_scz = df_check[df_check['PatientCat'] == 2]
similarity_columns = [('n_1', 'n_2'), ('n_1', 'n_3'), ('n_1', 'n_4'), ('n_1', 'n_5')]
for col in similarity_columns:        
    group_1 = df_scz[col[0]]
    group_2 = df_scz[col[1]]
    t_statistic, p_value = stats.ttest_ind(group_1, group_2)
    print(f'patients: word2vec similarity before {col[0]} and {col[1]}:')
    print(f'T-Statistic: {t_statistic}, P-Value: {p_value}')
    print('-------------------------')

## --------------------------------------------------------------------
# relation between TLI_DISORG and w2v similarity
## --------------------------------------------------------------------
similarity_columns = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for col in similarity_columns:
    r, p_value = pearsonr(df_check.loc[df_check['PatientCat'] == 2,'TLI_DISORG'], df_check.loc[df_check['PatientCat'] == 2,col])
    print(f'correlation between TLI and similarity for {col}:'
        f'\ncorrelation {r},p value: {p_value}')


df_high = df_check[df_check['TLI_DISORG']>=1]
for col in similarity_columns:
    r, p_value = pearsonr(df_high.loc[df_high['PatientCat'] == 2,'TLI_DISORG'], df_high.loc[df_high['PatientCat'] == 2,col])
    print(f'correlation between TLI and similarity for {col}:'
        f'\ncorrelation {r},p value: {p_value}')

for col in similarity_columns:
    plt.figure()  # Create a new figure for each plot
    sns.scatterplot(data=df_check, x='TLI_DISORG', y=col, hue='PatientCat', palette='viridis')
    slope, intercept = np.polyfit(df_check['TLI_DISORG'], df_check[col], 1)
    regression_line = slope * df_check['TLI_DISORG'] + intercept
    plt.plot(df_check['TLI_DISORG'], regression_line, color='red', label='Linear Regression')
    plt.savefig(f'w2v_{col}_vs_TLI_DISORG.png') 
    plt.close() 


## --------------------------------------------------------------------
# relation between topic measures and w2v similarity
## --------------------------------------------------------------------

similarity_columns = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for col in similarity_columns:
    r, p_value = pearsonr(df_check.loc[df_check['PatientCat'] == 2,'entropyApproximate'], df_check.loc[df_check['PatientCat'] == 2,col])
    print(f'correlation between entropyApproximate and similarity for {col}:'
        f'\ncorrelation {r},p value: {p_value}')
    

for col in similarity_columns:
    plt.figure()  # Create a new figure for each plot
    sns.scatterplot(data=df_check, x='entropyApproximate', y=col, hue='PatientCat', palette='viridis')
    slope, intercept = np.polyfit(df_check['entropyApproximate'], df_check[col], 1)
    regression_line = slope * df_check['entropyApproximate'] + intercept
    plt.plot(df_check['entropyApproximate'], regression_line, color='red', label='Linear Regression')
    plt.savefig(f'w2v_{col}_vs_entropyApproximate.png') 
    plt.close() 

#-------------------------------------------
# Check chronic patients: PatientCat = 3
#-------------------------------------------

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

fname_all = os.path.join(parent_folder,'TOPSY_all_spontaneous.csv')
df = pd.read_csv(fname_all)
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2), ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'n_segment', 
                          'entropyApproximate', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
filtered_df.dropna(inplace=True)
df_check = filtered_df.groupby('ID')[['PatientCat','entropyApproximate','TLI_DISORG','n_segment', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean()


# for each position, test group differences
similarity_columns = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
for col in similarity_columns:
    group_1 = df_check[df_check['PatientCat'] == 1][col]
    group_2 = df_check[df_check['PatientCat'] == 2][col]
    t_statistic, p_value = stats.ttest_ind(group_1.values, group_2.values)
    # Alternatively, perform Mann-Whitney U test (non-parametric test)
    # u_statistic, p_value = stats.mannwhitneyu(group_1, group_2)
    print(f'word2vec similarity before word n and {col}:')
    print(f'T-Statistic: {t_statistic}, P-Value: {p_value}')
    print('-------------------------')
    

df_melted = df.melt(id_vars=['PatientCat'], value_vars=['n_1', 'n_2', 'n_3', 'n_4', 'n_5'],
                    var_name='Condition', value_name='Value')

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Condition', y='Value', hue='PatientCat')
plt.title('Bar Plots for senN_x Conditions Separated by PatientCat')
plt.xlabel('Condition')
plt.ylabel('Value')
plt.legend(title='Patient Category')
plt.show()
