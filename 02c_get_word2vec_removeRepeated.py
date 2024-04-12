# Get word2vec semantic similarity for speech output
import json
import pandas as pd
import numpy as np
import nltk
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from matplotlib import pyplot as plt
import gensim.downloader as api
import seaborn as sns
from scipy.stats import pearsonr
from collections import Counter
from collections import defaultdict

model = api.load("word2vec-google-news-300")
from scipy.spatial.distance import pdist
nltk.download('punkt')
nltk.download('stopwords')
stop_words_new = [',','uh','um','...','um…','xx','uh…','and…','hm',"'s",
                  'mhmm','mmhmm','mhm',"''",'eh','re',"”",'v','hmm',"'m",'ish',
                  'umm','ii','yup','yes','ugh',"“",'ar','oh','h',"'re",'ohh',
                  'wow','lo','aw','ta','ah','na','ex',"'","’","‘",'yo','ok','ah','mm',
                  'na','ra','ha','ka','huh','bc','a.c','uhh','hey','gee',"n't",'nah']
stop_words = set(stopwords.words('english') + stop_words_new)


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
            text = re.sub(r'…', 'DOTDOTDOT', text)
            text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', 'DOTABBREVIATION', text)
            sentences = re.split(r'\.', text)
            sentences = [sentence.strip().replace('DOTDOTDOT', '...').replace('DOTABBREVIATION', '.') for sentence in sentences if sentence.strip()]
            processed_sections.append(sentences)
        sentence_list = [sentence for sublist in processed_sections for sentence in sublist]
    return sentence_list


def get_content_words(stim):
    stim_all = ' '.join(stim)
    words = word_tokenize(stim_all)
    cleaned_content_words = []
    for i, word in enumerate(words):                   
        cleaned_word = word.replace('-', ' ')  # Replace '-' with a space to split the word
        cleaned_word = cleaned_word.replace('—', ' ')
        cleaned_word = cleaned_word.replace('-', ' ')
        cleaned_word = cleaned_word.replace('stereotypicalprison', 'stereotypical prison')
        cleaned_word = cleaned_word.replace('almostof', 'almost of')
        cleaned_word = cleaned_word.replace('girlhas', 'girl has')
        cleaned_word = cleaned_word.replace('shelooks', 'she looks')        
        cleaned_word = cleaned_word.replace('grey', 'gray')
        cleaned_word = cleaned_word.replace('judgement', 'judgment')
        cleaned_words = cleaned_word.split()   # Split the cleaned word into multiple words
        for cleaned_subword in cleaned_words:
            cleaned_subword = cleaned_subword.replace('labour', 'labor')
            cleaned_subword = cleaned_subword.replace("'cause", 'because')
            cleaned_subword = cleaned_subword.replace('centre', 'center')
            cleaned_subword = cleaned_subword.replace('theatre', 'theater')
            cleaned_subword = cleaned_subword.replace('hholding', 'holding')                        
            if cleaned_subword.lower() not in stop_words and cleaned_subword not in string.punctuation:
                if cleaned_subword.endswith('...'):
                    cleaned_content_words.append(cleaned_subword[:-3])
                else:
                    cleaned_content_words.append(cleaned_subword.lower())
    x = nltk.pos_tag(cleaned_content_words)
    content_word_categories = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']     
    words_with_desired_tags = [word for word, tag in x if tag in content_word_categories]
 
    return words_with_desired_tags


def get_word2vec(content_words):
    '''get word2vec similarity values between word n and word n-1, ..., n-5 for each word
    input: a list of words
    outout: dataframe of five columns'''
    missing_words = [word for word in content_words if word not in model.key_to_index]
    content_words = [word for word in content_words if word not in missing_words]
    
    similarities = []
    for i in range(5, len(content_words)):
        word_n_minus_1 = content_words[i - 1]
        word_n_minus_2 = content_words[i - 2]
        word_n_minus_3 = content_words[i - 3]
        word_n_minus_4 = content_words[i - 4]
        word_n_minus_5 = content_words[i - 5]
        word_n = content_words[i]
        
        similarity_n_1 = model.similarity(word_n_minus_1, word_n)
        similarity_n_2 = model.similarity(word_n_minus_2, word_n)
        similarity_n_3 = model.similarity(word_n_minus_3, word_n)
        similarity_n_4 = model.similarity(word_n_minus_4, word_n)
        similarity_n_5 = model.similarity(word_n_minus_5, word_n)
        
        similarities.append([similarity_n_1, similarity_n_2, similarity_n_3, similarity_n_4, similarity_n_5])
    columns = ['similarity_n_1', 'similarity_n_2', 'similarity_n_3', 'similarity_n_4', 'similarity_n_5']
    similarity_df = pd.DataFrame(similarities, columns=columns)
    return similarity_df, missing_words


def remove_repeated_within_window(words, window_size=5):
    result = []
    last_occurrence = defaultdict(int)
    for i, word in enumerate(words):
        if last_occurrence[word] == 0 or i - last_occurrence[word] > window_size:
            result.append(word)
        last_occurrence[word] = i
    return result


def process_file(parent_folder, foldername, filename):
    fname = os.path.join(parent_folder, foldername, filename)
    stim = read_data_fromtxt(fname)
    content_words = get_content_words(stim)
    new_words = remove_repeated_within_window(content_words, window_size=5)
    df_similarity,_ = get_word2vec(new_words)
    return filename.split('.')[0][6:], df_similarity.mean(axis=0).to_numpy()

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



# Get list of missing words, and included content words
missing_words_list = {}
content_words_list = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname)
        content_words = get_content_words(stim)
        content_words_list[filename.split('.')[0][6:]] = content_words
        _,missing_words = get_word2vec(content_words)
        missing_words_list[filename.split('.')[0][6:]] = missing_words       
concatenated_values = [value for values in missing_words_list.values() for value in values]
all_content_words = [value for values in content_words_list.values() for value in values]
unique_words_all = set(all_content_words)
with open(os.path.join(parent_folder, 'removed_words_word2vec.txt'), 'w') as file:
    for word in concatenated_values:
        file.write(word  +'\n')
with open(os.path.join(parent_folder, 'unique_words.txt'), 'w') as file:
    for word in unique_words_all:
        file.write(word + '\n')

## --------------------------------------------------------------------
# Get word2vec similarity values
## --------------------------------------------------------------------
word2vec_similarity = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        id_stim, similarity_values = process_file(parent_folder, foldername, filename)
        word2vec_similarity[id_stim] = similarity_values
result_data = [(id_stim, *similarity_values) for id_stim, similarity_values in word2vec_similarity.items()]
columns = ['ID_stim', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']
result_df = pd.DataFrame(result_data, columns=columns)
result_df[['ID', 'stim']] = result_df['ID_stim'].str.split('_', expand=True)
result_df.drop(columns=['ID_stim'], inplace=True)
result_df = result_df[['ID', 'stim', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']]
result_df['ID'] = result_df['ID'].astype('int64')
fname = os.path.join(parent_folder,'word2vec_rmRepeated.csv')
result_df.to_csv(fname,index=False)

## --------------------------------------------------------------------
# Load results and statistical tests
## --------------------------------------------------------------------
fname = os.path.join(parent_folder,'word2vec_rmRepeated.csv')
df_w2v = pd.read_csv(fname)
columns_to_rename = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
new_names = {'n_1': 'rmRep_n_1', 'n_2': 'rmRep_n_2', 'n_3': 'rmRep_n_3',
             'n_4': 'rmRep_n_4', 'n_5': 'rmRep_n_5'}
df_w2v.rename(columns=new_names,inplace=True)

fname_var = os.path.join(parent_folder,'TOPSY_all.csv') #containing topic measures
df_var = pd.read_csv(fname_var)

df_merge = df_var.merge(df_w2v, on=['ID','stim'],how='outer')
filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')

fname_all = os.path.join(parent_folder,'TOPSY_all.csv')
filtered_df.to_csv(fname_all,index=False)

df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]
fname_goi = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df_goi.to_csv(fname_goi,index=False)


## --------------------------------------------------------------------
# Visualize
## --------------------------------------------------------------------
# Compare between two groups
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
fname_var = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df = pd.read_csv(fname_var)
index_to_remove = df[df['stim'] == 'Picture4'].index
df = df.drop(index_to_remove)
filtered_df = df.loc[(df['PatientCat'] == 1) | (df['PatientCat'] == 2), 
                         ['ID', 'PatientCat', 'PANSS Pos', 'TLI_DISORG', 'stim', 'n_sentence', 
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
df_check = filtered_df.groupby('ID')[['PatientCat','entropyApproximate','TLI_DISORG','n_sentence', 'n_1', 'n_2', 'n_3', 'n_4', 'n_5']].mean()
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