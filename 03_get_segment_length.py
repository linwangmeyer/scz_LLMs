# Get number of words and number of content words
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

def count_repetition(word_list, window_size):
    ''' It is used to identify the repeated words within a small window of text
        word_list: a list word to check for repetition
        window_size: a number that defines how many words to check for repetition'''
    word_with_repeated_counts = Counter()
    for i in range(len(word_list)):
        word = content_words[i]
        start_index = max(0, i - window_size)
        end_index = i
        # Extract the preceding words within the window
        window = content_words[start_index:end_index]
        if word in window and len(window) > 1:
            word_with_repeated_counts[word] += 1
    return word_with_repeated_counts
    
## --------------------------------------------------------------------
# Get data file names
## --------------------------------------------------------------------
parent_folder = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/sczTopic/stimuli/'
child_folders = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
text_file_list = []
folder_file = {}
for child_folder in child_folders:
    child_folder_path = os.path.join(parent_folder, child_folder)
    text_files = [file for file in os.listdir(child_folder_path) if file.endswith(".txt")]
    folder_file[child_folder] = text_files

## --------------------------------------------------------------------
# # Get number of total words, content words, 
# and repeated words within a window of 5 content words
## --------------------------------------------------------------------
count_results = {}
for foldername, filenames in folder_file.items():
    print(f'folder: {foldername}')
    for filename in filenames:
        print(f'file: {filename}')
        fname = os.path.join(parent_folder, foldername, filename)
        stim = read_data_fromtxt(fname) #each sentence is a list element
        
        all_words = ' '.join(stim).split()
        num_all_words = len(all_words) #number of all words, including stop words
        
        content_words = get_content_words(stim)
        num_content_words = len(content_words) #number of content words
        
        word_with_repeated_counts = count_repetition(content_words, 5) #get the number of repeated words within a window=5
        num_repetition = sum(word_with_repeated_counts.values())
        
        count_results[filename.split('.')[0][6:]] = [num_all_words,num_content_words,num_repetition]


# Get all data together
data_rows = [{'ID': key.split('_')[0], 
              'stim': key.split('_')[1], 
              'num_all_words': values[0], 
              'num_content_words': values[1], 
              'num_repeated_words': values[2]} 
             for key, values in count_results.items()]
df = pd.DataFrame(data_rows)

fname = os.path.join(parent_folder,'count_words.csv')
df.to_csv(fname,index=False)



## --------------------------------------------------------------------
# Combine with other measures
## --------------------------------------------------------------------
fname = os.path.join(parent_folder,'count_words.csv')
df_count = pd.read_csv(fname)
fname_var = os.path.join(parent_folder,'TOPSY_all.csv') #containing topic measures
df_var = pd.read_csv(fname_var)

df_merge = df_var.merge(df_count, on=['ID','stim'],how='outer')
filtered_df = df_merge.dropna(subset = df_merge.columns[1:].tolist(), how='all')

fname_all = os.path.join(parent_folder,'TOPSY_all.csv')
filtered_df.to_csv(fname_all,index=False)

df_goi = filtered_df.loc[(filtered_df['PatientCat']==1) | (filtered_df['PatientCat']==2)]
fname_goi = os.path.join(parent_folder,'TOPSY_TwoGroups.csv')
df_goi.to_csv(fname_goi,index=False)

## --------------------------------------------------------------------
# Get mean values separately for patient and controls
## --------------------------------------------------------------------