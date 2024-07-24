
# Get syntactic complexity for speech output
from bert_utils import read_data_fromtxt
from bert_utils import process_file_syntax, process_file_lexical
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import stanza
from nltk.tokenize import word_tokenize
import re

## --------------------------------------------------------------------
# Read data
## --------------------------------------------------------------------
# Get list of folder and file names
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.join(os.path.dirname(current_directory),'stimuli','Relabeld')
text_files = [file for file in os.listdir(parent_folder) if file.endswith(".txt")]

## --------------------------------------------------------------------
# Get the syntactic measures
## --------------------------------------------------------------------
mode_label = ['before_time_up','spontaneous', 'full_speech']
outputfile_label = ['1min', 'spontaneous', 'concatenated']

for k in range(3):
    mode = mode_label[k]
    outputfile = outputfile_label[k]

    syntax_measures = {}
    for filename in text_files:
        print(f'file: {filename}')
            
        id_stim, disfluency, syntax = process_file_syntax(parent_folder, filename, mode)
        
        syntax_measures[id_stim] = {'N_fillers': disfluency['fillers'],
                                    'N_immediate_repetation': disfluency['repetitions'],
                                    'false_starts': disfluency['false_starts'],
                                    'self_corrections': disfluency['self_corrections'],
                                    'length_utter': syntax['length_utter'],
                                    'subord_index': syntax['subord_index'],
                                    'clause_density': syntax['clause_density'],
                                    'dependency_distance': syntax['dependency_distance']
                                    }

    df = pd.DataFrame.from_dict(syntax_measures, orient='index').reset_index()
    df[['ID', 'stim']] = df['index'].str.split('_', expand=True)
    df.drop(columns=['index'], inplace=True)
    df['ID'] = df['ID'].astype('int64')
    df.dropna(subset=['stim'], inplace=True)
    
    fname = os.path.join(parent_folder,'analysis','syntax_measures_' + outputfile + '.csv')
    df.to_csv(fname, index=False)


## --------------------------------------------------------------------
# Get the lexical measures
## --------------------------------------------------------------------
mode_label = ['before_time_up','spontaneous', 'full_speech']
outputfile_label = ['1min', 'spontaneous', 'concatenated']

for k in range(3):
    mode = mode_label[k]
    outputfile = outputfile_label[k]

    lexical_measures = {}
    for filename in text_files:
        print(f'file: {filename}')
            
        stimID, lexical = process_file_lexical(parent_folder, filename, mode)
        lexical_measures[stimID] = {'content_function_ratio': lexical['content_function_ratio'],
                                    'type_token_ratio': lexical['type_token_ratio'],
                                    'average_word_frequency': lexical['average_word_frequency']}
            
    df = pd.DataFrame.from_dict(lexical_measures, orient='index').reset_index()
    df[['ID', 'stim']] = df['index'].str.split('_', expand=True)
    df.drop(columns=['index'], inplace=True)
    df['ID'] = df['ID'].astype('int64')

    df.loc[df['stim'] == 'Picture 1','stim']='Picture1'
    df = df.drop(df[df['stim'] == 'Picture4'].index)
    df.dropna(subset=['stim'], inplace=True)
    
    fname = os.path.join(parent_folder,'analysis','lexical_measures_' + outputfile + '.csv')
    df.to_csv(fname, index=False)




#----------------------------------------
# identify text with extreme values
def find_file(start_dir, target_file):
    for root, dirs, files in os.walk(start_dir):
        if target_file in files:
            return os.path.join(root, target_file)
    return None

# Define the starting directory and the target file name
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_directory)
start_dir = os.path.join(parent_folder,'stimuli','Relabeld')
picID = 'Picture3'
subID = '106'
target_file = f"TOPSY_{subID}_{picID}.txt"

# Find the file
file_path = find_file(start_dir, target_file)

# check the content of particular file if necessary
stim = read_data_fromtxt(file_path)
stim  


## --------------------------------------------------------------------
# Combine with subject info
## --------------------------------------------------------------------
# see script 05_combine_features.py