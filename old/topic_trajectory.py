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
parent_folder = r'/Users/linwang/Downloads'
child_folders = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
text_file_list = []
folder_file = {}
for child_folder in child_folders:
    child_folder_path = os.path.join(parent_folder, child_folder)
    text_files = [file for file in os.listdir(child_folder_path) if file.endswith(".txt")]
    folder_file[child_folder] = text_files

## --------------------------------------------------------------------
# Get data and conduct analysis: the approxiamte distribution approach
# different windows for entropy: patient=5, controls=20
fname = r'/Users/linwang/Downloads/test.txt'
with open(fname,'r') as file:
    stim2 = file.read()

window=30
topic_entropy_app = {}
appdistr_topic, _ = topic_model.approximate_distribution(stim2,window=window,use_embedding_model=True)
entropy_val = calculate_entropy_app(appdistr_topic[0])
topic_entropy_app['win30'] = entropy_val

fname = r'/Users/linwang/Downloads/test_short.txt'
with open (fname,'r') as file:
    stim = file.read()
stim_all = stim.split()

#---------------------------------
# controls: all available contexts
start_point = 20
window=20
increment = 2
topic_entropy_control = {}
for i in range(start_point, len(stim_all)-start_point, increment):
    selected_text = ' '.join(stim_all[:i+increment+start_point])
    print(selected_text)
    appdistr_topic, _ = topic_model.approximate_distribution(selected_text,window=window,use_embedding_model=True)
    entropy_val = calculate_entropy_app(appdistr_topic[0])
    topic_entropy_control[f'win{i}'] = entropy_val
plt.plot(topic_entropy_control.values())
    
#---------------------------------
# patients: all available contexts
start_point = 20
window=5
increment = 2
topic_entropy_patient = {}
for i in range(start_point, len(stim_all)-start_point, increment):
    selected_text = ' '.join(stim_all[:i+increment+start_point])
    print(selected_text)
    appdistr_topic, _ = topic_model.approximate_distribution(selected_text,window=window,use_embedding_model=True)
    entropy_val = calculate_entropy_app(appdistr_topic[0])
    topic_entropy_patient[f'win{i}'] = entropy_val

plt.plot(topic_entropy_control.values(),'b')
plt.plot(topic_entropy_patient.values(),'r')

plt.plot(zscore(np.array(list(topic_entropy_control.values()))),'b-',label='controls')
plt.plot(zscore(np.array(list(topic_entropy_patient.values()))),'r-',label='patients')
plt.legend()


## --------------------------------------------------------------------
# Get data and conduct analysis: the approxiamte distribution approach: window=20
# different windows for context
fname = r'/Users/linwang/Downloads/test_short.txt'
with open (fname,'r') as file:
    stim = file.read()
stim_all = stim.split()

#---------------------------------
# controls: all available contexts
start_point = 20
window=20
increment = 2
topic_entropy_control = {}
for i in range(start_point, len(stim_all)-start_point, increment):
    selected_text = ' '.join(stim_all[:i+increment+start_point])
    print(selected_text)
    appdistr_topic, _ = topic_model.approximate_distribution(selected_text,window=window,use_embedding_model=True)
    entropy_val = calculate_entropy_app(appdistr_topic[0])
    topic_entropy_control[f'win{i}'] = entropy_val
plt.plot(topic_entropy_control.values())


#---------------------------------
# patients: 20 words contexts
start_point = 20
window=20
increment = 2
topic_entropy_patient_partial = {}
for i in range(start_point, len(stim_all)-start_point, increment):
    selected_text = ' '.join(stim_all[i+increment:i+increment+start_point])
    print(selected_text)
    appdistr_topic, _ = topic_model.approximate_distribution(selected_text,window=window,use_embedding_model=True)
    entropy_val = calculate_entropy_app(appdistr_topic[0])
    topic_entropy_patient_partial[f'win{i}'] = entropy_val
plt.plot(topic_entropy_control.values(),'b')
plt.plot(topic_entropy_patient.values(),'r')

plt.plot(zscore(np.array(list(topic_entropy_control.values()))),'b-',label='controls')
plt.plot(zscore(np.array(list(topic_entropy_patient_partial.values()))),'r-',label='patients')
plt.legend()



## --------------------------------------------------------------------
# Get dominant topic and its associated probability for each sentence
# Get topic distribution of full speech output
## --------------------------------------------------------------------
fname = r'/Users/linwang/Downloads/test_short.txt'
with open (fname,'r') as file:
    stim = file.read()
stim_all = stim.split()

#---------------------------------
# controls: all available contexts
start_point = 20
window=20
increment = 2
topic_control = {}
topic_prob_control = {}
for i in range(start_point, len(stim_all)-start_point, increment):
    selected_text = ' '.join(stim_all[:i+increment+start_point])
    print(selected_text)
    domtopic, prob = topic_model.transform(selected_text)
    topic_control[f'win{i}'] = domtopic
    topic_prob_control[f'win{i}'] = prob
plt.plot(zscore(np.array(list(topic_control.values()))),'b-',label='dom topic')
plt.plot(zscore(np.array(list(topic_prob_control.values()))),'b--',label='prob')
plt.legend()

topic_control_array = np.array(list(topic_control.values())).flatten()
differences_control = np.concatenate(([1], np.diff(topic_control_array) != 0)).astype(int)
plt.plot(zscore(differences_control),'b-',label='dom topic')
plt.plot(zscore(np.array(list(topic_prob_control.values()))),'b--',label='prob')
plt.legend()


#---------------------------------
# patients: 20 words contexts
start_point = 20
window=20
increment = 2
topic_patient = {}
topic_prob_patient = {}
for i in range(start_point, len(stim_all)-start_point, increment):
    selected_text = ' '.join(stim_all[i+increment:i+increment+start_point])
    print(selected_text)
    domtopic, prob = topic_model.transform(selected_text)
    topic_patient[f'win{i}'] = domtopic
    topic_prob_patient[f'win{i}'] = prob
plt.plot(zscore(np.array(list(topic_patient.values()))),'r-',label='dom topic')
plt.plot(zscore(np.array(list(topic_prob_patient.values()))),'r--',label='prob')
plt.legend()

topic_patient_array = np.array(list(topic_patient.values())).flatten()
differences_patient = np.concatenate(([1], np.diff(topic_patient_array) != 0)).astype(int)
plt.plot(zscore(differences_patient),'r-',label='dom topic')
plt.plot(zscore(np.array(list(topic_prob_patient.values()))),'r--',label='prob')
plt.legend()