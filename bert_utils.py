
# utils.py
import os
import numpy as np
import pandas as pd
import math
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import re
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words_new = [',','uh','um','...','um…','xx','uh…','and…','hm',"'s",'..',
                  'mhmm','mmhmm','mhm',"''",'eh','re',"”",'v','hmm',"'m",'ish',
                  'umm','ii','yup','yes','ugh',"“",'ar','oh','h',"'re",'ohh',
                  'wow','lo','aw','ta','ah','na','ex',"'","’","‘",'yo','ok','ah','mm',
                  'na','ra','ha','ka','huh','bc','a.c.','a.c','uhh','hey','gee',"n't",'nah']
stop_words = set(stopwords.words('english') + stop_words_new)

import gensim.downloader as api
model_w2v = api.load("word2vec-google-news-300")

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#topic_model = BERTopic.load("davanstrien/chat_topics")
# for a list of pre-trained topics, see: https://huggingface.co/models?library=bertopic&sort=downloads

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
#-----------------------------------
# get data related functions
#-----------------------------------
def read_data_fromtxt(fname):
    '''input: the full path of the txt file
    output: a dictionary containing speech output from different speakers''' 
    with open(fname,'r') as file:
        stim = file.read()
    stim = stim.replace('\xa0', '')
    stim = stim.replace('\n', ' ')
    stim = stim.replace('<','')
    stim = stim.replace('>','') 
    input_sections = [section.strip() for section in stim.split('*')]
    processed_sections = {}
    for nitem, item in enumerate(input_sections): #seperate speech from speakers
        if len(item.split()) > 1:
            processed_sections[item[0]+str(nitem)] = ' '.join(item.split()[1:])
    return processed_sections


def get_speech_before_time_up(stim):
    '''Isolate patient speech prior to 1 min ends
    input: dictionary containing multiple turns
    output: list of patients speech'''
    selected_values = []
    for i in range(2, len(stim), 2):
        if 'time' in stim.get('E{}'.format(i), '') or 'minute' in stim.get('E{}'.format(i), '') or 'All right and now' in stim.get('E{}'.format(i), ''):
            # Combine the speech of previous patients
            for j in range(1, i, 2):
                selected_values.append(stim['P{}'.format(j)])
            break  # Stop iterating as soon as 'time' or 'minute' is found
    else:
        # If 'time' or 'minute' is not found, select all responses from the patient
        for key in stim:
            if key.startswith('P'):
                selected_values.append(stim[key])
    return selected_values


#-----------------------------------
# calculate entropy related functions
#-----------------------------------
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

#-----------------------------------
# get word2vec related functions
#-----------------------------------
def get_content_words(stim_all):
    words = word_tokenize(stim_all)
    cleaned_content_words = []
    for i, word in enumerate(words):                   
        cleaned_word = word.replace('-', ' ')  # Replace '-' with a space to split the word
        cleaned_word = cleaned_word.replace('—', ' ')
        cleaned_word = cleaned_word.replace('-', ' ')
        cleaned_word = cleaned_word.replace('…', ' ')
        cleaned_word = cleaned_word.replace('.', ' ')
        cleaned_word = cleaned_word.replace('background.behind', 'background behind')
        cleaned_word = cleaned_word.replace('background.and', 'background and')
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
        word = word_list[i]
        start_index = max(0, i - window_size)
        end_index = i
        # Extract the preceding words within the window
        window = word_list[start_index:end_index]
        if word in window and len(window) > 1:
            word_with_repeated_counts[word] += 1
    return word_with_repeated_counts
    
    
    
    
def get_word2vec(content_words):
    '''get word2vec similarity values between word n and word n-1, ..., n-5 for each word
    input: a list of words
    outout: dataframe of five columns'''
    missing_words = [word for word in content_words if word not in model_w2v.key_to_index]
    content_words = [word for word in content_words if word not in missing_words]
    
    similarities = []
    for i in range(5, len(content_words)):
        word_n_minus_1 = content_words[i - 1]
        word_n_minus_2 = content_words[i - 2]
        word_n_minus_3 = content_words[i - 3]
        word_n_minus_4 = content_words[i - 4]
        word_n_minus_5 = content_words[i - 5]
        word_n = content_words[i]
        
        similarity_n_1 = model_w2v.similarity(word_n_minus_1, word_n)
        similarity_n_2 = model_w2v.similarity(word_n_minus_2, word_n)
        similarity_n_3 = model_w2v.similarity(word_n_minus_3, word_n)
        similarity_n_4 = model_w2v.similarity(word_n_minus_4, word_n)
        similarity_n_5 = model_w2v.similarity(word_n_minus_5, word_n)
        
        similarities.append([similarity_n_1, similarity_n_2, similarity_n_3, similarity_n_4, similarity_n_5])
    columns = ['similarity_n_1', 'similarity_n_2', 'similarity_n_3', 'similarity_n_4', 'similarity_n_5']
    similarity_df = pd.DataFrame(similarities, columns=columns)
    return similarity_df, missing_words


def get_missing_words(folder_file, parent_folder, foldername, filename, mode):
    '''get lists of words that can't be found in word2vec model'''
    missing_words_list = {}
    content_words_list = {}
    for foldername, filenames in folder_file.items():
        print(f'folder: {foldername}')
        for filename in filenames:
            print(f'file: {filename}')
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
            
            content_words = get_content_words(stim_all)
            content_words_list[filename.split('.')[0][6:]] = content_words
            _,missing_words = get_word2vec(content_words)
            missing_words_list[filename.split('.')[0][6:]] = missing_words       
    concatenated_values = [value for values in missing_words_list.values() for value in values]
    all_content_words = [value for values in content_words_list.values() for value in values]
    unique_words_all = set(all_content_words)
    return concatenated_values, unique_words_all


def process_file_w2v(parent_folder, foldername, filename, mode):
    fname = os.path.join(parent_folder, foldername, filename)
    file_list = filename.split('.')[0][6:]
    
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
    
    content_words = get_content_words(stim_all)
    df_similarity,_ = get_word2vec(content_words)
    w2v_sim = df_similarity.mean(axis=0).to_numpy()
    
    #number of all words, including stop words
    num_all_words = len(stim_all.split())
    
    #number of content words
    num_content_words = len(content_words) 
    
    #get the number of repeated words within a window=5
    word_with_repeated_counts = count_repetition(content_words, 5) 
    num_repetition = sum(word_with_repeated_counts.values())
    
    return file_list, w2v_sim, num_all_words, num_content_words, num_repetition

    
#-----------------------------------
# get topic entropy related functions
#-----------------------------------
def process_file_topic(parent_folder, foldername, filename, mode, window):
    
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
    
    # approximate approach
    appdistr_topic, _ = topic_model.approximate_distribution(stim_all,window=window,use_embedding_model=True)
    topic_entropy_app = calculate_entropy_app(appdistr_topic[0])
    
    # dominant topic and then calculate entropy
    stim_sen = sent_tokenize(stim_all)
    domtopic, prob = topic_model.transform(stim_sen)
    topic_entropy_trans = cal_entropy_weighted(domtopic,prob)
    
    # get pairwise cosine similarity between identified topic embeddings normalized by number of sentences
    sim_matrix = cosine_similarity(np.array(topic_model.topic_embeddings_)[domtopic,:])
    labels = [topic_model.get_topic_info(label).Name.to_list()[0] for label in domtopic]
    blow_index = np.tril_indices(sim_matrix.shape[0], k=-1)
    topic_sim = np.mean(sim_matrix[blow_index])
    
    # Use find_topic to get similarity/probability of the whole input
    _, similarity = topic_model.find_topics(stim_all,top_n=2376)
    topic_entropy_sim = calculate_entropy_similarity(similarity)
    
    # Get number of sentences, number of words information
    n_sentence = len(stim_sen)
    n_word = len(stim_all.split())
    
    return file_list, topic_entropy_app, topic_entropy_trans, topic_sim, topic_entropy_sim, n_sentence, n_word