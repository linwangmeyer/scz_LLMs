# test bertopic
import sys
import csv
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
import math

from bertopic import BERTopic
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")


def generate_stim(prompts):
    responses = {}
    for prompt_key, prompt_value in prompts.items():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_value}
            ]
        )
        responses[prompt_key] = response.choices[0].message.content.strip().split('\n\n',1)[-1]
    return responses


def calculate_entropy_app(probabilities):
    """
    Calculate the entropy of a list of probability values.
    """
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy


def calculate_topic_entropy(stimulus, window=30):
    appdistr_topic, _ = topic_model.approximate_distribution(stimulus, window=window, use_embedding_model=True)
    return calculate_entropy_app(appdistr_topic[0])



#----------- Gegerate stimuli ---------
key_fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/my_api_key.txt'
with open(key_fname,'r') as file:
    key = file.read()

client = OpenAI(
    api_key=key,
)

prompts = {
    "hc": "Please write a paragraph with 200 words that describes a picture with a random topic.",
    "scz": "Please write a paragraph with 200 words that describes a picture with multiple topics."
}

responses = []
for _ in range(10):
    responses.append(generate_stim(prompts))

df = pd.DataFrame(responses)

df['nwor_hc'] = df['hc'].apply(lambda x: len(x.split()))
df['nwor_scz'] = df['scz'].apply(lambda x: len(x.split()))


#----------- Calculate entropy ---------
df['entropy_hc'] = df['hc'].apply(calculate_topic_entropy)
df['entropy_scz'] = df['scz'].apply(calculate_topic_entropy)

print(df[['entropy_hc','entropy_scz']])

list(df['hc'])
list(df['scz'])