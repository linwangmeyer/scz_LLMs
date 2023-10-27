# Data

We transcribed and analyzed speech data from 70 treatment-naive, first-episode psychosis patients and 34 demographically-matched controls who described three images for one minute each.

# Analysis

We used BERTopic and Word2Vec to probe message-level and meaning-level representations of the picture descriptions. We also visualized the results and conducted statistical tests to evaluate how participant group, positive thought disorder, sentence level and other variables affect these measures.

## BERTopic (’get_BERTopic.py’)

We used a pre-trained BERTopic model with 2376 common topics discussed on Wikipedia to probe higher-level topic representations of all picture descriptions. We obtained the topic distributions and quantified the entropy of the topic distribution for each picture description. There are three methods available (see below). We used the first method because it captures moment-by-moment topics in the descriptions.

### Approximate method

The whole document is first segmented into tokensets using a moving time window (we chose ~30 words for each window). For each of these tokensets, we used their embedding vectors and find out how similar they are to the representations of previously generated topics. For each topic, the similarity values between the topic and all tokensets are summed, giving rise to one value for one topic. These similarity values across all topics are then normalized (by dividing each similarity value with the sum of the absolute similarity value) to create a topic distribution for the entire document.

### Transform method

The model aims to assign the new document to a pre-existing topic that best aligns with its content. It goes through the BERTopic pipeline (embeddings -> dimensionality reduction -> clustering). However, the clustering process is slightly different from the training step. For new document, the model calculates the similarity between the embedding of the new document and the centroids of the pre-existing clusters (topics) from the training data. This similarity helps identify the most suitable cluster for the new document. 

We found that, although there were no differences in the topic entropy between schizophrenia and control participants, within the patient group, higher entropy strongly correlated with more severe positive thought disorder.

### Similarity method

This method merely calculates the embeddings of its inputs and compares that with the topic embeddings, where the topic embeddings are the average embeddings of all documents in the topic. The most similar topic embeddings are then selected. This is a bit more rough and is typically not intended for finding which topic a document belongs to.

## Word2Vec (’get_word2vec.py’)

We also used a pre-trained Word2Vec model with 300-dimensional word embeddings to probe lower-level meaning representations. We calculated cosine similarity between word pairs, focusing on the similarity between a word and its three preceding words (e.g., n & n-1, n & n-2, n & n-3, n & n-4, n& n-5) within each speech sample.

# Findings

- For patients with schizophrenia, greater topic entropy correlates with positive thought disorder scores.
- Patients with schizophrenia show greater local semantic associations than healthy controls.
