# Data

We transcribed and analyzed speech data from 70 treatment-naive, first-episode psychosis patients and 34 demographically-matched controls who described three images for one minute each.

# Aims of the project

This project aims to extract language features from speech samples. The extract features are used to predict (1) continuous Thought Language Index (TLI) impoverishment and disorganization scores; (2) Categorical Participant Group (Healthy controls vs. patients with first episodic psychosis)

# Extract language features

## Sentence-level coherence

```python
'senN_4', 'senN_3', 'senN_2', 'senN_1'
```

## Disfluency

```python
 
 'N_fillers', 'N_immediate_repetation', 'false_starts', 'self_corrections'
```

## Word-level association

```python
'n_1', 'n_2', 'n_3', 'n_4', 'n_5' (#similarity between every word and its preceeding N words)
```

## Lexical-level

```python
'type_token_ratio','average_word_frequency'
```

## Discourse topic-level

```python
'entropyApproximate' (#the diversity of the topic distribution)
's0_mean' (#similarity between everything sentence and the picture label),
'consec_mean' (#similarity between the current sentence and its previous sentence)
```
For more details on BERTopic, see my [post](https://wordpress.com/post/linlifejourney.wordpress.com/291).
## Syntactic complexity

```python
'clause_density', 'dependency_distance', 'content_function_ratio'
```

## Other relevant variables

```python
'n_segment', 'length_utter','num_all_words', 'num_content_words', 'num_repetition'
```

# Exploratory data analysis (EDA)

## Check for missing values —> ignore all cognitive function measures
<img src="https://github.com/user-attachments/assets/c1c89779-4d3e-4229-a972-9487f79a6b85" alt="01_EDA_MissingValues" width="50%"/>

# Feature selection

### Visualize the distribution of all variables and identify outliers

- Remove data points with `average_word_frequency < 4.5`, `N_fillers > 20`, `content_function_ratio > 2.0`
- Remove variables with skewed distributions: `N_immediate_repetition`
<img src="https://github.com/user-attachments/assets/4ebe40d9-1e0a-456a-96a9-ec675cf86a57" alt="02_EDA_DistributionOutlier" width="50%"/>

### Check pairwise correlation

Check pairwise correlation matrix to remove or combine variables that are highly correlated

- n_1, n_2, n_3, n_4 and n_5: calculate the means to represent local word associations
- sen_1, sen_2, sen_3, sen_4: calculate the means to represent local semantic coherence
- Use VIF to identify highly correlated variables (VIF > 10): 
'num_all_words', 'num_content_words', 'length_utter’

#### Before feature selection:
<img src="https://github.com/user-attachments/assets/d5dcc137-3071-4ed7-b101-1586162df17b" alt="03_EDA_PairwiseRawVars" width="50%"/>

#### After feature selection:
<img src="https://github.com/user-attachments/assets/84a3e3f1-c5cf-4cfa-b889-717ba15a6809" alt="04_EDA_PairwiseNewVars" width="50%"/>

## Visualize data patterns

### Continuous variables
Visualize how the continuous dependent variables correlate to the language features.
<img src="https://github.com/user-attachments/assets/eed104cc-7dd2-41b1-a26c-eb7d08b40a05" alt="05_EDA_pairplot_continuousVars" width="50%"/>

### Categorical variables
Visualize how the Categorical dependent variables correlate to the language features.
<img src="https://github.com/user-attachments/assets/69d63410-3e29-49ce-b68d-ed1abd607f7f" alt="06_EDA_byPateintCategory" width="50%"/>

# Model continuous measures (TIL_IMPOV and TIL_DISORG)

### Lasso Regression for Feature Selection

- Use cross-validation to identify the best hyperparameters for the lasso regression:

```python
Best alpha for IMPOV: 0.02848035868435799
Best Mean Absolute Error (IMPOV): 0.15227086130242762
Best alpha for DISORG: 0.05462277217684337
Best Mean Absolute Error (DISORG): 0.26148430772065
```

- Use the identified hyperparameter to test the model

```python
Mean Absolute Error (IMPOV): 0.33674784603513247
Mean Absolute Error (DISORG): 0.5009238250371127
R2 (IMPOV): 0.15227086130242762
R2 (DISORG): 0.26148430772065
```

### Most Predictive Variables

For TLI_IMPOV:

```python
'type_token_ratio', 'num_repetition', 'entropyApproximate', 'average_word_frequency', 'Age', 'Gender_M', 'self_corrections'
```

For TLI_DISORG:

```python
's0_mean', 'num_repetition', 'false_starts', 'type_token_ratio', 'clause_density', 'N_fillers', 'consec_mean', 'Age',
 'self_corrections', 'Gender_M', 'dependency_distance'
```

# Model categorical data (HC vs. FEP)

## Deal with unbalanced data (36 HC vs. 64 FEP)

- Use SMOTE to upsample data with less samples

## Try out different models

### Random forest

#### Model performance: 85% accuracy
<img width="500" alt="ML_05_Accuracy_RandomForest" src="https://github.com/user-attachments/assets/63cd2d87-53ba-4f18-8b2a-f4609a291576">

#### Ranking the predictors based on their importance
<img src="https://github.com/user-attachments/assets/d2992845-90c8-4ecc-b173-8ceeb44d291d" alt="ML_03_RandomForest_PatientCat_beta" width="50%"/>

### L1 regularized logistic regression

#### Model performance: 65% accuracy
<img width="500" alt="ML_06_FeatureImportance_LogisticRegression" src="https://github.com/user-attachments/assets/238abdd7-3476-4cf8-aeea-d8671a4080da">

#### Ranking the predictors based on their importance
<img src="https://github.com/user-attachments/assets/b9998774-b474-4b80-a917-8267d7f2490b" alt="ML_04_Lasso_PredictPANSS_Pos_beta" width="50%"/>

