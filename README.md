# C_N_reviews
This repository contains the code and graphs of the "Constructive and Non-constructive Reviews" research paper<br />
Authors: Prabhat Kumar Bharti, Priyanshu Patra<br />
Affiliation: Indian Institute of Technology (IIT) Patna, India<br />

------

### Folders in main repository:
- **[datasets](https://github.com/pripat-2002/C_N_reviews/tree/main/datasets) :** <br />
    - contains the [review dataset csv file](https://github.com/pripat-2002/C_N_reviews/blob/main/datasets/Final_review_dataset.csv) that we will be using to train our models 
    - contains the readymade [toxicbert features csv file](https://github.com/pripat-2002/C_N_reviews/blob/main/datasets/toxicbert.csv) for our convenience (since creating this takes at least 1 hour time).<br />
- **[graphs](https://github.com/pripat-2002/C_N_reviews/tree/main/graphs) :**<br />
    - contains the graphs and other pictorial charts obtained from the codes.
    
------

### Files in main repository:
- **[model hyperparameters textfile](https://github.com/pripat-2002/C_N_reviews/blob/main/model_hyperparameters.txt) :**<br />
  - model hyperparameters textfile talks about the ML/DL models used and also their hyperparameters, which can be adjusted according to our dataset, to obtain best results. <br />
- **[feature analysis notebook](https://github.com/pripat-2002/C_N_reviews/blob/main/features_review.ipynb) :**<br />
  - feature analysis notebook contains the codes that provide a detailed analysis of the 16 features (other than word embeddings) that we have used for creating the models. This can give us a basic idea how the features such as review length, linguistic features, sentiment and harshness of reviews varies for constructive (C) and non-constructive (N) reviews.<br />
- **[model prediction notebook](https://github.com/pripat-2002/C_N_reviews/blob/main/CN_baseline.ipynb) :**<br />
  - model prediction notebook contains the codes with detailed guiding comments, that create each of the six models we have used, prepare the embedding and labels matrix from the features for training and testing, train the models, and then show the results of the training on the testing dataset (baseline) as well as the ICLR dataset (qualitative analysis).

------

### Models we use for experiments:

- **SVC :** Support Vector Classifier <br />
  **Link:** https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

- **MNB :** Multinomial Naive Bayes Classifier <br />
  **Link:** https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
  
- **XGBoost :** XGBoost <br />
  **Link:** https://github.com/dmlc/xgboost

- **FNN :** Feed-forward Neural Network <br />
  
- **USE :** Universal Sentence Encoder with BiLSTM <br />
  **Link:** https://tfhub.dev/google/universal-sentence-encoder/4
  
- **BERT :** BERT with BiLSTM <br />
  **Link:** 
  - BertTokenizer: https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer 
  - TFBertModel: https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel

------

### Models we use for extracting features:

- **tfidf :** Tfidf Vectorizer <br />
  **Link:** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  
- **USE :** universal-sentence-encoder/4 <br />

- **BERT :** BertTokenizer + TFBertModel <br />

- **VADER :** VADER Sentiment <br />
  **Link:** https://github.com/cjhutto/vaderSentiment
  
- **toxicBERT :** ToxicBERT toxicity score <br />
  **Link:** https://huggingface.co/unitary/toxic-bert
  
------
