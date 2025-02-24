## Topic Modeling from Scratch

📌 Overview

This repository contains an implementation of topic modeling from scratch using Latent Dirichlet Allocation (LDA). The project includes a collapsed Gibbs sampler to estimate the topic distributions and infer latent topics from a collection of documents.

🛠 Features
* Implements LDA from scratch without using existing libraries like `gensim` or `scikit-learn`
* Uses collapsed Gibbs sampling for topic inference
* Provides document-topic and word-topic distributions

📂 Repository Structure 
```
📁 topic_modelling_from_scratch  
│── 📄 LDA.py   # Implementation of LDA with Gibbs Sampling  
│── 📄 main.py   # runner script   
│── 📁 utils/ 
  │── 📄 utils.py # helper functions such as reading from text files 
  │── 📄 evaluation.py # evaluation functions for the ML model  
  │── 📄 logistic_regression.py # Script to build, train and test the classification model 
│── 📁 data/  # folder containing different datasets for topic classification 
  │── 📁 20newsgroups/   # contains 884 documents  
  │── 📁 artificial/   # contains 10 documents  
│── 📁 config \
  │── 📄 config.py   # creates a config class to be used in runner scripts 
  │── 📄 config.json   # contains all the tunable parameters the user can 
│── 📄 requirements.txt # contains required Python packages and versions 
│── 📄 README.md     # This file
```
