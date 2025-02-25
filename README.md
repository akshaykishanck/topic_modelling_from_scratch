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
🚀 How to Run
1. Clone the Repository
```
git clone https://github.com/akshaykishanck/topic_modelling_from_scratch.git
cd topic_modelling_from_scratch
```
2. Create a Virtual Environment (Optional but Recommended)
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the experiment
```
python main.py
```
All the plots will be saved in the plots folder and the top 10 words per topic will be saved in topic_words.csv

📝 Notes \
Please make use of the config.json file to design your experiment. It comes with the following parameters
```
{   "main_data_path": "data/", # folder where you should be storing your data (documents)
    "plots_path": "plots/", # folder to where the plots of the experiment will be saved
    "runs": 30, # number of runs of the topic modelling experiment for every training fraction
    "number_of_iterations": 300, # number of iterations for the collapsed Gibbs Sampler
    "dataset": "20newsgroups",  # name of your dataset folder within the data folder
    "hyper_parameters": {
        "artificial": {
            "number_of_topics": 2,
            "alpha": 2.5,
            "beta": 0.01
        },
        "default": {
            "number_of_topics": 20, # number of topics 
            "alpha": 0.25, # dirchlet prior parameter for the topic distribution
            "beta": 0.01 # dirchlet prior parameter for word distribution
        }
    }
}
```




