# DataMining
MSBD5002 data mining assignments
---

## Required Packages
- sklearn
- pandas
- numpy
- matplotlib
- scipy
- torch 1.10

## Frequent Pattern Mining
### 1. [Hash Tree](frequent_pattern_mining/hash_tree.py)

Hash tree is a data structure to store frequent pattern for efficient counting and search. The implementation of hash tree can be found in `frequent_pattern_mining/hash_tree.py`. To build a hash tree, first need to put the item set in variable `candidate_str` in the format of "{1 2 3}, {1 4 5}, {1 2 4}". For simplicity, this implementation is only for max leaf size of 3, so as the depth of the final hash tree. If there are leaves which cannot further split, with leaf size larger than 3. In these cases, we will use linked list to append the additional item set when printing out the tree.

Then run the following command in terminal, a hash tree will be built and printed out. The printed tree rotates 90 degree and the indentation is related to the depth.

```
python3 frequent_pattern_mining/hash_tree.py
```

### 2. [Frequent Pattern Tree](frequent_pattern_mining/FP_tree.py)

Frequent pattern tree is used to mine frequent patterns within a transaction database. The item dataset is storeed in `frequent_pattern_mining/DataSetA.csv`. It contains 12,526 records and each record records every single transaction in the grocery store. Each line represents a single transaction with names of products. You could modify the minimum support threshold in `fp_tree = FPTree(data, min_support=2500, item="NULL", item_freq=0)`. The default minimum support threshold is 2500.

To run the program, run the following command in terminal.
```
python3 frequent_pattern_mining/FP_tree.py
```

## Classification
### 1. [Simple classification](classification/simple_classifiers.py)

In `classification/simple_classifiers.py`, I compared model performance of decision tree, KNN and random forest on the wine quality dataset under `classifcation/dataset`.

For decision tree, I tuned hyperparameters including criterion in ["gini", "entropy"] and max_depth in [5, 10, 15].  For KNN, I tuned the n_neighbors in [1, 5, 10]. For random forest, I tuned hyperparameters min_samples_split in [5, 10] and n_estimators in [50, 100]. 

To run the program, run the following command in terminal.
```
python3 classification/simple_classifiers.
```

### 2. [Adaboost](classification/adaboost.py)

The above classifers are directly imported from sklearn, while I also implemented the Adaboost from scratch. Adaboost is a boosting method, consisted of any weak classifier. In `classification/adaboost.py`, I used a simply binary weak classifier - predict 1 when x is larger than a threshold and predict -1 otherwise. The sign could be flipped to minimize the loss. When building each weak classifier, the weight of data is adjusted according to the classification accuracy of previous classifier.

The default number of boosting rounds is 10, you could modify in `adaboost = Adaboost(n_boosting=10)`. For simplicity, I only use 10 data points, you could change the data in model initialization.

To run the program, run the following command in terminal. The input, true label, predicted value, estimator's weight, error rate and threshold of each weak classifier will be printed out.
```
python3 classification/adaboost.py
```

### 3. [Neural Network](neural_network/main.py)

Aside from simple machine learing methods for classification, neural network is also widely used. I implemented a simple single layer neural network in Pytorch for binary classification (breast-cancer, diabets, iris and wine dataset) and a two-layer neural network for multi-class classification (hand-written digit images).

To run the program, run the following command in terminal. The model classification results on train and test are saved in `neural_network/pictures`.
```
python3 neural_network/main.py
```

## Clustering

### 1. [DBSCAN](clustering/DBSCAN.py)

DBSCAN is a clustering algorithm, yet could also be used to detect outliers. There are two most important hyperparameters, namely eps and minPts. In `clustering/DBSCAN.py`, I implemented the algorithm to detect the outliers in the `clustering/dataset/DBSCAN_Points.txt`. From the output images in `clustering/pictures/DBSCAN`, we could see that when eps=2.0 and minPts=15 or eps=2.5 and minPts=20, it could detect the outliers, colored in black. 

To run the program, run the following command in terminal. The clustering results on are saved in `clustering/pictures/DBSCAN`.
```
python3 clustering/DBSCAN.py
```

### 2. [Fuzzy KMeans in EM](clustering/FuzzyCluster_EM.py)

Expectation - Maximization (EM) algorithm is an iteratively method to find maximum likelihood. I implemented a fuzzy KMeans algo using the EM method in `clustering/FuzzyCluster_EM.py`. The fuzzy KMeans is a soft version of KMeans clustering, in which each point could belong to more than one clustering, with certain probabilities. There are three hyperparameters: number of iterations, epsilon to evaluate whether the stopping criterion is met and also the p -  the fuzzy degree.

To run the program, run the following command in terminal. The clustering results on are saved in `clustering/pictures/fuzzy_cluster`.
```
python3 clustering/FuzzyCluster_EM.py
```

### 3. [Nested Loop for Outlier Detection](clustering/NestedLoop.py)

Nested loop algorithm was introduced by Knorr & Ng in 1998, and this implementation is based on their paper - [Algorithms for Mining Distance-based Outliers in Larger Dataset](http://www.vldb.org/conf/1998/p392.pdf).

To run the program, run the following command in terminal. The clustering results on are saved in `clustering/pictures/NestedLoop`.
```
python3 clustering/NestedLoop.py
```

## [Community Detection](community_detection/main.py)

- Required packages
    - networkx==2.2
    - sklearn
    - pandas

- Dataset description

The file `community_detection/dataset/email-Eu-core.txt` represents the social network generated using email data from a large European research institution. There is an edge (u,v) in the network if person u sent person v at least one email. Each line in this file records an email communication link between members of the institution. The `community_detection/dataset/email-Eu-core-department-labels.txt` file provides the ground-truth department membership labels.B ased on the two files, detect 42 communities among 1005 members of the institution. 

- Algorithm used

I use an open-sourced package **networkx** to detect the communities and the Asynchronous Fluid Communities algorithm - asyn_fluidc provided by the packages. The algorithm is based on the simple idea of fluids interacting in an environment, expanding and pushing each other. Its initialization is random, so found communities may vary on different executions. An advantage of it is that it allows for the definition of number of communities to be found. Run ```python3 community_detection/main.py```, the output is saved in `result.csv`.

- Evaluation metric

For community detection, normalized mutual information is widely used for performance evaluation. The algorithm has a NMI of 0.6851.

## [Covid-19 Daily Prediction](covid_prediction/lstm.py)

- Required packages
    - matplotlib, plotly, pandas, numpy, sklearn
    - tensorflow

- Data description

The two datasets `covid_prediction/dataset/covid19_confirmed_global.txt` and `covid_prediction/dataset/covid19_deaths_global.txt` record the number of confirmed cases and deaths from 2022/01/22 to 2021/11/29 in different regions.

- Algorithm used

I ensembled three models to predict the daily covid confirmed and death cases in the US respectively. They are linear regression model with second-order polinomial, Bayesian ridge regression and LSTM.

Run the following code in terminal to get the predicted output.
```
python3 covid_prediction/linear_model.py
python3 covid_prediction/lstm.py
python3 covid_prediction/ensemble.py
python3 covid_prediction/plot.py
```

- Model performance

The model predicts US confirmed cases and deaths cases from 2021/11/30 to 2021/12/06. The prediction results are as below. ![results](covid_prediction/output/prediction_result.png?raw=true "Covid19 Confirmed and Death Cases vs Prediction Results in the US")


## [Handwritten Digit Classification](digit_classification/main.py)

- Required packages
    - torch==1.10
    - numpy, pandas, matplotlib
    - tqdm

- Data description

The train dataset can be downloaded via this [link](https://drive.google.com/file/d/1g-EcByqjG0nxZvVovOZEwHd9mTDuKuye/view?usp=sharing). Download the dataset and put it under the `digit_classification` folder. This dataset contains 9269 images about the handwritten digits. Each row is the data for one instance. The first 784 columns in a row correspond to the 784 features of an instance (a 28*28 image), and the last column is the lable, ranging from 0 to 9.

- Model architecture

I used 3 convolutional layers, with the number of filters increasing from 16 to 128. BatchNorm, Dropout and MaxPool layers are added after each Conv layer. Then two fully connected layers are added to flatten and convert the output into 10 units. The accuracy on train set could reach over 95% after training the CNN model for 20 epochs.

Run the following code in terminal to get the predicted output.
```
python3 digit_classification/main.py
```

## [Fraud Detection](fraud_detection/main.py)

- Required packages
    - numpy, pandas
    - lightgbm

- Data description

The fraud detection train set can be downloaded via this [link](https://drive.google.com/file/d/1IV2gZJc4MOlZTV3bdnb_kPuHCFVeVPuz/view?usp=sharing). Download and put it under the `fraud_detection` folder. The dataset contains 1,296,676 transaction records. Each record has 23 features.The feature description can be found in `fraud_detection/feature_description.txt`. It is very imbalanced, with fraud to non-fraud ratio of nearly 1:130.

- Feature engineering

I extracted the prefix of credit card number, month, hour of the transaction and also whether the transaction occurs at the weekend, computed age from date of birth and distretized the city population into seven levels.

- Model used

I used LightGBM for classification. To tackle the imbalanced dataset, I set the scale_pos_weight hyperparameter to be 120, so the model could better learn from the positive class. The model achieves a precision of 0.33, recall of 0.91 and AUC of 0.95 on the train set.

Run the following code in terminal to get the predicted output.
```
python3 fraud_detection/main.py
```

## [MovieLens Recommendation System](movieLens_Resys/main.py)

- Required packages
    - Tensorflow
    - DeepCTR

- Data description

There are 4 datasets: `movies.csv`, `rating_train.csv`, `rating_test.csv`, and `users.csv`. Specifically, `movies.csv` contains the titles and genres of 3,883 movies; `rating_train.csv` includes user ratings to different movies; `rating_test.csv` is the test set containing movies that you are required to predict ratings for; `users.csv` introduces the basic information of 6,040 users. Please see `DATA_DESCRIPTION.txt` for more detailed description.

- Model used

I used DeepFM provided by the DeepCTR package for building the recommendation system. Details of the DeepFM model could be found in this original [paper].(https://arxiv.org/abs/1703.04247) The implementation largely follows this [documentation].(https://deepctr-doc.readthedocs.io/en/latest/Examples.html#multi-value-input-movielens)

Run the following code in terminal to get the predicted output.
```
python3 movieLens_Resys/main.py
```

## [Sentiment Analysis on Twitter Data](twitter_sentiment_analysis/notebook.ipynb)

- Required packages
    - numpy, pandas, scipy, sklearn
    - matplotlib
    - htmk
    - pickle
    - nltk, download corpus (stopwords, wordnet, treebank, words)
    - wordcloud

- Data description

The dataset used here is the “Sentiment140”, which originated from Stanford University. It contains 1,600,000 tweets extracted using the twitter api. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiments . It contains the following 6 fields: target, id, date, flag, user and text. The data can be downloaded via this [link].(https://drive.google.com/file/d/1s2cUol3CbfczjG328vuvmAhjLJMxJMkM/view?usp=sharing)

- Data cleaning procedures
    - Decode HTML using htmk package
    - Convert all text to lowercase
    - Remove @user, url links, stop words, non-characters including the hashtag # and punctuations
    - Tokenization and stemming

- Wordcloud

We could then visualize the frequent words in the positive text and negative text. 

Frequent Words in the Positive Tweets. [pic](twitter_sentiment_analysis/output/positive_wordcloud.png)

Frequent Words in the Negative Tweets. [pic](twitter_sentiment_analysis/output/negative_wordcloud.png)

- Word Frequency

We could also count the term frequency using the `sklearn.feature_extraction.text.CountVectorizer`. The token frequency of positive tweets and negative tweets are saved in `twitter_sentiment_analysis/output`.