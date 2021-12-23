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
### 1. Hash Tree

Hash tree is a data structure to store frequent pattern for efficient counting and search. The implementation of hash tree can be found in `frequent_pattern_mining/hash_tree.py`. To build a hash tree, first need to put the item set in variable `candidate_str` in the format of "{1 2 3}, {1 4 5}, {1 2 4}". For simplicity, this implementation is only for max leaf size of 3, so as the depth of the final hash tree. If there are leaves which cannot further split, with leaf size larger than 3. In these cases, we will use linked list to append the additional item set when printing out the tree.

Then run the following command in terminal, a hash tree will be built and printed out. The printed tree rotates 90 degree and the indentation is related to the depth.

```
python3 frequent_pattern_mining/hash_tree.py
```

### 2. Frequent Pattern Tree

Frequent pattern tree is used to mine frequent patterns within a transaction database. The item dataset is storeed in `frequent_pattern_mining/DataSetA.csv`. It contains 12,526 records and each record records every single transaction in the grocery store. Each line represents a single transaction with names of products. You could modify the minimum support threshold in `fp_tree = FPTree(data, min_support=2500, item="NULL", item_freq=0)`. The default minimum support threshold is 2500.

To run the program, run the following command in terminal.
```
python3 frequent_pattern_mining/FP_tree.py
```

## Classification
### 1. Simple classification

In `classification/simple_classifiers.py`, I compared model performance of decision tree, KNN and random forest on the wine quality dataset under `classifcation/dataset`.

For decision tree, I tuned hyperparameters including criterion in ["gini", "entropy"] and max_depth in [5, 10, 15].  For KNN, I tuned the n_neighbors in [1, 5, 10]. For random forest, I tuned hyperparameters min_samples_split in [5, 10] and n_estimators in [50, 100]. 

To run the program, run the following command in terminal.
```
python3 classification/simple_classifiers.
```

### 2. Adaboost

The above classifers are directly imported from sklearn, while I also implemented the Adaboost from scratch. Adaboost is a boosting method, consisted of any weak classifier. In `classification/adaboost.py`, I used a simply binary weak classifier - predict 1 when x is larger than a threshold and predict -1 otherwise. The sign could be flipped to minimize the loss. When building each weak classifier, the weight of data is adjusted according to the classification accuracy of previous classifier.

The default number of boosting rounds is 10, you could modify in `adaboost = Adaboost(n_boosting=10)`. For simplicity, I only use 10 data points, you could change the data in model initialization.

To run the program, run the following command in terminal. The input, true label, predicted value, estimator's weight, error rate and threshold of each weak classifier will be printed out.
```
python3 classification/adaboost.py
```

### 3. Neural Network

Aside from simple machine learing methods for classification, neural network is also widely used. I implemented a simple single layer neural network in Pytorch for binary classification (breast-cancer, diabets, iris and wine dataset) and a two-layer neural network for multi-class classification (hand-written digit images).

To run the program, run the following command in terminal. The model classification results on train and test are saved in `neural_network/pictures`.
```
python3 neural_network/main.py
```

## Clustering

### 1. DBSCAN

DBSCAN is a clustering algorithm, yet could also be used to detect outliers. There are two most important hyperparameters, namely eps and minPts. In `clustering/DBSCAN.py`, I implemented the algorithm to detect the outliers in the `clustering/dataset/DBSCAN_Points.txt`. From the output images in `clustering/pictures/DBSCAN`, we could see that when eps=2.0 and minPts=15 or eps=2.5 and minPts=20, it could detect the outliers, colored in black. 

To run the program, run the following command in terminal. The clustering results on are saved in `clustering/pictures/DBSCAN`.
```
python3 clustering/DBSCAN.py
```

### 2. Fuzzy KMeans in EM

Expectation - Maximization (EM) algorithm is an iteratively method to find maximum likelihood. I implemented a fuzzy KMeans algo using the EM method in `clustering/FuzzyCluster_EM.py`. The fuzzy KMeans is a soft version of KMeans clustering, in which each point could belong to more than one clustering, with certain probabilities. There are three hyperparameters: number of iterations, epsilon to evaluate whether the stopping criterion is met and also the p -  the fuzzy degree.

To run the program, run the following command in terminal. The clustering results on are saved in `clustering/pictures/fuzzy_cluster`.
```
python3 clustering/FuzzyCluster_EM.py
```

### 3. Nested Loop for Outlier Detection

Nested loop algorithm was introduced by Knorr & Ng in 1998, and this implementation is based on their paper - [Algorithms for Mining Distance-based Outliers in Larger Dataset](http://www.vldb.org/conf/1998/p392.pdf).

To run the program, run the following command in terminal. The clustering results on are saved in `clustering/pictures/NestedLoop`.
```
python3 clustering/NestedLoop.py
```