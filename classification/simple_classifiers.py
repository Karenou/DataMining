import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Model:

    def __init__(self) -> None:
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def read_data(self, path: dict) -> None:
        """
        @param path: path of train and test csv files
        return a pandas dataframe
        """
        train_data = pd.read_csv(path["train"], sep=";")
        test_data = pd.read_csv(path["test"], sep=";")
        
        self.y_train = train_data["quality"]
        self.x_train = train_data.drop("quality", axis=1)
        self.y_test = test_data["quality"]
        self.x_test = test_data.drop("quality", axis=1)
    
    def get_classifier(self, param, model):
        """
        @param: a dictionary of hyperparameter
        @param model: name of model
        return a classifier
        """
        
        if model == "decision tree":
            clf = DecisionTreeClassifier(**param)
        elif model == "knn":
            clf = KNeighborsClassifier(**param)
        elif model == "random forest":
            clf = RandomForestClassifier(**param)
        else:
            print("please input the correct model name")
            return
    
        return clf
    
    def build_model(self, params, model="decision tree"):
        """
        @param params: a dictionary of hyperparameters
        @param model: name of model
        return model index and model metrics dictionary 
        """
        
        model_idx = {}
        metrics = {}
        grid = ParameterGrid(params)
        print("-----------start to build %s-------------" % model)

        for idx, param in zip(range(len(grid)), grid):
            model_idx[idx] = param
            metrics[idx] = {}
        
            start_time = datetime.now()
            clf = self.get_classifier(param, model=model)
            clf.fit(self.x_train, self.y_train)
            end_time = datetime.now()
            metrics[idx]["training_time"] = (end_time - start_time)
            
            y_pred = clf.predict(self.x_test)
            metrics[idx]["accuracy"] = round(accuracy_score(self.y_test, y_pred), 4)
            metrics[idx]["prediction"] = round(precision_score(self.y_test, y_pred, average="macro",zero_division=0), 4)
            metrics[idx]["recall"] = round(recall_score(self.y_test, y_pred, average="macro", zero_division=0), 4)
            metrics[idx]["F1"] = round(f1_score(self.y_test, y_pred, average="macro", zero_division=0), 4)
        
        return model_idx, metrics

model = Model()
path = {
    "train": "dataset/winequality_train.csv",
    "test": "dataset/winequality_test.csv"
}

model.read_data(path)

# build decision tree
dt_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5,10,15,20]
}

dt_idx, dt_metrics = model.build_model(dt_params, model="decision tree")

print("-----------decision tree model performance-------------")
for k, v in dt_metrics.items():
    print(dt_idx[k])
    print(v)


# build knn classifier
knn_params = {
    "n_neighbors": [1, 5, 10]
}

knn_idx, knn_metrics = model.build_model(knn_params, model="knn")

print("-----------knn model performance-------------")
for k, v in knn_metrics.items():
    print(knn_idx[k])
    print(v)


# build random forest 
rf_params = {
    "n_estimators": [50, 100],
    "min_samples_split": [5, 10]
}

rf_idx, rf_metrics = model.build_model(rf_params, model="random forest")

print("-----------random forest model performance-------------")
for k, v in rf_metrics.items():
    print(rf_idx[k])
    print(v)

