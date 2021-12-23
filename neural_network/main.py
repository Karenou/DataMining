import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

np.random.seed(0)
torch.manual_seed(0)

from net import Net
from data_utils import load_data, get_dataloader
from plot_utils import plot_history


def get_optimizer(opt, lr, model):
    if opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt =="adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)   
    else:
        print("please input correct optimizer name")
    return optimizer  


class Model:
    def __init__(self, path, file_name, params):
        """
        @param path: folder path to load data
        @param file_name: file_name of npz file
        @param params: a dictionary of hyperparameters
        """
        self.base_path = path
        self.file_name = file_name
        self.params = params
    
    def run(self):
        """
        main function to run the model
        """
        # load dataset
        x_train, y_train, x_val, y_val, x_test, y_test = load_data(self.base_path, self.file_name, self.params["num_class"])
        train_loader = get_dataloader(x_train, y_train, mode="train", batch_size=self.params["batch_size"], num_workers=0)

        # initial metrics
        best_acc, best_auc, best_n_hidden, best_model, history = 0, 0, None, None, None

        # set up param grid
        if self.params["num_class"] == 2:
            grid = ParameterGrid({"n_hidden_1": self.params["n_hidden_1"]})
        else:
            grid = ParameterGrid({"n_hidden_1": self.params["n_hidden_1"], "n_hidden_2": self.params["n_hidden_2"]})

        # start training 
        for param in grid:
            if self.params["num_class"] == 2:
                print("n_hidden: %d" % param["n_hidden_1"])
                model = Net(
                    x_train.shape[1], num_hidden_layer=self.params["num_hidden_layer"], 
                    n_hidden=param["n_hidden_1"], num_class=1
                    )
                criterion = nn.BCEWithLogitsLoss(reduction="sum")
            else:
                print("n_hidden: [%d, %d]" % (param["n_hidden_1"], param["n_hidden_2"]))
                model = Net(
                    x_train.shape[1], num_hidden_layer=self.params["num_hidden_layer"], 
                    n_hidden=[param["n_hidden_1"], param["n_hidden_2"]], num_class=self.params["num_class"]
                    )
                criterion = nn.CrossEntropyLoss(reduction="sum")

            optimizer = get_optimizer(self.params["optimizer"], self.params["lr"], model)
            model = self.train(
                train_loader, model, optimizer, criterion, self.params["epochs"], verbose=self.params["verbose"]
            )

            # evaluate performance on train set
            train_loss = self.compute_loss(model, criterion, x_train, y_train)
            train_pred_score, train_pred = self.predict(model, x_train, y_train)
            train_acc, train_auc, train_f1, _ = self.evaluate_metrics(y_train, train_pred, train_pred_score)
            print("train_set loss: %.4f, accuracy: %.4f, auc: %.4f, f1: %.4f" % (train_loss, train_acc, train_auc, train_f1))

            # evalute performance on val set
            val_loss = self.compute_loss(model, criterion, x_val, y_val)
            val_pred_score, val_pred = self.predict(model, x_val, y_val)
            val_acc, val_auc, val_f1, _ = self.evaluate_metrics(y_val, val_pred, val_pred_score)
            print("val_set loss: %.4f, accuracy: %.4f, auc: %.4f, f1: %.4f" % (val_loss, val_acc, val_auc, val_f1)) 
            
            # perform cross-validation to select the best parameter
            if val_acc > best_acc and val_auc > best_auc:
                if self.params["num_class"] == 2:
                    best_n_hidden = param["n_hidden_1"]
                else:
                    best_n_hidden = [param["n_hidden_1"], param["n_hidden_2"]]
                best_acc = val_acc
                best_auc = val_auc
                best_model = model
            
            print()
            
            if self.params["num_class"] == 2:
                history = self.update_history(history, param["n_hidden_1"], train_acc, train_auc, train_f1, val_acc, val_auc, val_f1)
            else:
                history = self.update_history(history, [param["n_hidden_1"], param["n_hidden_2"]], train_acc, train_auc, train_f1, val_acc, val_auc, val_f1)
        
        # plot the history 
        if not os.path.exists("./pictures"):
            os.makedirs("./pictures")
        plot_history(history, num_class=self.params["num_class"], ylim=self.params["figsize"], 
                    fig_save_path="./pictures/%s_history.png" % self.file_name
                    )

        if self.params["num_class"] == 2:
            print("The best performance is achieved when n_hidden = %d" % best_n_hidden)
        else:
            print("The best performance is achieved when n_hidden = [%s]" % ",".join(map(str, best_n_hidden)))

        # evaluate final performance on whole train set
        x_train = torch.cat([x_train, x_val], dim=0)
        y_train = torch.cat([y_train, y_val], dim=0)
        train_loss = self.compute_loss(best_model, criterion, x_train, y_train)
        train_pred_score, train_pred = self.predict(best_model, x_train, y_train)
        train_acc, train_auc, train_f1, _ = self.evaluate_metrics(y_train, train_pred, train_pred_score)
        print("train_set loss: %.4f, accuracy: %.4f, auc: %.4f, f1: %.4f" % (train_loss, train_acc, train_auc, train_f1))

        # evaluate final performance on test set
        test_loss = self.compute_loss(best_model, criterion, x_test, y_test)
        test_pred_score, test_pred = self.predict(best_model, x_test, y_test)
        test_acc, test_auc, test_f1, table = self.evaluate_metrics(y_test, test_pred, test_pred_score, conf_matrix=True)
        print("test_set loss: %.4f, accuracy: %.4f, auc: %.4f, f1: %.4f" % (test_loss, test_acc, test_auc, test_f1))
        print(table)

    def compute_loss(self, model, criterion, x, y):
        """
        compute the loss of the trained network on the whole train / test set
        @param model: trained model
        @param criterion: loss function
        @param x: input
        @param y: label
        """
        model.eval()
        logits = model(x)

        if self.params["num_class"] == 2:
            loss = criterion(logits, y.float())
        else:
            loss = criterion(logits, y.long().reshape(-1))

        return loss.item() / x.shape[0]

    def train(self, train_loader, model, optimizer, criterion, epochs, verbose=True):
        """
        @param train_loader: dataloader of the train_set
        @param model: a nn model
        @param optimizer: torch.optim
        @param criterion: loss function
        @param epochs: number of training epochs
        @param verbose: True to print out the loss in each epoch
        """

        for epoch in range(epochs):  
            running_loss = 0.0
            N = 0

            for i, data in enumerate(train_loader):
                inputs, labels = data

                optimizer.zero_grad()
                logits = model.forward(inputs.float())

                if self.params["num_class"] == 2:
                    loss = criterion(logits, labels.float())
                else:
                    loss = criterion(logits, labels.long().reshape(-1))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                N += inputs.shape[0]
            
            if verbose:
                print("Training loss at epoch %d: %.6f" % (epoch + 1, running_loss / N)) 
            
        print('Finished Training')
        return model

    def predict(self, model, x, y):
        """
        predict the class probabilities and label
        @param model: trained model
        @param x:
        @param y:
        """
        # set as eval mode
        model.eval()
        logits = model(x)
        
        if self.params["num_class"] == 2:
            pred_prob = torch.sigmoid(logits)
            pred_label = (pred_prob > 0.5) * 1
        else:
            pred_prob = F.softmax(logits, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)

        return pred_prob, pred_label

    def evaluate_metrics(self, y_true, y_pred, y_pred_score, conf_matrix=False):
        """
        compute the accuracy, auc, f1 score
        @param y_true: a tensor of true label
        @param y_pred: a tensor of predicted label
        @param y_preds_sore: a tensor of predicted probability
        @param conf_matrix: whether return the confusion matrix
        """
        if self.params["num_class"] == 2:
            f1 = f1_score(y_true.detach().numpy(), y_pred.detach().numpy(), average="binary")
            auc = roc_auc_score(y_true.detach().numpy(), y_pred_score.detach().numpy())
        else:
            f1 = f1_score(y_true.detach().numpy(), y_pred.detach().numpy(), average="macro")
            auc = roc_auc_score(y_true.detach().numpy(), y_pred_score.detach().numpy(), multi_class="ovr")
        
        acc = accuracy_score(y_true, y_pred)

        if conf_matrix:
            table = confusion_matrix(y_true.detach().numpy(), y_pred.detach().numpy())
        else:
            table = None
            
        return acc, auc, f1, table

    def update_history(self, history, n_hidden, train_acc, train_auc, train_f1, val_acc, val_auc, val_f1):
        """
        append the metrics in each epoch in the history
        """
        if history is None:
            history = {
            "n_hidden": [],
            "train_acc": [],
            "train_auc": [],
            "train_f1": [],
            "val_acc": [],
            "val_auc": [],
            "val_f1": []
        }

        history["n_hidden"].append(n_hidden)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["train_f1"].append(train_f1)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        return history




# binary class
base_path = "./datasets/bi-class"

print("---------------Breast cancer dataset------------------")
params = {
    "num_class": 2,
    "batch_size": 5,
    "epochs": 10,
    "num_hidden_layer": 1,
    "n_hidden_1": np.arange(1, 10, 1),
    "optimizer": "adam",
    "lr": 0.1,
    "verbose": True,
    "figsize": (0.94, 1.0)
}
model = Model(base_path, "breast-cancer", params)
model.run()


print("---------------Diabetes------------------")
params = {
    "num_class": 2,
    "batch_size": 5,
    "epochs": 10,
    "num_hidden_layer": 1,
    "n_hidden_1": np.arange(1, 10, 1),
    "optimizer": "adam",
    "lr": 0.15,
    "verbose": True,
    "figsize": (0.5, 1.0)
}
model = Model(base_path, "diabetes", params)
model.run()


print("---------------Iris------------------")
params = {
    "num_class": 2,
    "batch_size": 5,
    "epochs": 5,
    "num_hidden_layer": 1,
    "n_hidden_1": np.arange(1, 10, 1),
    "optimizer": "adam",
    "lr": 0.15,
    "verbose": True,
    "figsize": (0.5, 1.0)
}
model = Model(base_path, "iris", params)
model.run()


print("---------------Wine------------------")
params = {
    "num_class": 2,
    "batch_size": 5,
    "epochs": 10,
    "num_hidden_layer": 1,
    "n_hidden_1": np.arange(1, 10, 1),
    "optimizer": "adam",
    "lr": 0.05,
    "verbose": True,
    "figsize": (0., 1.0)
}
model = Model(base_path, "wine", params)
model.run()


# # multi-class
base_path = "./datasets/multi-class"
print("-----------------digits----------------")
params = {
    "num_class": 10,
    "batch_size": 5,
    "epochs": 10,
    "num_hidden_layer": 2,
    "n_hidden_1": [15, 25, 50], 
    "n_hidden_2": [5, 10, 20],
    "optimizer": "adam",
    "lr": 0.008,
    "verbose": True,
    "figsize": (0., 1.0)
}
model = Model(base_path, "digits", params)
model.run()