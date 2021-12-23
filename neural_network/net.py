import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_feature, num_hidden_layer=1, n_hidden=1, num_class=1):
        """        
        @param n_feature: number of features
        @param num_hidden_layer: int
        @param n_hidden: int if only one hidden layer, otherwise is a list of hidden units
        @param num_class: 1 for binary classification, num_class for multi-class classification
        """
        super(Net, self).__init__()
        
        if num_hidden_layer == 1:
            layer_list = [
                nn.Linear(n_feature, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, num_class)
            ]
        else:
            layer_list = []
            for i in range(num_hidden_layer + 1):
                if i == 0:
                    layer_list.append(nn.Linear(n_feature, n_hidden[i]))
                    layer_list.append(nn.ReLU())
                elif i == num_hidden_layer:
                    layer_list.append(nn.Linear(n_hidden[i-1], num_class))
                else:
                    layer_list.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
                    layer_list.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        logits = self.layers(x)
        return logits