import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

def to_tensor(x, y):
    """
    convert x and y from numpy array to torch tensor type
    @param x: np.array
    @param y: np.array
    """
    return torch.from_numpy(x).float(), torch.from_numpy(y).reshape(-1, 1)

def load_data(path, file_name, num_class=2):
    """
    @param path: base path
    @param file_name: name of the npz file
    @param num_class: if bi-class, use train_Y, test_Y; otherwise, use train_y and test_y
    load npz file, split to train, val and test data
    """
    data = np.load("%s/%s.npz" % (path, file_name))
    if num_class == 2:
        x_train, x_val, y_train, y_val = train_test_split(
            data["train_X"], data["train_Y"], 
            test_size=0.2, random_state=0
            )
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            data["train_X"], data["train_y"], 
            test_size=0.2, random_state=0
            )
    
    # convert from np.array to tensor
    x_train, y_train = to_tensor(x_train, y_train)
    x_val, y_val = to_tensor(x_val, y_val)
    if num_class == 2:
        x_test, y_test = to_tensor(data["test_X"], data["test_Y"])
    else:
        x_test, y_test = to_tensor(data["test_X"], data["test_y"])
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_dataloader(x, y, mode="train", batch_size=10, num_workers=2):
    """
    @param x: x in tensor
    @param y: y in tensor
    @param mode: train, val or test
    @param batch_size: us ein batch gradient descent
    @param num_workers
    """
    dataset = torch.utils.data.TensorDataset(x, y)
    
    if mode == "train":
        shuffle=True
    else:
        shuffle=False
    
    loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
    )

    return loader