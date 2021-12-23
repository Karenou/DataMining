import pandas as pd

import torch
from torch import nn

from model import *
from utils import plot_class_dist, prepare_data, calculat_acc


def train_model(train_data_loader, net, optimizer, n_epochs, model_save_path):
    for epoch in range(n_epochs):
        running_loss = 0.0
        net.train()
        correct, cnt = 0, 0
        for data in train_data_loader:
            imgs, label = data
            optimizer.zero_grad()
            output = net(imgs.float())
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            cnt += imgs.shape[0]
            correct += calculat_acc(output, label)
            
        print("epoch %d, loss: %.4f, accuracy: %.4f" % (epoch, running_loss, correct / cnt))
    
    torch.save({'model_state_dict': net.state_dict()}, model_save_path)


def predict(model, data_loader):
    model.eval()
    
    y_pred = None
    for data in data_loader:
        imgs, _ = data
        optimizer.zero_grad()
        output = model(imgs.float())
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        if y_pred is None:
            y_pred = output
        else:
            y_pred = torch.cat([y_pred, output], dim=0)
            
    return y_pred

# if use new data set, replace the path to load the data, and set isTrain = False
is_train = False
data = pd.read_csv("digit_classification/train_valid.csv", header=None)
labels = data.iloc[:, 784]

# save class distribution
plot_class_dist(labels, "digit_classification/output/class_distribution.png")

# reshape data in torch dataloader format
if is_train:
    shuffle = True
else:
    shuffle = False
data_loader = prepare_data(data, labels, n_dims=(28, 28), batch_size=64, shuffle=shuffle)

model = get_model("cnn")
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model_save_path = "digit_classification/output/model.pth"

if is_train:
    n_epochs = 20
    train_model(data_loader, model, optimizer, n_epochs, model_save_path)
else:
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_pred = predict(model, data_loader)
    y_pred = pd.DataFrame(y_pred.detach().numpy())
    print("Predict accuracy: %.4f" % (sum(labels == y_pred[0]) / len(labels)))
    y_pred.to_csv("digit_classification/output/predicted_results.csv", index=False, header=False)
