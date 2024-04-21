# Import Modules
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Read in data file
cna_raw = pd.read_csv('preprocessed_cna.csv')
cna_raw.rename(columns={cna_raw.columns[0]:'ID',}, inplace=True)
for i in range(1,(len(cna_raw.columns)-2)):
    cna_raw.rename(columns={cna_raw.columns[i]:str(i)}, inplace=True)
del cna_raw['ID']

## CREATE DATA INPUTS
class Dataset_fix(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data.iloc[:, :-1].values).float()
        self.targets = torch.from_numpy(data.iloc[:, -1].values).long()
    def __getitem__(self, index):
        x = self.data[index] 
        y = self.targets[index]
        return x, y
    def __len__(self):
        return len(self.data)

# Group data types
cna_grouped = cna_raw.groupby('group')
cna_test = cna_grouped.get_group('test')
del cna_test['group']
cna_train = cna_grouped.get_group('train')
del cna_train['group']
cna_val = cna_grouped.get_group('val')
del cna_val['group']

batch_size = 32
# Get cpu, gpu or mps device for training.
device = ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f"Using {device} device")

# Assign training + test data
train_data = Dataset_fix(cna_train)
test_data = Dataset_fix(cna_test)
val_data = Dataset_fix(cna_val)

# Create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

## CNN MODEL
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 512), ## first layer
            nn.ReLU(),
            nn.Linear(512, 512), ## second layer
            nn.ReLU(),
            nn.Linear(512, 1), ## third layer accomodated to the binary output
            nn.Sigmoid())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
        
model = NeuralNetwork().to(device)
print(model)

# Optimize model
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        y = y.view(-1, 1).float()
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    t_p, t_n, a_p, a_n = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_prob = torch.sigmoid(pred)
            pred_label = (pred_prob > 0.5).float().squeeze()
            y = y.float().squeeze()
            test_loss += loss_fn(pred_label, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    t_p += ((pred_label == 1) & (y ==1)).sum().item()
    t_n += ((pred_label == 0) & (y ==0)).sum().item()
    a_p += (y == 1).sum().item()
    a_n += (y == 0).sum().item()
    sensitivity = t_p / a_p
    specificity = t_n / a_n
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, \n Sensitivity: {sensitivity:.4f}, \n Specificity: {specificity:.4f}, \n Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Save model
torch.save(model.state_dict(), "model.pth") 
print("Saved PyTorch Model State to model.pth")

# Load model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

# Make predictions 
classes = ['0', '1']
##labels = {
##    0: 'living',
##    1: 'deceased'}
model.eval()
with torch.no_grad():
    for X, y in val_dataloader:
        X, y = X.to(device), y.to(device)
        pred_prob = torch.sigmoid(model(X))
        pred_label = (pred_prob > 0.5).long().squeeze()
        for pred, actual in zip(pred_label, y):
            predicted, actual = classes[pred.item()], classes[actual.item()]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
