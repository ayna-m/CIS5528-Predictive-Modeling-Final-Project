# Import Modules
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Read in data file
gene_raw = pd.read_csv('preprocessed_gene_expression.csv')
gene_raw.rename(columns={gene_raw.columns[0]:'ID',}, inplace=True)
for i in range(1,(len(gene_raw.columns)-2)):
    gene_raw.rename(columns={gene_raw.columns[i]:str(i)}, inplace=True)
del gene_raw['ID']

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
gene_grouped = gene_raw.groupby('group') 
gene_test = gene_grouped.get_group('test')
del gene_test['group']
gene_train = gene_grouped.get_group('train')
del gene_train['group']
gene_val = gene_grouped.get_group('val')
del gene_val['group']

batch_size = 990
# Get cpu, gpu or mps device for training.
device = ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f"Using {device} device")

# Assign training + test data
train_data = Dataset_fix(gene_train)
test_data = Dataset_fix(gene_test)

# Create data loaders
##labels = {
##    0: 'living',
##    1: 'deceased'}
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Check data sizes
for X, y in train_dataloader:
    print(f'Shape of X: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')

## CREATE CNN MODEL
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 10), ## first layer
            nn.ReLU(),
            nn.Linear(10, 20), ## second layer
            nn.ReLU(),
            nn.Linear(20, 2)) ## third layer 

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
model = NeuralNetwork().to(device)
print(model)

# Optimize model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #X = X.view(1, y.shape[0] * y.shape[0]) 
        #y = y.view(X.shape[0] * X.shape[1], 1) 
        # Compute prediction error
        pred = model(X) 
        loss = loss_fn(pred, y) 
        print('train3')
        # Backpropagation
        loss.backward()
        optimizer.step()
        print('train4')
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
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
classes = ['living', 'deceased']
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
