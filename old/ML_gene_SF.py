# Import Modules
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchviz import make_dot
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

# Read in data file
gene_raw = pd.read_csv('preprocessed_gene_expression.csv')
gene_raw.rename(columns={gene_raw.columns[0]:'ID',}, inplace=True)
gene_raw.columns = gene_raw.columns.astype(str)
del gene_raw['ID']
del gene_raw['group']

## CREATE DATA INPUTS
class Dataset_fix(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data.iloc[:, :-1].values).float()
        self.targets = torch.from_numpy(data.iloc[:, -1].values).float()
    def __getitem__(self, index):
        x = self.data[index] 
        y = self.targets[index]
        return x, y
    def __len__(self):
        return len(self.data)

batch_size = 32
# Get cpu, gpu or mps device for training.
device = ('cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu')
print(f'Using {device} device')

# Create data loaders
gene_test = gene_raw.sample(int(len(gene_raw)*0.5))
gene_train = gene_raw.sample(int(len(gene_raw)*0.25))
gene_val = gene_raw.sample(int(len(gene_raw)*0.25))
train_data = Dataset_fix(gene_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_data = Dataset_fix(gene_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
val_data = Dataset_fix(gene_val)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
all_data = Dataset_fix(pd.concat([gene_test, gene_train, gene_val]))
all_dataloaders = DataLoader(all_data, batch_size=batch_size)

## CREATE CNN MODEL
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1024, 512) 
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  #Sigmoid activation for binary classification
        return x

model = CNN().to(device)
print(model)

# Optimize model
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    epoch_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        y = y.view(-1, 1).float()
        batch_loss = loss_fn(pred, y)
        epoch_loss += batch_loss.item()
        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss_value, current = batch_loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]')
    return epoch_loss/size

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_targets = []
    t_p, t_n, a_p, a_n = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_prob = torch.sigmoid(model(X))
            pred_label = (pred_prob > 0.9).float().squeeze()
            batch_loss = loss_fn(pred_prob, y.view(-1,1).float())
            test_loss += batch_loss.item()
            all_preds.append(pred_prob.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            correct += (pred_label == y).sum().item()
            t_p += ((pred_label == 1) & (y == 1)).sum().item()
            t_n += ((pred_label == 0) & (y == 0)).sum().item()
            a_p += (y == 1).sum().item()
            a_n += (y == 0).sum().item()
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    correct /= size
    sensitivity = t_p / a_p
    specificity = t_n / a_n
        
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, \n Sensitivity: {sensitivity:.4f}, \n Specificity: {specificity:.4f} \n')
    return test_loss / size, all_preds, all_targets

def validate(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X, y in dataloader:
            pred = model(X)
            pred_prob = torch.sigmoid(pred)
            val_loss += loss_fn(pred, y.float().unsqueeze(1)).item()
    return val_loss / len(dataloader)

train_losses = []
test_losses = []
val_losses = []
epochs = 250
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_losses.append(train_loss)
    test_loss, _, _ = test(test_dataloader, model, loss_fn)
    test_losses.append(test_loss)
    val_loss = validate(val_dataloader, model, loss_fn)
    val_losses.append(val_loss)
    print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Val Loss: {val_loss:.4f} \n')

print('Done!')

# Save model
torch.save(model.state_dict(), 'model.pth') 
print('Saved PyTorch Model State to model.pth \n')

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth'))

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
        pred_label = (pred_prob > 0.9).long().squeeze()
        for pred, actual in zip(pred_label, y):
            predicted, actual = classes[int(pred.item())], classes[int(actual.item())]
            #print(f'Predicted: "{predicted}", Actual: "{actual}"')

## CREATE VISUALIZATIONS
dot = make_dot(model(X), params = dict(model.named_parameters()))
dot.render('model_graph', format='png', cleanup=True)

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
_, preds, targets = test(test_dataloader, model, loss_fn)
pred_labels = (preds > 0.9).astype(int)
plot_confusion_matrix(targets, pred_labels)

# Precision-recall Curve
precision, recall, _ = precision_recall_curve(targets, preds)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(targets, preds)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print('Done!')
