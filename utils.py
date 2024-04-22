import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def get_data(filename, num_features):
    data = pd.read_csv(filename)
    X = data.iloc[:, :num_features]
    y = data.iloc[:, num_features]
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    return X_train, X_test, y_train, y_test

def get_combined_data(filename, num_features):
    data = pd.read_csv(filename)
    data = data.sort_values(by=data.columns[0], ascending=True)
    data = data.drop(columns=data.columns[0])
    X = data.iloc[:, :num_features]
    y = data.iloc[:, num_features]
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1) 
    return X,y
    
def model_train(model, X_train, y_train, X_val, y_val):
    # Loss function and optimizer
    loss_fn = nn.BCELoss()  # Binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    n_epochs = 100   # Number of epochs to run
    batch_size = 64  # Size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    # Hold the best model
    best_acc = -np.inf   # Initialize to negative infinity
    best_weights = None
    
    # List to store losses
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # Take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_losses.append(loss.item()) 
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
                # Print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(loss), acc=float(acc))
        train_losses.append(np.mean(epoch_losses)) 
        
        # Evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    
    # Restore model and return best accuracy and losses
    model.load_state_dict(best_weights)
    return best_acc, train_losses

def get_acc_loss(model, X_train, y_train, name):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    acc_scores = []
    train_losses = []
    for train, test in kfold.split(X_train, y_train):       
        acc,train_loss = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
        print(f"{name} accuracy: %.2f" % acc)
        acc_scores.append(acc)
        train_losses.append(train_loss)
    deep_acc = np.mean(acc_scores)
    deep_std = np.std(train_losses)
    print(name, "%.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))
    torch.save(model.state_dict(), f'{name}_model.pth') 
    return model, acc_scores, train_losses

def plot_loss (train_losses, name):
    plt.plot(np.mean(train_losses, axis=0))
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.savefig(f'plots/{name}_loss.png')
    plt.close()

def plot_roc(fpr, tpr, auc_roc, name):
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.2f})') # ROC curve = TPR vs FPR
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f'plots/{name}_ROC_curve.png')
    plt.close()

def plot_pr(precision, recall, auc_pr, name):
    plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'plots/{name}_PR_curve.png')
    plt.close()

def eval(model, X_test, y_test, name):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred)
        plot_roc(fpr, tpr,auc_roc, name)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        auc_pr = average_precision_score(y_test, y_pred)
        plot_pr(precision, recall,auc_pr,name)
