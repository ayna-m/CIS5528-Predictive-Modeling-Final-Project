import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import utils
import torch.nn as nn

class Clinical(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 60)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5) 
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5) 
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5) 
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)  
        x = self.act2(self.layer2(x))
        x = self.dropout2(x) 
        x = self.act3(self.layer3(x))
        x = self.dropout3(x)  
        x = self.sigmoid(self.output(x))
        return x


class Gene(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(403, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.output = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.output(x))
        return x

class CNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(400, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.output = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.output(x))
        return x
 
clinical_X_train, clinical_X_test, clinical_y_train, clinical_y_test= utils.get_data("ML_project_data/preprocessed_clinical_data_v2.csv", 5)
clinical_model, clinical_acc_scores, clinical_train_losses = utils.get_acc_loss(Clinical(), clinical_X_train, clinical_y_train, 'clinical')
utils.plot_loss(clinical_train_losses,'clinical')
utils.eval(clinical_model, clinical_X_test, clinical_y_test, 'clinical')

gene_X_train, gene_X_test, gene_y_train, gene_y_test = utils.get_data("ML_project_data/preprocessed_gene_expression_v2.csv", 403)
gene_model, gene_acc_scores, gene_train_losses = utils.get_acc_loss(Gene(), gene_X_train, gene_y_train, 'gene')
utils.plot_loss(gene_train_losses,'gene')
utils.eval(gene_model, gene_X_train, gene_y_train, 'gene')

cna_X_train, cna_X_test, cna_y_train, cna_y_test = utils.get_data("ML_project_data/preprocessed_cna_v2.csv", 400)
cna_model, cna_acc_scores, cna_train_losses = utils.get_acc_loss(CNA(), cna_X_train, cna_y_train, 'cna')
utils.plot_loss(cna_train_losses,'cna')
utils.eval(cna_model, cna_X_train, cna_y_train, 'cna')

clinical_model = Clinical()
gene_model = Gene()
cna_model = CNA()

print('Initiated models')

# Load the model state dictionary
clinical_model.load_state_dict(torch.load('clinical_model.pth'))
gene_model.load_state_dict(torch.load('gene_model.pth'))
cna_model.load_state_dict(torch.load('cna_model.pth'))

print('Loaded models')

clinical_X_test, clinical_y_test = utils.get_combined_data("ML_project_data/preprocessed_clinical_comb_v2.csv", 5)
gene_X_test, gene_y_test = utils.get_combined_data("ML_project_data/preprocessed_gene_comb_v2.csv", 403)
cna_X_test, cna_y_test = utils.get_combined_data("ML_project_data/preprocessed_cna_comb_v2.csv", 400)

clinical_predictions = clinical_model(clinical_X_test).round()
gene_predictions = gene_model(gene_X_test).round()
cna_predictions = cna_model(cna_X_test).round()

# Calculate combined probabilities
combined_probabilities = []

for i in range(len(clinical_predictions)):
    predictions = [clinical_predictions[i], gene_predictions[i], cna_predictions[i]]
    majority_vote = Counter(predictions).most_common(1)[0][0]
    combined_probabilities.append([majority_vote.float()])

combined_probabilities_tensor = torch.tensor(combined_probabilities)

precision, recall, _ = precision_recall_curve(clinical_y_test, combined_probabilities_tensor.detach().numpy())

# Calculate ROC curve
fpr, tpr, _ = roc_curve(clinical_y_test, combined_probabilities_tensor.detach().numpy())

# Calculate area under the curves
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

utils.plot_roc(fpr, tpr, roc_auc, 'combined')
utils.plot_pr(recall, precision, pr_auc, 'combined')


tp = fp = fn = correct = 0


for i in range(len(clinical_predictions)):
    predictions = [clinical_predictions[i], gene_predictions[i], cna_predictions[i]]
    majority_vote = Counter(predictions).most_common(1)[0][0]  
    if majority_vote == 1 and clinical_y_test[i] == 1:
        tp += 1
        correct += 1
    elif majority_vote == 1 and clinical_y_test[i] == 0:
        fp += 1
    elif majority_vote == 0 and clinical_y_test[i] == 1:
        fn += 1
    elif majority_vote == 0 and clinical_y_test[i] == 0:
        correct += 1


precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
accuracy = correct / len(clinical_y_test)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

