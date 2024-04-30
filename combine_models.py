import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import Counter
import utils
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


class Clinical(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 128)
        self.act1 = nn.ReLU() 
        self.layer2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 128)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


class Gene(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(403, 1024)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(1024, 1024)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.output = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x

class CNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(400, 1024)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(1024, 1024)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(1024, 1024)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

 
clinical_X_train, clinical_X_test, clinical_y_train, clinical_y_test= utils.get_data("ML_project_data/preprocessed_clinical_data_v2.csv", 5)
clinical_model, clinical_acc_scores, clinical_train_losses = utils.get_acc_loss(Clinical(), clinical_X_train, clinical_y_train, 'clinical')
utils.plot_loss(clinical_train_losses,'clinical')
clinical_metrics = utils.eval(clinical_model, clinical_X_test, clinical_y_test, 'clinical')

gene_X_train, gene_X_test, gene_y_train, gene_y_test = utils.get_data("ML_project_data/preprocessed_gene_expression_v2.csv", 403)
gene_model, gene_acc_scores, gene_train_losses = utils.get_acc_loss(Gene(), gene_X_train, gene_y_train, 'gene')
utils.plot_loss(gene_train_losses,'gene')
gene_metrics = utils.eval(gene_model, gene_X_train, gene_y_train, 'gene')

cna_X_train, cna_X_test, cna_y_train, cna_y_test = utils.get_data("ML_project_data/preprocessed_cna_v2.csv", 400)
cna_model, cna_acc_scores, cna_train_losses = utils.get_acc_loss(CNA(), cna_X_train, cna_y_train, 'cna')
utils.plot_loss(cna_train_losses,'cna')
cna_metrics = utils.eval(cna_model, cna_X_train, cna_y_train, 'cna')

metrics = pd.DataFrame([clinical_metrics, gene_metrics, cna_metrics], 
                       columns = ['Precision', 'Recall', 'Specificy', 'Sensitivity', 'Accuracy'], 
                       index = ['Clinical_model', 'Gene_model', 'CNA_model'] )
metrics.to_csv('metric_v2.csv')

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

clinical_predictions = clinical_model(clinical_X_test)*1/3
gene_predictions = gene_model(gene_X_test)*1/3
cna_predictions = cna_model(cna_X_test)*1/3

# Calculate combined probabilities
combined_probabilities = []

for i in range(len(clinical_predictions)):
    predictions = [clinical_predictions[i], gene_predictions[i], cna_predictions[i]]
    majority_vote = sum(predictions).round()
    # majority_vote = Counter(predictions).most_common(1)[0][0]
    combined_probabilities.append([majority_vote])

combined_probabilities_tensor = torch.tensor(combined_probabilities)
tn, fp, fn, tp = confusion_matrix(clinical_y_test, combined_probabilities_tensor.detach().numpy()).ravel()
accuracy = (tp + tn)/(tn + fp + fn + tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(tn+fp)
sensitivity = tp/(tp+fn)

name = 'final'
print(f'Precision for {name} model:\t', precision)
print(f'Recall for {name} model:\t', recall)
print(f'Specificy for {name} model:\t', specificity)
print(f'Sensitivity for {name} model:\t', sensitivity)
print(f'Accuracy for {name} model:\t', accuracy)