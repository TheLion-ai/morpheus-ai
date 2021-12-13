from statistics import mean

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

import torch
from torch.nn.modules.loss import CrossEntropyLoss
from pytorch_tabnet.tab_model import TabNetClassifier

from torch_losses import *
from torch_losses import DiceLoss


def create_metrics_dict():
    metrics = {
        'accuracy': [],
        'f1': [],
        'class_report':{
            '0.0': {
                'precision': [],
                'recall': [],
                'f1-score': [],
                'support': []
            },
            '1.0': {
                'precision': [],
                'recall': [],
                'f1-score': [],
                'support': []
            },
            '2.0': {
                'precision': [],
                'recall': [],
                'f1-score': [],
                'support': []
            },
            'accuracy': [],
            'macro avg': {
                'precision': [],
                'recall': [],
                'f1-score': [],
                'support': []
            },
            'weighted avg': {
                'precision': [],
                'recall': [],
                'f1-score': [],
                'support': []
            }
        }
    }
    return metrics
    

def calc_metrics(y_true, y_pred, metrics):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics['accuracy'].append(acc)
    metrics['f1'].append(f1)
    for k1 in metrics['class_report'].keys():
        for k2 in metrics['class_report'][k1]:
            val = class_report[k1][k2]
            metrics['class_report'][k1][k2].append(val)
    return metrics


def get_avg_metrics(metrics):
    acc = round(mean(metrics['accuracy']), 2)
    f1 = round(mean(metrics['f1']), 2)

    class_report = {}
    print('     precision    recall    support')
    for k1 in metrics['class_report'].keys():
        class_report[k1] = {}
        for k2 in metrics['class_report'][k1]:
            val = mean(metrics['class_report'][k1][k2])*100
            val = round(val, 2)
            class_report[k1][k2] = val
    
    print(f'Accuracy: {acc}')
    print(f'F1: {f1}')
    print(f"classification report: {class_report}")

# Read dataset
df = pd.read_csv('../datasets/processed/uck_prognosis.csv')
df = df[['LYT', 'HGB', 'PLT', 'WBC', 'Age', 'Sex', 'target']]
df['Sex'] = df['Sex'].replace({'K': 0, 'M': 1}).astype('int64')
df['target'] = df['target'].replace({0: 0, 1: 1, 2: 1, 3: 2})
df.info()


# Get features and the target variable
X = df.copy().drop('target', axis=1)
y = df['target']


# Impute missing values with K-nearest neighbours
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)


# Scale features to <-1,1>
scalar = StandardScaler()
X = scalar.fit_transform(X)


# Train TabNet
metrics = create_metrics_dict()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train, test in skf.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    # oversample = SMOTE() 

    class_weights = compute_class_weight("balanced", classes= np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

    clf = TabNetClassifier( optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        scheduler_params={"step_size":10, # how to use learning rate scheduler
                        "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        n_d= 64,
        n_a=64,
        cat_emb_dim = 1,
        cat_idxs=[5],
        lambda_sparse=1e-2,
        n_independent=5
        )# This will be overwritten if using pretrain model)
    clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=5000,
            patience=60,
            loss_fn=CrossEntropyLoss(weight = class_weights),
            batch_size=64,
            eval_metric=['balanced_accuracy']
        )
    y_pred = clf.predict(X_test)
    metrics = calc_metrics(y_test, y_pred, metrics)

get_avg_metrics(metrics)
