import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.modules.loss import CrossEntropyLoss

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, auc, roc_curve

# Select dataset
df = pd.read_csv('datasets/processed/auxiliary/balanced_uck_detection.csv')
# df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)
df = df[['BAT', 'EOT', 'LYT', 'MOT', 'HGB', 'MCHC', 'MCV', 'PLT', 'WBC', 'Age', 'Sex', 'target']]
df['Sex'] = df['Sex'].astype('int64')


# Select features and the target variable
X = np.array(df.copy().drop("target", axis=1))
y = np.array(df["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

## ----------------------------------------
## For transfer learning uncomment lines below
# source = 'zenodo'
# target = 'uck'

# df_uck = pd.read_csv('datasets/processed/auxiliary/balanced_uck_detection.csv')
# df_uck = df_uck[['BAT', 'EOT', 'LYT', 'MOT', 'HGB', 'MCHC', 'MCV', 'PLT', 'WBC',
#        'Age', 'Sex', 'target']]
# df_zenodo = pd.read_csv('datasets/processed/auxiliary/zenodo_detection.csv')
# df_zenodo = df_zenodo[['BAT', 'EOT', 'LYT', 'MOT', 'HGB', 'MCHC', 'MCV', 'PLT', 'WBC',
#        'Age', 'Sex', 'target']]

# if target == 'uck':
#     df1 = df_uck
#     df2 = df_zenodo
# else:
#     df1 = df_zenodo
#     df2 = df_uck
# X = df1.copy().drop('target', axis=1)
# y = df1['target']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
# X2 = df2.copy().drop('target', axis=1)
# y2 = df2['target']

# X_train = np.concatenate((X_train, X2), axis=0) #.reshape(-1)
# y_train = np.concatenate((y_train, y2), axis=0).reshape(-1)
## ----------------------------------------


# Impute missing values with K-nearest neighbours
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)


# Scale values to <-1, 1>
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

class_weights = compute_class_weight("balanced", classes= np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()


# Train TabNet model
clf = TabNetClassifier( optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-2),
    scheduler_params={"step_size":10, # how to use learning rate scheduler
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
    n_d= 32,
    n_a=32,
    cat_emb_dim = 1
    
    )# This will be overwritten if using pretrain model)
clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=5000,
        patience=60,
        loss_fn=CrossEntropyLoss(weight = class_weights),
        batch_size=256, # UCK 2048 # zenodo 128
        eval_metric=['balanced_accuracy']#['balanced_accuracy']
    )

# Print metrics
y_pred = clf.predict(X_test) > 0.5

acc = accuracy_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Accuracy: {acc}")
print(f"Specificity: {tn / (tn+fp)}")
print(f"Sensitivity: {tp / (tp+fn)}")

f1 = f1_score(y_test, y_pred)

print(f'F1: {f1}')

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1, drop_intermediate=False)
auc_val = auc(fpr, tpr)
print(f'AUC: {auc_val}')

clf.save_model('tabnet_detection')
