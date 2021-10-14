import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import statsmodels.api as sm
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.modules.loss import CrossEntropyLoss

from torch_losses import *
from torch_losses import DiceLoss



df_desc = pd.read_csv("feature_desc.csv")
df = pd.read_csv("all_training.csv")
df.info()


# Take only CBC columns
cbc_columns = list(df_desc.loc[df_desc["CBC features"] == 1]["Acronym"])
cbc_columns = [c for c in cbc_columns if c not in ["MPV", "RDW"]]
print(f"CBC columns:\n{cbc_columns}\n")

df_cbc = df[cbc_columns]
print(f"Removed values: {len(df)} - {len(df_cbc)} = {len(df)-len(df_cbc)}")

bin_cols = ["Sex", "Suspect", "target"]

for col in cbc_columns:
    if col not in bin_cols:
        df_cbc[col] = df_cbc[col].fillna(df[col].mean())
    else:
        df_cbc[col] = df_cbc[col].astype(int)

cbc_cols_del = ["NET", "RBC", "HCT", "MCH", "NE", "EO", "LY", "BAT"]
df_cbc = df_cbc.drop(cbc_cols_del, axis=1)


X = np.array(df_cbc.copy().drop("target", axis=1))
y = np.array(df_cbc["target"])


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# unsupervised_model = TabNetPretrainer(
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     mask_type='entmax' # "sparsemax"
# )

# unsupervised_model.fit(
#     X_train=X_train,
#     eval_set=[X_test],
#     pretraining_ratio=0.8,
# )

class_weights = compute_class_weight("balanced", classes= np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

clf = TabNetClassifier(    optimizer_fn=torch.optim.Adam,
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
        max_epochs=200,
        patience=30,
        loss_fn=CrossEntropyLoss(weight = class_weights),
        batch_size=16,
        eval_metric=['balanced_accuracy']
        # metrics = ['accuracy']
        # from_unsupervised=unsupervised_model
    )
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print(f"Accuracy: {acc}")
print(f"Specificity: {tn / (tn+fp)}")
print(f"Sensitivity: {tp / (tp+fn)}")
