import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tab_transformer_pytorch import TabTransformer
from torch.nn.modules.loss import CrossEntropyLoss
import uuid


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import lib
from torch_losses import *
from torch_losses import DiceLoss


#https://github.com/lucidrains/tab-transformer-pytorch
#TODO separate features into categorical and continous

class CBCDataset(Dataset):
    def __init__(self, X, y):
        self.X_cont = X[:, 0:-1].astype(np.float32)
        self.X_cat = X[:, -1::].astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

df = pd.read_csv('datasets/processed/balanced_detection1.csv')
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df['Sex'] = df['Sex'].astype('int64')
X = np.array(df.copy().drop("target", axis=1))
y = np.array(df["target"])

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# class_weights = compute_class_weight("balanced", classes= np.unique(y), y=y)
# class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories = (1,),      # tuple containing the number of unique values within each category # bez przecinka po ostatim elemencie nie działa
    num_continuous = 10,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.2,                   # feed forward dropout
    mlp_hidden_mults = (20, 40, 80, 160, 1),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)



batch_size = 128
train_ds = CBCDataset(X_train, y_train)
valid_ds = CBCDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)

device = torch.device('cpu')

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

# to_device(model, device)




f1_prev = 0



# train_loop(model, train_dl, valid_dl, epochs=epochs, lr=lr, wd=0.00001)
criterion = nn.BCELoss() #(weight = class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer)
for epoch in range(10000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        # get the inputs; data is a list of [inputs, labels]
        x1, x2, labels = data

        # zero the parameter gradients
        

        # forward + backward + optimize
        outputs = model(x1, x2)
        loss = criterion(torch.nn.functional.sigmoid(torch.reshape(outputs, (-1,))), labels)
        # loss = criterion(torch.nn.functional.softmax(outputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            # print('[%d, %5d] loss: %.8f' %
            #       (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    scheduler.step(loss)

    y_test = y_train
    for i, data in enumerate(train_dl, 0):
        x1, x2, labels = data
        y_p = model(x1, x2)
        y_p = torch.nn.functional.sigmoid(torch.reshape(y_p, (-1,)))
        y_p = y_p.cpu().detach().numpy()
        if i == 0:
            y_pred = y_p
        else:
            y_pred = np.hstack((y_pred, y_p))

    y_pred = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1, drop_intermediate=False)
    auc_val = auc(fpr, tpr)
    print(f"Epoch {epoch}")
    print(f"Accuracy: {acc}")
    print(f"Specificity: {tn / (tn+fp)}")
    print(f"Sensitivity: {tp / (tp+fn)}")
    print(f'F1: {f1}')
    print(f'AUC: {auc_val}')
    # model._save_to_state_dict()
    # f1_prev = f1

print('Finished Training')

# X_cont = X_test[:, 0:-1].astype(np.float32)
# X_cat = X_test[:, -1::].astype(np.int64)
# y_pred = model(X_cat, X_cont)
