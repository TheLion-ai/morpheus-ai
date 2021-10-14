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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tab_transformer_pytorch import TabTransformer
from torch.nn.modules.loss import CrossEntropyLoss
import uuid

import lib
from torch_losses import *
from torch_losses import DiceLoss

#https://github.com/lucidrains/tab-transformer-pytorch
#TODO separate features into categorical and continous

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
df_cbc = df_cbc.drop(cbc_cols_del, axis=1).drop(["Sex", "Suspect"], axis=1)


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
cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories = [],      # tuple containing the number of unique values within each category
    num_continuous = 10,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

trainer = lib.Trainer(
    model=model, loss_function=F.cross_entropy,
    experiment_name=str(uuid.uuid4()),
    warm_start=False,
    Optimizer=torch.optim.Adam,
    # optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
    verbose=True,
    n_last_checkpoints=5
)

loss_history, err_history = [], []
best_val_err = 1.0
best_step = 0
early_stopping_rounds = 2500
report_frequency = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for batch in lib.iterate_minibatches(X_train,
                                     y_train,
                                     batch_size=512, 
                                     shuffle=True,
                                     epochs=float('inf')):
    
    # batch_x, batch_y = batch
    # batch_x = {""}
    
    metrics = trainer.train_on_batch(*batch, device=device)
    
    loss_history.append(metrics['loss'])

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')
        err = trainer.evaluate_classification_error(
            X_test,
            y_test,
            device=device,
            batch_size=128)
        
        if err < best_val_err:
            best_val_err = err
            best_step = trainer.step
            trainer.save_checkpoint(tag='best')
        
        err_history.append(err)
        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()
            
        plt.figure(figsize=[12, 6])
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.grid()
        plt.subplot(1,2,2)
        plt.plot(err_history)
        plt.grid()
        plt.show()
        print("Loss %.5f" % (metrics['loss']))
        print("Val Error Rate: %0.5f" % (err))
        
    if trainer.step > best_step + early_stopping_rounds:
        print('BREAK. There is no improvement for {} steps'.format(early_stopping_rounds))
        print("Best step: ", best_step)
        print("Best Val Error Rate: %0.5f" % (best_val_err))
        break
