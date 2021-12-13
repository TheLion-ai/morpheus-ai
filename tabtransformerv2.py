from tsai.imports import *

#export
# This is an unofficial TabTransformer implementation in Pytorch developed by Ignacio Oguiza - timeseriesAI@gmail.com based on:
# Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). 
# TabTransformer: Tabular Data Modeling Using Contextual Embeddings. 
# arXiv preprint https://arxiv.org/pdf/2012.06678
# Official repo: https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a

        
def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From fastai.layers
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    # From fastai.layers
    def __init__(self, ni, nf, std=0.01):
        super(Embedding, self).__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)
        

class SharedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, shared_embed=True, add_shared_embed=False, shared_embed_div=8):
        super().__init__()
        if shared_embed:
            if add_shared_embed:
                shared_embed_dim = embedding_dim
                self.embed = Embedding(num_embeddings, embedding_dim)
            else:
                shared_embed_dim = embedding_dim // shared_embed_div
                self.embed = Embedding(num_embeddings, embedding_dim - shared_embed_dim)
            self.shared_embed = nn.Parameter(torch.empty(1, 1, shared_embed_dim))
            trunc_normal_(self.shared_embed.data, std=0.01)
            self.add_shared_embed = add_shared_embed
        else: 
            self.embed = Embedding(num_embeddings, embedding_dim)
            self.shared_embed = None

    def forward(self, x):
        out = self.embed(x).unsqueeze(1)
        if self.shared_embed is None: return out
        if self.add_shared_embed:
            out += self.shared_embed
        else:
            shared_embed = self.shared_embed.expand(out.shape[0], -1, -1)
            out = torch.cat((out, shared_embed), dim=-1)
        return out


class FullEmbeddingDropout(nn.Module):
    '''From https://github.com/jrzaurin/pytorch-widedeep/blob/be96b57f115e4a10fde9bb82c35380a3ac523f52/pytorch_widedeep/models/tab_transformer.py#L153'''
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        mask = x.new().resize_((x.size(1), 1)).bernoulli_(1 - self.dropout).expand_as(x) / (1 - self.dropout)
        return mask * x

    
class _MLP(nn.Module):
    def __init__(self, dims, bn=False, act=None, skip=False, dropout=0., bn_final=False):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for i, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = i >= (len(dims) - 2)
            if bn and (not is_last or bn_final): layers.append(nn.BatchNorm1d(dim_in))
            if dropout and not is_last:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_out))
            if is_last: break
            layers.append(ifnone(act, nn.ReLU()))
        self.mlp = nn.Sequential(*layers)
        self.shortcut = nn.Linear(dims[0], dims[-1]) if skip else None

    def forward(self, x):
        if self.shortcut is not None: 
            return self.mlp(x) + self.shortcut(x)
        else:
            return self.mlp(x)


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k:int, res_attention:bool=False): 
        super().__init__()
        self.d_k,self.res_attention = d_k,res_attention
        
    def forward(self, q, k, v, prev=None, attn_mask=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                    # scores : [bs x n_heads x q_len x q_len]

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Attention mask (optional)
        if attn_mask is not None:                                     # mask with shape [q_len x q_len]
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores += attn_mask

        # SoftMax
        if prev is not None: scores = scores + prev

        attn = F.softmax(scores, dim=-1)                               # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                # context: [bs x n_heads x q_len x d_v]

        if self.res_attention: return context, attn, scores
        else: return context, attn


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int, res_attention:bool=False):
        """Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]"""
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.res_attention = res_attention

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k, self.res_attention)
        else:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k)

        
    def forward(self, Q, K, V, prev=None, attn_mask=None):

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            context, attn, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, attn_mask=attn_mask)
        else:
            context, attn = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)                                                           # context: [bs x q_len x d_model]

        if self.res_attention: return output, attn, scores
        else: return output, attn                                                            # output: [bs x q_len x d_model]

        
class _TabEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 res_dropout=0.1, activation="gelu", res_attention=False):

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        d_ff = ifnone(d_ff, d_model * 4)

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.layernorm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.layernorm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None, attn_mask=None):

        # Multi-Head attention sublayer
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.layernorm_attn(src) # Norm: layernorm 

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.layernorm_ffn(src) # Norm: layernorm

        if self.res_attention:
            return src, scores
        else:
            return src

    def _get_activation_fn(self, activation):
        if callable(activation): return activation()
        elif activation.lower() == "relu": return nn.ReLU()
        elif activation.lower() == "gelu": return nn.GELU()
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class _TabEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='gelu', res_attention=False, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([_TabEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout, 
                                                            activation=activation, res_attention=res_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, attn_mask=attn_mask)
            return output

        
class TabTransformer(nn.Module):
    def __init__(self, classes, cont_names, c_out, column_embed=True, add_shared_embed=False, shared_embed_div=8, embed_dropout=0.1, drop_whole_embed=False, 
                 d_model=32, n_layers=6, n_heads=8, d_k=None, d_v=None, d_ff=None, res_attention=True, attention_act='gelu', res_dropout=0.1, norm_cont=True,
                 mlp_mults=(20, 40, 80, 1), mlp_dropout=0., mlp_act=None, mlp_skip=False, mlp_bn=False, bn_final=False):

        super().__init__()
        n_cat = len(classes)
        n_classes = [len(v) for v in classes.values()]
        n_cont = len(cont_names)
        self.embeds = nn.ModuleList([SharedEmbedding(ni, d_model, shared_embed=column_embed, add_shared_embed=add_shared_embed, 
                                                     shared_embed_div=shared_embed_div) for ni in n_classes])
        n_emb = sum(n_classes)
        self.n_emb,self.n_cont = n_emb,n_cont
        self.emb_drop = None
        if embed_dropout:
            self.emb_drop = FullEmbeddingDropout(embed_dropout) if drop_whole_embed else nn.Dropout(embed_dropout)
        self.transformer = _TabEncoder(n_cat, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                       activation=attention_act, res_attention=res_attention, n_layers=n_layers)
        self.norm = nn.LayerNorm(n_cont) if norm_cont else None
        mlp_input_size = (d_model * n_cat) + n_cont
        hidden_dimensions = list(map(lambda t: int(mlp_input_size * t), mlp_mults))
        all_dimensions = [mlp_input_size, *hidden_dimensions, c_out]
        self.mlp = _MLP(all_dimensions, act=mlp_act, skip=mlp_skip, bn=mlp_bn, dropout=mlp_dropout, bn_final=bn_final)
        self.softmax = torch.nn.Softmax(1)
    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            if self.emb_drop is not None: x = self.emb_drop(x)
            x = self.transformer(x)
            x = x.flatten(1)
        if self.n_cont != 0:
            if self.norm is not None: x_cont = self.norm(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.mlp(x)

        x = self.softmax(x)
        return x


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
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
# from tab_transformer_pytorch import TabTransformer
from torch.nn.modules.loss import CrossEntropyLoss
import uuid

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import lib
from torch_losses import *
from torch_losses import DiceLoss

class CBCDataset(Dataset):
    def __init__(self, X, y):
        self.X_cont = X[:, 0:-1].astype(np.float32)
        self.X_cat = X[:, -1::].astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_cat[idx], self.X_cont[idx]), self.y[idx]

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

cont_mean_std = torch.randn(10, 2)



batch_size = 128
train_ds = CBCDataset(X_train, y_train)
valid_ds = CBCDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)

device = torch.device('cpu')

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

classes = {'Sex': ['#na#', 0.0, 0.93333334, 1.0],
 'WBC_na': ['#na#', False, True],
 'HGB_na': ['#na#', False, True],
 'MCV_na': ['#na#', False, True],
 'MCHC_na': ['#na#', False, True],
 'PLT_na': ['#na#', False, True],
 'LYT_na': ['#na#', False, True],
 'MOT_na': ['#na#', False, True],
 'EOT_na': ['#na#', False, True],
 'BAT_na': ['#na#', False, True],
 'Age_na': ['#na#', False, True]}

cont_names = ['WBC','HGB','MCV','MCHC','PLT','LYT','MOT','EOT','BAT','Age']

class MyModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = TabTransformer(classes, cont_names, 1)
        #     categories = (1,),      # tuple containing the number of unique values within each category # bez przecinka po ostatim elemencie nie działa
        #     num_continuous = 10,                # number of continuous values
        #     dim = 32,                           # dimension, paper set at 32
        #     dim_out = 1,                        # binary prediction, but could be anything
        #     depth = 6,                          # depth, paper recommended 6
        #     heads = 8,                          # heads, paper recommends 8
        #     attn_dropout = 0.1,                 # post-attention dropout
        #     ff_dropout = 0.2,                   # feed forward dropout
        #     mlp_hidden_mults = (20, 40, 80, 160, 1),          # relative multiples of each hidden dimension of the last mlp to logits
        #     mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        #     continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        # )
        self.loss = nn.BCELoss()
        self.val_loss = nn.BCELoss()
        self.lr = 0.001
        self.optimizer = torch.optim.Adam

    def forward(self, x):
        x1, x2 = x
        outputs = self.model(x1, x2)
        outputs = torch.nn.functional.sigmoid(torch.reshape(outputs, (-1,)))
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x
        preds = self(x)
        loss = self.loss(preds, y)

        # metrics = self.train_metrics(preds, y)
        # metrics = {**metrics, 'loss': loss}
        # self.log_dict(metrics, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-20,
            verbose=True
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x
        preds = self(x)
        val_loss = self.loss(preds, y)
        # metrics = self.val_metrics(preds, y)
        metrics = {'val_loss': val_loss}
        # self.log_dict(metrics, on_epoch=True)

        return {'val_loss': val_loss}

logger = CSVLogger(save_dir="", name='tabtr')
early_stopping = EarlyStopping('val_loss', patience=50)
checkpoint_callback = ModelCheckpoint(
                            monitor="val_loss",
                            dirpath=f"tabtr",
                            filename="-{epoch:02d}-{val_loss:.2f}-{accuracy:.2f}",
                            save_top_k=1,
                            mode="min",
                        )


trainer = Trainer(logger=logger, callbacks=[checkpoint_callback, early_stopping])

model = MyModel()
trainer.fit(model, train_dl, valid_dl)


X_cont = torch.from_numpy(X_test[128, 0:-1].astype(np.float32))
X_cat = torch.from_numpy(X_test[128, -1::].astype(np.int64))
preds = model((X_cat, X_cont))
y_pred = preds.cpu().detach().numpy()
y_test = y
y_pred = (y_pred > 0.5).astype(int)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1, drop_intermediate=False)
auc_val = auc(fpr, tpr)
print(f"Accuracy: {acc}")
print(f"Specificity: {tn / (tn+fp)}")
print(f"Sensitivity: {tp / (tp+fn)}")
print(f'F1: {f1}')
print(f'AUC: {auc_val}')



