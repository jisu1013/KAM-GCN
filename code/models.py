import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from load_graph import *
#from torch import nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import math
import sys
import scipy.sparse as sp


#####################NGCF-Pytorch###########################
class UIGCN(nn.module):
    def __init__(self, dataset, n_user, n_item, args):
        super(NGCF,self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.nfold = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = 'amazon-book'
        #self.emb_size = args.embed_size
        self.emb_size = 64
        self.layer_size = [64, 64, 64]
        self.layers = eval(self.layer_size)
        self.mess_dropout = [0.1, 0.1, 0.1]

        ####Init the weight of user-item####
        self.embedding_dict, self.weight_dict = self.init_weight()

        self.norm_adj = sp.load_npz('../data/'+self.dataset+'/s_pre_adj_mat.npz')        
        self.sparse_norm_adj = self.convert_sp_mat_to_sp_tensor(self.norm_adj).to(device)

    def init_weight(self):        
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,self.emb_size)))
        })
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})                                                            
        
        return embedding_dict, weight_dict
    
    def _convert_sp_mat_to_sp_tensor(self,X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row,col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index,data,torch.Size(coo.shape))
    
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat
    
    def create_gcn_embed(self):
        A_fold_hat = self.split_A_hat(self.norm_adj)
        embeddings = torch.cat([self.embedding_dict['user_emb'],self.embedding_dict['item_emb']],0)
        all_embeddings = [embeddings]

        for k in range(len(self.n_layers)):
            tmp_embed = []
            for f in range(self.n_fold):
                tmp_embed.append(torch.sparse.mm(A_fold_hat[f], embeddings))
            
            embeddings = torch.cat(tmp_embed, 0)
            embeddings = nn.LeakyReLU(torch.matmul(embeddings, self.weights['W_gc_%d'%k]) + self.weights['b_gc_%d'%k])
            embeddings = nn.Dropout(embeddings, 1-self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = torch.cat(all_embeddings,1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]
        return u_g_embeddings, i_g_embeddings

################################ AM-GCN ##########################

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1) #alpha_T^i
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.UIGCN = NGCF()
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj, i_g_embeddings):
        #emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph #Z_T     
        emb1 = i_g_embeddings
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph #Z_CT
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph #Z_CF
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph #Z_F
        Xcom = (com1 + com2) / 2 #Common Embedding Z_C
        
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1) 
        emb, att = self.attention(emb)
        #output = self.MLP(emb)
        return att, emb1, com1, com2, emb2, emb