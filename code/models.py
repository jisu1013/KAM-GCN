#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
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
class GCNEmbedding(nn.Module):
    def __init__(self, dataset, n_user, n_item):
        super(GCNEmbedding,self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_fold = 100
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        #self.emb_size = args.embed_size
        self.emb_size = 64
        self.layers = eval('[64,64,64]')
        #self.n_layers = len(self.layer_size)
        self.mess_dropout = [0.1, 0.1, 0.1]

        ####Init the weight of user-item####
        self.embedding_dict, self.weight_dict = self.init_weight()

        self.norm_adj = sp.load_npz('../data/'+self.dataset+'/ui_adj_mat.npz')        
        self.sparse_norm_adj = self.convert_sp_mat_to_sp_tensor(self.norm_adj)

    def init_weight(self):        
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,self.emb_size)))
        })
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        #print('layers: ',layers)
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})                                                            
        
        return embedding_dict, weight_dict
    
    def convert_sp_mat_to_sp_tensor(self,X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row,col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index,data,torch.Size(coo.shape))
    
    def split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_user + self.n_item) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_user + self.n_item
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self.convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat
    
    #def create_gcn_embed(self):
    def forward(self):
        A_fold_hat = self.split_A_hat(self.norm_adj)
        embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']],0)
        all_embeddings = [embeddings]

        for k in range(len(self.layers)):
            tmp_embed = []
            for f in range(self.n_fold):
                tmp_embed.append(torch.sparse.mm(A_fold_hat[f], embeddings))                      
            embeddings = torch.cat(tmp_embed, 0)          
            embeddings = nn.LeakyReLU(0.2)(torch.matmul(embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k])
            embeddings = nn.Dropout(self.mess_dropout[k])(embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            #print('norm_embeddings: ',norm_embeddings.shape)
            all_embeddings += [norm_embeddings]
    
        all_embeddings = torch.cat(all_embeddings, 1)

        u_g_embeddings = embeddings[:self.n_user, :]
        i_g_embeddings = embeddings[self.n_user:, :]
        
        return u_g_embeddings, i_g_embeddings

################################ AM-GCN ##########################

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc3(x, adj)
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
    def __init__(self, dataset, nhid1, nhid2, nhid3, n_user, n_item, dropout):
        super(SFGCN, self).__init__()
        self.dataset = dataset
        self.GCNEmb = GCNEmbedding(dataset, n_user, n_item)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #nfeature -> n_item
        self.SGCN1 = GCN(64, nhid1, nhid2, nhid3, dropout)
        self.SGCN2 = GCN(64, nhid1, nhid2, nhid3, dropout)
        self.CGCN = GCN(64, nhid1, nhid2, nhid3, dropout)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid3, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()
        self.n_user = n_user
        self.n_item = n_item
        '''
        self.MLP = nn.Sequential(
            nn.Linear(nhid3, 1, bias=False)
            #nn.LogSoftmax(dim=1)
        )
        '''
        self.predict_layer = nn.Linear(nhid3, 1, bias=False)

    def forward(self, sadj, fadj):
        user_embedding, item_embedding = self.GCNEmb()
        #emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph #Z_T     
        #x->item_embedding
        #print('item_embedding : ',item_embedding.shape)
        #print('user_embedding : ',user_embedding.shape)
        #print('sadj : ', sadj)
        #print('fadj : ', fadj)
        com1 = self.CGCN(item_embedding, sadj)  # Common_GCN out1 -- sadj structure graph #Z_CT
        com2 = self.CGCN(item_embedding, fadj)  # Common_GCN out2 -- fadj feature graph #Z_CF
        emb2 = self.SGCN2(item_embedding, fadj) # Special_GCN out2 -- fadj feature graph #Z_F
        Xcom = (com1 + com2) / 2 #Common Embedding Z_C        
        ##attention
        emb = torch.stack([emb2, Xcom], dim=1) 
        #emb, att = self.attention(emb)
        output = self.predict_layer(emb)
        return output, user_embedding, item_embedding, com1, com2, emb2, emb