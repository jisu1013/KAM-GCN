import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import SFGCN, UIGCN
from sklearn.metrics import f1_score
from utils import *
import argparse
from config import Config
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

if __name__ == "__main__":
    #.to(device)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(123)

    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    #parse.add_argument("-emb", "--embsize", help="embsize", type=int, required=True)
    args = parse.parse_args()
    #config_file = '../config/'+str(args.labelrate)+str(args.dataset)+'.ini'
    config_file = '../config/amazon-book.ini'
    config = Config(config_file)

    uii_adj, kii_adj = load_graph(config)
    #feature???
    uiigraph, kiigraph, idx_train, idx_test = load_data(config)
    
    f = open('../data/'+args.dataset+'/n_user_item.txt','r')
    n_user_item = f.readlines()
    f.close()
    u_g_embeddings, i_g_embeddings = UIGCN(int(n_user_item[0]),int(n_user_item[1]),args)
    
    model = SFGCN(nfeat = config.fdim,
            nhid1 = config.nhid1,
            nhid2 = config.nhid2,
            #nclass = config.class_num,
            n = int(n_user_item[1]),
            dropout = config.dropout)
    
    if device == "cuda":
        model.to(device)
        #features = features.to(device)
        uii_adj = uii_adj.to(device)
        kii_adj = kii_adj.to(device)
        #labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_test = idx_test.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_dacay=config.weight_decay)

    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output, att, emb1, com1, com2, emb2, emb = model(uii_adj, kii_adj)
        
        if (epoch == 0):
            print('output shape : {}'.format(output.shape))
            print('emb : {}'.format(emb.shape))
        
        path_output = './output/output_'+args.dataset+str(args.labelrate)+'.txt'
        if (epoch==0) & os.path.exists(path_output):
            os.remove(path_output)
        f_output = open(path_output,'a')
        print('e : {}'.format(epochs),file=f_output)
        print('output : {}'.format(output),file=f_output)
        f_output.close()
       
        path = './attention/attetion_'+args.dataset+str(args.labelrate)+'.txt'
        if (epoch==0) & os.path.exists(path):
            os.remove(path)
        f = open(path,'a')
        #alpha_T,alpha_F,alpha_C
        print('shape : {}'.format(att.shape),file=f)
        print('e : {}'.format(epochs)+str(att),file=f)
        f.close()
       
        loss_class =  F.nll_loss(output[idx_train], labels[idx_train]) #L_t
        loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
        loss_com = common_loss(com1,com2)

        path_loss = './losseff/losseff_'+args.dataset+str(args.labelrate)+'.txt'
        if (epoch==0) & os.path.exists(path_loss):
            os.remove(path_loss)
        f_loss = open(path_loss,'a')
        #loss_class,loss_dep,loss_com
        print('e : {}'.format(epochs)+' '+str(loss_class.item())+' '+str(loss_dep.item())+' '+str(loss_com.item()),file=f_loss)
        f_loss.close()
        
        loss = loss_class + config.beta * loss_dep + config.theta * loss_com #optimization objective
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model)
        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()))
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test
    
    def main_test(model):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb
    
    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))





    
