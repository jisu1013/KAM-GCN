import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from time import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from models import SFGCN
from sklearn.metrics import f1_score
from utils import *
import argparse
from config import Config
from metrics import *
from load_graph import UI_Graph, II_Graph
from log_helper import *
from helper import *

#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

if __name__ == "__main__":
    #.to(device)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    #device_list = [0,1]
    #torch.cuda.set_device(device_list[0])
    #if n_gpu > 0:
    #    torch.cuda.manual_seed_all(123)
    print("n_gpu: ", n_gpu)

    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    #parse.add_argument("-emb", "--embsize", help="embsize", type=int, required=True)
    args = parse.parse_args()
    #config_file = '../config/'+str(args.labelrate)+str(args.dataset)+'.ini'
    config_file = '../config/amazon-book.ini'
    config = Config(config_file)

    data = UI_Graph(config, args.dataset)
    uii_adj, kii_adj = load_graph(config, data.n_items)
    #uiigraph, kiigraph, _, _ = load_data(config)
    
    model = SFGCN(dataset = args.dataset,
            nhid1 = config.nhid1,
            nhid2 = config.nhid2,
            nhid3 = config.nhid3,
            n_user = data.n_users,
            n_item = data.n_items,
            dropout = config.dropout)
    model = nn.DataParallel(model,device_ids=[0,1])
    '''
    if n_gpu > 1:
        #torch.cuda.manual_seed_all(123)
        model = nn.DataParallel(_model,[0,1])        
        model = model.cuda()
        uii_adj = uii_adj.to(device)
        kii_adj = kii_adj.to(device) 
    '''   
    if device == "cuda":
        #model.to(device)       
        uii_adj = uii_adj.to(device)
        kii_adj = kii_adj.to(device)      
        #idx_train = idx_train.to(device)
        #idx_test = idx_test.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)    
    '''
    main
    
    train(model, config, args)
    predict(config, args)
    '''

    def evaluate(model, user_embed, item_embed, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
        model.eval()
        '''
        with torch.no_grad():
            att = model.compute_attention(train_graph)
        train_graph.edata['att'] = att
        '''
        n_users = len(test_user_dict.keys())
        item_ids_batch = item_ids.cpu().numpy()

        cf_scores = []
        precision = []
        recall = []
        ndcg = []

        with torch.no_grad():
            for user_ids_batch in user_ids_batches:
                cf_scores_batch = cf_score(user_embed, item_embed, user_ids_batch, item_ids) #(n_batch_users, n_eval_items)
                
                cf_scores_batch = cf_scores_batch.cpu()
                user_ids_batch = user_ids_batch.cpu().numpy()
                precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict, test_user_dict, user_ids_batch, item_ids_batch, K)

                cf_scores.append(cf_scores_batch.numpy())
                precision.append(precision_batch)
                recall.append(recall_batch)
                ndcg.append(ndcg_batch)
                #print(precision_batch, recall_batch, ndcg_batch)

        cf_scores = np.concatenate(cf_scores, axis=0)
        precision_k = sum(np.concatenate(precision)) / n_users
        recall_k = sum(np.concatenate(recall)) / n_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_users
        #print(cf_scores, precision_k, recall_k, ndcg_k)
        return cf_scores, precision_k, recall_k, ndcg_k

    def train(model, config, args):        
        '''
        KGAT
        '''
        #logging
        log_save_id = create_log_id(config.logging_path)
        logging_config(folder=config.logging_path, name='log{:d}'.format(log_save_id),no_console=False)
        #logging.info(config)
        #logging.info(model)

        data = UI_Graph(config, args.dataset)
        user_ids = list(data.test_user_dict.keys())
        # args-> config
        user_ids_batches = [user_ids[i: i+config.test_batch_size] for i in range(0, len(user_ids), config.test_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
        # GPU        
        if use_cuda:
            user_ids_batches = [d.to(device) for d in user_ids_batches]        
        item_ids = torch.arange(data.n_items, dtype=torch.long)        
        if use_cuda:
            item_ids = item_ids.to(device)        
        best_epoch = -1
        epoch_list = []
        precision_list = []
        recall_list = []
        ndcg_list = []

        for epoch in range(1, config.epochs+1):
            time0 = time()
            model.train()
            optimizer.zero_grad()
            output, user_embedding, item_embedding, com1, com2, emb2, emb = model(uii_adj, kii_adj)            
            logging.info('Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time()-time0))
            # except attention scores 
            # KGAT
            # train cf
            time1 = time()
            total_loss = 0
            n_cf_batch = data.n_train // config.cf_batch_size + 1

            for iter in range(1, n_cf_batch + 1):
                with torch.autograd.set_detect_anomaly(True):
                    time2 = time()
                    cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict)                    
                    if use_cuda:
                        cf_batch_user = cf_batch_user.to(device)
                        cf_batch_pos_item = cf_batch_pos_item.to(device)
                        cf_batch_neg_item = cf_batch_neg_item.to(device)                
                    #cf_batch_loss = model('calc_cf_loss', train_graph, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
                    cf_batch_loss = calc_cf_loss(config, user_embedding, item_embedding, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
                    #print('main emb2',emb2[cf_batch_pos_item])
                    loss_dep = (loss_dependence(item_embedding[cf_batch_pos_item], com1[cf_batch_pos_item], config.cf_batch_size) 
                                    + loss_dependence(emb2[cf_batch_pos_item], com2[cf_batch_pos_item], config.cf_batch_size))/2
                    loss_com = common_loss(com1[cf_batch_pos_item], com2[cf_batch_pos_item])
                    #print(cf_batch_loss, loss_dep, loss_com)
                    #cf_batch_loss.backward(retain_graph=True)
                    total_loss += cf_batch_loss
                    total_loss += loss_dep
                    total_loss += loss_com
                    total_loss.backward()
                    #loss_dep.backward()
                    #loss_com.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    #total_loss = (cf_batch_loss + loss_dep + loss_com)
                    #total_loss += cf_batch_loss.item()
                    if(iter % config.cf_print_every) == 0:
                        logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format
                                (epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), total_loss / iter))
                        #_, precision, recall, ndcg = evaluate(model, user_embedding, item_embedding, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, config.K)
                        #logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))
                        #save_model(model, config.logging_path, epoch, best_epoch)
                        #logging.info('save model on epoch {:04d}!'.format(epoch))

            logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format
                            (epoch, n_cf_batch, time() - time1, total_loss / n_cf_batch))            
            logging.info('CF Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))
            
            #evaluate CF
            if (epoch % config.evaluate_every) == 0:
                time1 = time()
                _, precision, recall, ndcg = evaluate(model, user_embedding, item_embedding, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, config.K)
                logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))

                epoch_list.append(epoch)
                precision_list.append(precision)
                recall_list.append(recall)
                ndcg_list.append(ndcg)
                best_recall, should_stop = early_stopping(recall_list, args.stopping_stpes)

                if should_stop:
                    break
                if recall_list.index(best_recall) == len(recall_list)-1:
                    save_model(model, config.logging_path, epoch, best_epoch)
                    logging.info('save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch
            
        save_model(model, config.logging_path, epoch)

        _, precision, recall, ndcg = evaluate(model, user_embedding, item_embedding, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, config.K)
        logging.info('Final Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))

        epoch_list.append(epoch)
        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)

        metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list]).transpose()
        metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K)]
        metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)
        '''
        KGAT
        '''
    def predict(config, args):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            torch.cuda.manual_seed_all(config.seed)

        data = UI_Graph(args.dataset)

        user_ids = list(data.test_user_dict.keys())
        user_ids_batches = [user_ids[i: i + config.test_batch_size] for i in range(0, len(user_ids), config.test_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
        if use_cuda:
            user_ids_batches = [d.to(device) for d in user_ids_batches]

        item_ids = torch.arange(data.n_items, dtype=torch.long)
        if use_cuda:
            item_ids = item_ids.to(device)

        #load model
        model = SFGCN(uii_adj, kii_adj)
        model = load_model(model, config.pretrain_model_path)
        if n_gpu > 1:
            print("Let's use", torch.cuda.device_count(), "GPU!")
            model = nn.DataParallel(model)        
        model.to(device)
        output, user_embedding, item_embedding, com1, com2, emb2, emb = model(uii_adj, kii_adj)
        
        cf_scores, precision, recall, ndcg = evaluate(model, user_embedding, item_embedding, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, config.K)
        np.save(config.save_dir + 'cf_scores.npy', cf_scores)
        print('CF Evaluation : Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))

    '''
    main
    '''
    train(model, config, args)
    #predict(config, args)


    



    
