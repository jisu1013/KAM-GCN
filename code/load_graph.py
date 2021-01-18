"""
LightGCN
"""
import numpy as np
from collections import defaultdict, OrderedDict
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class UI_Graph(object):
    """
    amazon-book dataset
    User-Item Interaction Graph
    Generate Adjacency matrix, topK Graph
    """
    def __init__(self, config, dataset):
        #self.batch_size = batch_size
        self.dataset = dataset        
        self.n_users = 0
        self.n_items = 0
        self.cf_batch_size = config.cf_batch_size
        train_file = '../data/'+dataset+'/train.txt'
        test_file = '../data/'+dataset+'/test.txt'
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [],[],[]
        self.n_train = 0
        self.n_test = 0
        #add - NGCF
        self.neg_pools = {}
        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid]*len(items))
                    trainItem.extend(items)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        #amazon-book
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    #l = l.strip('\n').split(' ')
                    #items = [int(i) for i in l[1:]]
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])

                    self.exist_users.append(uid)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid]*len(items))
                    testItem.extend(items)
                    '''
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    '''
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users,uid)
                    self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        #f = open('../data/'+dataset+'/n_user_item.txt','a')
        #f.write(str(self.n_users)+'\n'+str(self.n_items))
        #f.close()
        self.Graph = None
        print(f"{self.n_train} interactions for training")
        print(f"{self.n_test} interactions for testing")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                            shape=(self.n_users, self.n_items))
        #print(self.UserItemNet.shape)

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.        
        '''
        generate knn txt files
        '''
        #self.generate_knn(dataset)
        '''
        get sparse graph
        '''
        #self.getSparseGraph()
        '''
        get Userdict
        '''
        self.train_user_dict = self.load_cf(train_file)
        self.test_user_dict = self.load_cf(test_file)
        # pre-calculate
        #self._allPos = self.getUserPosItems(list(range(self.n_user)))
        #__testDict = self.__build_test()
        #print(f"{world.dataset} is ready to go")
    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break
            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items
    
    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break
            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict):
        exist_users = user_dict.keys()
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.sample(exist_users) for _ in range(self.cf_batch_size)]
        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        #print('batch_user: ',batch_user.shape)
        #print('batch_pos: ',batch_pos_item.shape)
        #print('batch_ng: ',batch_neg_item.shape)
        return batch_user, batch_pos_item, batch_neg_item

    def load_cf(self, filename):       
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                user_dict[user_id] = item_ids

        return user_dict

    def getUserdict(self, filename):
        user_dict = dict()
        
        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]
            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))
                user_dict[user_id] = item_ids                
        return user_dict

    def _convert_sp_mat_to_sp_tensor(self,X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row,col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index,data,torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        dataset = self.dataset
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz('../data/' + dataset + '/ui_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()                
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
               
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz('../data/' + dataset + '/ui_adj_mat.npz', norm_adj)                

                #if self.split == True:
                #    self.Graph = self._split_A_hat(norm_adj)
                #    print("done split matrix")
                #else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            
                #self.Graph = self.Graph.coalesce().to(world.device)
            print("don't split the matrix")

        return self.Graph

    def generate_knn(self,dataset):
        for topk in range(2,4):
            t1 = time()
            data = self.UserItemNet.copy()
            inds = []
            for i in range(self.UserItemNet.shape[1]):
                target_item = self.UserItemNet.getcol(i)                            
                compare = data.multiply(target_item)
                compare_sum = compare.sum(axis=0)
                #print('compare_sum='+str(compare_sum.shape))
                ind = np.argpartition(compare_sum.tolist(),-(topk+1))[0][-(topk+1):-1]
                inds.append(ind)
                if i % 1000 == 0:
                    print('topk: ',topk,' i: ',i)
            fname = '../data/'+dataset+'/uii_knn/c'+ str(topk)+'.txt'
            f = open(fname,'w')
            for i,v in enumerate(inds):
                for vv in v:
                    if vv == i:
                        pass
                    else:
                        f.write('{} {}\n'.format(i,vv))
            f.close()
            print('time:',time()-t1)

class II_Graph(object):
    """
    amazon-book dataset
    Item-Item Knowledge Graph
    Generate Adjacency matrix, topK Graph
    """
    def __init__(self,dataset):
        self.dataset = dataset
        path = '../data/'+ self.dataset
        self.kg_file = path + '/kg_final.txt'
        self.item_file = path + '/item_list.txt'
        self.entity_file = path + '/entity_list.txt'
        #self.load_data()
        
    def load_data(self):
        f_i = open(self.item_file, 'r')
        n_item = f_i.read().count('\n') - 1
        f_i.close()
    
        #f_e = open(self.entity_file, 'r')
        #n_entity = f_e.read().count('\n') - 1 #includes number of item
        #f_e.close()
    
        kg_np = np.loadtxt(self.kg_file, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        _dict = defaultdict(set)
        for kg in kg_np:
            if kg[0] < n_item and (kg[2] > n_item or kg[2] == n_item):
                _dict[kg[0]].add(kg[2])
            elif kg[2] < n_item and (kg[0] > n_item or kg[0] == n_item):
                _dict[kg[2]].add(kg[0])
        _dict = OrderedDict(sorted(_dict.items()))
        #print(_dict)
        _dist = list()
        for dic1 in _dict:
            tmp = list()
            for dic2 in _dict:
                if dic1 == dic2:
                    continue
                tmp.append((len(_dict[dic1] & _dict[dic2]), len(_dict[dic2]), dic1, dic2))
            _dist.append(tmp)
        #_dist = sorted(_dist, key=lambda x: x[2])

        for dist in _dist:
            tmp = sorted(sorted(dist, key=lambda x: x[1]), key=lambda x: x[0], reverse=True)
            #print(tmp[:5])
            for idx in range(2,4):
                f = open('../data/'+ self.dataset + '/kii_knn/c' + str(idx) + '.txt', 'a')
                #f.write(str(tmp[0][2]) + " ")
                for i in range(idx):
                    f.write(str(tmp[i][2]) + ' ' + str(tmp[i][3]) + '\n')
                #f.write("\n")
                f.close()


#UI_Graph('amazon-book')
#II_Graph('amazon-book')