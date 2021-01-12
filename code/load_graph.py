"""
LightGCN
"""
import numpy as np
from collections import defaultdict, OrderedDict
import random as rd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class UUI_Graph(object):
    """
    amazon-book dataset
    User-Item Interaction Graph
    Generate Adjacency matrix, topK Graph
    """
    def __init__(self,dataset):
        #self.batch_size = batch_size
        self.dataset = dataset        
        self.n_users = 0
        self.n_items = 0
        train_file = '../data/'+dataset+'/train.txt'
        test_file = '../data/'+dataset+'/test.txt'
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [],[],[]
        self.trainDataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid]*len(items))
                    trainItem.extend(items)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.trainDataSize += len(items)
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
                    self.testDataSize += len(items)
        self.n_items += 1
        self.n_users += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        f = open('../data/'+dataset+'/n_user_item.txt','a')
        f.write(str(self.n_users)+'\n'+str(self.n_items))
        f.close()
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                            shape=(self.n_users, self.n_items))
        print(self.UserItemNet.shape)
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        #_allPos = self.getUserPosItems(list(range(self.n_user)))
        #__testDict = self.__build_test()
        #print(f"{world.dataset} is ready to go")
        '''
        generate knn txt files
        '''
        #self.generate_knn(dataset)
        '''
        get sparse graph
        '''
        #self.getSparseGraph()
        
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
                pre_adj_mat = sp.load_npz('../data/' + dataset + '/' + 's_pre_adj_mat_' + dataset + '.npz')
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
                sp.save_npz('../data/' + dataset + '/' + 's_pre_adj_mat_' + dataset + '.npz', norm_adj)                

                #if self.split == True:
                #    self.Graph = self._split_A_hat(norm_adj)
                #    print("done split matrix")
                #else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            
                #self.Graph = self.Graph.coalesce().to(world.device)
            print("don't split the matrix")

        return self.Graph

    def generate_knn(self,dataset):
        for topk in range(5,9):
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
        path = "../data/"+dataset
        kg_file = path + "/kg_final.txt"
        item_file = path + "/item_list.txt"
        entity_file = path + "/entity_list.txt"
        load_data(kg_file, item_file, entity_file)
        
    def load_data(kg_file, item_file, entity_file):
        f_i = open(item_file, 'r')
        n_item = f_i.read().count('\n') - 1
        f_i.close()
    
        f_e = open(entity_file, 'r')
        n_entity = f_e.read().count('\n') - 1 #includes number of item
        f_e.close()

        kg_np = np.loadtxt(kg_file, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        _dict = defaultdict(set)
        for kg in kg_np:
            if kg[0] < n_item:
                _dict[kg[0]].add(kg[2])
            elif kg[2] < n_item:
                _dict[kg[2]].add(kg[0])
        _dict = OrderedDict(sorted(_dict.items()))
        print(_dict)
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
            print(tmp[:5])
            for idx in range(2, 10):
                f = open("./kNN/" + str(idx) + "-NN.txt", "a")
                f.write(str(tmp[0][2]) + " ")
                for i in range(idx):
                    f.write(str(tmp[i][3]) + " ")
                f.write("\n")
                f.close()


#UI_Graph('amazon-book')
#II_Graph('amazon-book')