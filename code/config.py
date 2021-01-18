import configparser
'''
from AM-GCN
'''
class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))
        
        #Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "n_epoch")
        self.stopping_steps = conf.getint("Model_Setup", "stopping_steps")
        self.lr = conf.getfloat("Model_Setup", "lr")
        #self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.topk = conf.getint("Model_Setup", "topk")
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.nhid3 = conf.getint("Model_Setup", "nhid3")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        #self.beta = conf.getfloat("Model_Setup", "beta")
        #self.theta = conf.getfloat("Model_Setup", "theta")
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")
        self.seed = conf.getint("Model_Setup", "seed")
        self.batch_size = conf.getint("Model_Setup", "batch_size")
        self.test_batch_size = conf.getint("Model_Setup", "test_batch_size")
        self.cf_batch_size = conf.getint("Model_Setup", "cf_batch_size")
        self.kg_batch_size = conf.getint("Model_Setup", "kg_batch_size")
        self.l2loss_lambda = conf.getfloat("Model_Setup", "l2loss_lambda")
        self.cf_l2loss_lambda = conf.getfloat("Model_Setup", "cf_l2loss_lambda")
        self.K = conf.getint("Model_Setup", "K")
        self.cf_print_every = conf.getint("Model_Setup", "cf_print_every")
        self.evaluate_every = conf.getint("Model_Setup", "evaluate_every")
        #self.layer_size = conf.get("Model_Setup","layer_size")
        # Dataset
        #self.n = conf.getint("Data_Setting", "n")
        #self.fdim = conf.getint("Data_Setting", "fdim")
        #self.class_num = conf.getint("Data_Setting", "class_num")
        self.entity_dim = conf.getint("Data_Setting", "entity_dim")
        self.relation_dim = conf.getint("Data_Setting", "relation_dim")
        self.uiigraph_path = conf.get("Data_Setting", "uiigraph_path")
        self.kiigraph_path = conf.get("Data_Setting", "kiigraph_path")
        #self.feature_path = conf.get("Data_Setting", "feature_path")
        #self.label_path = conf.get("Data_Setting", "label_path")
        self.test_path = conf.get("Data_Setting", "test_path")
        self.train_path = conf.get("Data_Setting", "train_path")
        self.logging_path = conf.get("Data_Setting", "logging_path")
        self.pretrain_model_path = conf.get("Data_Setting", "pretrain_model_path")
        self.pretrain_embedding_dir = conf.get("Data_Setting", "pretrain_embedding_dir")





        


