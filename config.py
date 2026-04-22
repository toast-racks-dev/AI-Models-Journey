class Config:
    def __init__(self):
        self.z_dim = 100         
        self.image_size = 32      
        self.image_channels = 3   
        self.batch_size = 128         
        self.total_steps = 50_000     
        self.lr_G = 2e-4              
        self.lr_D = 2e-4             
        self.betas = (0.5, 0.999)     
        self.n_dis = 1                
        self.loss = "bce"            
        self.use_lr_decay = True      
        self.sample_step = 500        
        self.sample_size = 64         
        self.logdir = "./logs/DCGAN_CIFAR10"
        self.fid_cache = "./stats/cifar10.train.npz"
        self.data_dir = "./data"
        self.seed = 0
        self.num_workers = 4
