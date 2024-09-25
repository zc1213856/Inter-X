'''
 @FileName    : module_utils.py
 @EditTime    : 2022-09-27 14:38:28
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import random
import torch
import pickle

def load_pkl(path):
    """"
    load pkl file
    """
    with open(path, 'rb') as f:
        param = pickle.load(f, encoding='iso-8859-1')
    return param

def readVclCamparams(path):
    camIns = []
    camExs = []
    with open(path, 'r') as f:
        line = f.readline()
        camIdx = 0
        while(line):
            if line == (str(camIdx)+'\n'):
                camIdx += 1
                camIn = []
                camEx = []
                for i in range(3):
                    line = f.readline().strip().split()
                    camIn.append([float(line[0]), float(line[1]), float(line[2])])
                camIns.append(camIn)
                _ = f.readline()
                for i in range(3):
                    line = f.readline().strip().split()
                    camEx.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
                camEx.append([0.0, 0.0, 0.0, 1.0])
                camExs.append(camEx)
            line = f.readline()
    return np.array(camIns), np.array(camExs)

def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree

def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g
