# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/14 14:18   xin      1.0         None
'''
from torch.utils import data
import os
import torch
import pickle
import scipy.sparse as sp
import numpy as np
from skimage.io import imread
import torchvision.transforms as T


class RS_Dataset(data.Dataset):
    def __init__(self, feature_node_path, edge_path, label_path):
        super(RS_Dataset,self).__init__()
        self.feature_node_path = feature_node_path
        self.edge_path = edge_path
        self.label_path = label_path
        self.filenames = os.listdir(self.feature_node_path)
        self.sample_num = len(self.filenames)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(os.path.join(self.feature_node_path, filename), 'rb') as f:
            node_feature = pickle.load(f)  # 反序列化
        with open(os.path.join(self.edge_path, filename), 'rb') as f:
            edge = pickle.load(f)  # 反序列化
        with open(os.path.join(self.label_path, filename), 'rb') as f:
            label = pickle.load(f)  # 反序列化
        adj = sp.coo_matrix(edge)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        node_feature = torch.FloatTensor(node_feature)
        label = torch.LongTensor(label)
        return node_feature, adj, label


class RS_Dataset_New(data.Dataset):
    def __init__(self, rs_img_path, edge_path, label_path, roi_path):
        super(RS_Dataset_New,self).__init__()
        self.rs_img_path = rs_img_path
        self.edge_path = edge_path
        self.label_path = label_path
        self.roi_path = roi_path
        self.filenames = os.listdir(self.rs_img_path)
        self.sample_num = len(self.filenames)
        self.transform = T.Compose([
            # T.Resize((256, 128)),

            T.ToTensor(),
            T.Normalize(mean=[0.5716390795822704,0.5191239166003989,0.4923358870147872],
                        std=[0.24454287910934064,0.2379462921336855,0.22901043133634436])
        ])

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        filename = self.filenames[index]
        rs_img = imread(os.path.join(self.rs_img_path, filename))

        with open(os.path.join(self.edge_path, filename.replace('tif', 'pkl')), 'rb') as f:
            edge = pickle.load(f)  # 反序列化
        with open(os.path.join(self.label_path, filename.replace('tif', 'pkl')), 'rb') as f:
            label = pickle.load(f)  # 反序列化
        with open(os.path.join(self.roi_path, filename.replace('tif', 'pkl')), 'rb') as f:
            roi = pickle.load(f)  # 反序列化
        adj = sp.coo_matrix(edge)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        # rs_img = torch.FloatTensor(rs_img)
        rs_img = self.transform(rs_img)
        label = torch.LongTensor(label)
        roi = torch.FloatTensor(roi)
        return rs_img, adj, label, roi



class RS_Dataset_New1(data.Dataset):
    def __init__(self, rs_img_path, edge_path, label_path, roi_path, obj_path):
        super(RS_Dataset_New1,self).__init__()
        self.rs_img_path = rs_img_path
        self.edge_path = edge_path
        self.label_path = label_path
        self.roi_path = roi_path
        self.obj_path = obj_path
        self.filenames = os.listdir(self.rs_img_path)
        self.sample_num = len(self.filenames)
        self.transform = T.Compose([
            # T.Resize((256, 128)),

            T.ToTensor(),
            T.Normalize(mean=[0.5716390795822704,0.5191239166003989,0.4923358870147872],
                        std=[0.24454287910934064,0.2379462921336855,0.22901043133634436])
        ])

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        filename = self.filenames[index]
        rs_img = imread(os.path.join(self.rs_img_path, filename))

        with open(os.path.join(self.edge_path, filename.replace('tif', 'pkl')), 'rb') as f:
            edge = pickle.load(f)  # 反序列化
        with open(os.path.join(self.label_path, filename.replace('tif', 'pkl')), 'rb') as f:
            label = pickle.load(f)  # 反序列化
        with open(os.path.join(self.roi_path, filename.replace('tif', 'pkl')), 'rb') as f:
            roi = pickle.load(f)  # 反序列化
        with open(os.path.join(self.obj_path, filename.replace('tif', 'pkl')), 'rb') as f:
            obj = pickle.load(f)  # 反序列化

        adj = sp.coo_matrix(edge)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        # rs_img = torch.FloatTensor(rs_img)
        rs_img = self.transform(rs_img)
        label = torch.LongTensor(label)
        roi = torch.FloatTensor(roi)
        obj = torch.FloatTensor(obj)
        return rs_img, adj, label, roi, obj