import torch
import pickle
import scipy.sparse as sp
import numpy as np
import json
from skimage.io import imsave
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_model(model, feature_node_file, edge_file, label_file, obj_mask_file, filename, save_path):
    model.eval()
    with open(feature_node_file, 'rb') as f:
        node_feature = pickle.load(f)  # 反序列化
    with open(edge_file, 'rb') as f:
        edge = pickle.load(f)  # 反序列化
    with open(label_file, 'rb') as f:
        label = pickle.load(f)  # 反序列化
    with open(obj_mask_file, 'r') as f:
        obj_json = json.load(f)
    adj = sp.coo_matrix(edge)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = torch.FloatTensor(np.array(adj.todense()))
    node_feature = torch.FloatTensor(node_feature)

    model = model.to(device)
    node_feature = node_feature.to(device)
    adj = adj.to(device)
    with torch.no_grad():
        output = model(node_feature, adj)
        output = torch.argmax(output, -1)
        result = np.zeros((224,224)).astype(np.uint8)
        for i, obj in enumerate(obj_json):
            feature_idx = obj['feature_idx']
            result[feature_idx] = output[i].numpy()
        imsave(os.path.join(save_path, filename+'tif'), result)





