from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from utils import setup_logger
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT, Encoder, Encoder_New
from dataset import RS_Dataset, RS_Dataset_New, RS_Dataset_New1
from torch.utils.data.dataloader import DataLoader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--train_path', type=str, default=r'D:\gat_dataset\train', help='train dataset path.')
parser.add_argument('--val_path', type=str, default=r'D:D:\gat_dataset\val', help='val dataset path.')
parser.add_argument('--output_path', type=str, default=r'D:\gat_dataset\output/gat_baseline', help='output path.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=10000, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

logger = setup_logger('baseline', args.output_path, 0)

if args.output_path and not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# train_feature_node_path = os.path.join(args.train_path, 'node_features')
train_img_path = os.path.join(args.train_path, 'imgs')
train_edge_path = os.path.join(args.train_path, 'edge_adjs')
train_label_path = os.path.join(args.train_path, 'labels')
train_roi_path = os.path.join(args.train_path, 'roi')
# train_obj_path = os.path.join(args.train_path, 'mask_objs')

# val_feature_node_path = os.path.join(args.val_path, 'node_features')
val_img_path = os.path.join(args.val_path, 'imgs')
val_edge_path = os.path.join(args.val_path, 'edge_adjs')
val_label_path = os.path.join(args.val_path, 'labels')
val_roi_path = os.path.join(args.val_path, 'roi')
# val_obj_path = os.path.join(args.val_path, 'mask_objs')
train_dataset = RS_Dataset_New(train_img_path, train_edge_path, train_label_path, train_roi_path)
val_dataset = RS_Dataset_New(val_img_path, val_edge_path, val_label_path, val_roi_path)

train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1
    )
val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1
    )


# Model and optimizer
# if args.sparse:
#     model = SpGAT(nfeat=3,
#                 nhid=args.hidden,
#                 nclass=12,
#                 dropout=args.dropout,
#                 nheads=args.nb_heads,
#                 alpha=args.alpha)
# else:
#     model = GAT(nfeat=3,
#                 nhid=args.hidden,
#                 nclass=12,
#                 dropout=args.dropout,
#                 nheads=args.nb_heads,
#                 alpha=args.alpha)
model = Encoder(output_size=(7, 7), spatial_scale=1.0, hidden=args.hidden, nclass=12,
                dropout=args.dropout, nb_heads=args.nb_heads,  alpha=args.alpha)
optimizer = optim.SGD(model.parameters(),
                       lr=args.lr)

if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch, train_loader, val_loader, logger=None):
    t = time.time()
    # model.eval()
    model.train()

    all_output = list()
    all_target = list()
    for rs_imgs, adj, labels, rois in train_loader:
        # rs_imgs = rs_imgs.squeeze(0)
        rois = rois.squeeze(0)
        labels = labels.squeeze(0)
        # objs = objs.squeeze(0)
        # print(features.shape, adj.shape, labels.shape)
        if args.cuda:
            rs_imgs = rs_imgs.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            rois = rois.cuda()
            # objs = objs.cuda()
        optimizer.zero_grad()
        output = model(rs_imgs, adj, rois)
        # print(output, labels)
        loss_train = F.nll_loss(output, labels)
        # print(loss_train.item())
        # acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        all_output.append(output)
        all_target.append(labels)
    all_output = torch.cat(all_output, 0)
    all_target = torch.cat(all_target, 0)
    loss = F.nll_loss(all_output, all_target)
    acc = accuracy(all_output, all_target)

    total_output = list()
    total_target = list()

    model.eval()
    for rs_imgs, adj, labels, rois in val_loader:
        # features = features.squeeze(0)
        rois = rois.squeeze(0)
        labels = labels.squeeze(0)
        # objs = objs.squeeze(0)
        if args.cuda:
            rs_imgs = rs_imgs.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            rois = rois.cuda()
            # objs = objs.cuda()
        with torch.no_grad():
            output = model(rs_imgs, adj, rois)
        total_output.append(output)
        total_target.append(labels)
    total_output = torch.cat(total_output, 0)
    total_target = torch.cat(total_target, 0)
    loss_val = F.nll_loss(total_output, total_target)
    acc_val = accuracy(total_output, total_target)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss.data.item()),
          'acc_train: {:.4f}'.format(acc.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    logger.info('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss.data.item()),
          'acc_train: {:.4f}'.format(acc.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()


# def compute_test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.data[0]),
#           "accuracy= {:.4f}".format(acc_test.data[0]))

if __name__ == "__main__":
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    logger.info('Trainer Built')
    for epoch in range(args.epochs):
        loss_values.append(train(epoch, train_loader, val_loader, logging))

        torch.save(model.state_dict(), os.path.join(args.output_path, '{}.pkl'.format(epoch)))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob(os.path.join(args.output_path, '*.pkl'))
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob(os.path.join(args.output_path, '*.pkl'))
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    # print('Loading {}th epoch'.format(best_epoch))
    # model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    # compute_test()
