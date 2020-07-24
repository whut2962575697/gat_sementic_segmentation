import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, GroupNorm2d

from backbones.resnet import resnet18
from roi_pooling.functions.roi_pooling import roi_pooling_2d


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class Encoder(nn.Module):
    def __init__(self, output_size, spatial_scale, hidden, nclass, dropout,nb_heads,  alpha):
        super(Encoder, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
        )

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 32, 1, 1),
        #     # nn.GroupNorm(8, 32),
        #     # nn.BatchNorm2d(8),
        #     # nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, 2, padding=1),
        #     # nn.GroupNorm(8, 32),
        #     # nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 1, 1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        # # nn.BatchNorm2d(8),
        # # nn.ReLU(inplace=True)
        # )
        # self.relu = nn.ReLU(inplace=True)
        # self.conv11 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(3, 64, 1, 1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        #
        # )
        #
        #
        #
        # self.conv12 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        # )
        #
        #
        self.max_pool = nn.MaxPool2d(2)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 1, 1),
        #     nn.Dropout2d(0.3),
        #     # nn.GroupNorm(8, 128),
        #     # nn.BatchNorm2d(8),
        #     # nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, 2, padding=1),
        #     # nn.GroupNorm(8, 128),
        #     nn.Dropout2d(0.3),
        #     # nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, 1, 1),
        #     # nn.GroupNorm(8, 256),
        #     nn.Dropout2d(0.3)
        #     # nn.InstanceNorm2d(128),
        #     # nn.BatchNorm2d(8),
        #     # nn.ReLU(inplace=True)
        # )
        # self.conv22 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(64, 256, 1, 1),
        #     # nn.GroupNorm(8, 256),
        #     nn.Dropout2d(0.3)
        #     # nn.InstanceNorm2d(128),
        # )
        #
        # self.conv23 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 256),
        #     # nn.InstanceNorm2d(128),
        #     nn.Dropout2d(0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 256),
        #     # nn.InstanceNorm2d(128),
        #     nn.Dropout2d(0.5)
        # )

        self.squeeze = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.base = resnet18()

        self.gat = GAT(nfeat=64+49,
                    nhid=hidden,
                    nclass=nclass,
                    dropout=0.5,
                    nheads=nb_heads,
                    alpha=alpha)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, adj, rois):
        # x1 = self.conv11(x)
        x = self.conv1(x)
        # x = self.relu(x+x1)
        # x = self.relu(self.conv12(x)+x)
        x = self.max_pool(x)
        # x2 = self.conv22(x)
        x = self.conv2(x)
        # x = self.relu(x + x2)
        # x = self.relu(self.conv23(x)+x)
        x = self.max_pool(x)
        # x = F.dropout(x, 0.5, training=self.training)

        # x = self.base(x)
        rois = rois*0.25
        roi_features = roi_pooling_2d(x, rois, self.output_size,
                           spatial_scale=self.spatial_scale)
        z = self.squeeze(roi_features)
        z = self.sigmoid(z)
        # print(z.shape)
        z = z.view((roi_features.shape[0], -1))

        roi_features = self.gap(roi_features)

        roi_features = roi_features.squeeze(-1).squeeze(-1)

        # print(roi_features.shape, z.shape)

        roi_features = torch.cat((roi_features, z), 1)

        adj = adj.squeeze(0)
        # print(roi_features.shape, adj.shape)
        y = self.gat(roi_features, adj)
        return y




class Encoder_New(nn.Module):
    def __init__(self, output_size, spatial_scale, hidden, nclass, dropout,nb_heads,  alpha):
        super(Encoder_New, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
        )

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 32, 1, 1),
        #     # nn.GroupNorm(8, 32),
        #     # nn.BatchNorm2d(8),
        #     # nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, 2, padding=1),
        #     # nn.GroupNorm(8, 32),
        #     # nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 1, 1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        # # nn.BatchNorm2d(8),
        # # nn.ReLU(inplace=True)
        # )
        # self.relu = nn.ReLU(inplace=True)
        # self.conv11 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(3, 64, 1, 1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        #
        # )
        #
        #
        #
        # self.conv12 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 64),
        #     # nn.InstanceNorm2d(32),
        # )
        #
        #
        self.max_pool = nn.MaxPool2d(2)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 1, 1),
        #     nn.Dropout2d(0.3),
        #     # nn.GroupNorm(8, 128),
        #     # nn.BatchNorm2d(8),
        #     # nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, 2, padding=1),
        #     # nn.GroupNorm(8, 128),
        #     nn.Dropout2d(0.3),
        #     # nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, 1, 1),
        #     # nn.GroupNorm(8, 256),
        #     nn.Dropout2d(0.3)
        #     # nn.InstanceNorm2d(128),
        #     # nn.BatchNorm2d(8),
        #     # nn.ReLU(inplace=True)
        # )
        # self.conv22 = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(64, 256, 1, 1),
        #     # nn.GroupNorm(8, 256),
        #     nn.Dropout2d(0.3)
        #     # nn.InstanceNorm2d(128),
        # )
        #
        # self.conv23 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 256),
        #     # nn.InstanceNorm2d(128),
        #     nn.Dropout2d(0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, 1, padding=1),
        #     # nn.GroupNorm(8, 256),
        #     # nn.InstanceNorm2d(128),
        #     nn.Dropout2d(0.5)
        # )

        self.squeeze = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.base = resnet18()

        self.gat = GAT(nfeat=64+49,
                    nhid=hidden,
                    nclass=nclass,
                    dropout=dropout,
                    nheads=nb_heads,
                    alpha=alpha)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, adj, rois, objs):
        # x1 = self.conv11(x)
        x = self.conv1(x)
        # x = self.relu(x+x1)
        # x = self.relu(self.conv12(x)+x)
        x = self.max_pool(x)
        # x2 = self.conv22(x)
        x = self.conv2(x)
        # x = self.relu(x + x2)
        # x = self.relu(self.conv23(x)+x)
        x = self.max_pool(x)
        # x = F.dropout(x, 0.5, training=self.training)

        # x = self.base(x)
        rois = rois*0.25
        roi_features = roi_pooling_2d(x, rois, self.output_size,
                           spatial_scale=self.spatial_scale)
        # print(roi_features.shape)

        objs = objs.unsqueeze(1)
        # print(objs.shape)
        roi_features = roi_features*objs

        z = self.squeeze(roi_features)
        z = self.sigmoid(z)
        # print(z.shape)
        z = z.view((roi_features.shape[0], -1))

        roi_features = self.gap(roi_features)

        roi_features = roi_features.squeeze(-1).squeeze(-1)

        # print(roi_features.shape, z.shape)

        roi_features = torch.cat((roi_features, z), 1)

        adj = adj.squeeze(0)
        # print(roi_features.shape, adj.shape)
        y = self.gat(roi_features, adj)
        return y

