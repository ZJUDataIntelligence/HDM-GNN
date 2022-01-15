import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from data_load import *
import argparse
import json
import global_var


config = global_var.get_value('config')


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization=True):
        super(BasicBlock, self).__init__()
        self.initialization = initialization
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation=config['DILATION'])

        if initialization:
            self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class GatingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization=True, activation=None):
        super(GatingBlock, self).__init__()

        self.activation = activation

        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, initialization)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.block2 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, initialization)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.block1(x)
        out1 = torch.sigmoid(self.bn1(out1))
        out2 = self.block2(x)
        out2 = self.bn2(out2)
        if self.activation != None:
            out2 = self.activation(out2)
        return out1 * out2


class TimeSequenceModel(torch.nn.Module):
    def __init__(self, activation=torch.tanh):
        super().__init__()
        self.gcnns = torch.nn.Sequential(GatingBlock(1, 1, 3,  activation=activation),
                                        GatingBlock(1, 1, 3, activation=activation))
        self.fc = nn.Linear(config['TIME_SERIES_LENGTH']-2*2,1)

        # test new model
        # self.fcnew1 = nn.Linear(config['TIME_SERIES_LENGTH'],1)
        # self.fcnew2 = nn.Linear(config['TIME_SERIES_LENGTH'], 1)
        # self.fcnew3 = nn.Linear(config['TIME_SERIES_LENGTH'], 1)
        # end test

    # input shape: (batch_size=area, channels=1, TIME_SERIES_LENGTH)
    # output shape: (area, 1, 1)
    def forward(self, x):
        out = self.gcnns(x)
        out = self.fc(out)
        return out
        # out = self.fcnew1(x) + self.fcnew1(x) + self.fcnew3(x)
        return out


class LinearAggregation(torch.nn.Module):
    def __init__(self, activation=torch.tanh):
        super().__init__()
        self.fc1 = nn.Linear(config['NUM_CLASS_CRIME']+config['NUM_CLASS_311']+config['NUM_CLASS_POI'],config['DIM_NODE_FEATURE'])
        # self.fc1 = nn.Linear(config['NUM_CLASS_CRIME'] , config['DIM_NODE_FEATURE'])
        # self.fc2 = nn.Linear(config['DIM_NODE_FEATURE'], config['DIM_NODE_FEATURE'])
    # input shape: (NUM_DAYS, NUM_AREAS, NUM_CLASS_CRIME+NUM_CLASS_311+NUM_CLASS_POI)
    # output shape: (NUM_DAYS, NUM_AREAS, DIM_NODE_FEATURE)
    def forward(self, x):
        # x = x[:,:,0:config['NUM_CLASS_CRIME']]
        out = self.fc1(x)
        # out = self.fc2(out)
        return out


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('area', 'bike', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5),
                ('area', 'taxi', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5),
                ('area', 'geo', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5),
                ('area', 'simi', 'area'): GATConv(-1, hidden_channels, heads=config['NUM_HEADS'], concat=False, dropout=0.5)
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    # x_dict shape: (263,1)
    # edge_index_dict shape: (263,263)
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['area'])


class STHGNN_S(torch.nn.Module):
    def __init__(self, time_series_length, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_agg = LinearAggregation()
        self.het_gnns = torch.nn.ModuleList()
        for _ in range(time_series_length):
            het_gnn = HeteroGNN(hidden_channels, out_channels, num_layers)
            self.het_gnns.append(het_gnn)
        self.t_model = TimeSequenceModel()
        self.lin1 = Linear(263, 100)
        self.lin2 = Linear(100, 263)
        # self.tpdata = torch.rand((7,263,1))


    # hst_data is a HSTData object
    def forward(self, hst_data):
        the_data = hst_data.hetero_data_list[0]

        #   node feature aggregation
        x_temp = torch.zeros((config['TIME_SERIES_LENGTH'], config['NUM_AREAS'],
                              config['NUM_CLASS_CRIME']+config['NUM_CLASS_311']+config['NUM_CLASS_POI'])).to('cuda')
        for i in range(config['TIME_SERIES_LENGTH']):
            x_temp[i,:,:] = torch.cat((the_data[i]['crime'].x, the_data[i]['a311'].x, the_data[i]['poi'].x), 1)

        agg_results = self.lin_agg(x_temp)
        for i in range(config['TIME_SERIES_LENGTH']):
            the_data[i]['area'].x = agg_results[i,:,:]

        # #   GNN模型
        # X = torch.zeros((config['TIME_SERIES_LENGTH'], config['NUM_AREAS'], 1)).to('cuda')
        # for i in range(len(self.het_gnns)):
        #     # het_gnn = self.het_gnns[i]
        #     # gpu_data = the_data[i]
        #     # x_t = het_gnn(gpu_data.x_dict, gpu_data.edge_index_dict)
        #     x_t = the_data[i]['area'].x
        #     X[i,:,:] = x_t

        X = agg_results

        #   时序模型
        a = X.permute(1,2,0)
        b = self.t_model(a)
        c = b.reshape(1,1,263)
        d = self.lin1(c)
        e = self.lin2(d)
        f = e.reshape(263,1,1)
        return f

# if __name__ == '__main__':
#     data = load_data()
#     data = data[0]
#     # a = torch.rand(263*263, 1, 7)
#     # model1 = HeteroGNN(hidden_channels=2, out_channels=1, num_layers=2)
#     # model2 = TimeSequenceModel()
#     # model = BasicBlock(1,3,3)
#     #model1 = nn.Conv1d(1,3,3)
#     # model2 = BasicBlock(1,3,3)
#
#     model = STHGNN(time_series_length=TIME_SERIES_LENGTH, hidden_channels=2, out_channels=1, num_layers=2)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#
#     b = model(data)
#     print(f'{b.shape}')