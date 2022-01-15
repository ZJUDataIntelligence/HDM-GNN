import random

import numpy as np
import torch
from torch_geometric.data import HeteroData
# from parameters_out import *
from utils import *
from HST_data import HSTData, WholeData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
import json
import global_var


config = global_var.get_value('config')


def my_norm(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


# taxi_data shape: (num_days, num_areas, num_areas), is numpy ndArray
# bike_data shape: (num_days, num_areas, num_areas), is numpy ndArray
# bike_data shape: (num_areas, num_areas), is numpy ndArray
# return: (taxi_edge_index_list, bike_edge_index_list, geo_edge_index)
#          taxi_edge_index_list: List of each day's taxi_edge_index, list length is num_days:
#                                     Each taxi_edge_index shape: (2, number of taxi edges), is pytorch tensor
#          bike_edge_index_list: List of each day's bike_edge_index, list length is num_days:
#                                     Each bike_edge_index shape: (2, number of taxi edges), is pytorch tensor
#          geo_edge_index shape: (2, number of geo_connections), is pytorch tensor
def edge_adjlist_construction(taxi_data, bike_data, geo_data, simi_data):
    taxi_data = torch.from_numpy(taxi_data).to(torch.float32)
    bike_data = torch.from_numpy(bike_data).to(torch.float32)
    geo_data = torch.from_numpy(geo_data).to(torch.float32)
    simi_data = torch.from_numpy(simi_data).to(torch.float32)

    taxi_edge_index_list = []
    bike_edge_index_list = []
    for day in range(config['NUM_DAYS']):
        print(day)
        taxi_edge_from = []
        taxi_edge_to = []
        bike_edge_from = []
        bike_edge_to = []
        for i in range(config['NUM_AREAS']):
            for j in range(config['NUM_AREAS']):
                if taxi_data[day,i,j] > config['TAXI_THRESHOLD']:
                    taxi_edge_from.append(i)
                    taxi_edge_to.append(j)
                if bike_data[day,i,j] > config['BIKE_THRESHOLD']:
                    bike_edge_from.append(i)
                    bike_edge_to.append(j)
        taxi_edge_index_oneday = torch.stack((torch.tensor(taxi_edge_from), torch.tensor(taxi_edge_to)), 0)
        taxi_edge_index_oneday = taxi_edge_index_oneday.type(torch.long)
        taxi_edge_index_list.append(taxi_edge_index_oneday)
        bike_edge_index_oneday = torch.stack((torch.tensor(bike_edge_from), torch.tensor(bike_edge_to)), 0)
        bike_edge_index_oneday = bike_edge_index_oneday.type(torch.long)
        bike_edge_index_list.append(bike_edge_index_oneday)

    geo_edge_from = []
    geo_edge_to = []
    for i in range(config['NUM_AREAS']):
        for j in range(config['NUM_AREAS']):
            if geo_data[i,j] > 0:
                geo_edge_from.append(i)
                geo_edge_to.append(j)
    geo_edge_index = torch.stack((torch.tensor(geo_edge_from), torch.tensor(geo_edge_to)), 0)
    geo_edge_index = geo_edge_index.type(torch.long)

    simi_edge_from = []
    simi_edge_to = []
    for i in range(config['NUM_AREAS']):
        for j in range(config['NUM_AREAS']):
            if simi_data[i, j] > config['SIMI_THRESHOLD']:
                simi_edge_from.append(i)
                simi_edge_to.append(j)
    simi_edge_index = torch.stack((torch.tensor(simi_edge_from), torch.tensor(simi_edge_to)), 0)
    simi_edge_index = simi_edge_index.type(torch.long)

    return taxi_edge_index_list, bike_edge_index_list, geo_edge_index, simi_edge_index


# return: a list of HeteroData() objects, the length of the list is NUM_DAYS.
#               For each HeteroData() object:
#               node type: 'area'
#               edge type: ('area', 'taxi', 'area'), ('area', 'bike', 'area'), ('area', 'geo', 'area')
def multi_graph_construction(crime_path, a311_path, poi_path, taxi_path, bike_path, geo_path, simi_path):
    crime_data = np.load(crime_path)
    a311_data = np.load(a311_path)
    poi_data = np.load(poi_path)
    crime_data = torch.from_numpy(crime_data).to(torch.float32)
    a311_data = torch.from_numpy(a311_data).to(torch.float32)
    poi_data = torch.from_numpy(poi_data).to(torch.float32)

    poi_data = poi_data.expand((config['NUM_DAYS'], config['NUM_AREAS'], config['NUM_CLASS_POI']))

    # crime_data = my_norm(crime_data)
    # a311_data = my_norm(a311_data)
    # poi_data = my_norm(poi_data)

    taxi_data = np.load(taxi_path)
    bike_data = np.load(bike_path)
    geo_data = np.load(geo_path)
    simi_data = np.load(simi_path)

    # taxi_edge_index_list, bike_edge_index_list, geo_edge_index, simi_edge_index = \
    #     edge_adjlist_construction(taxi_data, bike_data, geo_data, simi_data)
    # save_variable(taxi_edge_index_list, '/home/zh/temp/new_20nodefeatures_threshold5/taxi_edge_index_list')
    # save_variable(bike_edge_index_list, '/home/zh/temp/new_20nodefeatures_threshold5/bike_edge_index_list')
    # save_variable(geo_edge_index, '/home/zh/temp/new_20nodefeatures_threshold5/geo_edge_index')
    # save_variable(simi_edge_index, '/home/zh/temp/new_20nodefeatures_threshold5/simi_edge_index')

    # # 这里暂时先读取预处理过的结果，否则每次调试时间太长

    taxi_edge_index_list = load_variavle(config['TAXI_EDGE_INDEX'])
    bike_edge_index_list = load_variavle(config['BIKE_EDGE_INDEX'])
    geo_edge_index = load_variavle(config['GEO_EDGE_INDEX'])
    simi_edge_index = load_variavle(config['SIMI_EDGE_INDEX'])

    results = []
    if 'WITHOUT_E' in config and config['WITHOUT_E'] == 1:
        all_edge_index = load_variavle('/home/zh/temp/new_20nodefeatures_threshold5/all_edge_index')
        for day in range(config['NUM_DAYS'] - 1):
            data = HeteroData()
            data['crime'].x = crime_data[day, :, :]
            data['a311'].x = a311_data[day, :, :]
            data['poi'].x = poi_data[day, :, :]
            data['area'].x = torch.zeros((config['NUM_AREAS'], config['DIM_NODE_FEATURE']))
            data['area', 'taxi', 'area'].edge_index = taxi_edge_index_list[day]
            data['area', 'bike', 'area'].edge_index = bike_edge_index_list[day]
            data['area', 'geo', 'area'].edge_index = geo_edge_index
            data['area', 'simi', 'area'].edge_index = simi_edge_index
            data['area', 'all_edge', 'area'].edge_index = all_edge_index
            results.append(data)
    else:
        for day in range(config['NUM_DAYS']-1):
            data = HeteroData()
            data['crime'].x = crime_data[day,:,:]
            data['a311'].x = a311_data[day, :, :]
            data['poi'].x = poi_data[day, :, :]
            data['area'].x = torch.zeros((config['NUM_AREAS'], config['DIM_NODE_FEATURE']))
            data['area', 'taxi', 'area'].edge_index = taxi_edge_index_list[day]
            data['area', 'bike', 'area'].edge_index = bike_edge_index_list[day]
            data['area', 'geo', 'area'].edge_index = geo_edge_index
            data['area', 'simi', 'area'].edge_index = simi_edge_index
            results.append(data)
    return results

def multi_graph_construction2(embedding_path, crime_path, a311_path, poi_path, taxi_path, bike_path, geo_path, simi_path):
    embedding_data = np.load(embedding_path)
    crime_data = np.load(crime_path)
    a311_data = np.load(a311_path)
    poi_data = np.load(poi_path)

    embedding_data = torch.from_numpy(embedding_data).to(torch.float32)
    crime_data = torch.from_numpy(crime_data).to(torch.float32)
    a311_data = torch.from_numpy(a311_data).to(torch.float32)
    poi_data = torch.from_numpy(poi_data).to(torch.float32)

    poi_data = poi_data.expand((config['NUM_DAYS'], config['NUM_AREAS'], config['NUM_CLASS_POI']))

    # crime_data = my_norm(crime_data)
    # a311_data = my_norm(a311_data)
    # poi_data = my_norm(poi_data)

    taxi_data = np.load(taxi_path)
    bike_data = np.load(bike_path)
    geo_data = np.load(geo_path)
    simi_data = np.load(simi_path)

    # taxi_edge_index_list, bike_edge_index_list, geo_edge_index, simi_edge_index = \
    #     edge_adjlist_construction(taxi_data, bike_data, geo_data, simi_data)
    # save_variable(taxi_edge_index_list, '/home/zh/temp/new_20nodefeatures_threshold5/taxi_edge_index_list')
    # save_variable(bike_edge_index_list, '/home/zh/temp/new_20nodefeatures_threshold5/bike_edge_index_list')
    # save_variable(geo_edge_index, '/home/zh/temp/new_20nodefeatures_threshold5/geo_edge_index')
    # save_variable(simi_edge_index, '/home/zh/temp/new_20nodefeatures_threshold5/simi_edge_index')

    # # 这里暂时先读取预处理过的结果，否则每次调试时间太长
    taxi_edge_index_list = load_variavle('/home/zh/temp/new_20nodefeatures_threshold5/taxi_edge_index_list')
    bike_edge_index_list = load_variavle('/home/zh/temp/new_20nodefeatures_threshold5/bike_edge_index_list')
    geo_edge_index = load_variavle('/home/zh/temp/new_20nodefeatures_threshold5/geo_edge_index')
    simi_edge_index = load_variavle('/home/zh/temp/new_20nodefeatures_threshold5/simi_edge_index')

    results = []
    for day in range(config['NUM_DAYS']-1):
        data = HeteroData()
        data['crime'].x = embedding_data[day,:,:]
        data['a311'].x = a311_data[day, :, :]
        data['poi'].x = poi_data[day, :, :]
        data['area'].x = torch.zeros((config['NUM_AREAS'], config['DIM_NODE_FEATURE']))
        data['area', 'taxi', 'area'].edge_index = taxi_edge_index_list[day]
        data['area', 'bike', 'area'].edge_index = bike_edge_index_list[day]
        data['area', 'geo', 'area'].edge_index = geo_edge_index
        data['area', 'simi', 'area'].edge_index = simi_edge_index
        results.append(data)

    return results

# return: a list of HST_data() objects, the length of the list is (NUM_DAYS - TIME_SERIES_LENGTH + 1).
#               For each HST_data object:
#               has a 'hetero_data_list' attribute which contains a list of HeteroData() objects,
#               the length of the list is TIME_SERIES_LENGTH
def load_data(label_path, crime_path, a311_path, poi_path, taxi_path, bike_path, geo_path, simi_path, config):
    result = []
    ht_data_list = multi_graph_construction(crime_path, a311_path, poi_path, taxi_path, bike_path, geo_path, simi_path)
    labels = np.load(label_path)
    labels = torch.from_numpy(labels).to(torch.float32)
    for i in range(len(ht_data_list) - config['TIME_SERIES_LENGTH']):
        # hst_data = ht_data_list[i:i + TIME_SERIES_LENGTH]
        hst_data = HSTData(ht_data_list[i:i + config['TIME_SERIES_LENGTH']])
        hst_data.y = labels[i + config['TIME_SERIES_LENGTH']].reshape((263,1,1))
        result.append(hst_data)
    # random.shuffle(result)
    data_num = len(result)
    train_data = result[0:3*int(data_num/4)]
    test_data = result[3*int(data_num/4):data_num]
    return DataLoader(train_data,pin_memory=True,shuffle=False), DataLoader(test_data,pin_memory=True,shuffle=False)

def load_data2(embedding_path, label_path, crime_path, a311_path, poi_path, taxi_path, bike_path, geo_path, simi_path, config):
    result = []
    ht_data_list = multi_graph_construction2(embedding_path, crime_path, a311_path, poi_path, taxi_path, bike_path, geo_path, simi_path)
    labels = np.load(label_path)
    labels = torch.from_numpy(labels).to(torch.float32)
    for i in range(len(ht_data_list) - config['TIME_SERIES_LENGTH']):
        # hst_data = ht_data_list[i:i + TIME_SERIES_LENGTH]
        hst_data = HSTData(ht_data_list[i:i + config['TIME_SERIES_LENGTH']])
        hst_data.y = labels[i + config['TIME_SERIES_LENGTH']].reshape((263,1,1))
        result.append(hst_data)
    random.shuffle(result)
    data_num = len(result)
    train_data = result[0:3*int(data_num/4)]
    test_data = result[3*int(data_num/4):data_num]
    return DataLoader(train_data,pin_memory=True,shuffle=False), DataLoader(test_data,pin_memory=True,shuffle=False)


if __name__ == '__main__':
    a = multi_graph_construction()
    b = load_data()
    print('aaa')



