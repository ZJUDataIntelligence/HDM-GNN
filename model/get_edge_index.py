from torch_geometric.data import HeteroData
from utils import *
from HST_data import HSTData, WholeData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
import json

config_filename = '/home/zh/pycharm_projects/crime_prediction/config/ablation2/without-F.json'
with open(config_filename, 'r') as f:
    config = json.loads(f.read())

data_year = 2016
taxi_path = '/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_{}_total.npy'.format(data_year)
bike_path = '/home/zh/crime_prediction_data/nyc/bike_flow/bike_counts_{}.npy'.format(data_year)
simi_path = '/home/zh/crime_prediction_data/nyc/similarity_graph/simi_nyc_{}.npy'.format(data_year)


geo_path = '/home/zh/crime_prediction_data/nyc/geo_neighbor/geo_graph.npy'
output_path = '/home/zh/crime_prediction_data/nyc/edge_index/'

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


taxi_data = np.load(taxi_path)
bike_data = np.load(bike_path)
geo_data = np.load(geo_path)
simi_data = np.load(simi_path)

taxi_edge_index_list, bike_edge_index_list, geo_edge_index, simi_edge_index = \
    edge_adjlist_construction(taxi_data, bike_data, geo_data, simi_data)

save_variable(taxi_edge_index_list, output_path + '{}/taxi_edge_index_list'.format(data_year))
save_variable(bike_edge_index_list, output_path + '{}/bike_edge_index_list'.format(data_year))
save_variable(geo_edge_index, output_path + '{}/geo_edge_index'.format(data_year))
save_variable(simi_edge_index, output_path + '{}/simi_edge_index'.format(data_year))

