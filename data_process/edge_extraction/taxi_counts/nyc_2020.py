import numpy as np
import pandas as pd
import math
import datetime
import geopandas as gpd
from shapely.geometry import *
import time
from shapely import wkt

def get_day_list(year):
    begin = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    d = begin
    delta = datetime.timedelta(days=1)
    day_list = []
    while d <= end:
        day_list.append(d.strftime("%m/%d/%Y"))
        d += delta
    return day_list

id2line = np.load('/home/zh/crime_prediction_data/nyc/area_id_to_linenumber.npy',allow_pickle=True).item()

data_year = 2020
taxi_yellow = pd.read_csv('/home/zh/raw_data/New_York/taxi/{}/NYC_{}_Yellow_Taxi_Trip_Data.csv'.format(data_year,data_year), low_memory=False)
taxi_green = pd.read_csv('/home/zh/raw_data/New_York/taxi/{}/NYC_{}_Green_Taxi_Trip_Data.csv'.format(data_year,data_year), low_memory=False)
taxi_fhv = pd.read_csv('/home/zh/raw_data/New_York/taxi/{}/NYC_{}_For_Hire_Vehicles_Trip_Data.csv'.format(data_year,data_year), low_memory=False)
areas = gpd.read_file('/home/zh/crime_prediction_data/nyc/areas/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp')

taxi_yellow.drop(taxi_yellow[taxi_yellow['PULocationID']>263].index, inplace=True)
taxi_green.drop(taxi_green[taxi_green['PULocationID']>263].index, inplace=True)
taxi_fhv.drop(taxi_fhv[taxi_fhv['PULocationID']>263].index, inplace=True)

taxi_yellow.dropna(subset=['tpep_pickup_datetime', 'PULocationID', 'DOLocationID'], inplace=True)
taxi_green.dropna(subset=['lpep_pickup_datetime', 'PULocationID', 'DOLocationID'], inplace=True)
taxi_fhv.dropna(subset=['pickup_datetime', 'PULocationID', 'DOLocationID'], inplace=True)

day_list = get_day_list(data_year)
yellow_graph = np.zeros([len(day_list), 263, 263])
green_graph = np.zeros([len(day_list), 263, 263])
fhv_graph = np.zeros([len(day_list), 263, 263])

for index, row in taxi_yellow.iterrows():
    day_index = (
                datetime.datetime.strptime(row['tpep_dropoff_datetime'], "%m/%d/%Y %H:%M:%S %p").date() - datetime.date(
            data_year, 1, 1)).days
    pu_loc = int(row['PULocationID'])
    do_loc = int(row['DOLocationID'])
    if 0 <= day_index < len(day_list) and (pu_loc in id2line) and (do_loc in id2line):
        yellow_graph[day_index][id2line[pu_loc]][id2line[do_loc]] += 1

for index, row in taxi_green.iterrows():
    day_index = (datetime.datetime.strptime(row['lpep_pickup_datetime'], "%m/%d/%Y %H:%M:%S %p").date() - datetime.date(
        data_year, 1, 1)).days
    pu_loc = int(row['PULocationID'])
    do_loc = int(row['DOLocationID'])
    if 0 <= day_index < len(day_list) and (pu_loc in id2line) and (do_loc in id2line):
        green_graph[day_index][id2line[pu_loc]][id2line[do_loc]] += 1

for index, row in taxi_fhv.iterrows():
    day_index = (datetime.datetime.strptime(row['pickup_datetime'], "%m/%d/%Y %H:%M:%S %p").date() - datetime.date(
        data_year, 1, 1)).days
    pu_loc = int(row['PULocationID'])
    do_loc = int(row['DOLocationID'])
    if 0 <= day_index < len(day_list) and (pu_loc in id2line) and (do_loc in id2line):
        fhv_graph[day_index][id2line[pu_loc]][id2line[do_loc]] += 1


np.save('/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_{}_yellow.npy'.format(data_year), yellow_graph)
np.save('/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_{}_green.npy'.format(data_year), green_graph)
np.save('/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_{}_fhv.npy'.format(data_year), fhv_graph)

taxi_graph = yellow_graph + green_graph + fhv_graph
np.save('/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_{}_total.npy'.format(data_year), taxi_graph)