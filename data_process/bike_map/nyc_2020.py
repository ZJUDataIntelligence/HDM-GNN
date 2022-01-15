import numpy as np
import pandas as pd
import math
import datetime
import geopandas as gpd
from shapely.geometry import *
import time
from shapely import wkt
import os


def set_start_area_id(row):
    start_station_id = row['start station id']
    end_station_id = row['end station id']
    start_area_id = -1
    end_area_id = -1
    if start_station_id in bike_station_id_to_linenumber:
        start_area_id = bike_station_id_to_linenumber[start_station_id]
    if end_station_id in bike_station_id_to_linenumber:
        end_area_id = bike_station_id_to_linenumber[end_station_id]
    return start_area_id


def set_end_area_id(row):
    start_station_id = row['start station id']
    end_station_id = row['end station id']
    start_area_id = -1
    end_area_id = -1
    if start_station_id in bike_station_id_to_linenumber:
        start_area_id = bike_station_id_to_linenumber[start_station_id]
    if end_station_id in bike_station_id_to_linenumber:
        end_area_id = bike_station_id_to_linenumber[end_station_id]
    return end_area_id


input_path = '/home/zh/raw_data/New_York/bike/2020/'
input_files = os.listdir(input_path)
data_list = []
for file in input_files:
    df = pd.read_csv(input_path + file)
    data_list.append(df)
bike = pd.concat(data_list, ignore_index=True)

bike_station_id_to_linenumber = np.load('/home/zh/crime_prediction_data/nyc/bike_station_id_to_linenumber.npy',allow_pickle=True).item()

bike.insert(bike.shape[1],'start_area_id',-1)
bike.insert(bike.shape[1],'end_area_id',-1)

bike['start_area_id'] = bike.apply(set_start_area_id,axis=1)

bike['end_area_id'] = bike.apply(set_end_area_id,axis=1)

bike.to_csv('/home/zh/crime_prediction_data/nyc/bike_map/bike_2020.csv')