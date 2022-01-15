import numpy as np
import pandas as pd
import math
import datetime
import geopandas as gpd
from shapely.geometry import *
import time
from shapely import wkt
import os


data_year = 2020
input_path = '/home/zh/crime_prediction_data/nyc/bike_map/bike_2020.csv'


def get_day_list(year):
    begin = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    d = begin
    delta = datetime.timedelta(days=1)
    day_list = []
    while d <= end:
        day_list.append(d.strftime("%Y-%m-%d"))
        d += delta
    return day_list


day_list = get_day_list(data_year)
bike_graph = np.zeros([len(day_list), 263, 263])
bike = pd.read_csv(input_path, low_memory=False)

start_time = time.time()
for day in day_list:
    one_day_data = bike[bike['starttime'].apply(lambda x : x[0:10])==day]
    day_index = (datetime.datetime.strptime(day, "%Y-%m-%d").date() - datetime.date(data_year, 1, 1)).days
    for index, row in one_day_data.iterrows():
        start_area = row['start_area_id']
        end_area = row['end_area_id']
        bike_graph[day_index][start_area][end_area] += 1
    print('{}: {} data finished'.format(time.time()-start_time,day))

np.save('/home/zh/crime_prediction_data/nyc/bike_flow/bike_counts_2020.npy', bike_graph)