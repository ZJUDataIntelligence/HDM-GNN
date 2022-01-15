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
input_path = '/home/zh/crime_prediction_data/nyc/311_map/a311_2020.csv'

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

day_list = get_day_list(data_year)
a311_graph = np.zeros([len(day_list), 263])
a311 = pd.read_csv(input_path, low_memory=False)

start_time = time.time()
for day in day_list:
    one_day_data = a311[a311['Created Date'].apply(lambda x : x[0:10])==day]
    day_index = (datetime.datetime.strptime(day, "%m/%d/%Y").date() - datetime.date(data_year, 1, 1)).days
    for index, row in one_day_data.iterrows():
        area_id = row['area_id']
        a311_graph[day_index][area_id] += 1
    print('{}: {} data finished'.format(time.time()-start_time,day))

np.save('/home/zh/crime_prediction_data/nyc/311_counts/311_counts_2020.npy',a311_graph)