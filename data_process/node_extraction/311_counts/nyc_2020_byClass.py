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
a311_graph = np.zeros([len(day_list), 263, 20])
a311 = pd.read_csv(input_path, low_memory=False)

top_20_class_name = a311['Complaint Type'].value_counts().index.to_list()[0:20]

start_time = time.time()
for day in day_list:
    one_day_data = a311[a311['Created Date'].apply(lambda x : x[0:10])==day]
    day_index = (datetime.datetime.strptime(day, "%m/%d/%Y").date() - datetime.date(data_year, 1, 1)).days
    for index, row in one_day_data.iterrows():
        if row['Complaint Type'] in top_20_class_name:
            area_id = row['area_id']
            class_id = top_20_class_name.index(row['Complaint Type'])
            a311_graph[day_index][area_id][class_id] += 1
    print('{}: {} data finished'.format(time.time()-start_time,day))


np.save('/home/zh/crime_prediction_data/nyc/311_counts/311_counts_{}_byClass.npy'.format(data_year),a311_graph)