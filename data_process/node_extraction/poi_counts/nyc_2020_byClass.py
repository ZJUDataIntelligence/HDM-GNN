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
input_path = '/home/zh/crime_prediction_data/nyc/poi_map/poi_2020.csv'

poi = pd.read_csv(input_path, low_memory=False)

poi_graph = np.zeros([263,20])

top_20_class_name = poi['fclass'].value_counts().index.to_list()[0:20]

start_time = time.time()
area_id = 0
for index, row in poi.iterrows():
    if row['fclass'] in top_20_class_name:
        area_id = row['area_id']
        class_id = top_20_class_name.index(row['fclass'])
        if area_id>0:
            poi_graph[area_id][class_id] += 1
        area_id = 0

np.save('/home/zh/crime_prediction_data/nyc/poi_counts/poi_counts_{}_byClass.npy'.format(data_year),poi_graph)