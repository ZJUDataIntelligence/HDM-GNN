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

poi_graph = np.zeros(263)
poi = pd.read_csv(input_path, low_memory=False)

start_time = time.time()
area_id = 0
for index, row in poi.iterrows():
    area_id = row['area_id']
    if area_id>0:
        print(area_id)
        poi_graph[area_id] += 1
    area_id = 0

np.save('/home/zh/crime_prediction_data/nyc/poi_counts/poi_counts_2020.npy',poi_graph)