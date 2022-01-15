import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import *
import math
import time
import datetime

poi = gpd.read_file('/home/zh/raw_data/New_York/POI/new-york-200101-free/gis_osm_pois_free_1.shp', low_memory=False)
areas = gpd.read_file('/home/zh/crime_prediction_data/nyc/areas/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp')
poi.insert(poi.shape[1], 'area_id', -1)

start_time = time.time()
for index, row in poi.iterrows():
    pos = poi.geometry[index]
    try:
        b = areas.contains(pos)
        b_index = b[b == True].index.tolist()[0]
    except:
        poi.loc[index, 'area_id'] = -1
    else:
        poi.loc[index, 'area_id'] = b_index

    if index % 10000 == 0:
        print('{}: processed {} items'.format(time.time() - start_time, index))

df = pd.DataFrame(poi)
df.to_csv('/home/zh/crime_prediction_data/nyc/poi_map/poi_2020.csv')