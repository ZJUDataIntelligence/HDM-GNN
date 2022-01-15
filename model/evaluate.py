import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from data_load import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('/home/zh/temp/model_100_epoch.pth')

train_data_loader, test_data_loader = load_data(CRIME_LABEL_DATA_PATH,CRIME_DATA_PATH,A311_DATA_PATH,POI_DATA_PATH,TAXI_DATA_PATH,BIKE_DATA_PATH,GEO_DATA_PATH)

areas_path = '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'

areas = gpd.read_file(areas_path)

input_data = next(iter(test_data_loader))
input_data = input_data.to(device)

one_day_count = model(input_data)
one_day_count = np.floor(one_day_count.cpu().reshape(263).detach().numpy()).astype(int)

label = input_data.y
label = label.cpu().reshape(263).numpy().astype(int)

d = {'data_count': label, 'geometry': areas.geometry}
gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
base = areas.plot(color='white', edgecolor='black', alpha=0.3)
gdf.plot(column='data_count', ax = base, alpha = 0.5, cmap='OrRd', legend=True)
plt.show()