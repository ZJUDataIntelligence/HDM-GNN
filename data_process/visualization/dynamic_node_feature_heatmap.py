import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# mainly for crime_counts,311 data

# path of areas shapefile data
# nyc: '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'
# chicago: '/home/zh/raw_data/Chicago/region/chicago_Boundaries_Community_Areas/geo_export_4c5e5141-cac3-4432-91ed-98a5fc4a3138.shp'
# LA: '/home/zh/raw_data/Los_Angeles/region/Los_Angeles_Council_District_Boundaries/geo_export_ee48f79f-ef0a-4515-a620-1d0075f71a45.shp'
areas_path = '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'

data_counts_path = '/home/zh/crime_prediction_data/nyc/311_counts/311_counts_2020.npy'
# data_counts_path = '/home/zh/crime_prediction_data/nyc/crime_counts/crime_counts_2020.npy'

day = 10


areas = gpd.read_file(areas_path)
data_counts = np.load(data_counts_path)
one_day_count = data_counts[day]
d = {'data_count': one_day_count, 'geometry': areas.geometry}
gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
base = areas.plot(color='white', edgecolor='black', alpha=0.3)
gdf.plot(column='data_count', ax = base, alpha = 0.5, cmap='OrRd', legend=False)
plt.axis('off')
plt.savefig('/home/zh/Pictures/311_day10.svg')
plt.show()

