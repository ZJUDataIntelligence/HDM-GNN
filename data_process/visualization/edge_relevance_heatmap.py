import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# mainly for taxi_counts,bike_counts data

# path of areas shapefile data
# nyc: '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'
# chicago: '/home/zh/raw_data/Chicago/region/chicago_Boundaries_Community_Areas/geo_export_4c5e5141-cac3-4432-91ed-98a5fc4a3138.shp'
# LA: '/home/zh/raw_data/Los_Angeles/region/Los_Angeles_Council_District_Boundaries/geo_export_ee48f79f-ef0a-4515-a620-1d0075f71a45.shp'
areas_path = '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'

# for nyc taxi_counts data, there are 3 paths corresponding to yellow,green and fhv taxi_counts type
# data_counts_path = '/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_2020_total.npy'
data_counts_path = '/home/zh/crime_prediction_data/nyc/bike_flow/bike_counts_2020.npy'

day = 1

area_id = 33


areas = gpd.read_file(areas_path)
data_counts = np.load(data_counts_path)
# to plot the whole year data
# d = {'data_count': data_counts.sum(axis=0)[area_id], 'geometry': areas.geometry}
d = {'data_count': data_counts[day][area_id][0:263], 'geometry': areas.geometry}
gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
base = areas.plot(color='white', edgecolor='black', alpha=0.3)

plt.axis('off')

gdf.plot(column='data_count', ax = base, alpha = 0.9, cmap='OrRd', legend=False)
plt.savefig('/home/zh/Pictures/bike-area33-day1.svg')
plt.show()

print(data_counts.sum())
