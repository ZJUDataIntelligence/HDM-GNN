import matplotlib.pyplot as plt
import geopandas as gpd


# path of areas shapefile data
# nyc: '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'
# chicago: '/home/zh/raw_data/Chicago/region/chicago_Boundaries_Community_Areas/geo_export_4c5e5141-cac3-4432-91ed-98a5fc4a3138.shp'
# LA: '/home/zh/raw_data/Los_Angeles/region/Los_Angeles_Council_District_Boundaries/geo_export_ee48f79f-ef0a-4515-a620-1d0075f71a45.shp'
areas_path = '/home/zh/raw_data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp'

# id is the line number of the shapefile data
id = 65


areas = gpd.read_file(areas_path)
the_neighbors = areas[areas.geometry.intersects(areas.geometry[id])]
base = areas.plot(color='white', edgecolor='black')
base2 = the_neighbors.plot(ax=base, color='orange', edgecolor='black', alpha=1)
plt.axis('off')
areas[id:id+1].plot(ax = base2, color='red', alpha = 1)

plt.savefig('/home/zh/Pictures/areas.svg')
plt.show()