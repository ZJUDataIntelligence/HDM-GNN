# data_path
CRIME_DATA_PATH = '/home/zh/crime_prediction_data/nyc/crime_counts/crime_counts_2020_byClass.npy'
CRIME_LABEL_DATA_PATH = '/home/zh/crime_prediction_data/nyc/crime_counts/crime_counts_2020.npy'

A311_DATA_PATH = '/home/zh/crime_prediction_data/nyc/311_counts/311_counts_2020_byClass.npy'
POI_DATA_PATH = '/home/zh/crime_prediction_data/nyc/poi_counts/poi_counts_2020_byClass.npy'
TAXI_DATA_PATH = '/home/zh/crime_prediction_data/nyc/taxi_flow/taxi_counts_2020_total.npy'
BIKE_DATA_PATH = '/home/zh/crime_prediction_data/nyc/bike_flow/bike_counts_2020.npy'
GEO_DATA_PATH = '/home/zh/crime_prediction_data/nyc/geo_neighbor/geo_graph.npy'
SIMI_DATA_PATH = '/home/zh/crime_prediction_data/nyc/similarity_graph/simi_nyc_2020.npy'

# training
LEARNING_RATE = 0.005

NUM_HEADS = 4
HIDDEN_CHANNELS = 1
NUM_LAYERS = 1

#   parameter in huber loss function
DELTA = 5.0



# number of days in data(usually one year)
NUM_DAYS = 366
# number of areas(community/taxi zones)
NUM_AREAS = 263


# number of classes of crime data
# crime data shape: (num_day, num_areas, num_class_crime)
NUM_CLASS_CRIME = 20
# number of classes of 311 data
# 311 data shape: (num_days, num_areas, num_class_311)
NUM_CLASS_311 = 20
# number of classes of poi data
# poi data shape: (num_areas, num_class_poi)
NUM_CLASS_POI = 20

# dimension of the node feature after node feature aggregation
# node feature shape after aggregation: (num_days, num_areas, dim_node_feature)
DIM_NODE_FEATURE = 10

# threshold of taxi data, to determine weather an edge should be concluded in taxi_graph(taxi_edge_index in HetGraph)
TAXI_THRESHOLD = 5

# threshold of bike data, to determine weather an edge should be concluded in bike_graph(bike_edge_index in HetGraph)
BIKE_THRESHOLD = 5

# threshold of similarity data, to determine weather an edge should be concluded in simi_graph(simi_edge_index in HetGraph)
SIMI_THRESHOLD = 0.7

# number of days of data used to predict the next day's feature
TIME_SERIES_LENGTH = 7


# in gated cnn model for time-sequence predict





