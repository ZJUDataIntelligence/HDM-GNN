import numpy as np
import pandas as pd
import math
import datetime
import geopandas as gpd
from shapely.geometry import *
import time
from shapely import wkt
import matplotlib.pyplot as plt

areas = gpd.read_file('/mnt/E/zh/data/data/New_York/region/taxi_zones/geo_export_4307250c-7101-4b26-9c11-acc7be44d297.shp')