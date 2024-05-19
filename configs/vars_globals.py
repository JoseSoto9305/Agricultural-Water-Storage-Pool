import os

# Direcotories
DIRECTORY_IMAGES = './data/google_images'
DIRECTORY_PREDICTIONS = './data/prediction_images'
DIRECTORY_SHAPEFILES = './data/shapefiles'

# Coordinates Samples File
COORDINATES_FILE = os.path.join(DIRECTORY_SHAPEFILES, 'coordinates.shp')
COORDINATES_CRS_LATLONG = {'init': 'epsg:4326'}
COORDINATES_CRS_REPROJECTION = {'init': 'epsg:6372'}

# Image spatial parameters
IMAGE_HEIGHT = 552
IMAGE_WIDTH = 552
IMAGE_RESOLUTION_X = 0.5635245901639344
IMAGE_RESOLUTION_Y = 0.5635245901639344
IMAGE_CROP_SIZE = 20  # X pixels for each image side (left, right, upper, lower)
IMAGE_DTHR = 250

# Image filename rules 
IMAGE_EXT = '.png'
IMAGE_FMT_DATE = '%Y%m%d'
IMAGE_FMT_TIMESTAMP = '%Y%m%d_%H%M%S_%f'
IMAGE_PARTITION = 'date'
IMAGE_PREFIX = 'Image'
IMAGE_SUFFIX_DATA = '_data.png'
IMAGE_SUFFIX_PREDICTION = '_prediction.png'
