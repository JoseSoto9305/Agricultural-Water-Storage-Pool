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
IMAGE_CHANNELS = 3
IMAGE_RESOLUTION_X = 0.5635245901639344
IMAGE_RESOLUTION_Y = 0.5635245901639344
IMAGE_CROP_SIZE = 20  # X pixels for each image side (left, right, upper, lower)
IMAGE_DTHR = 250

# Neural Network
IMAGE_FIXED_HIGHT = IMAGE_HEIGHT - IMAGE_CROP_SIZE * 2
IMAGE_FIXED_WIDTH = IMAGE_WIDTH - IMAGE_CROP_SIZE * 2
NN_INPUT_SHAPE = (
    IMAGE_FIXED_HIGHT,
    IMAGE_FIXED_WIDTH,
    IMAGE_CHANNELS
)
NN_CLASSES = 2
NN_WEIGHTS_ID = '20181123_202857'
NN_WEIGHTS_path = f'./neural_network/Model/weights_{NN_WEIGHTS_ID}.h5'


# Image filename rules 
IMAGE_EXT = '.png'
IMAGE_FMT_DATE = '%Y%m%d'
IMAGE_FMT_TIMESTAMP = '%Y%m%d_%H%M%S_%f'
IMAGE_PARTITION = 'date'
IMAGE_PREFIX = 'Image'
IMAGE_SUFFIX_DATA = '_data.png'
IMAGE_SUFFIX_PREDICTION = f'_weights_{NN_WEIGHTS_ID}_prediction.png'
