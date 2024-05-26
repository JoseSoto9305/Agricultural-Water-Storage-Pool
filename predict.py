import os
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from keras.engine.training import Model
from skimage.io import imsave

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.image import cropping
from functions.image import im_rescaling
from functions.image import ImageGenerator
from functions.image import read_image
from functions.utils import load_coordinates
from functions.utils import load_json_config
from functions.utils import Timer
from neural_network.resnet import restore_model


logger = WriteLogger(name=__name__, level='DEBUG')
timer = Timer()

CONFIG_PATH = './configs/predict.json'


class ImageGeneratorPredictor(ImageGenerator):

    def _get_available_images(self) -> gpd.GeoDataFrame:
        logger(f'Getting available images for predictions; total records: {self.coords.shape[0]}')
        mask = (
            (self.coords['img_exists'] == True) &
            (self.coords['pred_exist'] == False)
        ) 
        logger(f'Available images for prediction={mask.sum()}')
        return self.coords[mask]

    def _get_image(self, img_path:str) -> np.array:
        # Load image and apply preprocessing
        data = read_image(img_path=img_path)
        data = cropping(data, crop_size=gl.IMAGE_CROP_SIZE)
        data = im_rescaling(data)
        return data

    def __getitem__(self, index:int) -> tuple:
        ini, end = self.indices[index]
        logger(f'Returning batch; indices from={ini}||to={end}')

        img_ids = self.coords.iloc[ini:end]['ID'].values
        img_paths = self.coords.iloc[ini:end]['img_path'].values
        X = np.empty((
            len(img_paths), 
            gl.IMAGE_FIXED_HIGHT, 
            gl.IMAGE_FIXED_WIDTH, 
            gl.IMAGE_CHANNELS
        ))
        for idx, img_path in enumerate(img_paths):
            X[idx] = self._get_image(img_path=img_path)
        logger(f'Total of imges for current batch={len(img_paths)}')
        return X, img_ids, img_paths


class ImagePredictor:

    def __init__(self):
        self.cfg = load_json_config(path=CONFIG_PATH)
        self.generator = self._set_image_generator()
        self.model = self._get_nn_model()

    def _set_image_generator(self) -> ImageGeneratorPredictor:
        generator = ImageGeneratorPredictor(
            batch_size=self.cfg['globals.batch_size']
        )
        return generator

    def _get_nn_model(self) -> Model:
        if len(self.generator) == 0:
            logger(f'Dont load model; not available images to process', level='debug')
            return None
        logger(f'Returning neural network model...')
        logger(f'Input shape={gl.NN_INPUT_SHAPE}')
        logger(f'N classes={gl.NN_CLASSES}')
        logger(f'Weights={gl.NN_WEIGHTS_PATH}')
        model = restore_model(
            input_shape=gl.NN_INPUT_SHAPE,
            n_classes=gl.NN_CLASSES,
            weights_path=gl.NN_WEIGHTS_PATH
        )
        return model

    def _update_coordinates_file(self, update:list) -> None:
        if not update:
            logger(f'Predictions not available, exit from function')
            return None
        
        update = pd.DataFrame(update)
        coords = load_coordinates()

        logger(f'Total of coordinates records to update: {update.shape[0]}')
        logger(f'Coordinates shape before update={coords.shape}', level='debug')
        coords = coords.merge(update, 
            left_on=['ID'], 
            right_on=['ID'], 
            how='left',
            suffixes=(None, '_DROP')
        )
        columns = [
            'pred_path', 
            'pred_exist', 
        ]
        for c in columns:
            mask = ~coords[f'{c}_DROP'].isna()
            coords.loc[mask, c] = coords.loc[mask, f'{c}_DROP']
        coords.drop([f'{c}_DROP' for c in columns], axis=1, inplace=True)
        coords['pred_exist'] = coords['pred_exist'].astype('int64')
        coords.to_file(driver='ESRI Shapefile', filename=gl.COORDINATES_FILE)
        logger(f'Coordinates shape after update={coords.shape}', level='debug')
        return None

    def run(self) -> None:
        logger(f'Starting predictions:.........')
        update = []
        for i in range(len(self.generator)):
            # Get batch data
            X, img_ids, img_paths = self.generator[i]
            # Make prediction
            y_pred = self.model.predict(X).reshape(
                X.shape[0],
                X.shape[1],
                X.shape[2],
                gl.NN_CLASSES
            )
            # Save prediction for each image in the current batch
            for j in range(X.shape[0]):
                path, filename = os.path.split(img_paths[j])
                filename = filename.replace(gl.IMAGE_SUFFIX_DATA, gl.IMAGE_SUFFIX_PREDICTION)

                partition = os.path.split(path)[-1]
                path = os.path.join(gl.DIRECTORY_PREDICTIONS, partition)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                path = os.path.join(path, filename)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    logger(f'Saving prediction at: {path}')
                    imsave(path, y_pred[j,...,1])
                update.append({
                    'ID': img_ids[j],
                    'pred_path': path,
                    'pred_exist': True
                })
        self._update_coordinates_file(update=update)
        return None


@timer.time
def main() -> None:
    try:
        logger(f'Starting agricultural water storage pool detector application at: {datetime.now()}')
        predictor = ImagePredictor()
        predictor.run()
        logger(f'Main application done successfully :)')
    except Exception as exc:
        logger('RuntiError at main application full traceback is show below:', level='error')
        raise exc
    return predictor


if __name__ == '__main__':
    p = main()