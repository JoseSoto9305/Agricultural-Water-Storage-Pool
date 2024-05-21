import os
import warnings

import pandas as pd
from skimage.io import imsave

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.image import ImageGenerator
from functions.utils import load_json_config
from functions.utils import load_coordinates
from functions.utils import Timer
from neural_network.resnet import restore_model


logger = WriteLogger(name=__name__, level='DEBUG')
timer = Timer()

CONFIG_PATH = './configs/predict.json'


class ImageGeneratorPredictor(ImageGenerator):

    def _get_available_images(self):
        mask = (
            (self.coords['img_exists'] == True) &
            (self.coords['pred_exist'] == False)
        ) 
        return self.coords[mask]


class ImagePredictor:

    def __init__(self):
        self.cfg = load_json_config(path=CONFIG_PATH)

    def update_coordinates_file(self, update:list):
        if not update:
            return None
        update = pd.DataFrame(update)
        coords = load_coordinates()
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
        return None

    @timer.time
    def run(self):
        generator = ImageGeneratorPredictor(
            batch_size=self.cfg['globals.batch_size']
        )
        model = restore_model(
            input_shape=gl.NN_INPUT_SHAPE,
            n_classes=gl.NN_CLASSES,
            weights_path=gl.NN_WEIGHTS_path
        )
        update = []
        for i in range(len(generator)):
            # Get batch data
            X, im_ids, im_paths = generator[i]
            # Make prediction
            y_pred = model.predict(X).reshape(
                X.shape[0],
                X.shape[1],
                X.shape[2],
                gl.NN_CLASSES
            )
            # Save prediction for each image in the current batch
            for j in range(X.shape[0]):
                path, filename = os.path.split(im_paths[j])
                filename = filename.replace(gl.IMAGE_SUFFIX_DATA, gl.IMAGE_SUFFIX_PREDICTION)

                partition = os.path.split(path)[-1]
                path = os.path.join(gl.DIRECTORY_PREDICTIONS, partition)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                path = os.path.join(path, filename)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    imsave(path, y_pred[j,...,1])
                update.append({
                    'ID': im_ids[j],
                    'pred_path': path,
                    'pred_exist': True
                })
        self.update_coordinates_file(update=update)
        return None


if __name__ == '__main__':
    predictor = ImagePredictor()
    predictor.run()
