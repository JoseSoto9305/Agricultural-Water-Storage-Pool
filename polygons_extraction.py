import os
from datetime import datetime
from typing import Generator

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from skimage import img_as_float
from shapely.geometry import Polygon
from skimage.measure import find_contours
from skimage.morphology import label

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.image import ImageGenerator
from functions.image import read_image
from functions.image import zero_padding
from functions.image_georeferencing import get_image_corners
from functions.utils import generate_id
from functions.utils import load_json_config
from functions.utils import parallel_process
from functions.utils import replace_templates
from functions.utils import Timer


logger = WriteLogger(name='polygons_extractor')
timer = Timer()
CONFIG_PATH = './configs/polygons_extraction.json'


def thr_prediction(img:np.array) -> np.array:
    img[img > gl.PREDICTION_THR] = 255
    img[img <= gl.PREDICTION_THR] = 0
    return img


class ImageGeneratorPolygonsExtraction(ImageGenerator):

    def __init__(self, batch_size:int=32):
        super().__init__(batch_size=batch_size)
        self.coords = self._set_image_corners()

    def _set_image_corners(self) -> gpd.GeoDataFrame:
        self.coords['_fixed_geo'] = self.coords[['center_x', 'center_y']].apply(
            lambda x: get_image_corners(
                center_x=x.center_x,
                center_y=x.center_y,
                as_polygon=True,
                as_fixed_size=True
            ), 
            axis=1
        )
        return self.coords

    def _get_available_images(self) -> gpd.GeoDataFrame:
        logger(f'Getting available images for polygons extraction; total records: {self.coords.shape[0]}')
        mask = (
            (self.coords['img_exists'] == True) &
            (self.coords['pred_exist'] == True)
        )
        logger(f'Available images for polygons extraction={mask.sum()}')
        return self.coords[mask]

    def _get_image(self, img_path:str) -> np.array:
        # Load image and apply preprocessing
        data = read_image(img_path=img_path)
        if len(data.shape) > 2:
            data = data[...,0]
        data = img_as_float(data)
        data = thr_prediction(data)
        return data

    def __getitem__(self, index:int) -> Generator:
        ini, end = self.indices[index]
        logger(f'Returning batch; indices from={ini}||to={end}')

        img_ids = self.coords.iloc[ini:end]['ID'].values
        img_paths = self.coords.iloc[ini:end]['pred_path'].values
        img_corners = self.coords.iloc[ini:end]['_fixed_geo'].values

        logger(f'Total of imges for current batch={len(img_paths)}')
        for idx in range(len(img_paths)):
            yield (
                self._get_image(img_path=img_paths[idx]), 
                img_ids[idx],
                img_corners[idx]
            )


def get_polygons(item:tuple) -> list:
    def _get_image_labels(img:np.array) -> tuple:
        labels = label(img)
        # Omit background labels
        polygons_indices = [i for i in np.unique(labels) if not i == 0]
        return labels, polygons_indices
    
    img, img_id, img_corners = item

    # Upper left image coordinate
    ul = (
        img_corners.exterior.xy[0][0], 
        img_corners.exterior.xy[1][0]
    )
    # Extract the labels in the image
    labels, polygons_indices = _get_image_labels(img)
    
    polygons = []
    count = 0
    for i in polygons_indices:
        # Detect polygon vertices in image
        mask = np.zeros((gl.IMAGE_FIXED_HEIGHT, gl.IMAGE_FIXED_WIDTH), dtype='uint8')
        mask[labels == i] =  255
        mask = zero_padding(mask, pad_size=gl.PREDICTION_PAD_SIZE)
        polygon = find_contours(mask, gl.PREDICTION_FIND_COUNTOURS)

        for pp in polygon:
            # Flip (y,x) to (x,y)
            pp = np.fliplr(pp)
            # Referencing the vertices into the image
            pp = np.array( 
                (
                    ul[0] + ((pp[:,0]-gl.PREDICTION_PAD_SIZE) * gl.IMAGE_RESOLUTION_X),
                    ul[1] - ((pp[:,1]-gl.PREDICTION_PAD_SIZE) * gl.IMAGE_RESOLUTION_Y)
                )
            ).T
            polygons.append({
                'ID': f'Polygon_{img_id}_{str(count).zfill(3)}',
                'ID_image': img_id,
                'geometry': Polygon(pp)
            })
            count += 1
    logger(f'Total of polygons found in current image: {len(polygons)}', level='debug')
    return polygons


def dissolve_polygons(polygons:gpd.GeoDataFrame):
    logger(f'Dissolving polygons; input shape={polygons.shape}')
    matrix = polygons.geometry.apply(
        lambda x: polygons.geometry.intersects(x)).values.astype(int)

    _, ids = connected_components(matrix, directed=False)
    dissolved = gpd.GeoDataFrame({
        'ID': ids,
        'geometry': polygons.geometry, 
    }, crs=polygons.crs).dissolve(by='ID')
    dissolved = dissolved.reset_index(drop=True)
    dissolved['ID_polygon'] = dissolved.apply(lambda _: generate_id(), axis=1)
    dissolved = dissolved[[
        'ID_polygon',
        'geometry'
    ]]
    logger(f'Total of polygons after apply connected components={dissolved.shape[0]}')
    return dissolved


class ImagePolygonsExtraction:

    def __init__(self):
        self.cfg = load_json_config(CONFIG_PATH)
        self.generator = self._set_image_generator()

    def _set_image_generator(self) -> ImageGeneratorPolygonsExtraction:
        generator = ImageGeneratorPolygonsExtraction(
            batch_size=self.cfg['globals.batch_size']
        )
        return generator

    def _save_relational_table(self, polygons:gpd.GeoDataFrame) -> None:
        logger(f'Saving relational table at: {gl.PREDICTION_RELATION_FILE}')
        logger(f'Total of polygons to save={polygons.shape[0]}')
        polygons.to_file(driver='ESRI Shapefile', filename=gl.PREDICTION_RELATION_FILE)
        return None

    def _save_dissolved_polygons(self, dissolved:gpd.GeoDataFrame) -> None:
        output_path = self.cfg['output.shapefile.path']
        path, filename = os.path.split(output_path)
        if not filename.endswith('.shp'):
            raise ValueError(f'Cannot save output file because filename doesn`t ends with `.shp`')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        crs = self.cfg.get('output.shapefile.projection', None)
        if crs is None:
            logger(f'Output CRS not provided, setting default output CRS')
            crs = gl.COORDINATES_CRS_LATLONG
        dissolved.to_crs(crs)

        output_path = replace_templates(
            value=output_path,
            replace_values={
                'NOW': datetime.now(),
                'NN_WEIGHTS_ID': gl.NN_WEIGHTS_ID
            }
        )
        logger(f'Saving dissolved polygons at:{output_path}')
        dissolved.to_file(driver='ESRI Shapefile', filename=output_path)        
        return None

    def run(self) -> None:
        logger(f'Starting extraction...........')
        polygons = []
        for i in range(len(self.generator)):
            pp = parallel_process(
                func=get_polygons,
                iterable=self.generator[i],
                class_pool='ProcessPoolExecutor'
            )
            [polygons.extend(p) for p in pp if p] # Remove empty images
        
        if not polygons:
            logger(f'Cannot find any polygons, exit from main function')
            return None

        polygons = gpd.GeoDataFrame(pd.DataFrame(polygons), 
                            crs=gl.COORDINATES_CRS_PROJECTION)
        self._save_relational_table(polygons=polygons)
        self._save_dissolved_polygons(dissolved=dissolve_polygons(polygons=polygons))
        return None


@timer.time
def main() -> None:
    try:
        logger(f'Starting extract polygons from prediction images application at: {datetime.now()}')
        extractor = ImagePolygonsExtraction()
        extractor.run()
        logger(f'Main application done successfully :)')
    except Exception as exc:
        logger('RuntimeError at main application full traceback is show below:', level='error')
        raise exc
    return None


if __name__ == '__main__':
    main()
