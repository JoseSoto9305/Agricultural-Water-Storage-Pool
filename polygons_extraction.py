import os

import numpy as np
import pandas as pd
import geopandas as gpd
from skimage import img_as_float

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.image import ImageGenerator, read_image
from functions.image_georeferencing import get_image_corners
from functions.image_georeferencing import dissolve_polygons
from functions.image_georeferencing import get_polygons
from functions.image_georeferencing import thr_prediction
from functions.utils import parallel_process
from functions.utils import load_json_config
from functions.utils import Timer


logger = WriteLogger(name=__name__, level='DEBUG')
timer = Timer()
CONFIG_PATH = './configs/polygons_extraction.json'


class ImageGeneratorPolygonsExtraction(ImageGenerator):

    def __init__(self, batch_size=32):
        super().__init__(batch_size)
        self.coords['_fixed_geo'] = self.coords[['center_x', 'center_y']].apply(
            lambda x: get_image_corners(
                center_x=x.center_x,
                center_y=x.center_y,
                as_polygon=True,
                as_fixed_size=True
            ), 
            axis=1
        )

    def _get_available_images(self) -> gpd.GeoDataFrame:
        mask = (
            (self.coords['img_exists'] == True) &
            (self.coords['pred_exist'] == True)
        ) 
        return self.coords[mask]

    def _get_image(self, im_path:str) -> np.array:
        # Load image
        data = read_image(im_path=im_path)

        # Image preprocessing
        if len(data.shape) > 2:
            data = data[...,0]
        data = img_as_float(data)
        data = thr_prediction(data)
        return data

    def __getitem__(self, index:int):
        ini, end = self.indices[index]
        im_paths = self.coords.loc[ini:end, 'pred_path'].values
        im_ids = self.coords.loc[ini:end, 'ID'].values
        im_corners = self.coords.loc[ini:end, '_fixed_geo'].values
        for idx in range(len(im_paths)):
            yield (
                self._get_image(im_path=im_paths[idx]), 
                im_ids[idx], 
                im_corners[idx]
            )


class ImagePolygonsExtraction:

    def __init__(self):
        self.cfg = load_json_config(CONFIG_PATH)

    def _save_relational_table(self, polygons:gpd.GeoDataFrame) -> None:
        polygons.to_file(driver='ESRI Shapefile', filename=gl.PREDICTION_RELATION)
        return None

    def _save_file(self, dissolved:gpd.GeoDataFrame) -> None:
        output_path = self.cfg['output.path']
        path, filename = os.path.split(output_path)
        if not filename.endswith('.shp'):
            raise ValueError(f'Cannot save output file because filename doesnt ends with `.shp`')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        crs = self.cfg.get('output.projection', None)
        if crs is None:
            crs = gl.COORDINATES_CRS_LATLONG
        dissolved.to_crs(crs)
        dissolved.to_file(driver='ESRI Shapefile', filename=output_path)        
        return None

    @timer.time
    def run(self) -> None:
        generator = ImageGeneratorPolygonsExtraction(
            batch_size=self.cfg['globals.batch_size']
        )
        polygons = []
        for i in range(len(generator)):
            pp = parallel_process(
                func=get_polygons,
                iterable=generator[i],
                class_pool='ProcessPoolExecutor'
            )
            [polygons.extend(p) for p in pp if p] # Remove empty images
        
        polygons = gpd.GeoDataFrame(pd.DataFrame(polygons), 
                            crs=gl.COORDINATES_CRS_REPROJECTION)
        self._save_relational_table(polygons=polygons)
        self._save_file(dissolved=dissolve_polygons(polygons=polygons))
        return None


if __name__ == '__main__':
    extractor = ImagePolygonsExtraction()
    extractor.run()
