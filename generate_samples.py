import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.image_georeferencing import get_image_corners
from functions.utils import generate_id
from functions.utils import get_batch_idxs
from functions.utils import load_json_config
from functions.utils import parallel_process
from functions.utils import Timer


logger = WriteLogger(name='generate_samples')
timer = Timer()
CONFIG_PATH = './configs/generate_samples.json'


class SamplesGeneratorStratifiedGrid:

    def __init__(self):
        self.cfg = load_json_config(CONFIG_PATH)
        self.coords = None
        self.batch_size = self.cfg['globals.batch_size']
        self.shape = self._load_input_shapefile()

    def _load_input_shapefile(self) -> gpd.GeoDataFrame:
        path = self.cfg['input.shapefile.path']
        logger(f'Loading input shapefile at: {path}')
        data = gpd.read_file(path)
        if data.crs is None:
            raise ValueError(f'Cannot process this file={path} because CRS is not Defined')
        if not data.crs.equals(gl.COORDINATES_CRS_PROJECTION):
            logger(f'Reproject shape to default CRS projection={gl.COORDINATES_CRS_PROJECTION}', level='debug')
            data = data.to_crs(gl.COORDINATES_CRS_PROJECTION)
        assert data.crs.equals(gl.COORDINATES_CRS_PROJECTION), f'Unexpected error; incorrect CRS in input shapefile; input={data.crs}; expected={gl.COORDINATES_CRS_PROJECTION}'
        if data.shape[0] == 0:
            raise ValueError(f'Empty file; cannot process this shapefile because is empty')
        logger(f'Shapefile loaded successfully')
        return data

    def _build_grid(self) -> np.array:
        xmin, ymin, xmax, ymax = self.shape.bounds.values[0]
        logger(f'Grid bounds: Xmin={xmin}||Xmax={xmax}  Ymin={ymin}||Ymax={ymax}')
        xs = np.arange(xmin, xmax, gl.IMAGE_DTHR)
        ys = np.arange(ymin, ymax, gl.IMAGE_DTHR)
        xs = xs.reshape(1, xs.size)
        ys = ys.reshape(ys.size, 1)
        coords = np.zeros((ys.shape[0] * xs.shape[1], 2))
        logger(f'Grid shape={coords.shape}')
        coords[:,0] = xs.repeat(ys.shape[0], axis=0).flatten()
        coords[:,1] = ys.repeat(xs.shape[1], axis=1).flatten()
        return coords

    def _process_batch(self, batch_idxs:tuple) -> gpd.GeoDataFrame:
        ini, end = batch_idxs
        logger(f'Processing batch; indices from={ini}||to={end}')
        df = gpd.GeoDataFrame(self.coords[ini:end], 
                            columns=['center_x', 'center_y'])
        df['geometry'] = df.apply(lambda x: get_image_corners(
            x.center_x, x.center_y, as_polygon=True), axis=1)
        df.crs = gl.COORDINATES_CRS_PROJECTION
        return gpd.sjoin(df, self.shape, how='inner', op='intersects')

    def _set_coordinates_file(self) -> gpd.GeoDataFrame:
        logger(f'Setting schema output file.....')
        self.coords = gpd.GeoDataFrame(
            pd.concat(self.coords, ignore_index=True),
            crs=gl.COORDINATES_CRS_PROJECTION
        )    
        # Get coordinates in latlong for the google map api
        centers = self.coords.apply(lambda x: 
                        Point(x.center_x, x.center_y), axis=1)
        centers = gpd.GeoSeries(centers, crs=gl.COORDINATES_CRS_PROJECTION)
        centers = centers.to_crs(gl.COORDINATES_CRS_LATLONG)

        self.coords = self.coords[[
            'center_x', 
            'center_y', 
            'geometry'
        ]]
        self.coords['ID'] = self.coords.apply(lambda _: generate_id(), axis=1)
        self.coords['center_lat'] = centers.y
        self.coords['center_long'] = centers.x
        self.coords['center'] = self.coords.apply(lambda x: f'{x.center_lat},{x.center_long}', axis=1)
        self.coords['timestamp'] = None
        self.coords['name'] = None
        self.coords['img_path'] = None
        self.coords['img_exists'] = False
        self.coords['pred_path'] = None
        self.coords['pred_exist'] = False
        self.coords = self.coords[[
            'ID',
            'center_lat',
            'center_long',
            'center_x', 
            'center_y', 
            'center',
            'timestamp',
            'name',
            'img_path',
            'img_exists',
            'pred_path',
            'pred_exist',  # 10 characters in ESRI column file
            'geometry'
        ]]
        logger(f'Total of samples to export: {self.coords.shape[0]}')
        return self.coords
    
    def _save_file(self) -> None:
        path = self.cfg['output.shapefile.path']
        if self.cfg['output.set_as_default']:
            logger(f'Setting output file as default coordinates file')
            path = gl.COORDINATES_FILE

        directory, filename = os.path.split(path)
        if not filename.endswith('.shp'):
            raise ValueError(f'Output filename={path} doesnt endswith `.shp`')
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            answer = input(f'Do you want to overwrite current file={path}? Y/n')
            if answer.lower() != 'y':
                raise FileExistsError(f'Cannot set file as default coordinates file; set a different path at `output.shapefile.path`')

        logger(f'Saving output at: {path}')
        self.coords.to_file(driver='ESRI Shapefile', filename=path)
        return None

    def run(self) -> None:
        logger(f'Generating stratified grid....')
        self.coords = self._build_grid()
        logger(f'Selecting grid samples that intersect with input shapefile; this will take a while....')
        logger(f'Processing samples with batch size={self.batch_size}')
        self.coords = parallel_process(
            func=self._process_batch,
            iterable=get_batch_idxs(
                iterable=self.coords, 
                batch_size=self.batch_size
            ),
            class_pool='Pool'
        )
        if not self.coords:
            logger(f'Cannot continue with application because coords is empty :(')
            return None

        self.coords = self._set_coordinates_file()
        self._save_file()
        return None


@timer.time
def main() -> None:
    try:
        logger(f'Starting Generate Samples application at: {datetime.now()}')
        generator = SamplesGeneratorStratifiedGrid()
        generator.run()
        logger(f'Main application done successfully :)')
    except Exception as exc:
        logger('RuntimeError at main application full traceback is show below:', level='error')
        raise exc
    return None


if __name__ == '__main__':
    main()
