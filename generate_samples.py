import os
import uuid

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.utils import parallel_process
from functions.utils import load_json_config
from functions.utils import get_batch_idxs
from functions.utils import Timer
from functions.image_georeferencing import get_image_corners


logger = WriteLogger(name=__name__, level='DEBUG')
timer = Timer()
CONFIG_PATH = './configs/generate_samples.json'


class SamplesGenerator:

    def __init__(self):
        self.cfg = load_json_config(CONFIG_PATH)
        self.batch_size = self.cfg['globals.batch_size']
        self.shape = gpd.read_file(self.cfg['input.shapefile.path'])
        if self.cfg['input.shapefile.is_latlong']:
            self.shape = self.shape.to_crs(gl.COORDINATES_CRS_REPROJECTION)
        self.coords = None

    def _stratified_grid_process_batch(self, batch_idxs):
        print(f'Processing batch={batch_idxs}')
        ini, end = batch_idxs
        df = gpd.GeoDataFrame(self.coords[ini:end], 
                            columns=['center_x', 'center_y'])
        df['geometry'] = df.apply(lambda x: get_image_corners(
            x.center_x, x.center_y, as_polygon=True), axis=1)
        df.crs = gl.COORDINATES_CRS_REPROJECTION
        return gpd.sjoin(df, self.shape, how='inner', op='intersects')

    def _set_coordinates_file(self):
        centers = self.coords.apply(lambda x: 
                        Point(x.center_x, x.center_y), axis=1)
        centers = gpd.GeoSeries(centers, crs=gl.COORDINATES_CRS_REPROJECTION)
        centers = centers.to_crs(gl.COORDINATES_CRS_LATLONG)

        self.coords = self.coords[[
            'center_x', 
            'center_y', 
            'geometry'
        ]]
        self.coords['ID'] = self.coords.apply(lambda x: uuid.uuid4().hex, axis=1)
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
            'pred_exist',  # 10 character in esri column file
            'geometry'
        ]]
        return self.coords
    
    def _set_as_default(self):
        if os.path.exists(gl.COORDINATES_FILE):
            answer = input(f'Do you want to overwrite current file={gl.COORDINATES_FILE}? Y/n')
            if answer.lower() != 'y':
                raise FileExistsError(f'Cannot set file as default coordinates file; set a different path at `output.path` configuration and set `output.set_as_default` to False')
        self.cfg['output.path'] = gl.COORDINATES_FILE
        return None

    def _save_file(self):
        path = self.cfg['output.path']
        self.coords.to_file(driver='ESRI Shapefile', filename=path)
        return None

    @timer.time
    def stratified_grid(self):
        xmin, ymin, xmax, ymax = self.shape.bounds.values[0]
        xs = np.arange(xmin, xmax, gl.IMAGE_DTHR)
        ys = np.arange(ymin, ymax, gl.IMAGE_DTHR)
        xs = xs.reshape(1,xs.size)
        ys = ys.reshape(ys.size,1)
        self.coords = np.zeros((ys.shape[0] * xs.shape[1], 2))
        self.coords[:,0] = xs.repeat(ys.shape[0], axis=0).flatten()
        self.coords[:,1] = ys.repeat(xs.shape[1], axis=1).flatten()
        self.coords = parallel_process(
            func=self._stratified_grid_process_batch,
            iterable=get_batch_idxs(
                iterable=self.coords, 
                batch_size=self.batch_size
            ),
            class_pool='ProcessPoolExecutor'
        )
        self.coords = gpd.GeoDataFrame(
            pd.concat(self.coords, ignore_index=True),
            crs=gl.COORDINATES_CRS_REPROJECTION)
        self.coords = self._set_coordinates_file()
        if self.cfg['output.set_as_default']:
            self._set_as_default()
        self._save_file()
        return None


if __name__ == '__main__':
    generator = SamplesGenerator()
    generator.stratified_grid()
