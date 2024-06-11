import os
import multiprocessing as mp
import urllib.parse
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from skimage.io import imread

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.utils import generate_id
from functions.utils import JsonConfig
from functions.utils import load_coordinates
from functions.utils import load_environment_variable
from functions.utils import load_json_config
from functions.utils import parallel_process
from functions.utils import Timer


logger = WriteLogger(name='image_downloader')
timer = Timer()

CONFIG_PATH = './configs/google_staticmap_image_downloader.json'
IMAGE_SIZE = f'{gl.IMAGE_HEIGHT}x{gl.IMAGE_WIDTH}'
STATUS_CODE_OK = 200


class ImageDownloadError(Exception):
    pass


class ImageDownloader:

    def __init__(self):
        self.execution = generate_id()
        logger(f'Image Download Execution ID={self.execution}', level='debug')
        self.cfg = load_json_config(CONFIG_PATH)
        self.cfg = self._load_api_key()
        self.timeout = self.cfg['input.api.timeout']

    def _create_output_directory(self):
        if not os.path.exists(gl.DIRECTORY_IMAGES):
            logger(f'Creating output directory at: {gl.DIRECTORY_IMAGES}', level='debug')
            os.makedirs(gl.DIRECTORY_IMAGES, exist_ok=True)
        return None

    def _get_pending_images(self, coords:gpd.GeoDataFrame) -> pd.Series:
        mask = coords['img_exists'] == False
        logger(f'Pending images to download={mask.sum()}/{coords.shape[0]}')
        return mask

    def _load_api_key(self) -> JsonConfig:
        logger(f'Loading Google Staticmap API key', level='debug')
        key = load_environment_variable(
            name=self.cfg['input.api.params.key.env_variable'],
            load_from_environ=self.cfg['input.api.params.key.load_from_environ']
        )
        self.cfg['input.api.params.key'] = key
        logger(f'API key retrieved successfully', level='debug')
        return self.cfg

    def _build_img_path(self, pid:int) -> tuple:
        # build filename
        t = datetime.now()
        date = t.strftime(gl.IMAGE_FMT_DATE)
        timestamp = t.strftime(gl.IMAGE_FMT_TIMESTAMP)
        partition = f'{gl.IMAGE_PARTITION}={date}'
        filename = f'{gl.IMAGE_PREFIX}_{pid}_{timestamp}{gl.IMAGE_SUFFIX_DATA}'
        
        # Build directory and path
        img_path = os.path.join(gl.DIRECTORY_IMAGES, partition)
        if not os.path.exists(img_path):
            os.makedirs(img_path, exist_ok=True)
        img_path = os.path.join(img_path, filename)
        return img_path, timestamp

    def _build_url(self, image_center:str) -> str:
        url = self.cfg['input.api.url']
        params = urllib.parse.urlencode(
            {**{
                'center': image_center,
                'size': IMAGE_SIZE
            }, **self.cfg['input.api.params']})
        return f'{url}?{params}'

    def _write_image(self, url:str, image_path:str) -> bool:
        error = False
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code != STATUS_CODE_OK:
                raise ImageDownloadError(f'Cannot download image; response={response.reason}|code={response.status_code}')
            with open(image_path, 'wb') as file:
                file.write(response.content)
                logger(f'Image saved at: {image_path}')
        except Exception as exc:
            error = True
            logger.exception(exc)
        return error
    
    def _check_image_corruption(self, image_path:str) -> bool:
        delate = False
        if os.path.exists(image_path):
            logger(f'Image exists at: {image_path}, checking if is not currupted', level='debug')
            try:
                im = imread(image_path)
                if im is None:
                    # TODO: Veriify if skimage returns None value?
                    delate = True
            except Exception as exc:
                logger.exception(exc)
                delate = True
            if delate:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger(f'Removing Image={image_path} because is corrupted', level='warning')
        return delate

    def _write_tmp_record(self, 
            pid:int,
            image_id:str,
            image_path:str,
            timestamp:str
        ) -> None:

        file = os.path.join(gl.DIRECTORY_IMAGES, f'.{self.execution}_image_info_{pid}.csv')
        if not os.path.exists(file):
            with open(file, 'w') as f:
                header = 'ID,timestamp,name,img_path,img_exists\n'
                f.write(header)
        logger(f'Writing record at file={file}', level='debug')
        with open(file, 'a') as f:
            name = os.path.split(image_path)[-1].replace(gl.IMAGE_SUFFIX_DATA, '')
            line = f'{image_id},{timestamp},{name},{image_path},True\n'
            f.write(line)
        return None

    def _download_image(self, row:np.array) -> None:
        image_id, image_center = row
        pid = mp.current_process().pid
        url = self._build_url(image_center=image_center)
        image_path, timestamp = self._build_img_path(pid=pid)
        error = self._write_image(url=url, image_path=image_path)
        delate = self._check_image_corruption(image_path=image_path)
        
        if error or delate:
            logger(f'Cannot save image info, because image download error or image corruption error at: {image_path}', level='warning')
            return None
        
        self._write_tmp_record(
            pid=pid,
            image_id=image_id,
            image_path=image_path,
            timestamp=timestamp
        )
        return None

    def _update_coordinates_file(self, coords:gpd.GeoDataFrame) -> None:
        logger(f'Updating coordinates file with download information of current execution....')

        # Loading temporal information
        files = [os.path.join(gl.DIRECTORY_IMAGES, file) for file in os.listdir(gl.DIRECTORY_IMAGES) 
                      if file.startswith(f'.{self.execution}')]
        if not files:
            logger(f'Image info tmp files not available for current execution; exit from function')
            return None
        data = []
        for file in files:
            data.append(pd.read_csv(file))
        data = pd.concat(data, ignore_index=True, axis=0)

        # Updating coords
        logger(f'Total of coordinates records to update: {data.shape[0]}')
        logger(f'Coordinates shape before update={coords.shape}', level='debug')
        coords = coords.merge(data, 
            left_on=['ID'], 
            right_on=['ID'], 
            how='left',
            suffixes=(None, '_DROP')
        )
        columns = [
            'timestamp',
            'name', 
            'img_path', 
            'img_exists', 
        ]
        for c in columns:
            mask = ~coords[f'{c}_DROP'].isna()
            coords.loc[mask, c] = coords.loc[mask, f'{c}_DROP']
        coords.drop([f'{c}_DROP' for c in columns], axis=1, inplace=True)
        coords['img_exists'] = coords['img_exists'].astype('int64')
        coords = coords.drop_duplicates(subset=['ID']) # Only if input file contains duplicates
        coords.to_file(driver='ESRI Shapefile', filename=gl.COORDINATES_FILE)
        logger(f'Coordinates shape after update={coords.shape}', level='debug')

        logger(f'Removing temporal files of current execution.....', level='debug')
        for file in files:
            try:
                logger(f'Removing temporal file={file}', level='debug')
                os.remove(file)
            except Exception as exc:
                logger(f'Cannot remove tmp file={file}', level='debug')
                logger.exception(exc)
        return None

    def run(self) -> None:
        logger(f'Downloading images at: {self.cfg["input.api.url"]}')
        self._create_output_directory()
        coords = load_coordinates()
        mask = self._get_pending_images(coords=coords)
        n = self.cfg['globals.max_images_to_download']
        logger(f'Maximun files to download for current execution={n}')
        parallel_process(
            func=self._download_image,
            iterable=coords.loc[mask, ['ID', 'center']].values[:n],
            class_pool='Pool',
            process=self.cfg['globals.n_process']
        )
        self._update_coordinates_file(coords=coords)
        return None


@timer.time
def main() -> None:
    try:
        logger(f'Starting Google Staticmap Image Downloader application at: {datetime.now()}')
        downloader = ImageDownloader()
        downloader.run()
        logger(f'Main application done successfully :)')
    except Exception as exc:
        logger('RuntimeError at main application full traceback is show below:', level='error')
        raise exc
    return None

if __name__ == '__main__':
    main()
