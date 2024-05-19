import os
import uuid
import urllib.parse
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from skimage.io import imread

from configs import vars_globals as gl
from functions.base_logger import WriteLogger
from functions.utils import JsonConfig
from functions.utils import Timer
from functions.utils import load_environment_variable
from functions.utils import load_json_config
from functions.utils import load_coordinates
from functions.utils import parallel_process


logger = WriteLogger(name=__name__, level='DEBUG')
timer = Timer()
CONFIG_PATH = './configs/google_staticmap_image_downloader.json'


class ImageDownloadError(Exception):
    pass


class ImageDownloader:

    def __init__(self):
        self.execution = uuid.uuid4()
        self.now = datetime.now()
        self.cfg = load_json_config(CONFIG_PATH)
        self.cfg = self._load_api_key()

    def _get_max_sample_mask(self, coords:pd.DataFrame) -> pd.Series:
        mask = coords['img_exists'] == False
        return mask

    def _load_api_key(self) -> JsonConfig:
        key = load_environment_variable(
            name=self.cfg['input.api.params.key.env_variable'],
            load_from_environ=self.cfg['input.api.params.key.load_from_environ']
        )
        self.cfg['input.api.params.key'] = key
        return self.cfg

    def _build_img_path(self, pid:int):
        t = datetime.now()
        date = t.strftime(gl.IMAGE_FMT_DATE)
        timestamp = t.strftime(gl.IMAGE_FMT_TIMESTAMP)
        partition = f'{gl.IMAGE_PARTITION}={date}'
        filename = f'{gl.IMAGE_PREFIX}_{pid}_{timestamp}{gl.IMAGE_SUFFIX_DATA}'
        path = os.path.join(gl.DIRECTORY_IMAGES, partition)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        return path, date, timestamp

    def _build_url(self, center:str):
        url = self.cfg['input.api.url']
        params = urllib.parse.urlencode(
            {**{
                'center': center,
                'size': f'{gl.IMAGE_HEIGHT}x{gl.IMAGE_WIDTH}'
            }, **self.cfg['input.api.params']})
        return f'{url}?{params}'

    def _download_image(self, row:np.array):
        _id, center = row
        pid = mp.current_process().pid
        url = self._build_url(center=center)
        path, _, timestamp = self._build_img_path(pid=pid)
        error = False
        timeout = self.cfg['input.api.timeout']
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code != 200:
                raise ImageDownloadError(f'Cannot download image at={url}; response={response.reason}|code={response.status_code}')
            with open(path, 'wb') as file:
                file.write(response.content)
                logger(f'Image saved at: {path}'.format(path))
        except Exception as exc:
            error = True
            logger.exception(exc)

        delate = False
        if os.path.exists(path):
            logger(f'Image exists at: {path}, checking if is not currupted', level='debug')
            try:
                im = imread(path)
                if im is None:
                    # TODO: Veriify if skimage returns None value?
                    delate = True
            except Exception as exc:
                logger.exception(exc)
                delate = True
            if delate:
                if os.path.exists(path):
                    os.remove(path)
                    logger(f'Removing Image={path} because is corrupted', level='warning')
        
        if error or delate:
            logger(f'Cannot save image info, because image download error or image corruption error')
            return None

        file = os.path.join(gl.DIRECTORY_IMAGES, f'.{self.execution}_image_info_{pid}.csv')
        if not os.path.exists(file):
            with open(file, 'w') as f:
                header = 'ID,timestamp,name,img_path,img_exists\n'
                f.write(header)

        with open(file, 'a') as f:
            name = os.path.split(path)[-1].replace(gl.IMAGE_SUFFIX_DATA, '')
            line = f'{_id},{timestamp},{name},{path},True\n'
            f.write(line)
        return None

    def _merge_csvs(self, coords:pd.DataFrame) -> None:
        files = [os.path.join(gl.DIRECTORY_IMAGES, f) for f in os.listdir(gl.DIRECTORY_IMAGES) 
                      if f.startswith(f'.{self.execution}')]
        if not files:
            logger(f'Image info tmp files not available for current execution')
            return None
        data = []
        for f in files:
            data.append(pd.read_csv(f))
        data = pd.concat(data, ignore_index=True, axis=0)
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
        coords.to_csv(gl.COORDINATES_FILE, index=False, header=True)
        for f in files:
            try:
                os.remove(f)
            except Exception as exc:
                logger(f'Cannot remove tmp file={f}', level='debug')
                logger.exception(exc)
        return None

    @timer.time
    def run(self) -> None:
        logger(f'Running google image downloader at: {self.now}')
        coords = load_coordinates()
        mask = self._get_max_sample_mask(coords=coords)
        parallel_process(
            func=self._download_image,
            iterable=coords.loc[mask, ['ID', 'center']]\
                           .values[:self.cfg['globals.max_images_to_download']],
            class_pool='Pool',
            process=self.cfg['globals.n_process']
        )
        self._merge_csvs(coords=coords)
        return None


if __name__ == '__main__':
    downloader = ImageDownloader()
    downloader.run()