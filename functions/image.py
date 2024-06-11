from typing import Any

import numpy as np
import geopandas as gpd
from skimage.io import imread

from functions.base_logger import WriteLogger
from functions.utils import get_batch_idxs
from functions.utils import load_coordinates


logger = WriteLogger(name=__name__, level='INFO')


def cropping(img:np.array, crop_size:int=20) -> np.array:
    logger(f'Cropping image={img.shape} with crop size={crop_size}', level='debug')
    new_img = img[crop_size:img.shape[0]-crop_size, 
                    crop_size:img.shape[1]-crop_size]
    logger(f'New image dimentions={new_img.shape}', level='debug')
    return new_img


def im_rescaling(
        img:np.array, 
        clip_min:float=0.0, 
        clip_max:float=1.0
    ) -> np.array:
    logger(f'Rescaling image with: clip_min={clip_min}, clip_max={clip_max}', level='debug')
    return (img-img.min()) * ((clip_max-clip_min) / 
                (img.max() - img.min())) + clip_min


def read_image(img_path:str) -> np.array:
    logger(f'Reading image at: {img_path}')
    img = imread(img_path)
    # Omit alpha channel
    if img.shape[-1] == 4:
        logger(f'Image contains alpha channel, removing....', level='debug')
        img = img[...,:3]
    return img


def zero_padding(img:np.array, pad_size:int=20) -> np.array:
    logger(f'Apply image={img.shape} padding with padding size={pad_size}', level='debug')
    # New image dimentions
    new_size = [img.shape[0] + (pad_size*2), 
                img.shape[1] + (pad_size*2)]    
    if len(img.shape) > 2:
        new_size.append(img.shape[-1])
    
    # Padding  
    new_img = np.zeros(new_size, dtype=img.dtype)
    new_img[pad_size:new_size[0]-pad_size, 
           pad_size:new_size[1]-pad_size,] = img
    logger(f'New image dimentions={new_img.shape}', level='debug')
    return new_img


class ImageGenerator:

    def __init__(self, batch_size:int=32):
        self.batch_size = batch_size
        self.coords = load_coordinates()
        self.coords = self._get_available_images()
        self.indices = get_batch_idxs(
            iterable=self.coords, 
            batch_size=self.batch_size
        )

    def _get_available_images(self) -> gpd.GeoDataFrame:
        mask = self.coords['img_exists'] == True
        return self.coords[mask]

    def _get_image(self, img_path:str) -> np.array:
        return None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index:int) -> Any:
        return None
