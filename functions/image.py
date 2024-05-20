import numpy as np

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.transform import resize

from configs import vars_globals as gl
from functions.utils import load_coordinates
from functions.utils import get_batch_idxs


def cropping(im, crop_size=20):
    return im[crop_size:im.shape[0]-crop_size, 
              crop_size:im.shape[1]-crop_size]


def im_rescaling(im, clip_min=0.0, clip_max=1.0):
    return (im-im.min()) * ((clip_max-clip_min) / 
             (im.max() - im.min())) + clip_min


def read_image(im_path, crop_size=None, new_shape=None):
    im = imread(im_path)
    # Omit alpha channel
    if im.shape[-1] == 4:
        im = im[...,:3]

    # Cropping the image
    if crop_size is not None:
        im = cropping(im, crop_size=crop_size)

    # Resize the image
    if new_shape is not None:
        im = img_as_ubyte(resize(im, new_shape))
    return im


def zero_padding(im, pad_size=20):
    # New image dimentions
    new_size = [im.shape[0] + (pad_size*2), 
                im.shape[1] + (pad_size*2)]    
    if len(im.shape) > 2:
        new_size.append(im.shape[-1])
    
    # Padding  
    new_im = np.zeros(new_size, dtype=im.dtype)
    new_im[pad_size:new_size[0]-pad_size, 
           pad_size:new_size[1]-pad_size,] = im
    return new_im


class ImageGenerator:

    def __init__(self, batch_size=32):
        self.coords = load_coordinates()
        self.coords = self._get_available_images()
        self.batch_size = batch_size
        self.indices = get_batch_idxs(
            iterable=self.coords, 
            batch_size=self.batch_size
        )

    def _get_available_images(self):
        mask = self.coords['img_exists'] == True
        return self.coords[mask]

    def _get_image(self, im_path:str):
        # Load image
        data = read_image(im_path=im_path)

        # Image preprocessing
        data = cropping(data, crop_size=gl.IMAGE_CROP_SIZE)
        data = im_rescaling(data)
        return data

    def __len__(self):
        return len(self.indices)

    def _data_generation(self, im_paths):
        X = np.empty((
            len(im_paths), 
            gl.IMAGE_FIXED_HIGHT, 
            gl.IMAGE_FIXED_WIDTH, 
            gl.IMAGE_CHANNELS
        ))
        for idx, im_path in enumerate(im_paths):
            features = self._get_image(im_path=im_path)
            X[idx] = features
        return X

    def __getitem__(self, index):
        ini, end = self.indices[index]
        im_paths = self.coords.loc[ini:end, 'img_path'].values
        im_ids = self.coords.loc[ini:end, 'ID'].values
        return self._data_generation(im_paths), im_ids, im_paths
