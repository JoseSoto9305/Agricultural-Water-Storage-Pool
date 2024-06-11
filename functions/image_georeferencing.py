from typing import Any

import numpy as np
from shapely.geometry import Polygon

from configs import vars_globals as gl
from functions.base_logger import WriteLogger


logger = WriteLogger(name=__name__)


def get_image_corners(
        center_x:float, 
        center_y:float, 
        close_polygon:bool=True, 
        as_polygon:bool=False,
        as_fixed_size:bool=False
    ) -> Any:
    logger(f'Get image corners at: X={center_x}||Y={center_y}', level='debug')
    if as_fixed_size:
        dx = (gl.IMAGE_RESOLUTION_X * gl.IMAGE_FIXED_WIDTH) / 2.0
        dy = (gl.IMAGE_RESOLUTION_Y * gl.IMAGE_FIXED_HEIGHT) / 2.0
    else:
        dx = (gl.IMAGE_RESOLUTION_X * gl.IMAGE_WIDTH) / 2.0
        dy = (gl.IMAGE_RESOLUTION_Y * gl.IMAGE_HEIGHT) / 2.0

    ul = (center_x - dx, center_y + dy)
    ur = (center_x + dx, center_y + dy)
    br = (center_x + dx, center_y - dy)
    bl = (center_x - dx, center_y - dy)

    if close_polygon:
        logger(f'Closing polygon enable', level='debug')
        corners = np.array((ul,ur,br,bl,ul))
    else:
        corners = np.array((ul,ur,br,bl))
    if as_polygon:
        logger(f'Returning as shapely.Polygon enable', level='debug')
        corners = Polygon(corners)
    return corners
