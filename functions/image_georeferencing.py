import numpy as np
from shapely.geometry import Polygon

from configs import vars_globals as gl


def get_image_corners(
        center_x:float, 
        center_y:float, 
        close_polygon:bool=True, 
        as_polygon:bool=False
    ):
    dx = (gl.IMAGE_RESOLUTION_X * gl.IMAGE_WIDTH) / 2.0
    dy = (gl.IMAGE_RESOLUTION_Y * gl.IMAGE_HEIGHT) / 2.0
    
    ul = (center_x - dx, center_y + dy)
    ur = (center_x + dx, center_y + dy)
    br = (center_x + dx, center_y - dy)
    bl = (center_x - dx, center_y - dy)

    if close_polygon:
        corners = np.array((ul,ur,br,bl,ul))
    else:
        corners = np.array((ul,ur,br,bl))
    if as_polygon:
        corners = Polygon(corners)
    return corners