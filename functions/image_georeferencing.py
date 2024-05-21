import uuid

import numpy as np
import geopandas as gpd
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon
from skimage.measure import find_contours
from skimage.morphology import label

from configs import vars_globals as gl
from functions.image import zero_padding


def get_image_corners(
        center_x:float, 
        center_y:float, 
        close_polygon:bool=True, 
        as_polygon:bool=False,
        as_fixed_size=False
    ):
    if as_fixed_size:
        dx = (gl.IMAGE_RESOLUTION_X * gl.IMAGE_FIXED_WIDTH) / 2.0
        dy = (gl.IMAGE_RESOLUTION_Y * gl.IMAGE_FIXED_HIGHT) / 2.0
    else:
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


def thr_prediction(im):
    im[im > gl.PREDICTION_THR] = 255
    im[im <= gl.PREDICTION_THR] = 0
    return im


def get_polygons(batch:tuple):
    def get_image_labels(im):
        labels = label(im)
        # Omit background labels
        polygons_index = [i for i in np.unique(labels) if not i == 0]
        return labels, polygons_index
    im, _id, corners = batch
    print(f'Processing image={_id}')
    # Upper left image coordinate
    ul = (
        corners.exterior.xy[0][0], 
        corners.exterior.xy[1][0]
    )

    # Extract the labels in the image
    labels, polygons_index = get_image_labels(im)
    
    polygons = []
    count = 0
    for i in polygons_index:
        # Detect polygon vertices in image
        mask = np.zeros((gl.IMAGE_FIXED_HIGHT, gl.IMAGE_FIXED_WIDTH), dtype='uint8')
        mask[labels == i] =  255
        mask = zero_padding(mask, pad_size=gl.PREDICTIONS_PAD_SIZE)
        polygon = find_contours(mask, gl.PREDICTIONS_FIND_COUNTOURS)

        for pp in polygon:
            # Flip (y,x) to (x,y)
            pp = np.fliplr(pp)
            # Referencing the vertices into the image
            pp = np.array( 
                (
                    ul[0] + ((pp[:,0]-gl.PREDICTIONS_PAD_SIZE) * gl.IMAGE_RESOLUTION_X),
                    ul[1] - ((pp[:,1]-gl.PREDICTIONS_PAD_SIZE) * gl.IMAGE_RESOLUTION_Y)
                )
            ).T
            polygons.append({
                'ID': f'Polygon_{_id}_{str(count).zfill(3)}',
                'ID_image': _id,
                'geometry': Polygon(pp)
            })
            count += 1
    return polygons


def dissolve_polygons(polygons:gpd.GeoDataFrame):
    matrix = polygons.geometry.apply(
        lambda x: polygons.geometry.intersects(x)).values.astype(int)

    _, ids = connected_components(matrix, directed=False)
    dissolved = gpd.GeoDataFrame({
        'ID': ids,
        'geometry': polygons.geometry, 
    }, crs=polygons.crs).dissolve(by='ID')
    dissolved = dissolved.reset_index(drop=True)
    dissolved['ID'] = dissolved.apply(lambda _: uuid.uuid4().hex, axis=1)
    dissolved = dissolved[[
        'ID',
        'geometry'
    ]]
    return dissolved
