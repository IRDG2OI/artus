"""Convert geojson file into coco file for deep learning purposes.

Annotations in the geojson file will be converted into pixel values with an
inverted affine transformation.
"""

import solaris.data.coco as solcoco
import os


def geojson_to_coco(tiled_geojsons, tiled_ortho, coco_dest_path, feature=None):
    """A function to convert a directory containing geojson to a single coco file.

    Args:
        tiled_geojsons (list) : a list containing paths to geojsons path, in geojson format.
        tiled_ortho (list): a list containing paths to raster tiles, in tif format.
        coco_dest_path (str): the path where the coco file will be saved. 
        feature (str): the geojson feature which is the class to affect to the polygons or bbox.

    Returns:
        A coco file saved at the coco_dest_path(.json)
    """
        
    if not os.path.exists(coco_dest_path):
        os.makedirs(coco_dest_path)
        
    coco_dict = solcoco.geojson2coco(
        image_src=tiled_ortho,
        label_src=tiled_geojsons,
        output_path=coco_dest_path,
        category_attribute=feature,
        matching_re=r'(\d+\.\d+_\-?\d+\.\d+)',
        verbose=0)
