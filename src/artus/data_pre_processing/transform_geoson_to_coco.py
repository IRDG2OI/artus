import solaris.data.coco as solcoco
import glob
import os


def geojson_to_coco(tiled_geojsons, tiled_ortho, coco_dest_path, feature=None):
    '''a function to convert a directory containing geojson to a coco file.
    # Inputs:
    - tiled_geojsons : a list containing geojsons files
    - tiled_ortho : a list containing tiles (tif format)
    # Output:
    - a coco file (.json)
    '''
        
    if not os.path.exists(coco_dest_path):
        os.makedirs(coco_dest_path)
        
    coco_dict = solcoco.geojson2coco(
        image_src=tiled_ortho,
        label_src=tiled_geojsons,
        output_path=coco_dest_path,
        category_attribute=feature,
        matching_re=r'(\d+\.\d+_\-?\d+\.\d+)',
        verbose=0)