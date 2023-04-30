from pylabel import importer
import numpy as np
import yaml
from shapely.geometry import Polygon
import solaris.vector.polygon as solpol
import pandas as pd
import os
import geopandas
import rasterio

def export(affine_transformed_gdf, dest_path, dest_name, site_by_site=bool):
    ''' Export the geopandas dataframe to geojson
    #Input:
    - affine_transformed_gdf : a geopandas dataframe transformed thanks to affine_transform()
    - dest_path : the destination path where the results will be exported (if directory does not exist, it will be created)
    - dest_name : the name of the file to store
    - site_by_site : if geopandas dataframe contains different acquisition site and if site_by_site=True then the 
    several geojsons will be exported for every acquisition site.

    #Output:
    a geojson file exported at the dest_path
    '''
    #check if dest_path exists (create it if needed)
    os.makedirs(dest_path, exist_ok=True)

    gdf_export = affine_transformed_gdf.filter(items=['img_filename', 'geometry', 'cat_name', 'cat_id'])

    if site_by_site:
        sites = gdf_export['img_filename']
        sites = [string[:5] for string in sites]

        for site in sites:
            gdf_export_ortho = gdf_export[gdf_export.img_filename.str.startswith(site)]
            gdf_export_ortho.to_file(os.path.join(dest_path, f'predictions_{site}.geojson'), driver='GeoJSON')  

    else:
        gdf_export.to_file(dest_path + dest_name + '.geojson', driver='GeoJSON')


