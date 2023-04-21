import os
import geopandas
import yaml
import rasterio
from rasterio.crs import CRS
import numpy as np
from crs_settings import check_crs
from tile import tile_ortho

def clip_annotated_ortho(annot_path, raster_path, matching_crs, dest_dir, tuple_tile_size, h_shift=0, v_shift=0, annot_type=['segm', 'bbox']):
    ''' A function that clip annotated raster into tiles.
    # Inputs:
    - annot_path : the path to the spatialannotations (shapefile or geojson)
    - raster_path : the path to an annotated raster (tif file)
    - matching_crs : the epsg_code matching the raster and annotation file
    - dest_dir : the destination directory where tiles and matching geojsons annotations will be saved
    - tuple_tile_size : a tuple indicating the length and width of the tiles produced
    - h_shift :  a float between 0 and 1 representing the fraction of shifting between horizontal neighbooring tiles
    - v_shift : a float between 0 and 1 representing the fraction of shifting between vertical neighbooring tiles
    - annot_type : whether the anntoations are segmentation or bounding boxes
    # Output : 
    - a directory at the dest_dir path containig tiles and matching geojsons for each tile containing annotations (if the tile did not include annotations then the geojson     file is not created)
    '''

    ortho = rasterio.open(raster_path)
    annotations = geopandas.read_file(annot_path)

    #Make sure CRS from raster and vector layers are set to the appropriate EPSG code
    if check_crs(ortho, matching_crs) and check_crs(annotations, matching_crs):

        #remove geometry in annotations that is not a polygon
        if annot_type=='segm':
            index_to_drop = annotations.loc[annotations.geometry.geometry.type!='Polygon'].index
            annotations = annotations.drop(index=index_to_drop)

        #crop orthomosaics into tiles
        tile_bounds = tile_ortho(
            ortho_path=raster_path,
            dest_dir=dest_dir,
            tuple_tile_size=tuple_tile_size,
            h_shift=h_shift,
            v_shift=v_shift
        )

        #clip annotation layer (shp) to the same boundaries as the clipped tiles
        geojsons_dir = os.path.join(dest_dir, 'geojsons/')
        if not os.path.exists(geojsons_dir):
            os.makedirs(geojsons_dir)
            
        for tile_bound in tile_bounds:
            annotations_clipped = annotations.clip(tile_bound, keep_geom_type=True)
            if len(annotations_clipped) > 0:
                if annot_type=='segm':
                    if not annotations_clipped.loc[annotations_clipped.geometry.geometry.type!='Polygon'].empty:
                        annotations_clipped = annotations_clipped.explode()

                annotations_clipped.to_file(f'{dest_dir}/geoms_{abs(np.round(tile_bound[0], 7))}_{abs(np.round(tile_bound[3],7))}.geojson', driver='GeoJSON')





