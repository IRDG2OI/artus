import solaris.tile as solt
import rasterio
import os
import geopandas
import yaml
from rasterio.crs import CRS
import numpy as np

from artus.prepare.crs_settings import check_crs


def tile_ortho(ortho_path, dest_dir, tuple_tile_size, h_shift=0.0, v_shift=0.0):
    ''' Tile a tif file
    #Inputs:
    - ortho_path: the path to tif file
    - dest_dir : the directory where the tile will be saved (create the directory if it does not exists)
    - src_tile_size: a tuple indicating the length and width of the tiles produced
    - h_shift : a float between 0 and 1 representing the fraction of shifting between horizontal neighbooring tiles
    - v_shift : a float between 0 and 1 representing the fraction of shifting between vertical neighbooring tiles
    #Outputs:
    - tiles with 3 color channels in tif format are saved in the dest_dir
    '''

    raster_tiler = solt.raster_tile.RasterTiler(
        dest_dir=dest_dir,
        src_tile_size=tuple_tile_size,
        use_src_metric_size=False,
        verbose=True,
        alpha=1,
        project_to_meters=True)

    raster_tiler.tile(
        ortho_path, 
        channel_idxs=[1,2,3])
    
    bounds = raster_tiler.tile_bounds
    
    #tile a second grid taking into account the shiftping between tiles
    if h_shift != 0.0 or v_shift!= 0.0:
        sample = rasterio.open(ortho_path)

        aoi_boundary = [
            sample.bounds.left + sample.transform[0]*(tuple_tile_size[1]*h_shift), 
            sample.bounds.bottom + sample.transform[0]*(tuple_tile_size[0]*v_shift), 
            sample.bounds.right - sample.transform[0]*(tuple_tile_size[1]*h_shift), 
            sample.bounds.top - sample.transform[0]*(tuple_tile_size[0]*v_shift)
            ]
        
        raster_tiler_shift = solt.raster_tile.RasterTiler(
            dest_dir=dest_dir,
            src_tile_size=tuple_tile_size,
            use_src_metric_size=False,
            verbose=True,
            alpha=1,
            aoi_boundary=aoi_boundary,
            project_to_meters=True)

        raster_tiler_shift.tile(
            ortho_path, 
            channel_idxs=[1,2,3])
    

        bounds = raster_tiler.tile_bounds + raster_tiler_shift.tile_bounds
    
    return bounds
    

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





        


