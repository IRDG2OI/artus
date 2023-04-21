import solaris.tile as solt
import rasterio
import os
import geopandas
import yaml


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
        


