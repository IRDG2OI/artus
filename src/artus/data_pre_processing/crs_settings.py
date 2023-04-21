import geopandas
import yaml
import rasterio
from rasterio.crs import CRS


def check_crs(obj, crs_code):
    ''' check if the layers provided are set to the correct CRS
    # Inputs :
    obj : a raster or vector layer read with rasterio or geopandas
    crs_code : the CRS number to check
    obj_type : can be a raster layer or a vector layer
    #Outputs : 
    the obj to the desired crs
    '''
    crs_goal = CRS.from_epsg(crs_code)
    
    if obj.crs == crs_goal:
        print('CRS already matches with desired CRS')
        checking = True
    else:
        checking = False
    return checking
    
