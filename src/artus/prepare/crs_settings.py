from rasterio.crs import CRS


def check_crs(obj, crs_code):
    """ Check if the layers provided are set to the correct CRS.

    Args:
        obj : a raster or vector layer read with rasterio or geopandas
        crs_code (str) : the CRS number to check

    Returns: 
        True or False. True means that CRS of the layer is the one expected.
    """
    crs_goal = CRS.from_epsg(crs_code)
    
    if obj.crs == crs_goal:
        print('CRS already matches with desired CRS')
        checking = True
    else:
        checking = False
    return checking
    
