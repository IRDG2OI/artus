from pylabel import importer
import numpy as np
import yaml
from shapely.geometry import Polygon
import solaris.vector.polygon as solpol
import pandas as pd
import os
import geopandas
import rasterio

def fiftyone2gdf(fiftyone_dataset):
    masks = fiftyone_dataset.values() #aller chercher automatiquement le champ qui contient les labels

    return 

def coco2gdf(coco_path):
    ''' Convert a coco fiel into a geopandas dataframe.
    #Input : 
    coco_path : the path to a coco.json file

    #Output :
    gdf : a geopandas dataframe with annotation masks converted into shapely polygons but still with pixel values.
    The polygons are indeed, not yet converted to spatial information.
    '''

    coco_dataset = importer.ImportCoco(path=coco_path, name="coco_dataset")
    predictions = coco_dataset.df

    #remove empty lines in dataset, unregular polygon coordinates and nested list in annotation masks
    predictions = predictions[predictions['cat_name'] != '']
    predictions['ann_segmentation'] = [predictions['ann_segmentation'].iloc[index][0] for index in range(len(predictions))]
    predictions = predictions[predictions['ann_segmentation'].str.len() > 4]


    #convert annotation polygons (masks) into shapely polygons
    predictions['geometry'] =  [np.array_split(mask, len(mask)/2) for mask in predictions['ann_segmentation']]

    predictions['geometry'] = [Polygon(mask) for mask in predictions['geometry']]

    gdf = geopandas.GeoDataFrame(predictions)

    return gdf


def get_transform(sample_dir, sample_name):
    ''' Get affine transformation for a tif sample
    #Input :
    sample_dir : the directory where the sample is located
    sample_name: the name of the sample with the extension (.tif)

    #Output:
    the affine.Affine shapely object with the 6 elements required for an affine trasnformation
    '''

    sample = rasterio.open(os.path.join(sample_dir, sample_name))
    
    return sample.transform

def affine_transform(gdf, epsg_code):
    ''' Aplly an affine trasnformation to every polygons (annotations) in a geopandas dataframe
    #Input :
    gdf : a geopandas dataframe containing a geometry column with shapely polygons in pixel value
    epsg_code : the epsg code for the polygons
    
    #Output:
    a geopandas dataframe with polygons spatialized to the desire CRS
    '''

    gdf['geometry'] = [solpol.convert_poly_coords(
        geom=polygon,
        affine_obj=transform) for polygon, transform in zip(gdf.geometry, gdf['transform'])]
    
    gdf = gdf.set_crs(epsg_code, inplace=True)

    return gdf






    
