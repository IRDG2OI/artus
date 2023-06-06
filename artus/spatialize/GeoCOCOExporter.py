"""A module to convert a coco file into a geojson file.

Annotations in the coco file are pixel values but if they match with a georeferenced raster
file (a tif file for example), they are spatialized with an affine transformation.

Typical usage examples:

    geojson_exporter = GeoCOCOExporter(
    coco_path = 'path/to/coco/file.json',
    sample_dir = 'path/to/tif/directory/', 
    epsg_code = 'EPSG:4326', 
    dest_path = 'path/to/export/directory/',
    dest_name = 'predictions.geojson'
)
    geojson_exporter.export()

"""

from pylabel import importer
import numpy as np
from shapely.geometry import Polygon
import solaris.vector.polygon as solpol
import geopandas
import rasterio
import os

class GeoCOCOExporter():
    """Class to convert a `coco.json` file into a geojson file. 

    Annotations in the coco file are pixel values. When converted to geojson format
    with an affine transformation, they become georeferenced.

    Attributes:
        coco_path (str): the relative or absolute path top a coco.json file
        epsg_code (str): the epsg_code (like EPSG:4326') to set the CRS of the geojson file
        dest_path (str): the destination path where the results will be exported. If the directory 
            does not exist yet, it will be created.
        dest_name (str): the name of the file to store
        sample_dir (str, optional): the path to the directory where the tif files are located. 
            If sample_name is already the full path of the sample, no need to provide sample_dir.

    """
    def __init__(self, coco_path, epsg_code, dest_path, dest_name, sample_dir=None):
        self.coco_path = coco_path
        self.epsg_code = epsg_code
        self.dest_path = dest_path
        self.dest_name = dest_name
        self.sample_dir = sample_dir

    def coco2gdf(self):
        """ Convert a coco file into a geopandas dataframe.

        Returns:
            gdf (:py:class: `geopandas.GeoDataFrame`): a dataframe with annotation masks converted into shapely polygons but still with pixel values.
                The polygons are indeed, not yet converted to spatial information.
        """

        coco_dataset = importer.ImportCoco(path=self.coco_path, name="coco_dataset")
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
    
    def get_transform(self, sample):
        """Get affine transformation for a tif sample.

        Args:
            sample (str): the path or name of the sample with the extension that should be .tif

        Returns:
            the :py:class:`affine.Affine` shapely object with the 6 elements required for an affine transformation
        """
        if self.sample_dir:
            sample = rasterio.open(os.path.join(self.sample_dir, sample))
        else:
            sample = rasterio.open(sample)
        return sample.transform
        
    def affine_transform(self, gdf):
        """ Apply an affine transformation to every polygons (annotations) in the 
        geometry column of a :py:class:`geopandas.Dataframe`

        Args:
            gdf : a g:py:class:`geopandas.Dataframe` containing a geometry column with 
                shapely polygons in pixel value.
        
        Returns:
            A :py:class:`geopandas.Dataframe` with polygons spatialized to the desire CRS
        """

        gdf['geometry'] = [solpol.convert_poly_coords(
            geom=polygon,
            affine_obj=transform) for polygon, transform in zip(gdf.geometry, gdf['transform'])]
        
        gdf = gdf.set_crs(self.epsg_code, inplace=True)

        return gdf
    
    def export(self):
        """ Export the geopandas dataframe to geojson.

        Returns:
            A geojson file exported at the dest_path + dest_name location.
        """
        #check if dest_path exists (create it if needed)
        os.makedirs(self.dest_path, exist_ok=True)

        gdf = self.coco2gdf()
        gdf['transform'] =  [self.get_transform(sample) for sample in gdf['img_filename']]
        affine_transformed_gdf = self.affine_transform(gdf)
        gdf_export = affine_transformed_gdf.filter(items=['img_filename', 'geometry', 'cat_name', 'cat_id'])
        gdf_export.to_file(self.dest_path + self.dest_name + '.geojson', driver='GeoJSON')
