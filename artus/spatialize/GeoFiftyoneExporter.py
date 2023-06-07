"""A module to convert a :class:`fiftyone.core.dataset` into a geojson file.

Annotations in the fiftyone dataset are masks or bounding box, mapped as pixel values 
but if they match with a georeferenced raster file (a tif file for example), 
they are spatialized with an affine transformation.

Typical usage examples:

    dataset = fo.load_dataset('dataset_name')

    geojson_exporter = GeoFiftyoneExporter(
    export_dir = '/path/to/directory/', 
    epsg_code = 'EPSG:4326', 
    label_type = 'detections',
    dest_name = 'spatial_predictions.geojson'
    )

    dataset.export(
    dataset_exporter=geojson_exporter,
    label_field='predictions',
    export_media=False
    )

"""

import pandas as pd
import fiftyone as fo
import fiftyone.utils.data as foud
from shapely import Point
import geopandas
import os
from artus.spatialize.GeoCOCOExporter import GeoCOCOExporter

class GeoFiftyoneExporter(foud.LabeledImageDatasetExporter, GeoCOCOExporter): 
    """Export a fiftyone dataset to a geospatial format (geojson).

     This is only possible if samples are raster format (i.e. tif).

     Datasets of this type are exported in the following format:

         <export_dir>/
             dest_name.geojson

     where ``dest_name.geojson`` is a GeoJson file containing labels.
    
    Attributes:
         export_dir: the directory to write the export
         label_type : the label_type of the concerned fiftyone field ('polylines' 
            for segmentation annotations or 'detections' for bbox annotations)
         epsg_code : the epsg code (for example : '4326' for world coordinates) in which the results will be exported
         dest_name : the file name of the geojson file with the extension (for example : 'spatial_predictions.geojson')
    """

    def __init__(self, export_dir, label_type, epsg_code, dest_name):
        super().__init__(export_dir=export_dir)
        self._data_dir = None
        self._labels_path = None
        self._labels = None
        self._image_exporter = None
        self.label_type = label_type
        self.epsg_code = epsg_code
        self.dest_name = dest_name
        self.sample_dir = None

    @property
    def requires_image_metadata(self):
        """Whether this exporter requires
         :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
         being exported.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) exported by this
         exporter.

         This can be any of the following:

         -   a :class:`fiftyone.core.labels.Label` class. In this case, the
             exporter directly exports labels of this type
         -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
             this case, the exporter can export a single label field of any of
             these types
         -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
             In this case, the exporter can handle label dictionaries with
             value-types specified by this dictionary. Not all keys need be
             present in the exported label dicts
         -   ``None``. In this case, the exporter makes no guarantees about the
             labels that it can export
        """
        return [fo.Detections, fo.Polylines]

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
         the dataset.

         This method is called when the exporter's context manager interface is
         entered, :func:`DatasetExporter.__enter__`.
        """
        self._labels_path = os.path.join(self.export_dir, self.dest_name)
        self._labels = []

        self.columns_names = ['img_filename', 'label' ,'confidence', 'geometry']
        
        self._image_exporter = foud.ImageExporter(
            False, export_path=self._data_dir, default_ext=".jpg",
        )
        self._image_exporter.setup()

    def log_collection(self, sample_collection):
        self.sample_collection = sample_collection

    def shapely_polygons(self, label, metadata):
        """Transform a :class:`fiftyone.core.label.Label` into 
        a shapely polygon. 

        Args:
            label (:class:`fiftyone.core.label.Label`): a detection or polylines field.
            metadata (:class:`fiftyone.core.metadata.ImageMetadata`): metadata of the sample

        Returns:
            :class:`shapely.Polygon`: a shapely polygon that is a square if label was 
            bouding boxes or a polygon if label was a polyline. Shapely polygons are still pixel values.
        """

        imw, imh = (metadata.width, metadata.height)
        if self.label_type=='detections':
            polygon = label.to_shapely(frame_size=(imw,imh))
        elif self.label_type=='polylines':
            polygon = label.to_shapely(frame_size=(imw,imh))
        return polygon

    def export_sample(self, image_or_path, label, metadata=None):
        """Exports the given sample to the dataset.
            
        If images in the dataset are tif then it will convert use the world coordinates for the masks or bbox...
        Otherwise, if images are simply georeferenced it will assigned every label on the image to a single GPS point
        GPS point must have been added to the dataset using :func:`artus.spatialize.LocationImporter.import_csv_locations`

        Args:
            image_or_path: an image or the path to the image on disk
            label: an instance of :meth:`label_cls`, or a dictionary mapping
                 field names to :class:`fiftyone.core.labels.Label` instances,
                 or ``None`` if the sample is unlabeled
            metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
                 instance for the sample. Only required when
                 :meth:`requires_image_metadata` is ``True``
        """

        sample = self.sample_collection[image_or_path]
        metadata = sample.metadata

        # Get field values
        for n_label in getattr(label, self.label_type):

            img_filename = sample.filepath
            label = n_label.label
            confidence = n_label.confidence
            
            if metadata.mime_type == "image/tiff":
                geometry = self.shapely_polygons(n_label, metadata) 
                   
            elif metadata.mime_type == "image/tiff" and self.sample_collection.has_field('location'):
                geometry = Point(sample.location.point)

            gdf_row = (img_filename , label , confidence, geometry)

            self._labels.append(gdf_row)
        
        
    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        exported.
        This method is called when the exporter's context manager interface is
        exited, :func:`DatasetExporter.__exit__`. Polygons are converted to a world
        coordinates with an affine transformation.

        Args:
            *args: the arguments to :func:`DatasetExporter.__exit__`
        """
        
        # Ensure the base output directory exists
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
            
        #convert _labels into gdf
        df = pd.DataFrame(data=self._labels, columns=self.columns_names)
        gdf = geopandas.GeoDataFrame(df, geometry='geometry')

        #convert pixel-values coordinates into geospatial coordinates
        if gdf['geometry'].geom_type.all() != 'Point':
            gdf['transform'] =  [self.get_transform(sample) for sample in gdf['img_filename']]
            gdf = self.affine_transform(gdf)        

        gdf = gdf.filter(items=['img_filename', 'label', 'confidence', 'geometry'])
        gdf.to_file(os.path.join(self.export_dir, self.dest_name), driver='GeoJSON')