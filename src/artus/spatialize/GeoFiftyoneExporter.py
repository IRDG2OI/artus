from pylabel import importer
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import fiftyone as fo
import fiftyone.utils.data as foud
import solaris.vector.polygon as solpol
import geopandas
import rasterio
import os

# class FiftyoneGeoExporter(foud.LabeledImageDatasetExporter): ctrl+maj+/
#     """Export a fiftyone dataset to a geospatial format.

#     Datasets of this type are exported in the following format:

#         <dataset_dir>/
#             models_predictions.geojson

#     where ``models_predictions.geojson`` is a GeoJson file.
    
#     Args:
#         export_dir: the directory to write the export
#         label_type : the label_type you want to export ('polylines' for segmentation annotations or 'classifications' for multi/mono label annotations)
#     """

#     def __init__(self, export_dir, label_type, eval_key):
#         super().__init__(export_dir=export_dir)
#         self._data_dir = None
#         self._labels_path = None
#         self._labels = None
#         self._image_exporter = None
#         self.label_type = label_type


#     @property
#     def requires_image_metadata(self):
#         """Whether this exporter requires
#         :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
#         being exported.
#         """
#         return False

#     @property
#     def label_cls(self):
#         """The :class:`fiftyone.core.labels.Label` class(es) exported by this
#         exporter.

#         This can be any of the following:

#         -   a :class:`fiftyone.core.labels.Label` class. In this case, the
#             exporter directly exports labels of this type
#         -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
#             this case, the exporter can export a single label field of any of
#             these types
#         -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
#             In this case, the exporter can handle label dictionaries with
#             value-types specified by this dictionary. Not all keys need be
#             present in the exported label dicts
#         -   ``None``. In this case, the exporter makes no guarantees about the
#             labels that it can export
#         """
#         return [fo.Detections, fo.Polylines]

#     def setup(self):
#         """Performs any necessary setup before exporting the first sample in
#         the dataset.

#         This method is called when the exporter's context manager interface is
#         entered, :func:`DatasetExporter.__enter__`.
#         """
#         self._labels_path = os.path.join(self.export_dir, "model_predictions.geojson")
#         self._labels = []

#         self.columns_names = ['image_filename', 'label' ,'confidence', 'mask_coord' , 'geometry']
        
#         # The `ImageExporter` utility class provides an `export()` method
#         # that exports images to an output directory with automatic handling
#         # of things like name conflicts
#         self._image_exporter = foud.ImageExporter(
#             False, export_path=self._data_dir, default_ext=".jpg",
#         )
#         self._image_exporter.setup()

#     def log_collection(self, sample_collection):
#         self.sample_collection = sample_collection

#     def masks_to_geometry(self, points):
#         if points.str.len() > 4:
#             #convert annotation polygons (masks) into shapely polygons
#             mask =  np.array_split(points, len(points)/2)
#             geometry = Polygon(mask)
#         return geometry

#     def export_sample(self, image_or_path, label, metadata=None):
#         """Exports the given sample to the dataset.

#         Args:
#             image_or_path: an image or the path to the image on disk
#             label: an instance of :meth:`label_cls`, or a dictionary mapping
#                 field names to :class:`fiftyone.core.labels.Label` instances,
#                 or ``None`` if the sample is unlabeled
#             metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
#                 instance for the sample. Only required when
#                 :meth:`requires_image_metadata` is ``True``
#         """
#         #out_image_path, _ = self._image_exporter.export(image_or_path)

#         sample = self.sample_collection[image_or_path]
#         self.gdf_rows = []

#         # Get field values
#         for n_label in getattr(label, self.label_type):

#             image = os.path.basename(sample.filepath)
#             label = n_label.label
#             confidence = n_label.confidence
#             mask_coord = n_label.points
#             geometry = label

#             gdf_row = (image , label , confidence , mask_coord , geometry)

#             self.gdf_rows.append(gdf_row)

#         # _labels contains all the rows to export in the darwincore.csv
#         self._labels.append((self.gdf_rows))

#     def close(self, *args):
#         """Performs any necessary actions after the last sample has been
#         exported.

#         This method is called when the exporter's context manager interface is
#         exited, :func:`DatasetExporter.__exit__`.

#         Args:
#             *args: the arguments to :func:`DatasetExporter.__exit__`
#         """
#         # Ensure the base output directory exists
#         basedir = os.path.dirname(self._labels_path)
#         if basedir and not os.path.isdir(basedir):
#             os.makedirs(basedir)

#         # Write the darwincore CSV file
#         with open(self._labels_path, "w") as f:
#             writer = csv.writer(f)
#             writer.writerow(self.columns_names)
#             for sample in self._labels:
#                 for row in sample:
#                     writer.writerow(row)
