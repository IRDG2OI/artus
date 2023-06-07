"""A module to import sample location from a CSV.

This module is dedicated to user that only have images georeferenced by a GPS point. It
will add the matching GPS point to every sample in the fiftyone dataset."""

import fiftyone as fo
import pandas as pd
import tqdm
import os

def import_csv_locations(location_path, fiftyone_dataset):
    """A function to import GPS locations from a CSV file and add it to  the matching sample in a fiftyone dataset.

    Args:
        location_path (str): a path to csv containing 3 columns called 'filename', 'latitude' and 'longitude'
        fiftyone_dataset (:class:`fiftyone.core.dataset`): a fiftyone dataset containing images

    Returns:
        :class:`fiftyone.core.dataset`: the same dataset with a geolocation point added to the samples listed in the csv file.
    """
    locations_df = pd.read_csv(location_path)

    for sample in tqdm(fiftyone_dataset):
        image_location = locations_df[locations_df['filename']==os.path.basename(sample['filepath'])]
        sample['location'] = fo.GeoLocation(point=[image_location['latitude'].iloc[0], image_location['longitude'].iloc[0]])
        sample.save()

    fiftyone_dataset.save()
    return fiftyone_dataset

