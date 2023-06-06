"""Create a fiftyone dataset with unlabeled or labeled images."""


import fiftyone as fo
import fiftyone.utils.coco as fouc
import json
import os

def create_or_load_dataset(dataset_name, dataset_type=['unlabeled', 'coco'], images_path=None, annotations_path=None, df_test=None, label_type=['segmentations', 'detections']):
    ''' Create a fiftyone labeled or unlabeled dataset.
    
    If the dataset_name provided is in the local 51 database : the dataset is loaded from the database otherwise,
    the dataset is created

    Args:
        dataset_name (str): the name of the dataset that you want to create or load
        dataset_type (str): can be 'unlabeled' for images without annotations or 'coco' if images have coco annotations.
        images_path (str): a path to the directory containing the images (can be tif, png or jpg images)
        annotations_path (str, optional): a path to the COCO annotations files
        df_test (str, optional): a path to the COCO annotations files for test images
        label_type : 'segmentations' for mask or 'detections' for bounding box annotations
    
    Returns : 
        A :class:`fiftyone.core.dataset` with at least 6 fields : id, coco_id, filepath, ground_truth annotations (if provided), tags and basic metadata.
    '''
    
    if dataset_name not in fo.list_datasets():
        if dataset_type == 'coco':
    # Create a 51 dataset with coco annotations
            importer = fouc.COCODetectionDatasetImporter(
                data_path=images_path,
                labels_path=annotations_path, 
                label_types=label_type,
                include_id=True,
                use_polylines=True,
                tolerance=2
                )
            
            dataset = fo.Dataset(
                name=dataset_name,
                persistent=True,
                overwrite=True
                )
            
            dataset.add_importer(
                dataset_importer=importer,
                label_field="ground_truth")

            dataset.persistent = True
            dataset.save()

            # Tag test samples for future usage
            if df_test:
                f = open(df_test)
                test_annot = json.load(f)

                for sample in dataset:
                    for images in test_annot['images']:
                        if os.path.basename(sample.filepath) == images['file_name']:
                            sample.tags.append('test')
                            sample.save()
        
        elif dataset_type == 'unlabeled':
            dataset = fo.Dataset.from_dir(
                name=dataset_name,
                persistent=True,
                overwrite=True,
                dataset_dir=images_path,
                dataset_type=fo.types.ImageDirectory,
                tags='test')
    
    else: 
        # load an existing dataset from the 51 database
        dataset = fo.load_dataset(dataset_name)
    
    dataset.compute_metadata()
    return dataset
