"""The module provides a splitter to prepare coco annotations files for deep learning training process.

The class: `COCOSplitter` splits a coco annnotations file into 3 files : the coco_train, 
coco_test and coco_val. Annotations are firstly parsed so under-reprensentated classes are removed.
Proportions of annotations per class are maintained in the splits.


Typical usage examples:

    `splitter = COCOSplitter(
        coco_path='/path/to/coco.json',
        export_dir='/path/to/export/coco/splits/',
        coco_train_name='coco_train',
        coco_test_name='coco_test',
        coco_val_name='coco_val',
        min_nb_occurences=50,
        train_pct=.8,
        val_pct=.1,
        test_pct=.1,
        batch_size=8
    )`
    `splitter.split()`

"""

from pylabel import importer, dataset
import os
from artus.evaluate_model.coco_stats import COCOStats

class COCOSplitter(COCOStats):
    """ Splits a coco file into a train, test and (optional) validation coco files. 
    
    Under-reprensentated classes are optionnally removed according the min_nb_occurrences set.
    Proportions of class annotations are kept into the splits. If no train_pct, val_pct or test_pct are set, 
    proportions of annotations in th splits will be set to 0.8, 0.1 and 0.1 respectively.

    Attributes:
        coco_path (str): a path to a coco file
        export_dir (str): a directory where the splits of the coco files will be exported
        coco_train_name (str): the name of the file with the annotations for training
        coco_test_name (str): the name of the file with the annotations for testing
        coco_val_name (str): the name of the file with the annotations for validation
        min_nb_occurrences (int): the minimum number of occurrences of a class to be 
        kept in the dataset to remove under representated classes
        train_pct (float): the fraction of annotations that will go into the train coco file
        val_pct (float): the fraction of annotations that will go into the validation coco file
        test_pct (float): the fraction of annotations that will go into the test coco file
        batch_size (int): number of images per batch process
    """

    def __init__(self, coco_path, export_dir, coco_train_name, coco_test_name, coco_val_name, min_nb_occurrences=None, train_pct=.8, val_pct=.1, test_pct=.1, batch_size=8):
        """Initializes the coco file splitter.

        Args:
            coco_path (str): a path to a coco file
            export_dir (str): a directory where the splits of the coco files will be exported
            coco_train_name (str): the name of the file with the annotations for training
            coco_test_name (str): the name of the file with the annotations for testing
            coco_val_name (str): the name of the file with the annotations for validation
            min_nb_occurrences (int): the minimum number of occurrences of a class to be 
            kept in the dataset to remove under representated classes
            train_pct (float): the fraction of annotations that will go into the train coco file
            val_pct (float): the fraction of annotations that will go into the validation coco file
            test_pct (float): the fraction of annotations that will go into the test coco file
            batch_size (int): number of images per batch process
        """        
        self.dataset = self.process_coco(coco_path, min_nb_occurrences)
        self.coco_path = coco_path
        self.export_dir = export_dir
        self.coco_train_name = coco_train_name
        self.coco_test_name = coco_test_name
        self.coco_val_name = coco_val_name
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.batch_size = batch_size

    def create_train_test_val_datasets(self):
        """Create dataset in the pylabel format to be populated by annotations aftersplitting.
        
        Returns:
            dataset_train: a training dataset in the pylabel format
            dataset_val: a validation dataset in the pylabel format
            dataset_test: a test dataset in the pylabel format
        """
        dataset_train = importer.ImportCoco(path=self.coco_path, name="trainset")
        dataset_val = importer.ImportCoco(path=self.coco_path, name="valset")
        dataset_test = importer.ImportCoco(path=self.coco_path, name="testset")
        return dataset_train, dataset_val, dataset_test
    
    def split_coco(self):
        """Splits coco files and export them in COCO format in the :py:attribute: `export_dir`.
        
        Returns:
            The proportions of class between the splits.
        """
        self.dataset.splitter.StratifiedGroupShuffleSplit(train_pct=self.train_pct, val_pct=self.val_pct, test_pct=self.test_pct, batch_size=self.batch_size)

        class_prop = self.dataset.analyze.ShowClassSplits()

        df_train = self.dataset.df.query("split == 'train'")
        df_val = self.dataset.df.query("split == 'val'")
        df_test = self.dataset.df.query("split == 'test'")
	
	
        dataset_train, dataset_val, dataset_test = self.create_train_test_val_datasets()
        
        # filter images that are moved in the other datasets
        dataset_train.df = dataset_train.df[dataset_train.df.img_filename.isin(df_train["img_filename"])].reset_index()
        # filter categories that have been deleted
        dataset_train.df = dataset_train.df[dataset_train.df.cat_name.isin(df_train["cat_name"])].reset_index()

        dataset_val.df = dataset_val.df[dataset_val.df.img_filename.isin(df_val["img_filename"])].reset_index()
        dataset_val.df = dataset_val.df[dataset_val.df.cat_name.isin(df_val["cat_name"])].reset_index()

        dataset_test.df = dataset_test.df[dataset_test.df.img_filename.isin(df_test["img_filename"])].reset_index()
        dataset_test.df = dataset_test.df[dataset_test.df.cat_name.isin(df_test["cat_name"])].reset_index()
        
        dataset_train.export.ExportToCoco(output_path=os.path.join(self.export_dir, self.coco_train_name))
        dataset_val.export.ExportToCoco(output_path=os.path.join(self.export_dir, self.coco_val_name))
        dataset_test.export.ExportToCoco(output_path=os.path.join(self.export_dir, self.coco_test_name))

        return class_prop

        
        

        
