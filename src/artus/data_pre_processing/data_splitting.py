from pylabel import importer, dataset
import numpy as np
import pandas as pd
import os

def rm_min_classes(dataset, min_nb_occurrences):
    '''remove classes that does not reach the minimum number of occurences/class
    #Inputs :
    - dataset : a coco file read with importer.ImportCoco() from pylabel
    - min_nb_occurrences : an integer that is the minimum number of occurrences of a class to be kept in the dataset (remove under representated classes)
    # Output : 
    - the same dataset with classes that have less than the minimum number of occurrences removed.
    '''
    grouped_by_class = dataset.df.groupby(by='cat_name', axis=0).count()
    under_represented_classes = grouped_by_class[grouped_by_class['img_folder'] < min_nb_occurrences].index
    deleted_classes=list(under_represented_classes.values)
    index_to_remove = dataset.df[dataset.df.cat_name.isin(deleted_classes)].index
    dataset.df.drop(index=index_to_remove, inplace=True)
    return dataset

def rm_tiles_without_annot(dataset):
    '''remove tiles without annotations
    #Input:
    - dataset : a coco file read with importer.ImportCoco() from pylabel
    # Output:
    - the same dataset with rows that do not contains any info in the 'cat_name' column removed.
    '''
    ind_images_without_annot = dataset.df.loc[dataset.df['cat_name']==''].index
    dataset.df.drop(index=ind_images_without_annot, inplace=True)
    return dataset


class COCOStats():
    ''' A class to explore the basic stats of a coco file'''
    def __init__(self, coco_path, min_nb_occurrences=None):
        self.dataset = self.process_coco(coco_path, min_nb_occurrences)

    def process_coco(self, coco_path, min_nb_occurrences):
        ''' remove the samples without annotations and removed underrepresetned classes if needed'''        
        dataset = importer.ImportCoco(path=coco_path, name="dataset")  
        dataset = rm_tiles_without_annot(dataset)
        if min_nb_occurrences:
            dataset = rm_min_classes(dataset, min_nb_occurrences)
        return dataset
    
    def get_class_stats(self):
        '''print the number of occurrences per classes'''
        print(f"Classes:{self.dataset.analyze.classes}")
        print(f"Number of classes: {self.dataset.analyze.num_classes}")
        print(f"Class counts:\n{self.dataset.analyze.class_counts}")
        
    def get_nb_images(self):
        print(f"Number of images: {self.dataset.analyze.num_images}")
    
    def export_stats(self, export_path):
        ''' Export the stats in csv format at the export_path'''
        self.dataset.analyze.class_counts.to_csv(export_path)


class COCOSplitter(COCOStats):
    ''' Split a coco file into a train, test and (optional) validation coco files. Proportions of class annotations are kept into the splits.
    # Inputs:
    - coco_path : a path to a coco file
    - export_dir : a directory where the splits of the coco files will be exported
    - coco_train_name : the name of the file with the annotations for training
    - coco_test_name : the name of the file with the annotations for testing
    - coco_val_name : the name of the file with the annotations for validation
    - min_nb_occurrences : an integer that is the minimum number of occurrences of a class to be kept in the dataset (remove under representated classes)
    - train_pct : the fraction (float) of annotations that will go into the train coco file
    - val_pct : the fraction (float) of annotations that will go into the validation coco file
    - test_pct : the fraction (float) of annotations that will go into the test coco file
    - batch_size
    # Outputs:
    Splits of the coco files are exported in COCO format in the export_dir mentionned.
    '''
    def __init__(self, coco_path, export_dir, coco_train_name, coco_test_name, coco_val_name, min_nb_occurrences=None, train_pct=.8, val_pct=.1, test_pct=.1, batch_size=8):
        self.dataset = self.process_coco(coco_path, min_nb_occurrences)
        self.export_dir = export_dir
        self.coco_train_name = coco_train_name
        self.coco_test_name = coco_test_name
        self.coco_val_name = coco_val_name
        
        
    def create_train_test_val_datasets(self):
        dataset_train = importer.ImportCoco(path=coco_path, name="trainset")
        dataset_val = importer.ImportCoco(path=coco_path, name="valset")
        dataset_test = importer.ImportCoco(path=coco_path, name="testset")
        return dataset_train, dataset_val, dataset_test
    
    def split_coco(self):
        self.dataset.splitter.StratifiedGroupShuffleSplit(train_pct=.8, val_pct=.1, test_pct=.1, batch_size=2)
        
        self.dataset.analyze.ShowClassSplits()

        df_train = dataset.Dataset(self.dataset.df.query("split == 'train'"))
        df_val = dataset.Dataset(self.dataset.df.query("split == 'val'"))
        df_test = dataset.Dataset(self.dataset.df.query("split == 'test'"))
        
        df_train.export.ExportToCoco(output_path=os.path.join(self.export_dir, self.coco_train_name, '.json'))
        df_val.export.ExportToCoco(output_path=os.path.join(self.export_dir, self.coco_val_name, '.json'))
        df_test.export.ExportToCoco(output_path=os.path.join(self.export_dir, self.coco_test_name, '.json'))
        

