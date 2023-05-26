from pylabel import importer, dataset
import numpy as np
import pandas as pd
import os
from artus.prepare.coco_splitting import rm_min_classes, rm_tiles_without_annot

class COCOStats():
    ''' A class to explore the basic stats of a coco file
    #Inputs:
    - coco_path : a path to a coco file
    - min_nb_occurrences : an integer that is the minimum number of occurrences of a class to be kept in the dataset (remove under representated classes)
    #Output:
    - statistics about the coco file (nb occurences per class, nb of images)
    - csv export of the statistics (export_stats() function)
    '''
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

