from pylabel import importer

def rm_min_classes(dataset, min_nb_occurrences):
    '''Remove classes that does not reach the minimum number of occurences/class
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
    ''' A class to explore the basic stats of a coco file
    # Inputs:
    - coco_path : a path to a coco file
    - min_nb_occurrences : an integer that is the minimum number of occurrences of a class to be kept in the dataset (remove under representated classes)
    # Output:
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

