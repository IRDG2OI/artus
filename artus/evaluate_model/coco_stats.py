"""The module provides basic descriptive statisitics on a coco file (coco.json).

The COCOStats class can be used to get basic statistics on a coco file after 
deleting images without annotations in a dataset and removing under-representated
classes with the min_nb_occurrences argument.

Typical usage examples:

  `cocostats = COCOStats(coco_path='/a/path/to/coco.json', min_nb_occurrences=50)`
  `nb_images = cocostats.get_nb_images()`
  `class_occ = cocostats.get_class_stats()`
  `cocostats.export_stats(export_path='/path/to/dir/export.csv)`
"""

from pylabel import importer

def rm_min_classes(dataset, min_nb_occurrences):
    """Remove classes that do not reach the minimum number of occurences per class.

    Allows to get rid of under-representated in a dataset before splitting a coco file
    for deep learning purposes.

    Args:
        dataset: a coco file read with :func:`importer.ImportCoco()` from pylabel. This function
        returns a dataframe where one row is one label
        min_nb_occurrences (int): an integer that is the minimum number of occurrences 
        of a class to be kept in the dataset

    Returns:
        The same dataframe format but classes that have less than the minimum number of 
        occurrences are removed.
    """
    grouped_by_class = dataset.df.groupby(by='cat_name', axis=0).count()
    under_represented_classes = grouped_by_class[grouped_by_class['img_folder'] < min_nb_occurrences].index
    deleted_classes=list(under_represented_classes.values)
    index_to_remove = dataset.df[dataset.df.cat_name.isin(deleted_classes)].index
    dataset.df.drop(index=index_to_remove, inplace=True)
    return dataset

def rm_tiles_without_annot(dataset):
    """Remove tiles without annotations.

    Args:
        dataset: a coco file read with importer.ImportCoco() from pylabel. This function
        returns a dataframe where one row is one label

    Returns:
        The same dataframe format but rows with empty column 'cat_name' are removed.
    """
    ind_images_without_annot = dataset.df.loc[dataset.df['cat_name']==''].index
    dataset.df.drop(index=ind_images_without_annot, inplace=True)
    return dataset


class COCOStats():
    """A class to explore the basic stats of a coco file.

    The COCOStats class can be used to get basic statistics on a coco file after 
    deleting images without annotations in a dataset and removing under-representated
    classes with the min_nb_occurrences argument. Statistics can be printed or exported 
    in a csv file.

    Attributes:
        coco_path (str): a path to a coco file
        min_nb_occurrences (int): an integer that is the minimum number of occurrences of a 
        class to be kept in the dataframe.
    """
    def __init__(self, coco_path, min_nb_occurrences=None):
        """Initializes the instance by reading the coco_path file and 
            removing under-representated class if min_nb_occurrences!=None.

        Args:
            coco_path (str): a path to a coco file
            min_nb_occurrences (int): an integer that is the minimum number of occurrences of a 
            class to be kept in the dataframe, remove under representated classes
        """
        self.dataset = self.process_coco(coco_path, min_nb_occurrences)

    def process_coco(self, coco_path, min_nb_occurrences):
        """Read coco file and removes the samples without annotations and underrepresetned classes if needed.
        
        Args:
            coco_path (str): a path to a coco file
            min_nb_occurrences (int): an integer that is the minimum number of occurrences of a 
            class to be kept in the dataframe, remove under representated classes.
        
        Returns:
            A dataframe where one row is one label.
        """        
        dataset = importer.ImportCoco(path=coco_path, name="dataset")  
        dataset = rm_tiles_without_annot(dataset)
        if min_nb_occurrences:
            dataset = rm_min_classes(dataset, min_nb_occurrences)
        return dataset
    
    def get_class_stats(self):
        """Print the number of occurrences per classes and number of classes in dataframe
            after removing images without annotations and under-representated classes."""
        print(f"Classes:{self.dataset.analyze.classes}")
        print(f"Number of classes: {self.dataset.analyze.num_classes}")
        print(f"Class counts:\n{self.dataset.analyze.class_counts}")
        
    def get_nb_images(self):
        """Print the number of images in dataframe after removing images without 
            annotations and under-representated classes."""
        print(f"Number of images: {self.dataset.analyze.num_images}")
    
    def export_stats(self, export_path):
        """ Export the number of occurrences per classin csv format.
         
        Statistics are exported after removing images without annotations and (optional) under-representated classes.
        
        Args:
            export_path (str): csv path to export the class counts.
        
        """
        self.dataset.analyze.class_counts.to_csv(export_path)

