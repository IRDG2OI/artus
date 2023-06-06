"""A module to handle a config file in yaml format and build a CfgNode.

An example of the config file expected can be found in artus's tutorials at
https://github.com/6tronl/artus-examples/blob/main/configs/x101_allsites_species_overlapping25_tiles5000_ITER3000.yml
"""

from detectron2 import model_zoo
import yaml


def read_config(config_path):
    """ Read a yaml file used to config a model for training or inference.

    Args:
        config_path (str) : the path to a config file in yaml format
    
    Returns:
        A python object 
    """
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
	

def add_config(cfg, config_path, device, train_dataset, test_dataset, output_dir=None, mode=['train', 'inference']):
    """Add config for DL model coming from a yaml config file.

    Args:
        cfg (CfgNode): a config file in detectron2 format
        config_path (str): the path to a config file following the format provided in the tutorials.
        device (str): 'cpu' or 'cuda'. Results of `"cuda" if torch.cuda.is_available() else "cpu"`
        train_dataset (str): name of the registered dataset for training
        test_dataset (str): name of the registered dataset for validation
        output_dir (str, optional): the path to the output directory. Defaults to None.
        mode (list, optional): Whether this config will be added in a training or inference use case. 
            Defaults to ['train', 'inference'].

    Returns:
        CfgNode: a custom config file for training a deep learning model
    """
    config = read_config(config_path)
        
    cfg.merge_from_file(model_zoo.get_config_file(config['MODEL']['URL']))

    if mode == 'train':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['MODEL']['URL'])
    elif mode == 'inference':
        cfg.MODEL.WEIGHTS = config['LOGS']['CHECKPOINT']

    cfg.DATASETS.TRAIN = train_dataset
    cfg.DATASETS.TEST = test_dataset

    cfg.TEST.EVAL_PERIOD = config['TEST']['EVAL_PERIOD']
    cfg.DATALOADER.NUM_WORKERS = config['DATALOADER']['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = config['SOLVER']['BASE_LR']
    cfg.SOLVER.MAX_ITER = config['SOLVER']['MAX_ITER']
    cfg.SOLVER.STEPS = config['SOLVER']['STEPS']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['MODEL']['ROI_HEADS']['NUM_CLASSES']
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
    cfg.MODEL_DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    cfg.INPUT.MIN_SIZE_TRAIN = (config['INPUT']['MIN_SIZE_TRAIN'],)
    cfg.INPUT.MAX_SIZE_TRAIN = config['INPUT']['MAX_SIZE_TRAIN']
    cfg.INPUT.MIN_SIZE_TEST = config['INPUT']['MIN_SIZE_TEST']
    cfg.INPUT.MAX_SIZE_TEST = config['INPUT']['MAX_SIZE_TEST']
    return cfg
    
 
