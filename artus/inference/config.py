"""
Read a config file and return a cfg to use a model with detectron2 library for the inference process.
"""

from detectron2 import model_zoo
from detectron2.config import get_cfg

from artus.train.config import read_config

def check_logs(config_path):
    """Check if the config file already contains path to a log directory 
    (which include a checkpoint to a model)

    Args:
        config_path (str): the path to a config file in yaml format

    Returns: 
        check_logs (bool): true if the config file includes path to a log directory, 
            false otherwise.
    """
    config = read_config(config_path)
    if config['LOGS']['CHECKPOINT']:
        check_logs = True
    else:
        check_logs = False
    return check_logs

def add_config(config_path, device):
    """ Add config for DL model coming from a yaml config file.

    The config is read and a cfg suitable for detectron2 package is returned.
    This cfg uses a model already fine tuned with detectron2. The path to this
    model'sa checkpoint should be mentionned in the LOGS section of the config file.

    Args:
        config_path (str): the path to a config file in yaml format
        device (str): 'cpu' or 'cuda'. Results of (Python("cuda" if torch.cuda.is_available() else "cpu"))
    
    Returns:
        a cfg to use a detectron model's checkpoint on unlabeled images (inference)
    """ 
    config = read_config(config_path)

    if check_logs(config_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config['MODEL']['URL']))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['MODEL']['ROI_HEADS']['NUM_CLASSES']
        cfg.MODEL.WEIGHTS = config['LOGS']['CHECKPOINT']
        cfg.DATALOADER.NUM_WORKERS = config['DATALOADER']['NUM_WORKERS']
        cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MIN_SIZE_TEST = config['INPUT']['MIN_SIZE_TEST']
        cfg.INPUT.MAX_SIZE_TEST = config['INPUT']['MAX_SIZE_TEST']
    return cfg

