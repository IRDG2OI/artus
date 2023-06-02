from detectron2 import model_zoo
from detectron2.config import get_cfg

from artus.train.config import read_config

def check_logs(config_path):
    '''
    Check if the config file already contains path to a log directory (which include a checkpoint to a model)
    # Input :
    - config_path : the path to a config file in yaml format
    # Output : 
    - Returns true if the config file includes path to a log directory, false otherwise.
    '''
    config = read_config(config_path)
    if config['LOGS']['CHECKPOINT']:
        check_logs = True
    else:
        check_logs = False
    return check_logs

def add_config(config_path, device):
    """
    Add config for DL model coming from a yaml config file.
    # Input :
    - config_path : the path to a config file in yaml format
    - device : 'cpu' or 'cuda'. Results of ("cuda" if torch.cuda.is_available() else "cpu")
    # Output : 
    - cfg to train a detectron model
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

