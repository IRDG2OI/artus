from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg

import os
import torch
import yaml
from datetime import datetime

from build_trainer import MyTrainer
from config import add_config


def train_model(config_path):
    ''' A function that trains a model with detectron2.
    # Inputs:
    - config_path : the path to a config file (yaml)
    # Outputs: 
    - a folder within the logs directory with the checkpoint (.pth) of the model and events file for visualization of the training and validation metrics with tensorboard.
    A subdirectory is also created with the inference resultst of the validation step.
    '''
    setup_logger()

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    #read config file for training
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #Set automatically the name of the current train session
    today = datetime.today().strftime('%Y%m%d_%H:%M:%S')
    session_name = today + os.path.basename(config_path)[:-4]

    #Register images and test/train labels
    register_coco_instances("trainset", {}, config['DATA_PATH']['COCO_TRAIN'], config['DATA_PATH']['IMAGES_DIR'])
    register_coco_instances("valset", {}, config['DATA_PATH']['COCO_VAL'], config['DATA_PATH']['IMAGES_DIR'])

    #Set the config parameters for the model
    cfg = get_cfg()
    cfg = add_config(
        cfg=cfg,
        config_path=config_path,
        output_dir=os.path.join(config['LOGS']['LOGS_DIR'], session_name),
        device=device,
        train_dataset = ("trainset",),
        test_dataset = ("valset",),
        mode='train'
        )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #Launch training session
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

