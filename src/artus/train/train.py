"""A module to train a deep learning model with a config file in the yaml format."""

from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg

import os
import torch
import yaml
from datetime import datetime

from artus.train.build_trainer import MyTrainer
from artus.train.config import add_config, read_config


def add_logs_path_to_config(config_path, checkpoint_path_dict):
    """Add logs paths to the config file to keep track of trained models.
    Args:
        config_path (str): the path to a config file (yaml)
        checkpoint_path_dict (dict): a dict including the 'CHECKPOINT' (the path to the checkpoint saved)

    Returns: 
        The same config file with the path to the final checkpoint added.
    """
    with open(config_path,'r') as config:
        config = yaml.safe_load(config) 
        config['LOGS'].update(checkpoint_path_dict)

    if config:
        with open(config_path,'w') as yamlfile:
            yaml.safe_dump(config, yamlfile) 


def train_model(config_path):
    """A function that trains a deep learning model with detectron2.

    Support multiple deep learning tasks : object detection and instance segmentation.

    Args:
        config_path : the path to a config file (yaml)

    Returns: 
    A folder within the logs directory with the checkpoint (.pth) of the model and events file 
    for visualization of the training and validation metrics with tensorboard.
    A subdirectory is also created with the inference resultst of the validation step.
    """
    setup_logger()

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    #read config file for training
    config = read_config(config_path)

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

    #Append logs information in the config file to keep track of trained model path
    checkpoint_path = os.path.join(config['LOGS']['LOGS_DIR'], session_name, 'model_final.pth')
    dict = {'CHECKPOINT' : checkpoint_path}
    add_logs_path_to_config(config_path, dict)




