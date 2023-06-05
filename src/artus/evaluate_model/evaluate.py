"""Performs COCO-style evaluation on a trained deep learning model.

To evaluate the model, a config file (yaml format) must be provided to set the model parameters.
Model is tested on the test_dataset indicated. The test set must be new to the model evaluated (not used
during training process). Results are exported in csv format at the desired location.

Typical usage examples:

  evaluate_model(config_path='/path/to/config.yml', csv_metrics_name='models_metrics.csv')
"""

from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
import os
import torch
from artus.evaluate_model.write_eval_results import ModelsMetricsFormat
from artus.train.build_trainer import MyTrainer
from artus.train.config import read_config, add_config


def evaluate_model(config_path, csv_metrics_name):
    """ A function that reads a config file and execute a COCO evaluation on a trained model.

    Config file must be written in yaml format and must follow the example here : 
    https://github.com/6tronl/artus-examples/blob/main/configs/x101_allsites_species_overlapping25_tiles5000_ITER3000.yml
    The config file contains information needed to set the detectron2 evaluator based on 
    the model trained. The model will be evaluated using the COCOEvaluator 
    (https://detectron2.readthedocs.io/en/latest/modules/evaluation.html?highlight=COCOEvaluator#detectron2.evaluation.COCOEvaluator).
    A csv will be exported containing metrics about the model (AP50, AP, AP75, APl and AP per class). 
    If a csv already exsists in the log directory mentionned in the config file, 
    then results will be appended to it.

    Args:
         config_path (str): the path to a config file in yaml format
        csv_metrics_name (str): a CSV name to write or append the results of the evaluation.
    """
    
    setup_logger()

    #read config file for training
    config = read_config(config_path)
    
    #Register images and test/train labels
    register_coco_instances("trainset", {}, config['DATA_PATH']['COCO_TRAIN'], config['DATA_PATH']['IMAGES_DIR'])
    register_coco_instances("testset", {}, config['DATA_PATH']['COCO_TEST'], config['DATA_PATH']['IMAGES_DIR'])

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    #Set the config parameters for the model
    cfg = get_cfg()
    cfg = add_config(
        cfg=cfg,
        config_path=config_path,
        output_dir=config['LOGS']['CHECKPOINT'],
        device=device,
        train_dataset = ("trainset",),
        test_dataset = ("testset",),
        mode='inference'
        )

    #Load model and weights
    trainer = MyTrainer(cfg)
    trainer.resume_or_load()

    #Evaluate final results
    evaluator = COCOEvaluator('testset', cfg, False, os.path.join(config['LOGS']['LOGS_DIR'], os.path.basename(config['LOGS']['CHECKPOINT'])))
    accuracy = trainer.test(cfg, trainer.model, evaluator)

    model_metrics = ModelsMetricsFormat(
        session=os.path.basename(config['LOGS']['CHECKPOINT']),
        eval_results=accuracy,
        export_dir=config['LOGS']['LOGS_DIR'],
        csv_name=csv_metrics_name
    )

    model_metrics.write_to_csv()