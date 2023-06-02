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
    ''' A function that reads a config file and execute a COCO evaluation on a trained model 
    (https://detectron2.readthedocs.io/en/latest/modules/evaluation.html?highlight=COCOEvaluator#detectron2.evaluation.COCOEvaluator).
    # Inputs:
    - config_path : the path to a config file (yaml)
    - csv_metrics_name : a CSV name to write or append the results of the evaluation.
    # Outputs: 
    - a csv file located into the log dir (mentionned in the config file) with the models metrics. If a models metrics wasq already present in the 
    logs directories then the metrics will be appended to it.
    '''
    
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