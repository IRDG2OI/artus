from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
import os
import torch
import yaml
from write_eval_results import ModelsMetricsFormat
from build_trainer import MyTrainer
from config import add_config

def evaluate_model(config_path, session_name, csv_metrics_name):
    ''' A function that reads a config file and execute a COCO evaluation on a trained model 
    (https://detectron2.readthedocs.io/en/latest/modules/evaluation.html?highlight=COCOEvaluator#detectron2.evaluation.COCOEvaluator).
    # Inputs:
    - config_path : the path to a config file (yaml)
    - session_name : a training session name, e.g. the name of a folder containing a model_final.pth checkpoint file to evaluate
    - csv_metrics_name : a CSV name to write or append the results of the evaluation.
    # Outputs: 
    - a csv file located into the log dir (mentionned in the config file) with the models metrics. If a models metrics wasq already present in the 
    logs directories then the metrics will be appended to it.
    '''
    
    setup_logger()

    #read config file for training
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    #Register images and test/train labels
    register_coco_instances("trainset", {}, config['DATA_PATH']['COCO_TRAIN'], config['DATA_PATH']['IMAGES_DIR'])
    register_coco_instances("testset", {}, config['DATA_PATH']['COCO_TEST'], config['DATA_PATH']['IMAGES_DIR'])

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    #Set the config parameters for the model
    cfg = get_cfg()
    cfg = add_config(
        cfg=cfg,
        config_path=config_path,
        output_dir=os.path.join(config['LOGS']['LOGS_DIR'], session_name),
        device=device,
        train_dataset = ("trainset",),
        test_dataset = ("testset",),
        mode='inference'
        )


    #Load model and weights
    trainer = MyTrainer(cfg)
    trainer.resume_or_load()

    #Evaluate final results
    evaluator = COCOEvaluator('testset', cfg, False, os.path.join(config['LOGS']['LOGS_DIR'], session_name))
    accuracy = trainer.test(cfg, trainer.model, evaluator)

    model_metrics = ModelsMetricsFormat(
        session=session_name,
        eval_results=accuracy,
        export_dir=config['LOGS']['LOGS_DIR'],
        csv_name=csv_metrics_name
    )

    model_metrics.write_to_csv()