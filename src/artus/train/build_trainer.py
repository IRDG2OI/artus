
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.config import CfgNode
from detectron2.evaluation import COCOEvaluator
import os
from artus.train.validation_hook import LossEvalHook

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    # To use the mapper inside the dataloader, you need to overwrite the build_train_loader method of the trainer
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg:CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


