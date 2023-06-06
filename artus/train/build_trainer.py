"""Build a  to train deep learning model."""

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.config import CfgNode
from detectron2.evaluation import COCOEvaluator
import os
from artus.train.validation_hook import LossEvalHook

class MyTrainer(DefaultTrainer):
    """A custom trainer that performs validation loop."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """_summary_

        Args:
            cfg (CfgNode): config file in detectron2 format
            dataset_name (str): name of the dataset
            output_folder (str, optional): Path to the directory where results of training
                and evaluation will be saved. Defaults to None.

        Returns:
            :class:`detectron2.evaluation.COCOEvaluator`: an evaluator for coco dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    # To use the mapper inside the dataloader, you need to overwrite the build_train_loader method of the trainer
    def build_train_loader(cls, cfg):
        """Build a train dataloader.

        Args:
            cfg (CfgNode): config file in detectron2 format

        Returns:
            :class:`torch.utils.data.DataLoader` : A train dataloader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg:CfgNode, dataset_name):
        """Build a test dataloader.

        Args:
            cfg (CfgNode): config file in detectron2 format
            dataset_name (str): the name of the dataset

        Returns:
            :class:`torch.utils.data.DataLoader` : A test dataloader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """Build a hook to make validation on the validation dataset.

        Returns:
            hooks: validation metrics
        """
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


