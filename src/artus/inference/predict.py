"""Module to use a model checkpoint and predict annotations on unlabeled images.

Annotations produced by the model (called predictions) are segmentation masks for 
instance segmentation tasks or bounding boxes for object detection tasks.
"""

from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
from torchvision.transforms import functional as func
import numpy as np
import cv2
from PIL import Image
import fiftyone as fo
import fiftyone.utils.labels as foul
from fiftyone import ViewField as F
import os
import cv2
from torchvision.ops import nms
import fiftyone as fo
from artus.inference.config import add_config
from joblib import Parallel, delayed
import multiprocessing
multiprocessing.set_start_method('spawn')
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"]=pow(2,50).__str__()

def build_predictor(config_path, device):
    """ Load a model in indeference mode.
    
    Args:
        config_path (str): the path to a config file in yaml format
        device (str): 'cpu' or 'cuda'. Results of `"cuda" if torch.cuda.is_available() else "cpu"`
    
    Returns 
        predictor : the model loaded to predict unlabeled images
    """
    cfg = add_config(config_path, device)
    predictor = DefaultPredictor(cfg)
    return predictor

def predict(predictor, filepath):
    """Predict masks or bounding boxes on an image.

    Args:
        predictor : a detectron2 predictor returned by `build_predictor()`
        filepath (str): the absolute or relative path to the image
    
    Returns:
        an ordered dict with the label predicted and the coordinates of the bbox or masks
    """
    image = cv2.imread(filepath)
    outputs = predictor(image)
    return outputs

def crop_mask(bbox, mask):
    """ Crop the mask within a bounding box.

    Args:
        bbox (list) : a list of the bounding box's corners for the predicted object.
        mask : the mask of the object predicted returned by detectron2. It is actually a mask for the whole image
        and 51 requires a mask within the bbox)

    Returns: 
        A cropped mask corresponding to the inside of the bbox for the predicted object."""

    x1, y1, x2, y2 = bbox.cpu().detach().numpy().astype(np.int32)
    return mask[y1:y2, x1:x2]

def filter_confidence_labels(dataset, fo_field, confidence_thr=0):
    """Filter labels that are smaller than the confidence threshold set.

    Args:
        dataset : a :class:`fiftyone.core.dataset`
        fo_field (str): a fiftyone field with model's predictions
        confidence_thr (float) : a threshold for confidence of model's predictions 

    Returns:
        dataset : the :class:`fiftyone.core.dataset` with labels smaller than the confidence threshold removed
    """ 
    conf_filter = fo.FilterLabels(fo_field, F("confidence") > confidence_thr)
    dataset = dataset.add_stage(conf_filter)
    return dataset

def filter_small_predictions(dataset, fo_field):
    """Filter predictions covering less than 3 pixels in a dataset.

    Predictions that are very small, i.e. smaller than 3 pixels, may trigger 
    problems when exporting the fiftyone dataset. It is best practises to remove them.

    Args:
        dataset : a :class:`fiftyone.core.dataset`
        fo_field (str): a fiftyone field with model's predictions

    Returns: 
        dataset : the :class:`fiftyone.core.dataset` with small predictions area removed
    """ 
    stage = fo.FilterLabels(fo_field, F("points").length() > 0)
    dataset = dataset.add_stage(stage)
    return dataset

def predict_on_sample(sample_filepath, device, predictor, nms_threshold, classes, type_of_preds=['segm', 'bbox']):
    """Use a loaded AI model to predict segmentations mask or bounding boxes on an image.

    Args:
        sample_filepath (str): the filepath to an image (jpg, png or tif file)
        predictor : the result of build_predictor() 
        device (str): "cpu" or "cuda". Results of `"cuda" if torch.cuda.is_available() else "cpu"`
        nms_threshold (float): the non-maximum-suppression threshold.
        classes (list): a list of the classes that can be predicted by the AI model.
        type_of_preds (str): indicate if semgentation masks "segm" expected or bounding boxes "bbox"

    Returns:
        a fiftyone field :class:``fiftyone.core.detections`` containing predictions for the sample
    """
    image = Image.open(sample_filepath)
    image = func.to_tensor(image).to(device)
    c, h, w = image.shape

    preds = predict(predictor, sample_filepath)

    preds = apply_nms(preds, nms_threshold, type_of_preds)
    
    labels = preds['instances'].pred_classes.cpu().detach().numpy()
    scores = preds['instances'].scores.cpu().detach().numpy()
    boxes = preds['instances'].pred_boxes
    if type_of_preds=='segm':
        masks = preds['instances'].pred_masks.cpu().detach().numpy()

    detections = []
    if type_of_preds=='segm':

        for label, score, box, mask in zip(labels, scores, boxes, masks):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                mask = crop_mask(box, mask)

                detections.append(
                    fo.Detection(
                        label=classes[label],
                        bounding_box=rel_box,
                        confidence=score, 
                        mask=mask
                    )
                )
    else:
        for label, score, box in zip(labels, scores, boxes):
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                    
            detections.append(
                fo.Detection(
                label=classes[label],
                bounding_box=rel_box,
                confidence=score)
                )   
            
    return fo.Detections(detections=detections)



def add_predictions_to_dataset(dataset, predictor, device, classes, predictions_field, tags=None, type_of_preds=['segm', 'bbox'], nms_threshold=1, confidence_thr=0):
    """Add predictions to :class:`fiftyone.core.dataset`. 

    All the images :class:`fiftyone.core.dataset` that have the tag indicated will be selected to be predicted by an AI model.
    This function uses the GPU is available or parallelize through multiple CPU to predict on all the images of a dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset`  
        predictor: the result of :py:func:`artus.inference.build_predictor()` 
        device (str): "cpu" or "cuda". Results of `"cuda" if torch.cuda.is_available() else "cpu"`
        classes (list): a list of the classes predicted by the model
        predictions_field (str): the name of the fiftyone field that will be added to be dataset
        tags (list or str): a variable or a list of variables (optional) to filter the samples on which to predict
        type_of_preds (str): indicate if semgentation masks "segm" expected or bounding boxes "bbox"
        confidence_thr (float): a threshold between 0 and 1. Predictions with lower confidence than this threshold will be filtered out 
            of the :class:`fiftyone.core.dataset`

    Returns:
        a :class:`fiftyone.core.dataset` containing predictions for the tagged samples.
    """
    num_cores = 0
    detections=[]
    if tags:
        test_set = dataset.match_tags(tags)
    else:
        test_set=dataset

    with fo.ProgressBar() as pb: 
        if device == "cpu":      
            for sample in pb(test_set.values('filepath')):
                detections.append(predict_on_sample(sample, device, predictor, nms_threshold, classes, type_of_preds))

        else:
            for sample in pb(test_set.values('filepath')):
                detections.append(predict_on_sample(sample, device, predictor, nms_threshold, classes, type_of_preds))

    if type_of_preds=='segm':
        test_set.set_values("predictions_with_bbox", detections)
        
        #Transform bbox and mask in polylines
        foul.instances_to_polylines(
            test_set,
            in_field='predictions_with_bbox',
            out_field=predictions_field
            )
        
        test_set.delete_sample_field('predictions_with_bbox')

    else: 
        test_set.set_values(predictions_field, detections)

    test_set = filter_small_predictions(test_set, predictions_field)
    
    if confidence_thr>0:
        test_set = filter_confidence_labels(test_set, predictions_field, confidence_thr)

    test_set.save()
    print(test_set)

    return test_set

def apply_nms(outputs, nms_threshold, type_of_preds=['segm', 'bbox']) :
    """Apply non-maximum threshold on predictions made on a sample.

    Non Maximum Suppression NMS is a technique to select one entity, i.e. bounding boxes or masks,
    out of many overlapping entities based on the IoU (intersection over union).The function 
    applies the NMS technique for predictions of different class types, the default Detectron2 
    parameter MODEL.ROI_HEADS.NMS_THRESH_TEST applies NMS only for same class types instances
    `https://github.com/facebookresearch/detectron2/issues/978`

    Args:
        outputs (list[dict]): a list of dict in the "outputs" inference format of detectron2 lib 
        nms_threshold (float): Overlap threshold used for non-maximum suppression, suppress boxes or
            masks with IoU >= threshold
        type_of_preds (str): indicate if predictions are semgentation masks "segm" or bounding boxes "bbox"

    Returns:
        res (list[dict]): the "outputs" inference format of detectron2 lib 
    """

    detections = outputs['instances']
    pred_boxes = detections.pred_boxes.tensor
    scores = detections.scores
    pred_classes = detections.pred_classes
    if type_of_preds == 'segm':
        pred_masks = detections.pred_masks
    image_size = detections.image_size
    # Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).
    # see :
    # https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html#torchvision.ops.nms
    keep_idx = nms(pred_boxes, scores, nms_threshold)
    res = Instances(image_size)
    res.pred_boxes = Boxes(pred_boxes[keep_idx])
    res.scores = scores[keep_idx]
    res.pred_classes = pred_classes[keep_idx]
    if type_of_preds == 'segm':
        res.pred_masks = pred_masks[keep_idx]
    res = {"instances": res}

    
    
