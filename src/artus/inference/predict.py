from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
from torchvision.transforms import functional as func
import numpy as np
import cv2
from PIL import Image
import fiftyone as fo
import fiftyone.utils.labels as foul
import yaml
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"]=pow(2,50).__str__()
import cv2
from torchvision.ops import nms
import fiftyone as fo

from joblib import Parallel, delayed
import multiprocessing
multiprocessing.set_start_method('spawn')


def build_predictor(config_name, device):
    """
    Add config for DL model coming from a yaml config file.
    """
    
    with open(os.path.join(models_inference_config_dir, config_name)) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['MODEL']['URL']))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['MODEL']['ROI_HEADS']['NUM_CLASSES']
    cfg.MODEL.WEIGHTS = config['MODEL']['WEIGHTS']
    cfg.DATALOADER.NUM_WORKERS = config['DATALOADER']['NUM_WORKERS']
    cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TEST = config['INPUT']['MIN_SIZE_TEST']
    cfg.INPUT.MAX_SIZE_TEST = config['INPUT']['MAX_SIZE_TEST']
    predictor = DefaultPredictor(cfg)

    return predictor

def predict(predictor, filepath):
    image = cv2.imread(filepath)
    outputs = predictor(image)
    return outputs


def crop_mask(bbox, mask):
    ''' Input : 
        bbox : the bounding box of the object predicted.
        mask : the mask of the object predicted returned bu detectron2 (it is actually a mask for the whole image
        and 51 requires a mask within the bbox)

        Output: 
        a cropped mask corresponding to the inside of the bbox for the predicted object.'''

    x1, y1, x2, y2 = bbox.cpu().detach().numpy().astype(np.int32)
    return mask[y1:y2, x1:x2]


def predict_on_sample(sample_filepath, device, predictor, nms_threshold, classes, type_of_preds=['segm', 'bbox']):
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


def add_predictions_to_dataset(dataset, predictor, device, classes, predictions_field, tags=None, type_of_preds=['segm', 'bbox'], nms_threshold=1):
    '''Add predictions to a fiftyone dataset

    Inputs:
    - dataset : a fiftyone dataset
    - predictor : the result of build_predictor() 
    - device : whether to load data on cpu or gpu
    - classes : a list of the classes predicted by the model
    - tags : a varaible or a list of varaibles (optional) to filter the samples on which to predict
    - type_of_preds : indicate if semgentation masks expeted of bounding boxes

    Outputs :
    - a dataset containing predictions for test samples
    '''
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

    # Save predictions to dataset
    if type_of_preds=='segm':
        test_set.set_values("predictions_with_bbox", detections)
        #Transform bbox and mask in polylines

        foul.instances_to_polylines(
            test_set,
            in_field='predictions_with_bbox',
            out_field=predictions_field
            )
        
        dataset.delete_sample_field('predictions_with_bbox')

    else: 
        test_set.set_values(predictions_field, detections)
        test_set.save()

    dataset.save()
    print(dataset)

    return dataset

def apply_nms(outputs, nms_treshold, type_of_preds=['segm', 'bbox']) :
    ''' 
    Inputs :
    1.outputs = list[dict] in the "outputs" inference format of detectron2 lib, 
    see : https://detectron2.readthedocs.io/en/latest/tutorials/models.html
    2.nms_treshold = Overlap threshold used for non-maximum suppression (suppress boxes with
                      IoU >= this threshold)
    
    Outputs :
    The function applies the NMS technique for predictions of different class types,
    the default Detectron2 parameter MODEL.ROI_HEADS.NMS_THRESH_TEST applies NMS only
    for same class types instances :
    see :
    https://github.com/facebookresearch/detectron2/issues/978
    1.res = list[dict] in the "outputs" inference format of detectron2 lib 
    '''

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
    keep_idx = nms(pred_boxes, scores, nms_treshold)
    res = Instances(image_size)
    res.pred_boxes = Boxes(pred_boxes[keep_idx])
    res.scores = scores[keep_idx]
    res.pred_classes = pred_classes[keep_idx]
    if type_of_preds == 'segm':
        res.pred_masks = pred_masks[keep_idx]
    res = {"instances": res}
    

