# import argparse
# parser = argparse.ArgumentParser(description='')
# args = parser.parse_args()


import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator

import pandas as pd
import shutil
setup_logger()

import PIL

from ensemble_boxes import nms
import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.modeling.postprocessing import detector_postprocess

class myPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, detected_instances=None, do_postprocess=None):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            if(detected_instances is not None):
                predictions = self.model.inference([inputs], detected_instances, do_postprocess)[0]
            else:
                predictions = self.model([inputs])[0]
            return predictions

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=1) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=0) == 0  # Missed objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ, THRESHOLDS):
    MIN_PIXELS = 30

    take = pred['instances'].scores >= THRESHOLDS
    pred_masks = pred['instances'].pred_masks[take].cpu().numpy()

    pred_masks_filter=[mask for mask in pred_masks if mask.sum() >= MIN_PIXELS]
    pred_masks = np.array(pred_masks_filter)
    
    
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    if not isinstance(ious, np.ndarray):
        # print("no mask after filter")
        assert len(ious)==0
        return 0
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

class myDetect():
    def __init__(self, predictors, thrs, dataset_name):
        self.thrs = thrs
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in self.dataset_dicts}
        self.scores=[]
        self.base_model = predictors[0]
        self.shsy5y_model = predictors[1]
        self.astro_model = predictors[2]
        self.cort_model = predictors[3]

    def detect(self):
        for inp in self.dataset_dicts:
            img = cv2.imread(inp['file_name'])
            out = self.base_model(img)
            pred_class = torch.mode(out['instances'].pred_classes)[0]
            if(pred_class==0):
                predictor = self.shsy5y_model
                thr = self.thrs[1]
            elif(pred_class==1):
                predictor = self.astro_model
                thr = self.thrs[2]
            elif(pred_class==2):
                predictor = self.cort_model
                thr = self.thrs[3]
            else:raise "ERRR"
            out = predictor(img)

            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                sc = score(out, targ, thr)
                self.scores.append(sc)
        print(np.mean(self.scores))


if __name__ == "__main__":
    output_dir = 'tes1'
    if os.path.exists(f'{output_dir}/map.csv'): os.remove(f'{output_dir}/map.csv')
    if os.path.exists(f'{output_dir}/compare.csv'): os.remove(f'{output_dir}/compare.csv')
    dataDir=Path('sartorius-cell-instance-segmentation/')
    register_coco_instances('sartorius_val',{},f'val_1.json', dataDir)


    cfg_dicts=[{'ckpt': 'exp3_bh1_aug_ms3_R50_p2000/model_0014139.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml','thr':[.35,.45,.75], 'opt':[1000,2000,2000,3]},
               {'ckpt': '1_R101/model_0011879.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':0.25, 'opt':[1000,2000,2000,1]},
               {'ckpt': '2_R101/model_0002441.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':0.4, 'opt':[800,1500,1500,1]},
               {'ckpt': '3_R101/model_0005039.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':0.7, 'opt':[500,1000,1000,1]}]
    models, thrs = [],[]
    for _cfg in cfg_dicts:
        cfg = get_cfg()
        cfg.OUTPUT_DIR = output_dir
        cfg.INPUT.MASK_FORMAT='bitmask'
        cfg.merge_from_file(model_zoo.get_config_file(_cfg['config']))
        cfg.DATASETS.TEST = ("sartorius_val",)
        cfg.DATALOADER.NUM_WORKERS = 12
        
        cfg.INPUT.MIN_SIZE_TEST = 1184
        cfg.INPUT.MAX_SIZE_TEST = 2000

        cfg.TEST.DETECTIONS_PER_IMAGE = _cfg['opt'][0]
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = _cfg['opt'][1]
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = _cfg['opt'][2]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = _cfg['opt'][3]

        cfg.MODEL.WEIGHTS = _cfg['ckpt']
        model = myPredictor(cfg)
        models.append(model)
        thrs.append(_cfg['thr'])

    MD = myDetect(models, thrs, 'sartorius_val')
    MD.detect()