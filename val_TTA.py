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

import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.modeling.postprocessing import detector_postprocess

class myPredictor:
    def __init__(self, cfg, weight):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weight)

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
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ, THRESHOLDS):
    MIN_PIXELS = [30, 30, 30]

    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take].cpu().numpy()

    pred_masks_filter=[mask for mask in pred_masks if mask.sum() >= MIN_PIXELS[pred_class]]
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

class myTTA():
    def __init__(self, predictor, dataset_name, step):
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in self.dataset_dicts}
        self.model = predictor
        self.step = step

        self.thr_iter=4
        self.scores = []
        init_thr = np.array([0.35,0.45,0.78])
        for i in range(self.thr_iter):
            for j in range(self.thr_iter):
                for k in range(self.thr_iter):
                    thr = np.array([0.05*i, 0.05*j, 0.03*k])
                    THRS = init_thr + thr
                    self.scores.append([THRS])

    def detect(self):
        init_thr = np.array([0.35,0.45,0.78])
        number=0
        
        for i in range(self.thr_iter):
            for j in range(self.thr_iter):
                for k in range(self.thr_iter):
                    thr = np.array([0.05*i, 0.05*j, 0.03*k])
                    THRS = init_thr + thr
                    for inp in tqdm.tqdm(self.dataset_dicts):
                        img = cv2.imread(inp['file_name'])
                        input = [{"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)), "height": img.shape[0], "width": img.shape[1]}]
                        with torch.no_grad():
                            out = self.model(input)[0]

                        if len(out['instances']) == 0:
                            self.scores[number].append(0)    
                        else:
                            targ = self.annotations_cache[inp['image_id']]
                            self.scores[number].append(score(out, targ, THRS))
                    number+=1
                    print(number)

        thrs, results = [],[]
        for l in self.scores:
            thrs.append(l[0])
            results.append(l[1:])
        results = np.array(results).mean(1)
        thrs = np.array(thrs)

        max_thr = thrs[np.argmax(results)]
        max_res = np.max(results)
        compare_pd = pd.DataFrame({'thr0':[max_thr[0]], 
                                   'thr1':[max_thr[1]], 
                                   'thr2':[max_thr[2]], 
                                   'result':[max_res],
                                   'step':[self.step]})
        if(os.path.exists(f'{output_dir}/compare2.csv')):
            compare_pre = pd.read_csv(f'{output_dir}/compare2.csv')
            compare_pre = compare_pre.append(pd.DataFrame({'thr0':[max_thr[0]], 
                                                            'thr1':[max_thr[1]], 
                                                            'thr2':[max_thr[2]], 
                                                            'result':[max_res],
                                                            'step':[self.step]}))
            compare_pre.to_csv(f'{output_dir}/compare2.csv', index=False, float_format='%.4f')
        else:
            compare_pd.to_csv(f'{output_dir}/compare2.csv', index=False, float_format='%.4f')


            

from detectron2.projects import _PROJECTS
_PROJECTS.update({"vovnet": "vovnet-detectron2-master"})
from detectron2.projects.vovnet import  add_vovnet_config

if __name__ == "__main__":
    # cfg_dict = {'exp': '1_V99_p2000_flipud_all',
    #             'config':'/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml',
    #             'opt':[1000,2000,2000,1],
    #             }

    cfg_dict = {'exp': 'exp7_V99',
                'config':'/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml',
                'opt':[1000,2000,2000,3],
                }

    output_dir = cfg_dict['exp']
    if os.path.exists(f'{output_dir}/compare2.csv'): os.remove(f'{output_dir}/compare2.csv')
    dataDir=Path('sartorius-cell-instance-segmentation/')

    register_coco_instances('sartorius_val',{},f'val_7.json', dataDir)
    # register_coco_instances('sartorius_val',{},f'val_7.json', dataDir)

    record = pd.read_csv(os.path.join(cfg_dict['exp'], 'compare.csv'))
    cfg = get_cfg()

    add_vovnet_config(cfg)
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.merge_from_file(cfg_dict['config'])
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 12
    
    cfg.INPUT.MIN_SIZE_TEST = 1184
    cfg.INPUT.MAX_SIZE_TEST = 2000

    cfg.TEST.DETECTIONS_PER_IMAGE = cfg_dict['opt'][0]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = cfg_dict['opt'][1]
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg_dict['opt'][2]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_dict['opt'][3]

    # cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
    # cfg.TEST.AUG.MIN_SIZES = (640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184)
    cfg.TEST.AUG.MIN_SIZES = (672, 736, 800, 864, 928, 992, 1056, 1120, 1184)
    cfg.TEST.AUG.MAX_SIZE = 2000


    # cfg.MODEL.PIXEL_MEAN = [127.973, 127.973, 127.973]
    # cfg.MODEL.PIXEL_STD = [13.260, 13.260, 13.260]

    from detectron2.modeling import GeneralizedRCNNWithTTA

    step = record['step'].tolist()

    for s in tqdm.tqdm(step):
        weight = os.path.join(output_dir, f'model_{s:07d}.pth')
        model = myPredictor(cfg, weight)
        # MD = myDetect(model, 'sartorius_val', s)
        # MD.detect()
        predictor = GeneralizedRCNNWithTTA(cfg, model.model)   
        # predictor = GeneralizedRCNNWithTTA(cfg, model.model, batch_size=6)   
        # predictor = GeneralizedRCNNWithTTA(cfg, model.model, batch_size=18)   
        predictor.model.eval()
        MTTA = myTTA(predictor, 'sartorius_val', s)
        MTTA.detect()