import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('exp', help='')
# parser.add_argument('--aug_color', action="store_true", help='')
parser.add_argument('--aug_color', default=True, help='')
parser.add_argument('--flipud', action="store_true", help='')
parser.add_argument('--p', type=int, help='')
parser.add_argument('--fgbg', default=0.25, type=float, help='')
parser.add_argument('--DBatch', default=512, type=int, help='')
# parser.add_argument('--image_mean', default=[103.530, 116.280, 123.675], type=float, nargs='+', help='')
# parser.add_argument('--image_std', default=[1.0, 1.0, 1.0], type=float, nargs='+', help='')

parser.add_argument('--lr', default=0.001, type=float, help='')
parser.add_argument('--batch', default=1, type=int, help='')
parser.add_argument('--step', default=14999, type=int, help='')

parser.add_argument('--fix-resize', action="store_true", help='')

parser.add_argument('--init_thr', type=float, help='')
parser.add_argument('--thr_iter', default=8, type=int, help='')
parser.add_argument('--thr_inv', default=0.05, type=float, help='')
parser.add_argument('--eval_perior', type=int, help='')

args = parser.parse_args()


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

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        # self.thr_iter=35
        self.thr_iter = args.thr_iter
        self.scores = []
        init_thr = args.init_thr
        for i in range(self.thr_iter):
            # thr = 0.02*i
            thr = args.thr_inv*i
            THRS = init_thr + thr
            self.scores.append([THRS])
                    

    def process(self, inputs, outputs):
        
        init_thr = args.init_thr
        number=0
        for i in range(self.thr_iter):
            # thr = 0.02*i
            thr = args.thr_inv*i
            THRS = init_thr + thr
            for inp, out in zip(inputs, outputs):
                if len(out['instances']) == 0:
                    self.scores[number].append(0)    
                else:
                    targ = self.annotations_cache[inp['image_id']]
                    self.scores[number].append(score(out, targ, THRS))
            number+=1

    def evaluate(self):
        thrs, results = [],[]
        for l in self.scores:
            thrs.append(l[0])
            results.append(l[1:])
        results = np.array(results).mean(1)
        thrs = np.array(thrs)
        results_pd = pd.DataFrame({'thrs':thrs, 
                                   'ep1':results})
        if(os.path.exists(f'{output_dir}/map.csv')):
            results_pre = pd.read_csv(f'{output_dir}/map.csv')
            ep = 'ep' + str(len(results_pre.columns))
            results_pre[ep] = results
            results_pre.to_csv(f'{output_dir}/map.csv', index=False, float_format='%.4f')
        else:
            results_pd.to_csv(f'{output_dir}/map.csv', index=False, float_format='%.4f')
        
        global step_
        step_ = step_ + itv
        max_thr = thrs[np.argmax(results)]
        max_res = np.max(results)
        compare_pd = pd.DataFrame({'thrs':[max_thr],  
                                   'result':[max_res],
                                   'step':[step_-1]})
        if(os.path.exists(f'{output_dir}/compare.csv')):
            compare_pre = pd.read_csv(f'{output_dir}/compare.csv')
            compare_pre = compare_pre.append(pd.DataFrame({'thrs':[max_thr], 
                                                            'result':[max_res],
                                                            'step':[step_-1]}))
            compare_pre.to_csv(f'{output_dir}/compare.csv', index=False, float_format='%.4f')
        else:
            compare_pd.to_csv(f'{output_dir}/compare.csv', index=False, float_format='%.4f')
        
        
        return {"Best MaP IoU": max_res,
                "Best Threshold 0": max_thr,
                }

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

from detectron2.projects import _PROJECTS
_PROJECTS.update({"vovnet": "vovnet-detectron2-master"})
from detectron2.projects.vovnet import  add_vovnet_config

if __name__ == "__main__":
    output_dir = args.exp
    # output_dir = 'exp1_bh1_aug'

    if os.path.exists(f'{output_dir}/map.csv'): os.remove(f'{output_dir}/map.csv')
    if os.path.exists(f'{output_dir}/compare.csv'): os.remove(f'{output_dir}/compare.csv')
    dataDir=Path('sartorius-cell-instance-segmentation/')
    cfg = get_cfg()
    add_vovnet_config(cfg)
    cfg.OUTPUT_DIR = output_dir
    cfg.INPUT.MASK_FORMAT='bitmask'
    if(args.exp[0]=='1'):
        jsname= 'shsy5y_1'
    elif(args.exp[0]=='2'):
        jsname= 'astro_2'
    elif(args.exp[0]=='3'):
        jsname= 'cort_3'
    else:
        raise"error"
    # register_coco_instances('sartorius_train',{}, f'train_{jsname}.json', dataDir)
    register_coco_instances('sartorius_train',{}, f'train_{jsname}_all.json', dataDir)
    register_coco_instances('sartorius_val',{},f'val_{jsname}.json', dataDir)
    # register_coco_instances('sartorius_train',{}, f'train_1.json', dataDir)
    # register_coco_instances('sartorius_val',{},f'val_1.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')
    
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml")
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.SEED=7
    
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = 'https://dl.dropbox.com/s/qx45cnv718k4zmn/mask_V_99_eSE_ms_3x.pth'
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = int(args.batch)
    cfg.SOLVER.BASE_LR = float(args.lr)
    cfg.SOLVER.MAX_ITER = int(args.step)    
    # cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.SOLVER.BASE_LR = 0.001
    # cfg.SOLVER.MAX_ITER = 19999    
    # cfg.SOLVER.STEPS = []        
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.TEST.EVAL_PERIOD = args.eval_perior*(len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH)  # Once per epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = args.eval_perior*(len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH)
    step_ = 0
    itv= args.eval_perior*(len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH)

    print(args)
    cfg.INPUT.RANDOM_COLOR = args.aug_color
    cfg.INPUT.RANDOM_FLIPUD = args.flipud
    if(args.fix_resize):
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)
        cfg.INPUT.MAX_SIZE_TRAIN = 1333
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

    # cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992,)
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184,)
    # cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376,)
    cfg.INPUT.MAX_SIZE_TRAIN = 2000
    # cfg.INPUT.MIN_SIZE_TEST = 992
    cfg.INPUT.MIN_SIZE_TEST = 1184
    # cfg.INPUT.MIN_SIZE_TEST = 1376
    cfg.INPUT.MAX_SIZE_TEST = 2000

    
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = args.p
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = args.p
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = args.p
    print(cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN)
    if(args.p==1000):
        cfg.TEST.DETECTIONS_PER_IMAGE = 500
    elif(args.p==1500):
        cfg.TEST.DETECTIONS_PER_IMAGE = 800
    elif(args.p==2000):
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    else:
        raise"error"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.DBatch
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION =  args.fgbg

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    # cfg.SOLVER.STEPS = (3000,)

    # cfg.MODEL.PIXEL_MEAN = [127.973, 127.973, 127.973]
    # cfg.MODEL.PIXEL_STD = [13.260, 13.260, 13.260]

    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
