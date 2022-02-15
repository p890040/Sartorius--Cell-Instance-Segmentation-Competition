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

def score_ensemble(pred, targ):

    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred]
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


def nms_predictions(classes, scores, bboxes, masks, iou_th=.5, shape=(520, 704)):
    he, wd = shape[0], shape[1]
    boxes_list = [[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he]
                  for x in bboxes]
    scores_list = [x for x in scores]
    labels_list = [x for x in classes]
    nms_bboxes, nms_scores, nms_classes = nms(
        boxes=[boxes_list], 
        scores=[scores_list], 
        labels=[labels_list], 
        weights=None,
        iou_thr=iou_th
    )
    nms_masks = []
    for s in nms_scores:
        nms_masks.append(masks[scores.index(s)])
    nms_scores, nms_classes, nms_bboxes, nms_masks = zip(*sorted(zip(nms_scores, nms_classes, nms_bboxes, nms_masks), reverse=True))
    for x in nms_bboxes:
        x[0] = x[0]*wd
        x[1] = x[1]*he
        x[2] = x[2]*wd
        x[3] = x[3]*he
    return np.array(nms_classes), np.array(nms_scores), np.array(nms_bboxes), np.array(nms_masks)

def EnsembleDetect(predictors, thrs, dataset_name):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
    MIN_PIXELS = [30, 30, 30]

    thr_iter=4
    init_thr = np.array([0.35,0.55,0.75])
    out_scores=[]
    for i in range(thr_iter):
        for j in range(thr_iter):
            for k in range(thr_iter):
                thr = np.array([0.05*i, 0.05*j, 0.05*k])
                THRS = init_thr + thr
                out_scores.append([THRS])


    
    for inp in tqdm.tqdm(dataset_dicts):
        classes = []
        scores = []
        bboxes = []
        masks = []
        for predictor, thr in zip(predictors, thrs):
            img = cv2.imread(inp['file_name'])
            pred = predictor(img)
            
            pred_class = torch.mode(pred['instances'].pred_classes)[0]
            take = pred['instances'].scores >= thr[pred_class]
            pred_masks = pred['instances'].pred_masks[take].cpu().numpy()

            keeps=[]
            pred_masks_filter=[]
            for i, mask in enumerate(pred_masks):
                 if mask.sum() >= MIN_PIXELS[pred_class]:
                     pred_masks_filter.append(mask)
                     keeps.append(i)
            pred_masks = np.array(pred_masks_filter)
            classes.extend(pred['instances'].pred_classes[take].cpu().numpy()[keeps,...].tolist())
            scores.extend(pred['instances'].scores[take].cpu().numpy()[keeps,...].tolist())
            bboxes.extend(pred['instances'].pred_boxes[take].tensor.cpu()[keeps,...].numpy().tolist())
            masks.extend(pred_masks)
            assert len(classes) == len(masks) , 'ensemble lenght mismatch'

        classes, scores, bboxes, masks = nms_predictions(classes, scores, bboxes, masks)

        from detectron2.structures import Instances, Boxes
        # res = Instances((520, 704))
        res = Instances((1184, 1603))
        #images.tensor.shape
        #torch.Size([1, 3, 1184, 1632])
        res.pred_boxes = Boxes(bboxes)
        res.pred_boxes.scale(scale_x = 1603/704, scale_y = 1184/520)
        res.scores = torch.tensor(scores)
        res.pred_classes = torch.tensor(classes)
        res = [res]
        
        new_masks=[]
        for predictor, thr in zip(predictors, thrs):
            new_pre = predictor(img, res, False)
            new_mask = new_pre.pred_masks
            new_masks.append(new_mask)
        all_pred_masks = torch.stack(new_masks, dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        res[0].pred_masks = avg_pred_masks.cpu()
        res[0] = detector_postprocess(res[0], 520, 704)
        masks = res[0].pred_masks.cpu().numpy()
 

        
        number=0
        for i in range(thr_iter):
            for j in range(thr_iter):
                for k in range(thr_iter):
                    thr = np.array([0.05*i, 0.05*j, 0.05*k])
                    THRS = init_thr + thr
                    if(len(classes)==0):
                        out_scores[number].append(0)
                    else:
                        targ = annotations_cache[inp['image_id']]
                        sc = score_ensemble(masks, targ)
                        out_scores[number].append(sc)
                    number+=1
                    
    thrs, results = [],[]
    for l in out_scores:
        thrs.append(l[0])
        results.append(l[1:])
    results = np.array(results).mean(1)
    thrs = np.array(thrs)
    results_pd = pd.DataFrame({'thr0':thrs[:,0], 
                                'thr1':thrs[:,1], 
                                'thr2':thrs[:,2], 
                                'ep1':results})
    if(os.path.exists(f'{output_dir}/map.csv')):
        results_pre = pd.read_csv(f'{output_dir}/map.csv')
        ep = 'ep' + str(len(results_pre.columns)-2)
        results_pre[ep] = results
        results_pre.to_csv(f'{output_dir}/map.csv', index=False, float_format='%.3f')
    else:
        results_pd.to_csv(f'{output_dir}/map.csv', index=False, float_format='%.3f')
    
    step_ = 1
    max_thr = thrs[np.argmax(results)]
    max_res = np.max(results)
    compare_pd = pd.DataFrame({'thr0':[max_thr[0]], 
                                'thr1':[max_thr[1]], 
                                'thr2':[max_thr[2]], 
                                'result':[max_res],
                                'step':[step_-1]})
    if(os.path.exists(f'{output_dir}/compare.csv')):
        compare_pre = pd.read_csv(f'{output_dir}/compare.csv')
        compare_pre = compare_pre.append(pd.DataFrame({'thr0':[max_thr[0]], 
                                                        'thr1':[max_thr[1]], 
                                                        'thr2':[max_thr[2]], 
                                                        'result':[max_res],
                                                        'step':[step_-1]}))
        compare_pre.to_csv(f'{output_dir}/compare.csv', index=False, float_format='%.3f')
    else:
        compare_pd.to_csv(f'{output_dir}/compare.csv', index=False, float_format='%.3f')
    

    #     if(len(classes)==0):
    #         out_scores.append(0)
    #     else:
    #         targ = annotations_cache[inp['image_id']]
    #         sc = score_ensemble(masks, targ)
    #         out_scores.append(sc)
        
    # print(np.mean(out_scores))

class myDetect():
    def __init__(self, predictor, dataset_name):
        self.predictor=predictor
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in self.dataset_dicts}
        self.thr = np.array([0.35, 0.55, 0.75])
        self.scores=[]

    def detect(self):
        for inp in self.dataset_dicts:
            img = cv2.imread(inp['file_name'])
            out = self.predictor(img)
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                sc = score(out, targ, self.thr)
                self.scores.append(sc)
        print(np.mean(self.scores))

if __name__ == "__main__":
    output_dir = 'tes1'
    if os.path.exists(f'{output_dir}/map.csv'): os.remove(f'{output_dir}/map.csv')
    if os.path.exists(f'{output_dir}/compare.csv'): os.remove(f'{output_dir}/compare.csv')
    dataDir=Path('sartorius-cell-instance-segmentation/')
    register_coco_instances('sartorius_val',{},f'val_1.json', dataDir)


    cfg_dicts=[{'ckpt': 'exp1_bh1_aug_ms3_p2000/model_0016159.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml','thr':[.35, .55, .75]},
               {'ckpt': 'exp1_bh1_aug_ms3_R101_p2000/model_0017169.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':[.35, .35, .85]},
               {'ckpt': 'exp1_bh1_aug_ms3_X101_p2000/model_0007069.pth','config':'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml','thr':[.25, .45, .85]}]
    models, thrs = [],[]
    for _cfg in cfg_dicts:
        cfg = get_cfg()
        cfg.OUTPUT_DIR = output_dir
        cfg.INPUT.MASK_FORMAT='bitmask'
        cfg.merge_from_file(model_zoo.get_config_file(_cfg['config']))
        cfg.DATASETS.TEST = ("sartorius_val",)
        cfg.DATALOADER.NUM_WORKERS = 12
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.INPUT.MIN_SIZE_TEST = 1184
        cfg.INPUT.MAX_SIZE_TEST = 2000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        cfg.MODEL.WEIGHTS = _cfg['ckpt']
        model = myPredictor(cfg)
        models.append(model)
        thrs.append(np.array(_cfg['thr']))
    EnsembleDetect(models, thrs, 'sartorius_val')

    # model = DefaultPredictor(cfg)
    # mymodel = myDetect(model, 'sartorius_val')
    # mymodel.detect()