#%%
import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
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
setup_logger()

dataDir=Path('sartorius-cell-instance-segmentation/')
cfg = get_cfg()
cfg.INPUT.MASK_FORMAT='bitmask'
register_coco_instances('val_shsy5y_1',{},'val_shsy5y_1.json', dataDir)
register_coco_instances('val_astro_2',{},'val_astro_2.json', dataDir)
register_coco_instances('val_cort_3',{},'val_cort_3.json', dataDir)
metadata_1 = MetadataCatalog.get('val_shsy5y_1')
metadata_2 = MetadataCatalog.get('val_astro_2')
metadata_3 = MetadataCatalog.get('val_cort_3')
train_ds_1 = DatasetCatalog.get('val_shsy5y_1')
train_ds_2 = DatasetCatalog.get('val_astro_2')
train_ds_3 = DatasetCatalog.get('val_cort_3')

#%%
# img_name = '1d9cf05975eb.png'
# d = [k for k in train_ds if img_name in k['file_name']][0]
data_type=1
if(data_type==1):
    d = train_ds_1[10]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_1)
elif(data_type==2):
    d = train_ds_2[0]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_2)
elif(data_type==3):
    d = train_ds_3[0]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_3)
print(d["file_name"])
out = visualizer.draw_dataset_dict(d)
f, axarr = plt.subplots(1,2, dpi=800)
axarr[0].imshow(out.get_image()[:, :, ::-1])
axarr[0].axis('off')
# plt.figure(figsize = (20,15))
# plt.imshow(out.get_image()[:, :, ::-1])

# cfg_dicts=[{'ckpt': 'exp3_bh1_aug_ms3_R50_p2000/model_0014139.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml','thr':[.35,.45,.75], 'opt':[1000,2000,2000,3]},
#             {'ckpt': '1_R101/model_0011879.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':0.25, 'opt':[1000,2000,2000,1]},
#             {'ckpt': '2_R101/model_0002441.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':0.4, 'opt':[800,1500,1500,1]},
#             {'ckpt': '3_R101/model_0005039.pth','config':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml','thr':0.7, 'opt':[500,1000,1000,1]}]

cfg_dicts=[{'ckpt': 'exp3_VOV/model_0015149.pth','config':'/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml','thr':[.35,.35,.85], 'opt':[1000,2000,2000,3]},
           {'ckpt': '1_V99_p2000_2/model_0008252.pth','config':'/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml','thr':0.34, 'opt':[1000,2000,2000,1]},
           {'ckpt': '2_V99_p2000_flipud/model_0006437.pth','config':'/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml','thr':0.56, 'opt':[1000,2000,2000,1]},
           {'ckpt': '3_V99_p2000_flipud/model_0006439.pth','config':'/home/solomon/public/Pawn/detectron2-0.6/projects/vovnet-detectron2-master/configs/mask_rcnn_V_99_FPN_3x.yaml','thr':0.7, 'opt':[1000,2000,2000,1]},
          ]

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import torch
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
from detectron2.projects import _PROJECTS
_PROJECTS.update({"vovnet": "vovnet-detectron2-master"})
from detectron2.projects.vovnet import  add_vovnet_config

models, thrs = [],[]
for _cfg in cfg_dicts:
    cfg = get_cfg()
    add_vovnet_config(cfg)
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.merge_from_file(_cfg['config'])
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

predictor = models[data_type]
thr = thrs[data_type]
pred = predictor(img)
MIN_PIXELS=30

take = pred['instances'].scores >= thr
pred_masks = pred['instances'].pred_masks[take].cpu().numpy()
pred_classes = pred['instances'].pred_classes[take].cpu().numpy()
scores = pred['instances'].scores[take].cpu().numpy()
pred_boxes = pred['instances'].pred_boxes[take].tensor.cpu().numpy()

pred_masks_filter=[]
pred_classes_filter=[]
scores_filter=[]
pred_boxes_filter=[]
for i, mask in enumerate(pred_masks):
        if mask.sum() >= MIN_PIXELS:
            pred_masks_filter.append(mask)
            pred_classes_filter.append(pred_classes[i])
            scores_filter.append(scores[i])
            pred_boxes_filter.append(pred_boxes[i])
pred_masks = np.array(pred_masks_filter)
pred_classes = np.array(pred_classes_filter)
scores = np.array(scores_filter)
pred_boxes = np.array(pred_boxes_filter)

from detectron2.structures import Instances, Boxes
# res = Instances((520, 704))
res = Instances((520, 704))
res.pred_boxes = Boxes(pred_boxes)
# res.pred_boxes.scale(scale_x = 1603/704, scale_y = 1184/520)
res.scores = torch.tensor(scores)
res.pred_classes = torch.tensor(pred_classes)
res.pred_masks = torch.tensor(pred_masks)

if(data_type==1):
    visualizer_det = Visualizer(img[:, :, ::-1], metadata=metadata_1)
elif(data_type==2):
    visualizer_det = Visualizer(img[:, :, ::-1], metadata=metadata_2)
elif(data_type==3):
    visualizer_det = Visualizer(img[:, :, ::-1], metadata=metadata_3)
vis_detection = visualizer_det.draw_instance_predictions(predictions=res)
axarr[1].imshow(vis_detection.get_image()[:, :, ::-1])
axarr[1].axis('off')
plt.show()
# plt.figure(figsize = (20,15))
# plt.imshow(vis_detection.get_image()[:, :, ::-1])

# %%
