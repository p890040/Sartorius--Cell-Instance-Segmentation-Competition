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

dataDir=Path('/home/solomon/public/Pawn/Datasets/Shoes/Images/')
cfg = get_cfg()
# cfg.INPUT.MASK_FORMAT='bitmask'
register_coco_instances('sartorius_train',{}, os.path.join(dataDir, 'train.json'), dataDir)
# register_coco_instances('sartorius_val',{},'.val_b.json', dataDir)
metadata = MetadataCatalog.get('sartorius_train')
train_ds = DatasetCatalog.get('sartorius_train')

#%%
# img_name = '42d8ecbc95a1_13.png'
# d = [k for k in train_ds if img_name in k['file_name']][0]
d = train_ds[1]
print(d["file_name"])
img = cv2.imread(d["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
out = visualizer.draw_dataset_dict(d)
plt.figure(figsize = (20,15))
plt.imshow(out.get_image()[:, :, ::-1])

# %%
