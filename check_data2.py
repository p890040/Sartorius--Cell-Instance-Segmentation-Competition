
import numpy as np
import json
import os
import cv2
import pycocotools.mask as mask_util
import shutil
import tqdm

def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py

    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask_util.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation #, [x, y, w, h], area


def polygonFromMask_D2(mask):
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    segmentation = [segment.astype(np.int32).flatten().tolist() for segment in res]
    return segmentation


# with open('train_1.json') as f:
#     train = json.load(f)

# dict_map = {k['id']:i for i, k in enumerate(train['images'])}

# for i, anno in enumerate(tqdm.tqdm(train['annotations'])):
#     maskedArr = mask_util.decode(anno['segmentation'])
#     # print(maskedArr.mean())
#     # segments = polygonFromMask(maskedArr)
#     segments = polygonFromMask_D2(maskedArr)
#     train['annotations'][i]['segmentation'] = segments
#     train['annotations'][i]['image_id'] = dict_map[train['annotations'][i]['image_id']]
#     train['annotations'][i]['angle'] = -1

# for i in tqdm.tqdm(range(len(train['images']))):
#     train['images'][i]['id'] = dict_map[train['images'][i]['id']]

# with open('trainval.json', 'w') as f:
#     json.dump(train, f)


with open('val_1.json') as f:
    train = json.load(f)

dict_map = {k['id']:i for i, k in enumerate(train['images'])}

for i, anno in enumerate(tqdm.tqdm(train['annotations'])):
    maskedArr = mask_util.decode(anno['segmentation'])
    # print(maskedArr.mean())
    # segments = polygonFromMask(maskedArr)
    segments = polygonFromMask_D2(maskedArr)
    train['annotations'][i]['segmentation'] = segments
    train['annotations'][i]['image_id'] = dict_map[train['annotations'][i]['image_id']]
    train['annotations'][i]['angle'] = -1

for i in tqdm.tqdm(range(len(train['images']))):
    train['images'][i]['id'] = dict_map[train['images'][i]['id']]

with open('trainval_val.json', 'w') as f:
    json.dump(train, f)





