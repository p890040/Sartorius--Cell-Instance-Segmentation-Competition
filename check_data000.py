
import numpy as np
import json
import os
import cv2
import pycocotools.mask as mask_util
import shutil
import tqdm


def merge_json(dicts):
    new_annos = []
    new_imgs=[]
    anno_id = 1
    for i, dict_ in enumerate(dicts):
        cat_id = i+1
        for d in dict_['annotations']:
            d['id'] = anno_id
            d['category_id'] = cat_id
            new_annos.append(d)
            anno_id+=1
        for d in dict_['images']:
            new_imgs.append(d)
            
    new_dict = {'categories' : [{'id': 1, 'name': 'shsy5y'}, {'id': 2, 'name': 'astro'}, {'id': 3, 'name': 'cort'}],
                'images': new_imgs,
                'annotations': new_annos,
                }        
    return new_dict

with open('train_shsy5y_1.json') as f:
    train_shsy5y_1 = json.load(f)
with open('train_astro_2.json') as f:
    train_astro_2 = json.load(f)
with open('train_cort_3.json') as f:
    train_cort_3 = json.load(f)

with open('val_shsy5y_1.json') as f:
    val_shsy5y_1 = json.load(f)
with open('val_astro_2.json') as f:
    val_astro_2 = json.load(f)
with open('val_cort_3.json') as f:
    val_cort_3 = json.load(f)

train = merge_json([train_shsy5y_1, train_astro_2, train_cort_3])
val = merge_json([val_shsy5y_1, val_astro_2, val_cort_3])

with open(r'train_7.json', 'w') as f:
    json.dump(train, f)

with open(r'val_7.json', 'w') as f:
    json.dump(val, f)


# c1, c2, c3=0,0,0
# for t in train['annotations']:
#     if(t['category_id']==1):
#         c1+=1
#     if(t['category_id']==2):
#         c2+=1
#     if(t['category_id']==3):
#         c3+=1

# d1, d2, d3=0,0,0
# for t in val['annotations']:
#     if(t['category_id']==1):
#         d1+=1
#     if(t['category_id']==2):
#         d2+=1
#     if(t['category_id']==3):
#         d3+=1

# with open('train_1.json') as f:
#     train_1 = json.load(f)
    
# with open('val_1.json') as f:
#     val_1 = json.load(f)

# e1, e2, e3=0,0,0
# for t in train_1['annotations']:
#     if(t['category_id']==1):
#         e1+=1
#     if(t['category_id']==2):
#         e2+=1
#     if(t['category_id']==3):
#         e3+=1

# f1, f2, f3=0,0,0
# for t in val_1['annotations']:
#     if(t['category_id']==1):
#         f1+=1
#     if(t['category_id']==2):
#         f2+=1
#     if(t['category_id']==3):
#         f3+=1

# print(c1+d1)
# print(c2+d2)
# print(c3+d3)

# print(e1+f1)
# print(e2+f2)
# print(e3+f3)

