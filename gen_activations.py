import torch
import numpy as np
import glob
from omegaconf import OmegaConf
from PIL import Image
import random
import time
import os

#from dataloader import load_metadata_dicts

from effdet.data.transforms import transforms_coco_eval
from timm.models import load_checkpoint
from effdet.efficientdet import EfficientDet
from effdet.config.model_config import default_detection_model_configs


IMG_SIZE = 640
BATCH_SIZE = 16

#img_size_tensor = torch.tensor(IMG_SIZE,dtype=torch.float32)



config=dict(
    name='efficientdet_d0',
    backbone_name='efficientnet_b0',
    image_size=(IMG_SIZE, IMG_SIZE),
    fpn_channels=64,
    fpn_cell_repeats=3,
    box_class_repeats=3,
    pad_type='',
    redundant_bias=False,
    backbone_args=dict(drop_path_rate=0.1),#, checkpoint_path="efficientnet_b0_ra-3dd342df.pth"),
    #url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth',
)

config = OmegaConf.create(config)

h = default_detection_model_configs()
h.update(config)
h.num_levels = h.max_level - h.min_level + 1

# create the base model
model = EfficientDet(h)
load_checkpoint(model,"efficientdet_d0-f3276ba8.pth")

model_config = model.config
print(model_config['num_classes'])

model.eval()

#lvis_sample,lvis_bboxes,lvis_cat_ids,epic_sample,epic_bboxes,epic_cat_ids,epic_cats = load_metadata_dicts()


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
imagenet_mean = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_MEAN],device=torch.device('cuda')).view(1, 3, 1, 1)
imagenet_std = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_STD],device=torch.device('cuda')).view(1, 3, 1, 1)

model.to('cuda')

transform = transforms_coco_eval(IMG_SIZE,interpolation='bilinear',use_prefetcher=True)

levels = [3,4,5,6,7]


'''web_imgs = glob.glob("/home/ubuntu/web_images/*/*")

for ix in range(0,len(web_imgs),BATCH_SIZE):
    img_batch = []
    batch_paths = []
    for img_path in web_imgs[ix:ix+BATCH_SIZE]:
        try:
            if not os.path.exists(img_path[:img_path.rfind('/')].replace('web_images','LVIS/support_dataset/support_features')):
                os.mkdir(img_path[:img_path.rfind('/')].replace('web_images','LVIS/support_dataset/support_features'))

            img_load = Image.open(img_path).convert('RGB')
            width,height = img_load.size
            #np.save(img_path[:img_path.find('.')].replace('downloads','support_features')+'_size.npy',np.array([width,height]))
            img_trans,target = transform(img_load,{})
            img_batch.append(torch.from_numpy(img_trans))
            batch_paths.append(img_path)
        except Exception as e:
            print(e)
            continue

    batch_size = len(img_batch)
    if batch_size == 0: continue
    img_batch = torch.stack(img_batch)
    img_batch = (img_batch.to('cuda').float()-imagenet_mean)/imagenet_std
    print('Web:',ix)

    feats= model(img_batch,mode='bb')

    for f_ix,feat in enumerate(feats):
        np_feat = feat.detach().cpu().numpy()
        for b_ix,img_path in enumerate(batch_paths):
            while True:
                try:
                    np.save(img_path[:img_path.find('.')].replace('web_images','LVIS/support_dataset/support_features')+'_feat'+str(levels[f_ix])+'.npy',np_feat[b_ix])
                    break
                except Exception as e:
                    print('---------------',e)
                    time.sleep(2)'''

'''for a_ix,act in enumerate(activs):
    np_activ = act.detach().cpu().numpy()
    for b_ix,img_path in enumerate(batch_paths):
        while True:
            try:
                np.save(img_path[:img_path.find('.')].replace('downloads','support_features')+'_activ'+str(levels[a_ix])+'.npy',np_activ[b_ix])
                break
            except Exception as e:
                print('---------------',e)
                time.sleep(2)'''



lvis_imgs = glob.glob("/home/ubuntu/LVIS/train2017/*")
#lvis_done = glob.glob("/home-mscluster/dvanniekerk/LVIS/train_activ_640/*feat5.npy*")
#lvis_imgs = glob.glob("/home/petrus/LVIS/train2017/*")
print(len(lvis_imgs))

for ix in range(0,len(lvis_imgs),BATCH_SIZE):
    img_batch = []
    for img_path in lvis_imgs[ix:ix+BATCH_SIZE]:
        #if img_path[:img_path.find('.')].replace('train2017','train_activ_'+str(IMG_SIZE))+'_feat5.npy' in lvis_done:
        #    continue
        
        img_load = Image.open(img_path).convert('RGB')
        width,height = img_load.size
        np.save(img_path.replace('train2017','train_feats_'+str(IMG_SIZE))[:-4]+'size.npy',np.array([width,height]))
        #img_trans,target = transform(img_load,{})
        #img_batch.append(torch.from_numpy(img_trans))

    print(ix)

'''    batch_size = len(img_batch)
    if batch_size == 0: continue

    img_batch = torch.stack(img_batch)
    img_batch = (img_batch.to('cuda').float()-imagenet_mean)/imagenet_std
    if ix % 1000 == 0:
        print('LVIS:',ix)

    feats = model(img_batch,mode='bb')

    for f_ix,feat in enumerate(feats):
        np_feat = feat.detach().cpu().numpy()
        for b_ix,img_path in enumerate(lvis_imgs[ix:ix+BATCH_SIZE]):
            while True:
                try:
                    np.save(img_path.replace('train2017','train_activ_'+str(IMG_SIZE))[:-4]+'_feat'+str(levels[f_ix])+'.npy',np_feat[b_ix].astype(np.float16))
                    break
                except Exception as e:
                    print('---------------',e)
                    time.sleep(2)'''

'''for a_ix,act in enumerate(activs):
    np_activ = act.detach().cpu().numpy()
    for b_ix,img_path in enumerate(lvis_imgs[ix:ix+BATCH_SIZE]):
        while True:
            try:
                np.save(img_path.replace('train2017','train_activ_'+str(IMG_SIZE))[:-4]+'_activ'+str(levels[a_ix])+'.npy',np_activ[b_ix])
                break
            except Exception as e:
                print('---------------',e)
                time.sleep(2)'''


