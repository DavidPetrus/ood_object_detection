import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
#import cv2
from collections import defaultdict
import random
import time
import csv
import ast
import os
import glob
import gc

from effdet.data.transforms import transforms_coco_eval, transforms_coco_train, clip_boxes_
from effdet.loss import DetectionLoss
from effdet.anchors import Anchors, AnchorLabeler

from absl import flags

FLAGS = flags.FLAGS

random.seed(time.time())

INP_SIZE = 256


class PretrainDataset(torch.utils.data.IterableDataset):

    def __init__(self,model_config,n_way,num_sup,num_qry,lvis_sample,lvis_val_sample,lvis_bboxes,lvis_cats,lvis_train_cats,lvis_val_cats):
        self.lvis_sample = lvis_sample
        self.lvis_val_sample = lvis_val_sample
        self.lvis_bboxes = lvis_bboxes
        self.lvis_cats = lvis_cats
        self.lvis_train_cats = lvis_train_cats
        self.lvis_val_cats = lvis_val_cats
        self.lvis_img_size_load = {}

        self.n_way = n_way
        self.num_sup = num_sup
        self.num_qry = num_qry

        self.supp_size = FLAGS.img_size

        if FLAGS.large_qry:
            self.qry_img_size = 640
        else:
            self.qry_img_size = 256

        if FLAGS.fpn:
            self.levels = [3,4,5]
            self.feats = 'feat'
        else:
            self.levels = [3,4,5,6,7]
            self.feats = 'activ'

        if FLAGS.ubuntu:
            self.feat_dir = 'train_feats'
        else:
            self.feat_dir = 'train_activ'

        self.val_freq = int(FLAGS.val_freq/max(FLAGS.num_workers,1))
        self.num_val_cats = int(FLAGS.num_val_cats/max(FLAGS.num_workers,1))
        self.exp = FLAGS.exp
        self.n_way = FLAGS.n_way
        self.num_workers = FLAGS.num_workers

        self.anchors = Anchors.from_config(model_config)
        self.anchor_labeler = AnchorLabeler(self.anchors, FLAGS.num_train_cats, match_threshold=0.5)

        if FLAGS.random_trans:
            self.train_transform = transforms_coco_train(self.qry_img_size,use_prefetcher=True)
        else:
            self.train_transform = transforms_coco_eval(self.qry_img_size,interpolation='bilinear',use_prefetcher=True)

        self.transform = transforms_coco_eval(self.qry_img_size,interpolation='bilinear',use_prefetcher=True)

    def __iter__(self):
        val_count = 1
        num_val_iters = 0
        val_iter = False

        for i in range(1000000):
            if (not val_iter and val_count % self.val_freq == 0):
                val_iter = True
                val_count += 1
            elif val_iter and num_val_iters < self.num_val_cats:
                num_val_iters += 1
            else:
                val_iter = False
                num_val_iters = 0
                val_count += 1

            query_img_batch = []
            query_lab_batch = []
            qry_bbox_ls = []
            qry_cls_ls = []

            print("Val:",val_iter,i,val_count)
            cats = random.sample(self.lvis_train_cats,self.num_qry)
            missed = 0
            query_imgs = []
            batch_cats = []
            for cat in cats:
                if val_iter:
                    if cat not in self.lvis_val_sample.keys():
                        print("Missing Cateogry!!",cat)
                        missed += 1
                        continue
                    query_imgs.extend(random.sample(list(self.lvis_val_sample[cat]),1+missed))
                    missed = 0
                else:
                    query_imgs.extend(random.sample(list(self.lvis_sample[cat]),1))

            for img_path in query_imgs:
                cat_idxs = []
                cls_targets = []
                for lv_ix,lv_cat in enumerate(self.lvis_cats[img_path]):
                    if lv_cat in self.lvis_train_cats: 
                        cat_idxs.append(lv_ix)
                        cls_targets.append(self.lvis_train_cats.index(lv_cat))

                batch_cats.extend(cls_targets)

                img_bboxes = np.asarray(self.lvis_bboxes[img_path])[cat_idxs].astype(np.float32)
                img_bboxes[:,2:] = img_bboxes[:,:2]+img_bboxes[:,2:]
                img_bboxes = np.concatenate([img_bboxes[:,1:2],img_bboxes[:,0:1],img_bboxes[:,3:],img_bboxes[:,2:3]],axis=1)
                
                img_cat_ids = np.array(cls_targets)

                target = {'bbox': img_bboxes, 'cls': img_cat_ids, 'target_size': 640}
                try:
                    img_load = Image.open(img_path).convert('RGB')
                except:
                    img_load = Image.open(img_path.replace('val2017','train2017')).convert('RGB')

                if not val_iter:
                    img_trans,target = self.train_transform(img_load,target)
                else:
                    img_trans,target = self.transform(img_load,target)

                query_img_batch.append(torch.from_numpy(img_trans))
                qry_bbox_ls.append(torch.from_numpy(target['bbox']))
                qry_cls_ls.append(torch.from_numpy(target['cls']+1))

            q_cls_targets, q_box_targets, q_num_positives = self.anchor_labeler.batch_label_anchors(qry_bbox_ls, qry_cls_ls)
            query_lab_batch = {'cls':qry_cls_ls, 'bbox': qry_bbox_ls, 'cls_anchor':q_cls_targets, 'bbox_anchor':q_box_targets, 'num_positives':q_num_positives}

            #yield list(map(torch.stack, zip(*support_img_batch))), supp_cls_lab, list(map(torch.stack, zip(*query_img_batch))), query_lab_batch, task_cats, val_iter
            yield torch.stack(query_img_batch), query_lab_batch, val_iter, set(batch_cats)


def load_metadata_dicts():

    if FLAGS.ubuntu:
        base_path = "/home/ubuntu/"
        feat_dir = 'train_feats'
    else:
        base_path = '/home-mscluster/dvanniekerk/'
        feat_dir = 'train_activ'

    if FLAGS.fpn:
        levels = [3,4,5]
        feats = 'feat'
    else:
        levels = [3,4,5,6,7]
        feats = 'activ'

    if FLAGS.large_qry:
        qry_img_size = 640
    else:
        qry_img_size = 256

    start = time.time()

    lvis_all_cats = {}
    with open(base_path+"LVIS/lvis_train_cats.csv",'r') as fp:
        csv_reader = csv.DictReader(fp)
        for row in csv_reader:
            #if row['name'] in cats_not_to_incl: continue
            lvis_all_cats[int(row['id'])] = int(row['image_count'])

    lvis_all_cats = {k: v for k, v in sorted(lvis_all_cats.items(), key=lambda item: item[1])}
    lvis_train_cats = list(lvis_all_cats.keys())[-FLAGS.num_train_cats:]
    lvis_val_cats = list(lvis_all_cats.keys())[-500:-400]

    eval_cats = []
    cat2id = {}
    with open(base_path+"LVIS/lvis_train_cats.csv",'r') as fp:
        csv_reader = csv.DictReader(fp)
        for row in csv_reader:
            cat2id[row['name']] = int(row['id'])
            if int(row['id']) in lvis_train_cats:
                eval_cats.append({'id':lvis_train_cats.index(int(row['id']))+1,'name':row['name']})

    lvis_cats = {}
    lvis_bboxes = {}
    with open(base_path+'LVIS/lvis_pre_annots.txt','r') as fp: lines = fp.readlines()
    for line in lines:
        splits = line.split(';')
        lvis_cats[splits[0].replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+splits[0][-14:]] = ast.literal_eval(splits[1])
        lvis_bboxes[splits[0].replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+splits[0][-14:]] = ast.literal_eval(splits[2])

    with open(base_path+'LVIS/lvis_val_annots.txt','r') as fp: lines = fp.readlines()
    for line in lines:
        splits = line.split(';')
        lvis_cats[splits[0].replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+splits[0][-14:]] = ast.literal_eval(splits[1])
        lvis_bboxes[splits[0].replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+splits[0][-14:]] = ast.literal_eval(splits[2])

    lvis_sample = {}
    added = 0
    not_added = 0
    with open(base_path+'LVIS/lvis_pre_sample.txt','r') as fp: lines = fp.readlines()
    for line in lines:
        splits = line.split(';')
        if cat2id[splits[0]] not in lvis_train_cats: continue
        #if FLAGS.fpn and FLAGS.large_qry:
        cat_imgs = []
        imgs = ast.literal_eval(splits[1])
        for img in set(imgs):
            #if not os.path.exists(img.replace('train2017','train_activ_640').replace('.jpg','_feat5.npy')):
            #    continue
            add_to_sample = True
            set_cats = set(lvis_cats[img.replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+img[-14:]])
            for img_cat in set_cats:
                if img_cat in lvis_val_cats:
                    add_to_sample = False

            if add_to_sample:
                added += 1
            else:
                not_added += 1
                continue

            cat_imgs.append(img.replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+img[-14:])

        lvis_sample[cat2id[splits[0]]] = cat_imgs

    print(len(lvis_sample.keys()))
    print(added)
    print(not_added)

    lvis_val_sample = {}
    added = 0
    not_added = 0
    with open(base_path+'LVIS/lvis_val_sample.txt','r') as fp: lines = fp.readlines()
    for line in lines:
        splits = line.split(';')
        if cat2id[splits[0]] not in lvis_train_cats: continue
        #if FLAGS.fpn and FLAGS.large_qry:
        cat_imgs = []
        imgs = ast.literal_eval(splits[1])
        for img in set(imgs):
            #if not os.path.exists(img.replace('train2017','train_activ_640').replace('.jpg','_feat5.npy')):
            #    continue
            add_to_sample = True
            set_cats = set(lvis_cats[img.replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+img[-14:]])
            for img_cat in set_cats:
                if img_cat in lvis_val_cats:
                    add_to_sample = False

            if add_to_sample:
                added += 1
            else:
                not_added += 1
                continue

            cat_imgs.append(img.replace('/home-mscluster/dvanniekerk/','/home/ubuntu/')[:-14]+'00'+img[-14:])

        lvis_val_sample[cat2id[splits[0]]] = cat_imgs

    print(len(lvis_val_sample.keys()))
    print(added)
    print(not_added)

    print(time.time()-start)

    return lvis_sample,lvis_val_sample,lvis_bboxes,lvis_cats,lvis_train_cats,lvis_val_cats,eval_cats

#colors = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,255,255)]
#count = 0





'''inner_steps = 5
train_iter = 0
for sup_imgs,sup_class,sup_reg,qry_imgs,qry_class,qry_reg in dataset:
    print('-------------------------',train_iter)
    fast_weights = [var*1 for var in base_model.trainable_variables]
    fast_weights = []
    with tf.GradientTape() as qry_tape:
        for var in base_model.trainable_variables:
            weight_tensor = var*1
            qry_tape.watch(weight_tensor)
            fast_weights.append(weight_tensor)

        with tf.GradientTape() as sup_tape:
            y_pred = base_model(sup_imgs, training=True)  # Forward pass
            sup_loss = base_model.compiled_loss({'regression': sup_reg,'classification': sup_class}, y_pred, regularization_losses=base_model.losses)

        gradients = sup_tape.gradient(sup_loss,base_model.trainable_variables)

        for var_ix in range(len(base_model.trainable_variables)):
            fast_model.trainable_variables[var_ix].assign(fast_weights[var_ix] - 0.01*gradients[var_ix])

        #for grad,var in zip(gradients,fast_model.trainable_variables):
        #    var = var - 0.01*grad

        #for inner_idx in range(inner_steps):
        #    sup_loss = fast_model.train_on_batch(sup_imgs, y={'regression': sup_reg,'classification': sup_class})
        #    print("Support {}: Loss {}; Class Loss: {}; Reg Loss {}".format(inner_idx,sup_loss[0],sup_loss[1],sup_loss[2]))
        
        #qry_loss = fast_model.evaluate(qry_imgs, y={'regression': qry_reg,'classification': qry_class}, verbose=0)
        qry_pred = fast_model(qry_imgs)
        qry_loss = fast_model.compiled_loss({'regression': qry_reg,'classification': qry_class}, qry_pred, regularization_losses=fast_model.losses)
        print(qry_loss)
        #print("Query {}: Loss {}; Class Loss: {}; Reg Loss {}".format(inner_idx,qry_loss[0],qry_loss[1],qry_loss[2]))

    gradients = qry_tape.gradient(qry_loss,fast_weights)
    print(gradients)


    train_iter += 1'''


'''for sup_imgs,_,sup_labs,qry_imgs,__,qry_labs,task_cats in meta_generator:
    sup_img_batch = list(sup_imgs[0].numpy().astype(np.uint8))
    qry_img_batch = list(qry_imgs[0].numpy().astype(np.uint8))
    print(count)
    print([epic_cats[tc] for tc in task_cats])
    for img_sup,lab_sup,img_qry,lab_qry in zip(sup_img_batch,sup_labs,qry_img_batch,qry_labs):
        img = cv2.cvtColor(img_sup, cv2.COLOR_RGB2BGR)
        for bbox,cat in zip(lab_sup['bboxes'].astype(np.int32),lab_sup['labels'].astype(np.int32)):
            cv2.rectangle(img,tuple(bbox[:2]),tuple(bbox[2:]),colors[cat],2)
            cv2.putText(img,'{}'.format(epic_cats[task_cats[cat]]), (bbox[0], bbox[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        cv2.imshow('a',img)
        cv2.waitKey(0)

        img = cv2.cvtColor(img_qry, cv2.COLOR_RGB2BGR)
        for bbox,cat in zip(lab_qry['bboxes'].astype(np.int32),lab_qry['labels'].astype(np.int32)):
            cv2.rectangle(img,tuple(bbox[:2]),tuple(bbox[2:]),colors[cat],2)
            cv2.putText(img,'{}'.format(epic_cats[task_cats[cat]]), (bbox[0], bbox[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        cv2.imshow('a',img)
        cv2.waitKey(0)

    if (count := count+1) > 10:
        break'''


#dataset = tf.data.Dataset.from_generator(epic_generator,args=[lvis_sample,lvis_bboxes,lvis_cat_ids,epic_sample,epic_bboxes,epic_cat_ids],(tf.float32,tf.int32))


'''start = time.time()
lvis_cat_ids = {}
with open(base_path+"LVIS/lvis_train_cats.csv",'r') as fp:
    csv_reader = csv.DictReader(fp)
    for row in csv_reader:
        if int(row['image_count']) >= 38 and row['name'] != 'cornet':
            lvis_cat_ids[int(row['id'])] = (row['name'],int(row['image_count']),int(row['instance_count']))

# Load LVIS data
lvis_base = base_path+"LVIS/train2017/"
lvis_imgs_not_exhau = {}
with open(base_path+"LVIS/lvis_train_imgs.csv",'r') as fp:
    csv_reader = csv.DictReader(fp)
    for row in csv_reader:
        not_exhau_ls = ast.literal_eval(row['not_exhaustive_category_ids'])
        lvis_imgs_not_exhau[lvis_base+f"{int(row['id']):012}.jpg"] = not_exhau_ls


lvis_cats = defaultdict(list)
lvis_bboxes = defaultdict(list)
lvis_sample = defaultdict(list)
with open(base_path+"LVIS/lvis_train_annots.csv",'r') as fp:
    csv_reader = csv.DictReader(fp)
    for row in csv_reader:
        img_path = lvis_base+f"{int(row['image_id']):012}.jpg"
        cat_id = int(row['category_id'])
        bbox = ast.literal_eval(row['bbox'])
        if cat_id in lvis_cat_ids.keys() and cat_id not in lvis_imgs_not_exhau[img_path]:
            cat_name = lvis_cat_ids[cat_id][0]
            lvis_cats[img_path].append(cat_name)
            lvis_bboxes[img_path].append(bbox)
            lvis_sample[cat_name].append(img_path.replace("/home/petrus/","/home-mscluster/dvanniekerk/"))

with open(base_path+'LVIS/lvis_annots.txt','w') as fp:
    for img_path in lvis_cats.keys():
        fp.write("{};{};{};\n".format(img_path.replace("/home/petrus/","/home-mscluster/dvanniekerk/"),lvis_cats[img_path],lvis_bboxes[img_path]))

with open(base_path+'LVIS/lvis_sample.txt','w') as fp:
    for cat_name in lvis_sample.keys():
        fp.write("{};{};\n".format(cat_name,lvis_sample[cat_name]))

print(time.time()-start)'''
