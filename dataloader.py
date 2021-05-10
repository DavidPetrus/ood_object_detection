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


class MetaEpicDataset(torch.utils.data.IterableDataset):

    def __init__(self,model_config,n_way,num_sup,num_qry,lvis_sample,web_sample,lvis_bboxes,lvis_cats,lvis_train_cats,lvis_val_cats):
        self.lvis_sample = lvis_sample
        self.web_sample = web_sample
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

        self.val_freq = int(FLAGS.val_freq/max(FLAGS.num_workers,1))
        self.num_val_cats = 2*int(FLAGS.num_val_cats/max(FLAGS.num_workers,1))
        self.exp = FLAGS.exp
        self.n_way = FLAGS.n_way
        self.num_workers = FLAGS.num_workers
        self.supp_aug = FLAGS.supp_aug
        self.num_zero = FLAGS.num_zero_images

        if FLAGS.random_trans:
            self.train_transform = transforms_coco_train(self.qry_img_size,use_prefetcher=True)
        else:
            self.train_transform = transforms_coco_eval(self.qry_img_size,interpolation='bilinear',use_prefetcher=True)

        self.anchors = Anchors.from_config(model_config)
        self.anchor_labeler = AnchorLabeler(self.anchors, self.n_way, match_threshold=0.5)

        self.transform = transforms_coco_eval(self.qry_img_size,interpolation='bilinear',use_prefetcher=True)

    def __iter__(self):
        val_count = 1
        num_val_iters = 0
        val_iter = False

        for i in range(1000000):
            if not val_iter and val_count % self.val_freq == 0:
                val_iter = True
                val_count += 1
            elif val_iter and num_val_iters < self.num_val_cats:
                num_val_iters += 1
            else:
                val_iter = False
                num_val_iters = 0
                val_count += 1


            support_img_batch = []
            support_lab_batch = []
            query_img_batch = []
            query_lab_batch = []
            supp_cls_lab = []
            qry_bbox_ls = []
            qry_cls_ls = []
            proj_bbox_ls = []
            proj_cls_ls = []
            if val_iter:
                task_cats = random.sample(self.lvis_val_cats,self.n_way)
                cat_ls = self.lvis_val_cats
            else:
                task_cats = random.sample(self.lvis_train_cats,self.n_way)
                cat_ls = self.lvis_train_cats

            #print("Val:",val_iter,task_cats,i,val_count)
            #try:
            for cat_ix,cat in enumerate(task_cats):
                support_imgs = random.sample(list(self.web_sample[cat]),self.num_sup)
                for img_path in support_imgs:
                    img_load = Image.open(img_path).convert('RGB')
                    width,height = img_load.size
                    if self.supp_aug and not val_iter:
                        img_trans,_ = self.train_transform(img_load,{'target_size':self.supp_size},(0.8, 1.5))
                    else:
                        img_trans,_ = self.transform(img_load,{'target_size':self.supp_size})
                        
                    support_img_batch.append(torch.from_numpy(img_trans))
                    img_cat = task_cats.index(cat)
                    supp_cls_lab.append(img_cat)

                query_imgs = random.sample(list(self.lvis_sample[cat]),self.num_qry)

                for img_path in query_imgs:
                    cat_idxs = []
                    proj_idxs = []
                    img_cat_ids = []
                    for lv_ix,lv_cat in enumerate(self.lvis_cats[img_path]):
                        if lv_cat in cat_ls:
                            img_cat_ids.append(cat_ls.index(lv_cat))
                            proj_idxs.append(lv_ix)

                    for lv_ix,lv_cat in enumerate(img_cat_ids):
                        if cat_ls[lv_cat] in task_cats: 
                            cat_idxs.append(lv_ix)

                    img_cat_ids = np.array(img_cat_ids)

                    #img_bboxes = np.asarray(self.lvis_bboxes[img_path])[cat_idxs].astype(np.float32)
                    img_bboxes = np.asarray(self.lvis_bboxes[img_path])[proj_idxs].astype(np.float32)
                    img_bboxes[:,2:] = img_bboxes[:,:2]+img_bboxes[:,2:]
                    img_bboxes = np.concatenate([img_bboxes[:,1:2],img_bboxes[:,0:1],img_bboxes[:,3:],img_bboxes[:,2:3]],axis=1)
                    
                    #img_cats = np.asarray(self.lvis_cats[img_path])[cat_idxs]
                    target = {'bbox': img_bboxes, 'cls': img_cat_ids, 'target_size': 640}
                    img_load = Image.open(img_path).convert('RGB')
                    if not val_iter:
                        img_trans,target = self.train_transform(img_load,target,(0.4,1.7))
                        if len(target['bbox']) < len(proj_idxs):
                            new_cat_idxs = []
                            ix_delta = 0
                            for v_ix,val in enumerate(target['valid_indices']):
                                if val:
                                    if v_ix in cat_idxs:
                                        new_cat_idxs.append(v_ix-ix_delta)
                                else:
                                    ix_delta += 1

                            cat_idxs = new_cat_idxs
                    else:
                        img_trans,target = self.transform(img_load,target)

                    query_img_batch.append(torch.from_numpy(img_trans))
                    qry_bbox_ls.append(torch.from_numpy(target['bbox'][cat_idxs]))
                    qry_cls_ls.append(torch.from_numpy(np.ones(len(cat_idxs),dtype=target['cls'].dtype)))

                    proj_bbox_ls.append(torch.from_numpy(target['bbox']))
                    proj_cls_ls.append(torch.from_numpy(target['cls']+1))

            z_ix = 0
            while z_ix < self.num_zero:
                if val_iter:
                    cat = random.sample(self.lvis_val_cats,1)[0]
                else:
                    cat = random.sample(self.lvis_train_cats,1)[0]

                if cat in task_cats: continue

                img_path = random.sample(list(self.lvis_sample[cat]),1)[0]
                target = {'target_size': 640}
                img_load = Image.open(img_path).convert('RGB')
                if not val_iter:
                    img_trans,target = self.train_transform(img_load,target,(0.4,1.7))
                else:
                    img_trans,target = self.transform(img_load,target)

                query_img_batch.append(torch.from_numpy(img_trans))
                qry_bbox_ls.append(torch.from_numpy(np.zeros([0,4]).astype(np.float32)))
                qry_cls_ls.append(torch.from_numpy(np.array([]).astype(np.int64)))

                z_ix += 1

            supp_tup = list(zip(support_img_batch,supp_cls_lab))
            random.shuffle(supp_tup)
            support_img_batch,supp_cls_lab = zip(*supp_tup)
            supp_cls_lab = F.one_hot(torch.LongTensor(supp_cls_lab),num_classes=self.n_way)

            #qry_tup = list(zip(query_img_batch,qry_bbox_ls,qry_cls_ls))
            #random.shuffle(qry_tup)
            #query_img_batch,qry_bbox_ls,qry_cls_ls = zip(*qry_tup)

            q_cls_targets, q_box_targets, q_num_positives = self.anchor_labeler.batch_label_anchors(qry_bbox_ls, qry_cls_ls)
            query_lab_batch = {'cls':qry_cls_ls, 'bbox': qry_bbox_ls, 'cls_anchor':q_cls_targets, 'bbox_anchor':q_box_targets, 'num_positives':q_num_positives}

            p_cls_targets, p_box_targets, p_num_positives = self.anchor_labeler.batch_label_anchors(proj_bbox_ls, proj_cls_ls)
            proj_lab_batch = {'cls':proj_cls_ls, 'bbox': proj_bbox_ls, 'cls_anchor':p_cls_targets, 'bbox_anchor':p_box_targets, 'num_positives':p_num_positives}

            #yield list(map(torch.stack, zip(*support_img_batch))), supp_cls_lab, list(map(torch.stack, zip(*query_img_batch))), query_lab_batch, task_cats, val_iter
            yield torch.stack(support_img_batch), supp_cls_lab, torch.stack(query_img_batch), query_lab_batch, proj_lab_batch, task_cats, val_iter


def load_metadata_dicts(base_path):

    if FLAGS.large_qry:
        qry_img_size = 640
    else:
        qry_img_size = 256

    start = time.time()

    cats_not_to_incl = ['peach','yogurt','crumb','stirrup','hook','zucchini','cherry','pea_(food)']

    lvis_all_cats = {}
    with open(base_path+"LVIS/lvis_train_cats.csv",'r') as fp:
        csv_reader = csv.DictReader(fp)
        for row in csv_reader:
            if row['name'] in cats_not_to_incl: continue
            lvis_all_cats[row['name']] = int(row['image_count'])

    lvis_all_cats = {k: v for k, v in sorted(lvis_all_cats.items(), key=lambda item: item[1])}
    lvis_train_cats = list(lvis_all_cats.keys())[-FLAGS.num_train_cats:]
    lvis_val_cats = list(lvis_all_cats.keys())[-FLAGS.num_train_cats-FLAGS.num_val_cats-len(cats_not_to_incl):-FLAGS.num_train_cats-len(cats_not_to_incl)]

    lvis_cats = {}
    lvis_bboxes = {}
    with open(base_path+'LVIS/lvis_annots.txt','r') as fp: lines = fp.readlines()
    for line in lines:
        splits = line.split(';')
        lvis_cats[splits[0].replace('/home-mscluster/dvanniekerk/',base_path)] = ast.literal_eval(splits[1])
        lvis_bboxes[splits[0].replace('/home-mscluster/dvanniekerk/',base_path)] = ast.literal_eval(splits[2])

    lvis_sample = {}
    added = 0
    not_added = 0
    with open(base_path+'LVIS/lvis_sample.txt','r') as fp: lines = fp.readlines()
    for line in lines:
        splits = line.split(';')
        if splits[0] not in lvis_train_cats and splits[0] not in lvis_val_cats: continue
        cat_imgs = []
        imgs = ast.literal_eval(splits[1])
        for img in set(imgs):
            add_to_sample = True
            if splits[0] in lvis_train_cats:
                set_cats = set(lvis_cats[img.replace('/home-mscluster/dvanniekerk/',base_path)])
                for img_cat in set_cats:
                    if img_cat in lvis_val_cats:
                        add_to_sample = False

            if add_to_sample:
                added += 1
            else:
                not_added += 1
                continue

            cat_imgs.append(img.replace('/home-mscluster/dvanniekerk/',base_path))

        lvis_sample[splits[0]] = cat_imgs
    
    web_sample = {}
    for cat in lvis_sample.keys():
        web_sample[cat] = glob.glob(base_path+"web_images/"+cat.replace('_',' ')+"/*")

    print(len(lvis_sample.keys()))
    print(added)
    print(not_added)

    print(time.time()-start)

    return lvis_sample,web_sample,lvis_bboxes,lvis_cats,lvis_train_cats,lvis_val_cats





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
