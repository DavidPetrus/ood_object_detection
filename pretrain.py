import torch

import numpy as np
#import cv2
from omegaconf import OmegaConf
import datetime

from timm.models import load_checkpoint
from effdet.efficientdet import EfficientDet, AnchorNet
from effdet.helpers import load_pretrained
from effdet.config.model_config import default_detection_model_configs
from effdet.distributed import all_gather_container
from effdet.bench import _post_process
from effdet.anchors import Anchors, AnchorLabeler, generate_detections
from effdet.loss import DetectionLoss, SupportLoss
from effdet.evaluation.detection_evaluator import ObjectDetectionEvaluator

from collections import OrderedDict, defaultdict

from absl import flags
from absl import app

import higher

import wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('exp','','')
flags.DEFINE_bool('ubuntu',True,'')

flags.DEFINE_integer('log_freq',50,'')
flags.DEFINE_integer('num_workers',16,'')
flags.DEFINE_integer('max_lvis_load',4000,'')
flags.DEFINE_integer('num_preloads',5,'')
flags.DEFINE_integer('num_preload_cats',100,'')
flags.DEFINE_integer('num_train_cats',250,'')
flags.DEFINE_integer('num_val_cats',50,'')
flags.DEFINE_integer('num_val_imgs',20,'')
flags.DEFINE_integer('val_freq',100,'')
flags.DEFINE_integer('n_way',2,'')
flags.DEFINE_integer('num_sup',25,'')
flags.DEFINE_integer('num_qry',10,'')
flags.DEFINE_integer('meta_batch_size',2,'')
flags.DEFINE_integer('img_size',256,'')

flags.DEFINE_string('model','d0','')
flags.DEFINE_string('bb','b0','')
flags.DEFINE_string('optim','adam','')
flags.DEFINE_bool('freeze_bb_bn',True,'')
flags.DEFINE_bool('train_bb',True,'')
flags.DEFINE_bool('train_fpn',True,'')
flags.DEFINE_bool('fpn',True,'')
flags.DEFINE_bool('large_qry',True,'')
flags.DEFINE_bool('train_mode',True,'')
flags.DEFINE_float('meta_lr',0.001,'')
flags.DEFINE_float('inner_lr',0.003,'')
flags.DEFINE_integer('steps',1,'')
flags.DEFINE_float('gamma',0.,'')
flags.DEFINE_float('bbox_coeff',50.,'')
flags.DEFINE_float('alpha',0.15,'')
flags.DEFINE_integer('supp_level_offset',2,'')




def main(argv):

    from preloader import PretrainDataset, load_metadata_dicts

    wandb.init(project="domain_generalization",name=FLAGS.exp)
    wandb.save("infer.py")
    wandb.save("effdet/efficientdet.py")
    wandb.save("effdet/loss.py")
    wandb.save("dataloader.py")
    wandb.config.update(flags.FLAGS)

    qry_img_size = 640 if FLAGS.large_qry else 256

    if FLAGS.model == 'd0':
        model_name = 'efficientdet_d0'
        bb_name = 'efficientnet_b0'
        load_ckpt = 'efficientdet_d0-f3276ba8.pth'
        config=dict(
            name=model_name,
            backbone_name=bb_name,
            image_size=(qry_img_size, qry_img_size),
            fpn_channels=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
            pad_type='',
            redundant_bias=False,
            backbone_args=dict(drop_path_rate=0.),# checkpoint_path="tf_efficientnet_b1_ns-99dd0c41.pth"),
            #url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth',
        )
    elif FLAGS.model == 'd1':
        model_name = 'efficientdet_d1'
        bb_name = 'efficientnet_b1'
        load_ckpt = 'efficientdet_d1-bb7e98fe.pth'
        config=dict(
            name='efficientdet_d1',
            backbone_name='efficientnet_b1',
            image_size=(640, 640),
            fpn_channels=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
            pad_type='',
            redundant_bias=False,
            backbone_args=dict(drop_path_rate=0.),
            #url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth',
        )
    elif FLAGS.model == 'd3':
        model_name = 'tf_efficientdet_d3'
        bb_name = 'tf_efficientnet_b3'
        load_ckpt = 'tf_efficientdet_d3_47-0b525f35.pth'
        config=dict(
            name='tf_efficientdet_d3',
            backbone_name='tf_efficientnet_b3',
            image_size=(640, 640),
            fpn_channels=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
            backbone_args=dict(drop_path_rate=0.),
            #url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth',
        )

    

    config = OmegaConf.create(config)

    h = default_detection_model_configs()
    h.update(config)
    h.num_levels = h.max_level - h.min_level + 1

    # create the base model
    model = EfficientDet(h)
    state_dict = torch.load(load_ckpt)
    load_state_dict = state_dict
    model.load_state_dict(load_state_dict, strict=True)
    #load_checkpoint(model,"efficientdet_d0-f3276ba8.pth")
    model.reset_head(num_classes=FLAGS.num_train_cats)

    model_config = model.config
    print(model_config['num_classes'])
    num_anchs = int(len(model_config.aspect_ratios) * model_config.num_scales)

    anchors = Anchors.from_config(model_config).to('cuda')

    lvis_sample,lvis_val_sample,lvis_bboxes,lvis_cats,lvis_train_cats,lvis_val_cats,eval_cats = load_metadata_dicts()
    cat_ls = [ec['name'] for ec in eval_cats]
    dataset = PretrainDataset(model_config,FLAGS.n_way,FLAGS.num_sup,FLAGS.num_qry,lvis_sample,lvis_val_sample,lvis_bboxes,lvis_cats,lvis_train_cats,lvis_val_cats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=FLAGS.num_workers, pin_memory=True)

    loss_fn = DetectionLoss(model_config)


    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    imagenet_mean = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_MEAN],device=torch.device('cuda')).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_STD],device=torch.device('cuda')).view(1, 3, 1, 1)
    model.to('cuda')

    if not FLAGS.train_mode:
        model.eval()
    elif FLAGS.freeze_bb_bn:
        def set_bn_eval(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
            
        model.apply(set_bn_eval)

    #if FLAGS.fpn:
    if FLAGS.optim == 'adam':
        meta_optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=FLAGS.meta_lr)
    elif FLAGS.optim == 'nesterov':
        meta_optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=FLAGS.meta_lr, momentum=0.9, nesterov=True)
    #else:
    #    meta_optimizer = torch.optim.Adam([{'params': model.class_net.parameters()},{'params': model.box_net.parameters()}, \
    #        {'params': anchor_net.parameters()}], lr=FLAGS.meta_lr)
    #meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #wandb.watch(model)


    evaluator = ObjectDetectionEvaluator(eval_cats, evaluate_corlocs=True)
    category_metrics = defaultdict(list)

    iter_metrics = {'qry_loss': 0., 'qry_class_loss': 0., 'qry_bbox_loss': 0., 'mAP': 0., 'CorLoc': 0.}
    val_metrics = {'val_qry_loss': 0., 'val_qry_class_loss': 0., 'val_qry_bbox_loss': 0., 'val_mAP': 0., 'val_CorLoc': 0.}
    t_ix = 0
    train_iter = 0
    log_count = 0
    prev_val_iter = False
    meta_norm = 0.
    for task in loader:
        meta_optimizer.zero_grad()

        evaluator.clear()
        
        qry_imgs, qry_labs, val_iter = task

        qry_cls_anchors = [cls_anchor.to('cuda:0') for cls_anchor in qry_labs['cls_anchor']]
        qry_bbox_anchors = [bbox_anchor.to('cuda:0') for bbox_anchor in qry_labs['bbox_anchor']]
        qry_num_positives = qry_labs['num_positives'].to('cuda:0')
        qry_imgs = (qry_imgs.to('cuda').float()-imagenet_mean)/imagenet_std

        with torch.set_grad_enabled(FLAGS.train_bb and not val_iter):
            feats= model(qry_imgs, mode='bb')

        with torch.set_grad_enabled(not val_iter):
            class_out,box_out = model(feats, mode='fpn_and_head')
            qry_loss, qry_class_loss, qry_box_loss = loss_fn(class_out, box_out, qry_cls_anchors, qry_bbox_anchors, qry_num_positives)


        with torch.no_grad():
            class_out_post, box_out_post, indices, classes = _post_process(class_out, box_out, num_levels=model_config.num_levels, 
                num_classes=model_config.num_classes, max_detection_points=model_config.max_detection_points)

            for b_ix in range(FLAGS.num_qry):
                detections = generate_detections(class_out_post[b_ix], box_out_post[b_ix], anchors.boxes, indices[b_ix], classes[b_ix],
                    None, qry_img_size, max_det_per_image=100, soft_nms=False).cpu().numpy()
                evaluator.add_single_ground_truth_image_info(b_ix,{'bbox': qry_labs['bbox'][b_ix].numpy(), 'cls': qry_labs['cls'][b_ix].numpy()})
                bboxes_yxyx = np.concatenate([detections[:,1:2],detections[:,0:1],detections[:,3:4],detections[:,2:3]],axis=1)
                evaluator.add_single_detected_image_info(b_ix,{'bbox': bboxes_yxyx, 'scores': detections[:,4], 'cls': detections[:,5]})

            if not val_iter: log_count += 1
            '''
            map_metrics = evaluator.evaluate([])
            if not val_iter:
                iter_metrics['qry_loss'] += qry_loss
                iter_metrics['qry_class_loss'] += qry_class_loss
                iter_metrics['qry_bbox_loss'] += qry_box_loss
                iter_metrics['mAP'] += map_metrics['Precision/mAP@0.5IOU']
                iter_metrics['CorLoc'] += map_metrics['Precision/meanCorLoc@0.5IOU']
                for metric_key in list(map_metrics.keys())[2:]:
                    category_metrics[metric_key].append(map_metrics[metric_key])
                log_count += 1
            else:
                val_metrics['val_qry_loss'] += qry_loss
                val_metrics['val_qry_class_loss'] += qry_class_loss
                val_metrics['val_qry_bbox_loss'] += qry_box_loss
                val_metrics['val_mAP'] += map_metrics['Precision/mAP@0.5IOU']
                val_metrics['val_CorLoc'] += map_metrics['Precision/meanCorLoc@0.5IOU']
                for metric_key in list(map_metrics.keys())[2:]:
                    category_metrics['val'+metric_key].append(map_metrics[metric_key])'''

        if not val_iter:
            iter_meta_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            meta_norm += iter_meta_norm
            print("Meta Norm:",iter_meta_norm)
            #torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            meta_optimizer.step()
            train_iter += 1

        if (not val_iter and prev_val_iter):
            log_metrics = {'iteration':train_iter}
            for met_key in val_metrics.keys():
                log_metrics[met_key] = val_metrics[met_key]/FLAGS.num_val_cats
            wandb.log(log_metrics)
            print("Validation Iteration {}:".format(train_iter),log_metrics)

            for key in category_metrics:
                if len(category_metrics[key]) > 0 and 'val' in key:
                    print(key,sum(category_metrics[key])/len(category_metrics[key]))
                    np.save('per_cat_metrics/'+FLAGS.exp+key.replace('/','_')+str(train_iter)+'.npy',np.array(category_metrics[key]))
                    category_metrics[key] = []

            val_metrics = {'val_supp_class_loss': 0., 'val_qry_loss': 0., 'val_qry_class_loss': 0., 'val_qry_bbox_loss': 0., 'val_mAP': 0., 'val_CorLoc': 0.}
        elif not val_iter and (log_count >= FLAGS.log_freq):
            log_metrics = {'iteration':train_iter}
            for met_key in iter_metrics.keys():
                log_metrics[met_key] = iter_metrics[met_key]/log_count
            log_metrics['meta_norm'] = meta_norm/(log_count/FLAGS.meta_batch_size)
            meta_norm = 0.
            wandb.log(log_metrics)
            print("Train iteration {}:".format(train_iter),log_metrics)

            for key in category_metrics:
                if len(category_metrics[key]) > 0 and 'val' not in key:
                    print(key,sum(category_metrics[key])/len(category_metrics[key]))
                    np.save('per_cat_metrics/'+FLAGS.exp+key.replace('/','_')+str(train_iter)+'.npy',np.array(category_metrics[key]))
                    category_metrics[key] = []

            iter_metrics = {'supp_class_loss': 0.,'qry_loss': 0., 'qry_class_loss': 0., 'qry_bbox_loss': 0., 'mAP': 0., 'CorLoc': 0.}
            
            log_count = 0

        prev_val_iter = val_iter

        #except Exception as e:
        #    print("!!!!!!!!!!!!!",e)
        #    with open(FLAGS.exp+'train.txt','a') as fp:
        #        fp.write(str(e))
        #    continue


        #for pars in model.named_parameters():
        #    if pars[1].grad is None:
        #        continue
        #    print(pars[0],pars[1].grad.abs().min())



    '''for supp_ix in range(FLAGS.n_way*FLAGS.num_sup):
            img_show = np.moveaxis(supp_imgs[0,supp_ix].int().numpy(), 0, 2).astype(np.uint8)
            img_show = cv2.cvtColor(img_show,cv2.COLOR_BGR2RGB)
            for annot in supp_labs['bbox'][supp_ix][0].numpy():
                bbox = annot.astype(np.int32)
                cv2.rectangle(img_show,(bbox[1],bbox[0]),(bbox[3],bbox[2]),(255,0,0),2)

            cv2.imshow('a',img_show)
            cv2.waitKey(0)

        for qry_ix in range(FLAGS.n_way*NUM_QRY):
            img_show = np.moveaxis(qry_imgs[0,qry_ix].int().numpy(), 0, 2).astype(np.uint8)
            img_show = cv2.cvtColor(img_show,cv2.COLOR_BGR2RGB)
            for annot in qry_labs['bbox'][qry_ix][0].numpy():
                bbox = annot.astype(np.int32)
                cv2.rectangle(img_show,(bbox[1],bbox[0]),(bbox[3],bbox[2]),(255,0,0),2)

            cv2.imshow('a',img_show)
            cv2.waitKey(0)'''



    '''class_outs, box_outs = [[],[],[],[],[]], [[],[],[],[],[]]
        for supp_idx in range(0,len(supp_imgs),BATCH_SIZE):
            class_out, box_out = model(supp_imgs[supp_idx:supp_idx+BATCH_SIZE])
            #class_out, box_out, indices, classes = _post_process(class_out, box_out, num_levels=model_config.num_levels, 
            #    num_classes=model_config.num_classes, max_detection_points=model_config.max_detection_points)
            #detections = generate_detections(class_out[0], box_out[0], anchors.boxes, indices[0], classes[0],
            #    None, IMG_SIZE, max_det_per_image=100, soft_nms=False)
            for l,cls_l,box_l in zip(range(5),class_out,box_out):
                class_outs[l].append(cls_l)
                box_outs[l].append(box_l)
            print(supp_idx)

        for l in range(5):
            class_outs[l] = torch.cat(class_outs[l],dim=1)
            box_outs[l] = torch.cat(box_outs[l],dim=1)
            print(class_outs[l].shape)
            print(box_outs[l].shape)'''






    '''img_show = np.moveaxis(supp_imgs[0].int().numpy(), 0, 2).astype(np.uint8)
        img_show = cv2.cvtColor(img_show,cv2.COLOR_BGR2RGB)
        for annot in supp_labs[0]['bbox'][0].numpy():
            print(annot)
            bbox = annot.astype(np.int32)
            cv2.rectangle(img_show,tuple(bbox[:2]),tuple(bbox[2:]),(255,0,0),2)

        cv2.imshow('a',img_show)
        cv2.waitKey(0)'''



    '''
    amp_autocast = torch.cuda.amp.autocast
    scaler = torch.cuda.amp.GradScaler()
    def unscale_grad_list(grads, get_scale=scaler._get_scale_async):
        inv_scale = 1./get_scale()
        print(inv_scale)
        grad_params = [p * inv_scale for p in grads]
        for gr_ix in range(len(grad_params)):
            print(grad_params[gr_ix].abs().min())
            grad_params[gr_ix][torch.isnan(grad_params[gr_ix])] = 0.
            grad_params[gr_ix][torch.isinf(grad_params[gr_ix])] = 0.
            
        return grad_params
    '''

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)