import numpy as np
from skimage import transform

import os
import sys
import time

sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import caffe
import base64
import cv2
import csv
import random
import json

from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms

MIN_BOXES = 36
MAX_BOXES = 36

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'features': pool5[keep_boxes]
    }   
  

caffe.set_device(0)
caffe.set_mode_gpu()
net = None
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = 'resnet101_faster_rcnn_final_iter_320000.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)
im_file = 'data/demo/000542.jpg'

result = get_detections_from_im(net, im_file, 0)

print result['features']

np.save('test.npy', result['features'])