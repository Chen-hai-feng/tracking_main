#!/usr/bin/env python
# encoding: utf-8
from tracker import Tracker3D
import os
import numpy as np
import time
from detdata import KittiTrackingDataset
from vision import *
from kitti_data_base import *
from box_op import *
import cv2
type = 'Car'
seq_id = 3

detections_path = '/home/ran/project/rospy_test/src/talker/scripts/data/point_rcnn/training/' #检测文件(h,w,l,x,y,z,yaw) format
#calib_path = './'+str(seq_id).zfill(4)+'.txt' #相机参数
calib_path = '/home/ran/project/rospy_test/src/talker/scripts/0003.txt'#相机参数
tracker = Tracker3D(box_type="Kitti", tracking_features=False) #跟踪器
dataset = KittiTrackingDataset(seq_id=seq_id, ob_path=detections_path, calib_path=calib_path, type=type)
all_time = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./'+'{}.mp4'.format(str(seq_id).zfill(4)),fourcc, 10.0,(1242,375))

for i in range(len(dataset)):
    P2, V2C,objects, det_scores, pose = dataset[i] #pose是None不用管
    mask = det_scores > 0 #检测得分大于0的
    objects = objects[mask] #三维(h,w,l,x,y,z,yaw)
    det_scores = det_scores[mask]
    #print('object')
    #print(objects)
    start = time.time()
    a1 = time.time()
    bbs,ids = tracker.tracking(objects[:, :7],
                     features=None,
                     scores=det_scores,
                     pose=pose,
                     timestamp=i)
    a2 = time.time()
    print('time:{}'.format(a2-a1))
    ddd_boxes = np.zeros(shape=(bbs.shape))
    #print('box3d')
    #print(box3d)
    bbs[:, 6] = -bbs[:, 6] - np.pi / 2
    bbs[:, 2] -= bbs[:, 5] / 2
    bbs[:, 0:3] = velo_to_cam(bbs[:, 0:3], V2C)[:, 0:3]
    box2ds= []
    for bb in bbs:
        box2d = bb3d_2_bb2d(bb, P2)
        box2ds.append(box2d)
    box2ds = np.array(box2ds).reshape(-1,4)
    #print(box2ds)
    #将输出bbox的顺序改成hwlxyzry
    ddd_boxes[:,0] = bbs[:,5] #h
    ddd_boxes[:,1] = bbs[:,4] #w
    ddd_boxes[:,2] = bbs[:,3] #l
    ddd_boxes[:,3] = bbs[:,0] #x
    ddd_boxes[:,4] = bbs[:,1] #y
    ddd_boxes[:,5] = bbs[:,2] #z
    ddd_boxes[:,6] = bbs[:,6] #ry
    end = time.time()
    all_time += end - start
    online_im = vison(ddd_boxes=ddd_boxes,dd_bboxes=box2ds,ids=ids,seq=seq_id,index=i) #可视化
    cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
    cv2.imshow('result', online_im)
    out.write(online_im)
    cv2.waitKey(0)

