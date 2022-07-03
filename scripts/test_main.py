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
import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math

tracker = Tracker3D(box_type="Kitti", tracking_features=False) #跟踪器
count = 0

def bboxcallback(data):
    global count
    objects = []
    for i in range(len(data.markers)):
        objects.append([data.markers[i].scale.z, data.markers[i].scale.x, data.markers[i].scale.y, \
                        data.markers[i].pose.position.x, data.markers[i].pose.position.y, data.markers[i].pose.position.z, data.markers[i].pose.orientation.x])
    objects = np.array(objects)
    score = [6] * len(objects)
    if len(objects) == 0:
        return
    bbs,ids = tracker.tracking(objects[:, :7],
                     features=None,
                     scores=score,
                     pose=None,
                     timestamp=count)
    idMarkerArray = MarkerArray()

    for i in range(len(bbs)):
        data.markers[i].id = ids[i]
        data.markers[i].color.g = 1.0

        #显示id
        idMarker = Marker()
        idMarker.header.frame_id = "rslidar"
        idMarker.ns = 'tracking'
        idMarker.type = Marker.TEXT_VIEW_FACING
        idMarker.action = Marker.ADD
        idMarker.scale.x = 2
        idMarker.scale.y = 2
        idMarker.scale.z = 2
        idMarker.pose.position.x = bbs[i][0]
        idMarker.pose.position.y = bbs[i][1]
        idMarker.pose.position.z = bbs[i][2] + 3
        idMarker.color.a = 1.0
        idMarker.color.r = 1.0
        idMarker.color.g = 0.0
        idMarker.color.b = 0.0
        idMarker.id = ids[i] 
        idMarker.lifetime = rospy.Duration(0.2)
        idMarker.text = str(ids[i])

        idMarkerArray.markers.append(idMarker)
    # Publish the MarkerArray
    id_pub.publish(idMarkerArray)
    count += 1

def listener():
    rospy.Subscriber('/bbox_info', MarkerArray, bboxcallback)
    rospy.spin()
    
if __name__ == '__main__':
    # 跟踪节点名
    rospy.init_node('tracking')
    # 发布的跟踪id topic
    topic = '/detection/lidar_detector/objects'
    id_pub = rospy.Publisher('track_id', MarkerArray, queue_size=1)
    listener()
