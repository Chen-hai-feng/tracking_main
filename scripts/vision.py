# encoding: utf-8
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
img_dir = '/home/ran/project/rospy_test/src/talker/scripts/' #跟踪图像的文件夹
calib_dir = '/home/ran/project/rospy_test/src/talker/scripts/'  #跟踪相机参数
def vison(ddd_boxes,dd_bboxes,ids,seq,index):#输入序列号和对应的序列图像大小
    img_path_seq = img_dir+'/{}'.format(str(seq).zfill(4))
    calib_path_seq = calib_dir+'/{}.txt'.format(str(seq).zfill(4))
    calib = read_clib(calib_path_seq)
    idss = sorted(np.arange(9999))
    colors = get_color(idss)
    img_path_frame = img_path_seq+'/{}.png'.format(str(index).zfill(6))
    img0 = cv2.imread(img_path_frame)
    online_im = plot_tracking_ddd(
        img0,
        online_tlrbs=dd_bboxes ,
        ddd_boxes = ddd_boxes,
        calib=calib,
        online_ids = ids ,
        frame_id=int(index),
        colors=colors,
    )
    return online_im

def generate_objects_color_map(color_map_name='rainbow'):
    """
    generate a list of random colors based on the specified color map name.
    reference  https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :param color_map_name: (str), the name of objects color map, such as "rainbow", "viridis","brg","gnuplot","hsv"
    :return: (list), a list of random colors
    """
    color_map = []
    np.random.seed(4)

    x = 0
    for i in range(10000):
        if x > 1:
            x = np.random.random() * 0.5
        color_map.append(x)
        x += 0.2
    cmp = plt.get_cmap(color_map_name)
    color_map = cmp(color_map)
    color_map = color_map[:, 0:3] * 255
    color_map = color_map.astype(np.int).tolist()
    return color_map

def generate_objects_colors(object_ids,color_map_list):
    """
    map the object indices into colors
    :param object_ids: (array or list(N,)), object indices
    :param color_map_list: (list(K,3)), color map list
    :return: (list(N,3)), a list of colors
    """
    assert len(color_map_list)>len(object_ids), "the color map list must longer than object indices list !"

    if len(object_ids)==0:
        return []
    else:
        colors={}
        for i in object_ids:
            colors[i] = (color_map_list[i])
        return colors

def get_color(num):
    colormap = generate_objects_color_map()
    colors = generate_objects_colors(num,colormap)
    return colors
def get_files(path,rule):
    all = []
    for fpathe, dirs, fs in os.walk(path):  # os.walk获取所有的目录
        for f in fs:
            if f.endswith(rule):  # 判断是否是".sfx"结尾
                all.append(f.split('.')[0])
    return sorted(all)
def read_clib(calib_path):
    f = open(calib_path, "r")
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line.strip().split(" ")[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
def plot_tracking_ddd(
    image,
    online_tlrbs,
    ddd_boxes,
    online_ids,
    scores=None,
    frame_id=0,
    fps=0.0,
    ids2=None,
    calib=None,
    colors=None
):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.0)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.0))
    line_thickness = 1
    radius = max(5, int(im_w / 140.0))
    cv2.putText(
        im,
        "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(ddd_boxes)),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    for i, box3d in enumerate(ddd_boxes):
        tlrb = online_tlrbs[i]
        x1, y1, x2, y2 = tlrb
        intbox = tuple(map(int, (x1, y1, x2, y2)))

        obj_id = int(online_ids[i])
        id_text = "{}".format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = colors[obj_id]
        dim = box3d[:3]
        loc = box3d[3:-1]
        rot = box3d[-1]
        box_3d = compute_box_3d(dim, loc, rot)
        box_2d = project_to_image(box_3d, calib)
        im = draw_box_3d(im, box_2d, c=color, same_color=True)
        cv2.putText(
            im,
            id_text,
            (intbox[0]+30, intbox[1]),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            color,
            thickness=2,
        )

    return im
def plot_tracking(image, online_tlrbs, online_ids, scores=None, frame_id=0, fps=0.0, ids2=None,colors=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    #print(im_h, im_w)  ###375 1242
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.0)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.0))
    line_thickness = 4
    radius = max(5, int(im_w / 140.0))
    cv2.putText(
        im,
        "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(online_tlrbs)),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )
    for i, tlrb in enumerate(online_tlrbs):
        x1, y1, x2, y2 = tlrb
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        obj_id = int(online_ids[i])
        id_text = "{}".format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        _line_thickness = 10 if obj_id <= 0 else line_thickness
        color = colors[obj_id]
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            id_text,
            (intbox[0]+30, intbox[1]),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            color,
            thickness=3,
        )
    return im


