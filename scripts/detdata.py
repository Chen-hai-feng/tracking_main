import numpy as np
import re
from kitti_data_base import *
import os


class KittiTrackingDataset:
    def __init__(self,seq_id,ob_path = None,calib_path = None,type=["Car"]):
        self.seq_name = str(seq_id).zfill(4)
        self.calib_path = calib_path
        self.type = type
        self.ob_path = ob_path+'/'+self.seq_name+'/'
        self.all_ids = os.listdir(self.ob_path)
        self.P2, self.V2C = read_calib(calib_path)



    def __len__(self):
        return len(self.all_ids)-1
    def __getitem__(self, item):

        name = str(item).zfill(6)

        pose = None

        if self.ob_path is not None:
            ob_path = os.path.join(self.ob_path, name + '.txt')
            if not os.path.exists(ob_path):
                objects = np.zeros(shape=(0, 7))
                det_scores = np.zeros(shape=(0,))
            else:
                objects_list = []
                det_scores = []
                with open(ob_path) as f:
                    for each_ob in f.readlines():
                        infos = re.split(' ', each_ob)
                        if infos[0] in self.type:
                            objects_list.append(infos[8:15])
                            det_scores.append(infos[15])
                if len(objects_list)!=0:
                    objects = np.array(objects_list,np.float32)
                    objects[:, 3:6] = cam_to_velo(objects[:, 3:6], self.V2C)[:, :3]
                    det_scores = np.array(det_scores,np.float32)
                else:
                    objects = np.zeros(shape=(0, 7))
                    det_scores = np.zeros(shape=(0,))
        else:
            objects = np.zeros(shape=(0,7))
            det_scores = np.zeros(shape=(0,))

        return self.P2,self.V2C,objects,det_scores,pose
