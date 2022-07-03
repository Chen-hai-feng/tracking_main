# encoding: utf-8
from trajectory import Trajectory
from box_op import *
import numpy as np
def find_martrix_max_value(data_matrix):
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    return max(new_data)
class Tracker3D:
    def __init__(self,tracking_features=False,
                    bb_as_features=False,
                    box_type='Kitti',):
        """
        initialize the the 3D tracker
        Args:
            tracking_features: bool, if tracking the features
            bb_as_features: bool, if tracking the bbs
            box_type: str, box type, available box type "OpenPCDet", "Kitti", "Waymo"
        """
        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type

        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

    def tracking(self,bbs_3D = None,
                 features = None,
                 scores = None,
                 pose = None,
                 timestamp = None
                 ):
        """
        tracking the objects at the given timestamp
        Args:
            bbs: array(N,7) or array(N，7*k), 3D bounding boxes or 3D tracklets
                for tracklets, the boxes should be organized to [[box_t; box_t-1; box_t-2;...],...]
            features: array(N,k), the features of boxes or tracklets
            scores: array(N,), the detection score of boxes or tracklets
            pose: array(4,4), pose matrix to global scene
            timestamp: int, current timestamp, note that the timestamp should be consecutive

        Returns:
            bbs: array(M,7), the tracked bbs
            ids: array(M,), the assigned IDs for bbs
        """
        self.current_bbs = bbs_3D
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp
        self.trajectores_prediction()
        if self.current_bbs is None:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            if len(self.current_bbs) == 0:
                return np.zeros(shape=(0,7)),np.zeros(shape=(0))

            else:
                self.current_bbs = convert_bbs_type(self.current_bbs,self.box_type)
                self.current_bbs = register_bbs(self.current_bbs,self.current_pose)
                ids = self.association()
                bbs,ids = self.trajectories_update_init(ids)
                return np.array(bbs),np.array(ids)



    def trajectores_prediction(self):
        """
        predict the possible state of each active trajectories, if the trajectory is not updated for a while,
        it will be deleted from the active trajectories set, and moved to dead trajectories set
        Returns:

        """
        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                if self.active_trajectories[key].consecutive_missed_num>=12:
                    dead_track_id.append(key)
                    continue
                if len(self.active_trajectories[key])-self.active_trajectories[key].consecutive_missed_num == 1 \
                    and len(self.active_trajectories[key])>= 5 :
                    dead_track_id.append(key)
                self.active_trajectories[key].state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra

    def compute_cost_map(self):
        """
        compute the cost map between detections and predictions
        Returns:
              cost, array(N,M), where N is the number of detections, M is the number of active trajectories
              all_ids, list(M,), the corresponding IDs of active trajectories
        """
        all_ids = []

        all_predictions = []
        all_detections = []

        for key in self.active_trajectories.keys():
            all_ids.append(key)
            state = np.array(self.active_trajectories[key].trajectory[self.current_timestamp].predicted_state)
            state = state.reshape(-1)

            pred_score = np.array([self.active_trajectories[key].trajectory[self.current_timestamp].prediction_score])

            state = np.concatenate([state,pred_score])
            all_predictions.append(state)

        for i in range(len(self.current_bbs)):
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]
            label=1
            new_tra = Trajectory(init_bb=box,
                                 init_features=features,
                                 init_score=score,
                                 init_timestamp=self.current_timestamp,
                                 label=label,
                                 tracking_features=self.tracking_features,
                                 bb_as_features=self.bb_as_features)

            state = new_tra.trajectory[self.current_timestamp].predicted_state
            state = state.reshape(-1)
            all_detections.append(state)

        all_detections = np.array(all_detections)
        all_predictions = np.array(all_predictions)

        det_len = len(all_detections)
        pred_len = len(all_predictions)

        all_detections = all_detections.reshape((det_len,1,-1))
        all_predictions = all_predictions.reshape((1,pred_len,-1))

        all_detections = np.tile(all_detections,(1,pred_len,1))
        all_predictions = np.tile(all_predictions,(det_len,1,1))

        dis = (all_detections[...,0:3]-all_predictions[...,0:3])**2
        dis = np.sqrt(dis.sum(-1))
        #print(find_martrix_max_value(dis))
        #print('*'*30)
        cost = dis/all_predictions[...,-1]
        #print(find_martrix_max_value(cost))
        #print('*'*30)
        return cost,all_ids

    def association(self):
        """
        greedy assign the IDs for detected state based on the cost map
        Returns:
            ids, list(N,), assigned IDs for boxes, where N is the input boxes number
        """
        if len(self.active_trajectories) == 0:
            ids = []
            for i in range(len(self.current_bbs)):
                ids.append(self.label_seed)
                self.label_seed+=1
            return ids
        else:
            cos_motion, all_ids = self.compute_cost_map() #计算代价矩阵得到代价矩阵和所有的活跃ID #检测*轨迹
            ids = self.association_single(cos_motion,all_ids)
            return ids
    def association_single(self,cos_motion,all_ids):
        """
        :param cos_motion:
        :param cos_apperance:
        :param all_ids: 所有轨迹的ID
        :return:
        """
        import matching
        ids = []
        matches_motion, u_detection_motion, u_track_motion = matching.linear_assignment(
            cos_motion, thresh=10
        )
        if(len(matches_motion)==0):
            for i in range(len(self.current_bbs)): #对于每一个检测的特征 #N*M
                ids.append(self.label_seed)
                self.label_seed += 1
        else:
            match_motion_det = list(matches_motion[:, 0])
            match_motion_track = list(matches_motion[:, 1])
            for i in range(len(self.current_bbs)): #对于每一个检测的特征 #N*M
                if i in match_motion_det:
                    ids.append(all_ids[match_motion_track[match_motion_det.index(i)]])
                else:
                    ids.append(self.label_seed)
                    self.label_seed+=1
        return ids

    def trajectories_update_init(self,ids):
        """
        update a exiting trajectories based on the association results, or init a new trajectory
        Args:
            ids: list or array(N), the assigned ids for boxes
        """
        assert len(ids) == len(self.current_bbs)

        valid_bbs = []
        valid_ids = []

        for i in range(len(self.current_bbs)):
            label = ids[i]
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]

            if label in self.active_trajectories.keys() and score>5:
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp)
                valid_bbs.append(box)
                valid_ids.append(label)
            elif score>5:
                new_tra = Trajectory(init_bb=box,
                                     init_features=features,
                                     init_score=score,
                                     init_timestamp=self.current_timestamp,
                                     label=label,
                                     tracking_features=self.tracking_features,
                                     bb_as_features=self.bb_as_features)
                self.active_trajectories[label] = new_tra
                valid_bbs.append(box)
                valid_ids.append(label)
            else:
                continue
        if len(valid_bbs)==0:
            return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            return np.array(valid_bbs),np.array(valid_ids)


    def post_processing(self):
        """
        globally filter the trajectories
        Args:

        Returns: dict(Trajectory)

        """
        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            track.filtering()
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            track.filtering()
            tra[key] = track

        return tra




