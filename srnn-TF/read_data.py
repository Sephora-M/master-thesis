from scipy.io import loadmat
from scipy.spatial import distance
import os
import numpy as np

NUM_VIDEOS = 927
NUM_FRAMES = 10
NUM_ACTIVITIES = 21


def get_pos_imgs(joint_positions_path='/local/home/msephora/master-thesis/master-thesis/data/annotations/JHMDB/joint_positions'):
    actions = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick',
               'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand',
               'swing_baseball', 'throw', 'walk', 'wave']
    all_data = {}
    num_video=0
    for i in range(len(actions)):
        all_videos = os.listdir(os.path.join(joint_positions_path,actions[i]))
        pos_imgs = {}
        for video in all_videos:
            mat_path = os.path.join(joint_positions_path,actions[i],video,'joint_positions.mat')
            if os.path.isfile(mat_path):
                num_video += 1
                joint = loadmat(os.path.join(joint_positions_path, actions[i], video,'joint_positions.mat'))
                pos_imgs[video] = joint['pos_img'][:,[2,1,11,12,13,14],:]
        all_data[i] = pos_imgs

    return all_data, num_video


def extract_features(all_data, num_considered_frames=NUM_FRAMES):
    temp_features_names = ['face-face','belly-belly','rightArm-rightArm','leftArm-leftArm','rightLeg-rightLeg','leftLeg-leftLeg']
    st_features_names = ['face-leftArm','face-rightArm','face-belly','belly-leftArm','belly-rightArm',
                             'belly-rightLeg','belly-leftLeg']

    joints = {'face-face' : 0,'belly-belly' : 1,'rightArm-rightArm' : 2,'leftArm-leftArm' : 3,'rightLeg-rightLeg' : 4,'leftLeg-leftLeg' : 5}

    joints_pairs = {'face-leftArm' : [0,3],'face-rightArm': [0,2],'face-belly' : [0,1],'belly-leftArm': [1,3],'belly-rightArm': [1,2],
                             'belly-rightLeg': [1,4],'belly-leftLeg': [1,5]}
    temp_features = {}


    for name in temp_features_names:
        temp_features[name] = np.empty((NUM_VIDEOS,num_considered_frames-1,3))
        current_video = 0
        for action_id in all_data:
            for video in all_data[action_id]:
                temp_features[name][current_video,:,:] = extract_temp_features(all_data[action_id][video], joints[name])
                current_video += 1
    st_features = {}
    for name in st_features_names:
        st_features[name] = np.empty((NUM_VIDEOS,num_considered_frames,1))
        current_video = 0
        for action_id in all_data:
            for video in all_data[action_id]:
                st_features[name][current_video,:,:]  = extract_st_features(all_data[action_id][video], joints_pairs[name])
                current_video += 1

    action_classes = np.zeros((NUM_VIDEOS,NUM_ACTIVITIES))
    current_video = 0
    for action_id in all_data:
        for _ in all_data[action_id]:
            action_classes[current_video,action_id] = 1
            current_video += 1

    return [temp_features, st_features, action_classes]

def extract_temp_features(pos_img, joint_id):
    """Extract the temporal features for a video for a joint
    given N > NUM_FRAMES frames, the features are the positions of the joint at
    NUM_FRAMES frames (linear space in N) as well as the distances between
    consecutive frames
    """
    _ , _ , num_frames = pos_img.shape

    frames_chosen = np.linspace(0,num_frames-1,NUM_FRAMES).astype(int)
    temp_features = np.empty((NUM_FRAMES-1,3))

    for i in range(NUM_FRAMES-1):
        dist = distance.euclidean(pos_img[:,joint_id,frames_chosen[i]], pos_img[:,joint_id,frames_chosen[i+1]])
        temp_features[i,0] = pos_img[0,joint_id,i] # position x
        temp_features[i,1] = pos_img[1,joint_id,i] # position y
        temp_features[i,2] = dist

    return temp_features


def extract_st_features(pos_img, joints_id):
    """
    Extract the spatio temporal features for a video for a pair of joints
    the features is the distance between the two joint at each frame
    :param pos_img: the joint positions info for a video
    :param joints_id: the id of the pair of joints to consider
    :return: the spatio temporal features
    """

    _ , _ , num_frames = pos_img.shape

    frames_chosen = np.linspace(0,num_frames-1,NUM_FRAMES).astype(int)
    st_features = np.empty((NUM_FRAMES,1))

    for i in range(NUM_FRAMES):
        dist = distance.euclidean(pos_img[:,joints_id[0],frames_chosen[i]], pos_img[:,joints_id[1],frames_chosen[i]])
        st_features[i,0] = dist

    return st_features


