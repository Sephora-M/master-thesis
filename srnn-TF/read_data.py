from scipy.io import loadmat
from scipy.spatial import distance
import os, math
import numpy as np

NUM_TEMP_FEATURES=3
NUM_ST_FEATURES=2

def get_splits(splits_path='/local/home/msephora/master-thesis/master-thesis/srnn-TF/data/JHMDB/sub_splits', ind_split = 1):
    train = np.array([],dtype=str)
    test = np.array([],dtype=str)

    ind_split = str(ind_split) + '.txt'
    splits_dic = {1 : train, 2: test}
    all_splits = os.listdir(splits_path)
    for splits in all_splits:
        if splits.strip().split('_')[-1] == ind_split:
            file = open(os.path.join(splits_path,splits))
            for line in file:
                type = line.strip().split(' ')
                name = type[0].split('.')[0]
                if len(name)>30:
                    name = name[0:22]+name[-8:]
                split = int(type[1])
                splits_dic[split] = np.append(splits_dic[split], name)
            file.close()
    return splits_dic

def get_pos_imgs(joint_positions_path='/local/home/msephora/master-thesis/master-thesis/srnn-TF/data/JHMDB/joint_positions',
                 sub_activities=True, validation_proportion=0.20, normalized=False, ind_split=1):
    if sub_activities:
        actions = ['catch', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick',
               'pullup', 'push', 'run', 'shoot_ball',
               'swing_baseball','walk']
    else:
        actions = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick',
               'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand',
               'swing_baseball', 'throw', 'walk', 'wave']

    splits = get_splits(ind_split=ind_split)
    #print(splits)
    train_data = {}
    train_data_size=0
    valid_data = {}
    valid_data_size=0
    for i in range(len(actions)):
        all_videos = os.listdir(os.path.join(joint_positions_path,actions[i]))
        train_pos_imgs = {}
        valid_pos_imgs = {}
        for video in all_videos:
            mat_path = os.path.join(joint_positions_path,actions[i],video,'joint_positions.mat')
            if os.path.isfile(mat_path):
                joint = loadmat(os.path.join(joint_positions_path, actions[i], video,'joint_positions.mat'))
                #print(video)
                if video in splits[1]: #np.random.random() > validation_proportion:
                    train_data_size += 1
                    if normalized:
                        train_pos_imgs[video] = joint['pos_world'][:,[2,1,11,12,13,14],:]
                    else:
                        train_pos_imgs[video] = joint['pos_img'][:,[2,1,11,12,13,14],:]
                elif video in splits[2]:
                    valid_data_size += 1
                    if normalized:
                        valid_pos_imgs[video] = joint['pos_world'][:,[2,1,11,12,13,14],:]
                    else:
                        valid_pos_imgs[video] = joint['pos_img'][:,[2,1,11,12,13,14],:]

        train_data[i] = train_pos_imgs
        valid_data[i] = valid_pos_imgs

    #print("Num videos:")
    #print(train_data_size)
    return train_data, train_data_size, valid_data, valid_data_size


def get_pos_imgsMRS(joint_positions_path='/local/home/msephora/master-thesis/master-thesis/data/MSRAction3D/MSRAction3DSkeleton(20joints)'):
    train_split = ['s01','s03','s05','s07','s09']

    train_data = {}
    train_data_size=0
    valid_data = {}
    valid_data_size=0
    actions = {}
    for i in range(1,21):
        if i<10:
            action_name = 'a0' + str(i)
        else:
            action_name = 'a'  + str(i)
        actions[action_name]=i-1
        train_data[i-1] = {}
        valid_data[i-1] = {}

    all_videos = os.listdir(joint_positions_path)
    for video in all_videos:
        file_path = os.path.join(joint_positions_path,video)
        if os.path.isfile(file_path):
            split = video.split('_')
            action = actions[split[0]]
            if split[1] in train_split:
                train_data[action][video] = read_pos_img(file_path)[:,[19,6,10,9,16,15],:]
                train_data_size += 1
            else:
                valid_data[action][video] = read_pos_img(file_path)[:,[19,6,10,9,16,15],:]
                valid_data_size += 1

    return train_data, train_data_size, valid_data, valid_data_size


def read_pos_img(file_path):
    """
    read the joint locations from a file and return a matrix similar to
    pos_img for the JHMDB data set
    :param file_path: path to the file to read
    :return: a 3D matrix containing the x and y location for each joint in each frame
    """
    file = open(file_path)
    num_frames = sum(1 for line in file)
    num_frames /= 20
    file.close()
    file = open(file_path)
    pos_img = np.zeros((2,20,num_frames))

    for frame in range(num_frames):
        for joint in range(20):
            coors = file.readline().strip().split('  ')
            if len(coors) == 4:
                pos_img[0][joint][frame] = float(coors[0])
                pos_img[1][joint][frame] = float(coors[1])
    file.close()
    return pos_img


def extract_features(all_data, num_video, num_activities, num_considered_frames):
    print("Num videos:")
    print(num_video)
    temp_features_names = ['face-face','belly-belly','rightArm-rightArm','leftArm-leftArm','rightLeg-rightLeg','leftLeg-leftLeg']
    st_features_names = ['face-leftArm','face-rightArm','face-belly','belly-leftArm','belly-rightArm',
                             'belly-rightLeg','belly-leftLeg']

    joints = {'face-face' : 0,'belly-belly' : 1,'rightArm-rightArm' : 2,'leftArm-leftArm' : 3,'rightLeg-rightLeg' : 4,'leftLeg-leftLeg' : 5}

    joints_pairs = {'face-leftArm' : [0,3],'face-rightArm': [0,2],'face-belly' : [0,1],'belly-leftArm': [1,3],'belly-rightArm': [1,2],
                             'belly-rightLeg': [1,4],'belly-leftLeg': [1,5]}
    temp_features = {}


    for name in temp_features_names:
        temp_features[name] = np.empty((num_video,num_considered_frames,NUM_TEMP_FEATURES))
        current_video = 0
        for action_id in all_data:
            for video in all_data[action_id]:
                temp_features[name][current_video,:,:] = extract_ntraj(all_data[action_id][video], joints[name], num_considered_frames)
                current_video += 1
    st_features = {}
    for name in st_features_names:
        st_features[name] = np.empty((num_video,num_considered_frames,NUM_ST_FEATURES))
        current_video = 0
        for action_id in all_data:
            for video in all_data[action_id]:
                st_features[name][current_video,:,:]  = extract_st_features(all_data[action_id][video], joints_pairs[name], num_considered_frames)
                current_video += 1

    action_classes = np.zeros((num_video,num_activities))
    print('Num activities')
    print(num_activities)
    current_video = 0
    for action_id in all_data:
        for _ in all_data[action_id]:
            action_classes[current_video,action_id] = 1
            current_video += 1

    return [temp_features, st_features, action_classes]


def extract_temp_features(pos_img, joint_id, num_frames):
    """Extract the temporal features for a video for a joint
    given N > NUM_FRAMES frames, the features are the positions of the joint at
    NUM_FRAMES frames (linear space in N) as well as the distances between
    consecutive frames
    """
    _ , _ , tot_num_frames = pos_img.shape

    frames_chosen = np.linspace(0, tot_num_frames - 1, num_frames).astype(int)
    temp_features = np.empty((num_frames, 3))

    for i in range(num_frames-1):
        dist = distance.euclidean(pos_img[:,joint_id,frames_chosen[i]], pos_img[:,joint_id,frames_chosen[i+1]])
        temp_features[i,0] = pos_img[0,joint_id,frames_chosen[i]] # position x
        temp_features[i,1] = pos_img[1,joint_id,frames_chosen[i]] # position y
        temp_features[i,2] = dist
    temp_features[num_frames - 1, 0] = pos_img[0, joint_id, frames_chosen[num_frames - 1]] # position x
    temp_features[num_frames - 1, 1] = pos_img[1, joint_id, frames_chosen[num_frames - 1]] # position y
    temp_features[num_frames - 1, 2] = 0.0

    return temp_features


def extract_st_features(pos_img, joints_id, num_frames):
    """
    Extract the spatio temporal features for a video for a pair of joints
    the features is the distance between the two joint at each frame
    :param pos_img: the joint positions info for a video
    :param joints_id: the id of the pair of joints to consider
    :return: the spatio temporal features
    """

    _ , _ , tot_num_frames = pos_img.shape

    frames_chosen = np.linspace(0, tot_num_frames - 1, num_frames).astype(int)
    st_features = np.empty((num_frames, NUM_ST_FEATURES))

    for i in range(num_frames):
        dist = distance.euclidean(pos_img[:,joints_id[0],frames_chosen[i]], pos_img[:,joints_id[1],frames_chosen[i]])
        d = pos_img[:,joints_id[0],frames_chosen[i]] - pos_img[:,joints_id[1],frames_chosen[i]]
        ort = math.atan2(d[1],d[0])*180.0/math.pi
        st_features[i,0] = dist
        st_features[i,1] = ort

    return st_features


def extract_ntraj(pos_img, joint_id, num_frames):

    _ , _ , tot_num_frames = pos_img.shape
    relative_pos = pos_img #normalize_positions(pos_img)
    frames_chosen = np.linspace(0, tot_num_frames - 1, num_frames).astype(int)

    temp_features = np.empty((num_frames, NUM_TEMP_FEATURES))

    for i in range(num_frames-1):
        d = pos_img[:,joint_id,frames_chosen[i+1]] - pos_img[:,joint_id,frames_chosen[i]]
        dist = distance.euclidean(pos_img[:,joint_id,frames_chosen[i+1]], pos_img[:,joint_id,frames_chosen[i]])
        ort = math.atan2(d[1],d[0])*180.0/math.pi
        # dist = distance.euclidean(pos_img[:,joint_id,frames_chosen[i]], pos_img[:,joint_id,frames_chosen[i+1]])
        # temp_features[i,0] = pos_img[0,joint_id,frames_chosen[i]] # position x
        # temp_features[i,1] = pos_img[1,joint_id,frames_chosen[i]] # position y
        # temp_features[i,2] = dist
        temp_features[i,0] = dist #  dx
        #temp_features[i,1] = ort
        temp_features[i,1] = relative_pos[0,joint_id,frames_chosen[i]] #  relative x
        temp_features[i,2] = relative_pos[1,joint_id,frames_chosen[i]] #  relative y

    temp_features[num_frames - 1,0] = 0.0
    #temp_features[num_frames - 1,1] = 0.0
    temp_features[num_frames - 1,1] = relative_pos[0,joint_id,frames_chosen[num_frames - 1]] #  realative x
    temp_features[num_frames - 1,2] = relative_pos[1,joint_id,frames_chosen[num_frames - 1]] #  reative y
    return temp_features

def normalize_positions(pos_img):
    """
    Returns the relative positions of normalized joint positions w.r.t to the puppet center
    :param pos_img:
    :return:
    """
    torso_positions = (pos_img[:,0,:] + pos_img[:,1,:])/2
    torso_positions=np.reshape(torso_positions,[2,1,pos_img.shape[2]])
    return pos_img - np.tile(torso_positions, [1 ,6, 1])

