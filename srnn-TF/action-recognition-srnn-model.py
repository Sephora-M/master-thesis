import tensorflow as tf
import numpy as np
import sys
import os
import cPickle

def jointModel(num_sub_activities, num_affordances, inputJointFeatures,
               inputHumanFeatures, inputObjectFeatures):

    shared_input_layer = TemporalInputFeatures(inputJointFeatures)
    shared_hidden_layer = tf.nn.rnn_cell.LSTMCell(128)

    shared_layers = [shared_input_layer, shared_hidden_layer]

    human_layers = [ConcatenateFeatures(inputHumanFeatures), tf.nn.rnn_cell.LSTMCell(256),
                    'softmax']

    object_layers = [ConcatenateFeatures(inputObjectFeatures), tf.nn.rnn_cell.LSTMCell(256),
                     'softmax']

    trY_1 = T.lmatrix()
    trY_2 = T.lmatrix()
    sharedrnn = SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)
    return sharedrnn

    return None

if __name__ == '__main__':
    index = sys.argv[1]
    fold = sys.argv[2]

    main_path = '/home/saiwen/sephora/master-thesis/activity-anticipation/'

    path_to_dataset = '{1}/dataset/{0}'.format(fold, main_path)
    path_to_checkpoints = '{1}/checkpoints/{0}'.format(fold, main_path)

    if not os.path.exists(path_to_checkpoints):
        os.mkdir(path_to_checkpoints)

    test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index, path_to_dataset)))
    Y_te_human = test_data['labels_human']
    Y_te_human_anticipation = test_data['labels_human_anticipation']
    X_te_human_disjoint = test_data['features_human_disjoint']
    X_te_human_shared = test_data['features_human_shared']

    train_data = cPickle.load(open('{1}/train_data_{0}.pik'.format(index, path_to_dataset)))
    Y_tr_human = train_data['labels_human']
    Y_tr_human_anticipation = train_data['labels_human_anticipation']
    X_tr_human_disjoint = train_data['features_human_disjoint']
    X_tr_human_shared = train_data['features_human_shared']

    Y_tr_objects = train_data['labels_objects']
    Y_tr_objects_anticipation = train_data['labels_objects_anticipation']
    X_tr_objects_disjoint = train_data['features_objects_disjoint']
    X_tr_objects_shared = train_data['features_objects_shared']

    num_sub_activities = int(np.max(Y_tr_human) - np.min(Y_tr_human) + 1)
    num_affordances = int(np.max(Y_tr_objects) - np.min(Y_tr_objects) + 1)
    num_sub_activities_anticipation = int(np.max(Y_tr_human_anticipation) - np.min(Y_tr_human_anticipation) + 1)
    num_affordances_anticipation = int(np.max(Y_tr_objects_anticipation) - np.min(Y_tr_objects_anticipation) + 1)
    inputJointFeatures = X_tr_human_shared.shape[2]
    inputHumanFeatures = X_tr_human_disjoint.shape[2]
    inputObjectFeatures = X_tr_objects_disjoint.shape[2]
    assert (inputJointFeatures == X_tr_objects_shared.shape[2])

    print '#human sub-activities ', num_sub_activities
    print '#object affordances ', num_affordances
    print '#human sub-activities-anticipation ', num_sub_activities_anticipation
    print '#object affordances_anticipation ', num_affordances_anticipation
    print 'shared features dim ', inputJointFeatures
    print 'human features dim ', inputHumanFeatures
    print 'object features dim ', inputObjectFeatures

    epochs = 300
    batch_size = X_tr_human_disjoint.shape[1]
    learning_rate_decay = 0.97
    decay_after = 5

    use_pretrained = False
    train_more = False

    global rnn
    if not use_pretrained:
        if not os.path.exists('{1}/{0}/'.format(index, path_to_checkpoints)):
            os.mkdir('{1}/{0}/'.format(index, path_to_checkpoints))

        rnn = jointModel(num_sub_activities, num_affordances, inputJointFeatures, inputHumanFeatures,
                         inputObjectFeatures)
        rnn.train_model(X_tr_human_shared, X_tr_human_disjoint, Y_tr_human, X_tr_objects_shared,
                     X_tr_objects_disjoint, Y_tr_objects, 1, '{1}/{0}/'.format(index, path_to_checkpoints), epochs,
                     batch_size, learning_rate_decay, decay_after)

    # else:
    #     checkpoint = sys.argv[3]
    #     # Prediction
    #     rnn = load('{2}/{0}/checkpoint.{1}'.format(index, checkpoint, path_to_checkpoints))
    #     if train_more:
    #         rnn.train_model(X_tr_human_shared, X_tr_human_disjoint, 1, '{1}/{0}/'.format(index, path_to_checkpoints), epochs, batch_size,
    #                      learning_rate_decay, decay_after)
