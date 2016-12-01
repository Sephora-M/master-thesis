import tensorflow as tf
import numpy as np
import sys
import os
import cPickle as pickle
from SRNN import SRNN_model

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_boolean("l2_reg", False, "Adds a L2 regularization term")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-6,
                          "Epsilon used for numerical stability in Adam optimizer.")
tf.app.flags.DEFINE_float("reg_factor", 1.0,
                          "Lambda for l2 regulariation.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("num_activities", 2, "Number of decoders, i.e. number of context chords")

tf.app.flags.DEFINE_string("data_file", "JSB_Chorales.pickle", "Data file name")

tf.app.flags.DEFINE_boolean("GD", False, "Uses Gradient Descent with adaptive learning rate")
tf.app.flags.DEFINE_string("train_dir", "bach/JSB_Chorales_128batch_2layers_512units", "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_valid_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_epochs", 20,
                            "Maximium number of epochs for trainig.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("test_model", False,
                            "Evaluate an existing model on test data")

FLAGS = tf.app.flags.FLAGS

def extract_features(file_name, get_train_data=True, get_valid_data=False, get_test_data=False):

    if not get_train_data ^ get_valid_data ^ get_test_data or get_train_data & get_valid_data & get_test_data:
        raise ValueError("Only one of training_data, valid_data and test_data must be True")


    dataset = pickle.load(open(file_name,'rb'))
    train_data = dataset['train']
    valid_data = dataset['valid']
    test_data = dataset['test']
    if get_train_data:
        return train_data
    if get_valid_data:
        return valid_data
    if get_test_data:
        return test_data



def create_SRNN_model(session, forward_only, result_file=None, batch_size=None,same_param=False):
    if batch_size is None:
        batch_size = FLAGS.batch_size

    model = SRNN_model(FLAGS.num_activities, FLAGS.num_frames, FLAGS.num_units, FLAGS.max_gradient_norm, batch_size, FLAGS.learning_rate,
                                       FLAGS.learning_rate_decay_factor, FLAGS.adam_epsilon, FLAGS.GD, forward_only=forward_only, l2_regularization=FLAGS.l2_reg, weight_decay=FLAGS.reg_factor)
    if not same_param:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
            if result_file is not None:
                result_file.write("Continue training existing model! ")
            print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
            model.saver.restore(session, checkpoint.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
    return model

def main(_):
    data = extract_features(FLAGS.data_file)
    train_data, valid_data, test_data = data
    result_file = open(FLAGS.train_dir + "/results.txt", 'a+')

    with tf.Session() as sess:
        with tf.name_scope("Train"):
            result_file.write("Creating SRNN model \n")
            with tf.variable_scope("Model"):
                print("Creating SRNN model ")
                # model = create_SRNN_model(sess,forward_only)


if __name__ == '__main__':
	tf.app.run()

