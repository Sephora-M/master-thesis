import tensorflow as tf
import numpy as np
import sys, random
import os, time
import cPickle as pickle
from SRNN import SRNN_model

"""
import action_recognition_srnn as ma
tr,va,te = ma.synthetic_data(4,2,0)

"""

NUM_ACTIVITIES = 20

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_boolean("l2_reg", False, "Adds a L2 regularization term")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-6,
                          "Epsilon used for numerical stability in Adam optimizer.")
tf.app.flags.DEFINE_float("reg_factor", 1.0,
                          "Lambda for l2 regulariation.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_units", 16, "Size of each model layer.")
#tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("num_activities", NUM_ACTIVITIES, "Number of decoders, i.e. number of context chords")
tf.app.flags.DEFINE_integer("num_frames", 5, "Number of frames in each example.")
tf.app.flags.DEFINE_string("data_file", "JSB_Chorales.pickle", "Data file name")

tf.app.flags.DEFINE_boolean("GD", False, "Uses Gradient Descent with adaptive learning rate")
tf.app.flags.DEFINE_string("train_dir", "models", "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_valid_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_epochs", 20,
                            "Maximium number of epochs for trainig.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",5,
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

def randomSample():
    temp_features_names = ['face-face','rightArm-rightArm','leftArm-leftArm','rightLeg-rightLeg','leftLeg-leftLeg','belly-belly']
    st_features_names = ['face-leftArm','face-rightArm','face-belly','belly-leftArm','belly-rightArm',
                             'belly-rightLeg','belly-leftLeg']
    temp_features = {}
    for name in temp_features_names:
        temp_features[name] = np.random.rand(5)
    st_features = {}
    for name in st_features_names:
        st_features[name] = np.random.rand(5)
    #train_data = np.append(train_data, [temp_features, st_features, random.randint(0, NUM_ACTIVITIES)])
    train_data_sample = np.array([{},{},-1])
    train_data_sample[0] = temp_features
    train_data_sample[1] = st_features
    train_data_sample[2] = random.randint(0, NUM_ACTIVITIES)
    return train_data_sample

def synthetic_data(train_size, valid_size, test_size,num_frames,num_features):
    temp_features_names = ['face-face','rightArm-rightArm','leftArm-leftArm','rightLeg-rightLeg','leftLeg-leftLeg','belly-belly']
    st_features_names = ['face-leftArm','face-rightArm','face-belly','belly-leftArm','belly-rightArm',
                             'belly-rightLeg','belly-leftLeg']
    train_data = np.array({})

    #for i in range(train_size):
    temp_features = {}
    for name in temp_features_names:
        temp_features[name] = np.random.rand(train_size,num_frames,num_features)
    st_features = {}
    for name in st_features_names:
        st_features[name] = np.random.rand(train_size,num_frames,num_features)
    #train_data = np.append(train_data, [temp_features, st_features, random.randint(0, NUM_ACTIVITIES)])
    train_data_sample = np.array([{},{},-1])
    train_data_sample[0] = temp_features
    train_data_sample[1] = st_features
    train_data_sample[2] = random.randint(0, NUM_ACTIVITIES)

    train_data = np.append(train_data,train_data_sample)

    train_data = np.delete(train_data,0)
    #train_data = np.reshape(train_data,(train_size,3))

    valid_data = np.array({})
    for _ in range(valid_size):
        temp_features = {}
        for name in temp_features_names:
            temp_features[name] = np.random.rand(num_frames,num_features)
        st_features = {}
        for name in st_features_names:
            st_features[name] = np.random.rand(num_frames,num_features)
        data_sample = np.array([{},{},-1])
        data_sample[0] = temp_features
        data_sample[1] = st_features
        data_sample[2] = random.randint(0, NUM_ACTIVITIES)

        valid_data = np.append(valid_data,data_sample)
    valid_data = np.delete(valid_data,0)
    valid_data = np.reshape(valid_data,(valid_size,3))

    test_data = np.array({})
    for _ in range(test_size):
        temp_features = {}
        for name in temp_features_names:
            temp_features[name] = np.random.rand(num_frames,num_features)
        st_features = {}
        for name in st_features_names:
            st_features[name] = np.random.rand(num_frames,num_features)
        data_sample = np.array([{},{},-1])
        data_sample[0] = temp_features
        data_sample[1] = st_features
        data_sample[2] = random.randint(0, NUM_ACTIVITIES)

        test_data = np.append(test_data,data_sample)
    test_data = np.delete(test_data,0)
    test_data = np.reshape(test_data,(test_size,3))

    return train_data, valid_data, test_data


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
    #data = extract_features(FLAGS.data_file)
    data = synthetic_data(20,10,10,5,5)
    train_data, valid_data, test_data = data
    train_data_size = train_data.shape[0]
    valid_data_size = valid_data.shape[0]
    test_data_size = test_data.shape[0]

    steps_per_epoch = int(len(train_data)/FLAGS.batch_size)

    result_file = open(FLAGS.train_dir + "/results.txt", 'a+')

    with tf.Session() as sess:
        with tf.name_scope("Train"):
            result_file.write("Creating SRNN model \n")
            with tf.variable_scope("Model"):
                print("Creating SRNN model ")
                print("with %d units and %d bach-size." % (FLAGS.num_units, FLAGS.batch_size))

                model = create_SRNN_model(sess,False,result_file)
                if FLAGS.max_train_data_size:
                    train_data = train_data[:FLAGS.max_train_data_size]
                if FLAGS.max_valid_data_size:
                    valid_data = test_data[:FLAGS.max_valid_data_size]
                if FLAGS.max_test_data_size:
                    test_data = test_data[:FLAGS.max_test_data_size]

                checkpoint_path = FLAGS.train_dir + "/srnn_model.ckpt"

                step_time, ckpt_loss,epoch_loss = 0.0, 0.0,0.0
                current_step = 0
                current_epoch = divmod(model.global_step.eval(),steps_per_epoch)[0]

                result_file.write(
                " %d batch size %d number of steps to complete one epoch \n" % (FLAGS.batch_size, steps_per_epoch))

                previous_losses,previous_train_loss, previous_eval_loss = [],[],[]
                best_train_loss, best_val_loss = np.inf, np.inf
                train_batch_id = 1

                while int(model.global_step.eval()/steps_per_epoch) < FLAGS.max_epochs:
                    start_time = time.time()
                    #batch = train_data[random.sample(range(train_data_size))]
                    print(train_data.shape)
                    print(train_data[0])
                    batch = np.array(random.sample(train_data,FLAGS.batch_size))
                    #print(batch)
                    print(batch[:,0])
                    temp_input_batch = batch[:,0]
                    st_input_batch = batch[:,1]
                    target_batch = batch[:,-1]
                    step_loss, _ = model.step(sess,temp_input_batch, st_input_batch, target_batch, False)

                    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    ckpt_loss += step_loss / steps_per_epoch
                    epoch_loss += step_loss/steps_per_epoch

                    current_step += 1
                    train_batch_id += 1
                    if current_step % FLAGS.steps_per_checkpoint == 0:
                        # Print statiistics for the previous epoch
                        print ("batch no %d epoch %d" % (train_batch_id,current_epoch))
                        print ("global step %d learning rate %.4f step-time %.3f loss %.4f "
                               % (model.global_step.eval(), model.learning_rate.eval(),
                                  step_time, ckpt_loss))
                        result_file.write("global step %d learning rate %.4f step-time %.3f loss %.4f \n"
                               % (model.global_step.eval(), model.learning_rate.eval(),
                                  step_time, ckpt_loss))


                        # Decrease learning rate if no improvement was seen over last 3 times.
                        if FLAGS.GD:
                            if len(previous_losses) > 2 and ckpt_loss > max(previous_losses[-3:]):
                                sess.run(model.learning_rate_decay_op)
                        previous_losses.append(ckpt_loss)
                        step_time, ckpt_loss = 0.0, 0.0

                    if model.global_step.eval() % steps_per_epoch == 0:
                        print ("epoch 5d finished" % (current_epoch))
                        result_file.write("epoch  %d finished \n" % (current_epoch))

                        previous_train_loss.append(epoch_loss)
                        print("  avg train batch:  loss %.4f " % (epoch_loss))
                        result_file.write("  avg train batch:  loss %.4f  \n" % (epoch_loss))
                        epoch_loss = 0.0

                    # TODO: validation and testing


if __name__ == '__main__':
	tf.app.run()

