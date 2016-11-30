import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class SRNN(object):

    # SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)

    def __int__(self,temp_inputs, st_inputs, num_classes, num_frames ,num_units, max_gradient_norm, batch_size, learning_rate,
                                       learning_rate_decay_factor, adam_epsilon,  GD, forward_only=False, l2_regularization=False, weight_decay=0):
        """
        Create S-RNN model
        edgeRNNs: dictionary with keys as RNN name and value is a list of layers
        nodeRNNs: dictionary with keys as RNN name and value is a list of layers
        nodeToEdgeConnections: dictionary with keys as nodeRNNs name and value is another
                dictionary whose keys are edgeRNNs the nodeRNN is connected to and value is a list
                of size-2 which indicate the features to choose from the unConcatenateLayer
        edgeListComplete:
        cost:
        nodeLabels:
        learning_rate:
        clipnorm:
        update_type:
        weight_decay:

        return:
        """

        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.adam_epsilon = adam_epsilon
        self.GD = GD
        self.weight_decay = weight_decay

        input_F, input_rA, input_lA, input_lL, input_rL, input_B = temp_inputs
        input_FlA, input_FrA, input_FB, input_BlA, input_BrA, input_BlL, input_BrL = st_inputs

        weights = {'out' : tf.Variable(tf.random_normal([num_units,num_classes]))}
        biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}

        # TODO: please re-write in a more elegant way, thank you
        edgesRNN = {}
        states = {}

        # temporal edges
        edgesRNN['face-face'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['face-face'] = tf.zero_state([batch_size, edgesRNN['face-face'].state_size])

        edgesRNN['rightArm-rightArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['rightArm-rightArm'] = tf.zero_state([batch_size, edgesRNN['rightArm-rightArm'].state_size])

        edgesRNN['leftArm-leftArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['leftArm-leftArm'] = tf.zero_state([batch_size, edgesRNN['leftArm-leftArm'].state_size])

        edgesRNN['leftLeg-leftLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['leftLeg-leftLeg'] = tf.zero_state([batch_size, edgesRNN['leftLeg-leftLeg'].state_size])

        edgesRNN['rightLeg-rightLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['rightLeg-rightLeg'] = tf.zero_state([batch_size, edgesRNN['rightLeg-rightLeg'].state_size])

        edgesRNN['belly-belly'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['belly-belly'] = tf.zero_state([batch_size, edgesRNN['belly-belly'].state_size])

        #spatio edges
        edgesRNN['face-leftArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['face-leftArm'] = tf.zero_state([batch_size, edgesRNN['face-leftArm'].state_size])

        edgesRNN['face-rightArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['face-rightArm'] = tf.zero_state([batch_size, edgesRNN['face-rightArm'].state_size])

        edgesRNN['face-belly'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['face-belly'] = tf.zero_state([batch_size, edgesRNN['face-belly'].state_size])

        edgesRNN['belly-leftArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['belly-leftArm'] = tf.zero_state([batch_size, edgesRNN['belly-leftArm'].state_size])

        edgesRNN['belly-rightArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['belly-rightArm'] = tf.zero_state([batch_size, edgesRNN['belly-rightArm'].state_size])

        edgesRNN['belly-leftLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['belly-leftLeg'] = tf.zero_state([batch_size, edgesRNN['belly-leftLeg'].state_size])

        edgesRNN['belly-rightLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        states['belly-rightLeg'] = tf.zero_state([batch_size, edgesRNN['belly-rightLeg'].state_size])

        # nodes
        nodesRNN = {}
        nodesRNN['face']  = tf.nn.rnn_cell.LSTMCell(num_units)
        states['face'] = tf.zero_state(batch_size, nodesRNN[''].state_size)

        nodesRNN['arms']  = tf.nn.rnn_cell.LSTMCell(num_units)
        states['arms'] = tf.zero_state(batch_size, nodesRNN['arms'].state_size)

        nodesRNN['legs']  = tf.nn.rnn_cell.LSTMCell(num_units)
        states['legs'] = tf.zero_state(batch_size, nodesRNN['legs'].state_size)

        nodesRNN['belly']  = tf.nn.rnn_cell.LSTMCell(num_units)
        states['belly'] = tf.zero_state(batch_size, nodesRNN['belly'].state_size)

        fullbodyRNN = tf.nn.rnn_cell.LSTMCell(num_units)
        states['fullbody'] = tf.zero_state(batch_size, fullbodyRNN.state_size)


        # connect the edgesRNN to the corresponding nodeRNN
        with tf.variable_scope("SRNN"):
            #TODO: proceed in batches!
            for time_step in range(num_frames):
                output_FF, states['face-face'] = edgesRNN['face-face'](input_F[:,time_step], states['face-face'])
                output_rArA, states['rightArm-rightArm'] = edgesRNN['rightArm-rightArm']( input_rA[:,time_step], states['rightArm-rightArm'] )
                output_lAlA, states['leftArm-leftArm'] = edgesRNN['leftArm-leftArm']( input_lA[:,time_step], states['leftArm-leftArm'] )
                output_lLlL, states['leftLeg-leftLeg'] = edgesRNN['leftLeg-leftLeg']( input_lL[:,time_step],states['leftLeg-leftLeg'] )
                output_rLrL, states['rightLeg-rightLeg'] = edgesRNN['rightLeg-rightLeg']( input_rL[:,time_step],states['rightLeg-rightLeg'] )
                output_BB, states['belly-belly'] = edgesRNN['belly-belly']( input_B[:,time_step],states['belly-belly'] )

                output_FlA, states['face-leftArm'] = edgesRNN['face-leftArm']( input_FlA[:,time_step], states['face-leftArm'] )
                output_FrA, states['face-rightArm'] = edgesRNN['face-rightArm']( input_FrA[:,time_step], states['face-rightArm'] )
                output_FB, states['face-belly'] = edgesRNN['face-belly']( input_FB[:,time_step], states['face-belly'] )
                output_BlA, states['belly-leftArm'] = edgesRNN['belly-leftArm']( input_BlA[:,time_step], states['belly-leftArm'] )
                output_BrA, states['belly-rightArm'] = edgesRNN['belly-rightArm']( input_BrA[:,time_step], states['belly-rightArm'] )
                output_BlL, states['belly-leftLeg'] = edgesRNN['belly-leftLeg']( input_BlL[:,time_step], states['belly-leftLeg'] )
                output_BrL, states['belly-rightLeg'] = edgesRNN['belly-rightLeg']( input_BrL[:,time_step], states['belly-rightLeg'] )

                node_input_F = tf.concat(1,[output_FF, output_FB, output_FlA, output_FrA])
                node_input_B = tf.concat(1,[output_BB, output_FB, output_BlA, output_BlL, output_BrA, output_BrL])
                node_input_A = tf.concat(1,[output_rArA,output_lAlA, output_BrA, output_BlA, output_FrA, output_FlA])
                node_input_L = tf.concat(1,[output_rLrL,output_lLlL, output_BrL, output_BlL])

                output_F = rnn.rnn(nodesRNN['face'], node_input_F, dtype=tf.float32)
                output_A = rnn.rnn(nodesRNN['arms'], node_input_A, dtype=tf.float32)
                output_L = rnn.rnn(nodesRNN['legs'], node_input_L, dtype=tf.float32)
                output_B = rnn.rnn(nodesRNN['belly'], node_input_B, dtype=tf.float32)

                fullbody_input = tf.concat(1,[output_F,output_A,output_B,output_L])

                output_Full, states['fullbody'] = fullbodyRNN(fullbody_input,states['fullbody'])

                self.output = tf.matmul(output_Full, weights['out']) + biases['out']



    def train_model(self,trX_shared_1, trX_1, trY_1, trX_shared_2, trX_2, trY_2, snapshot_rate=1,
                    path=None, epochs=30, batch_size=50, learning_rate_decay=0.97, decay_after=10):

         # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

