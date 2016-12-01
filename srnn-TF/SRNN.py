import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class SRNN_model(object):

    # SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)

    def __int__(self, num_classes, num_frames ,num_units, max_gradient_norm, batch_size, learning_rate,
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
        self.max_grad_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.adam_epsilon = adam_epsilon
        self.GD = GD
        self.weight_decay = weight_decay

        # TODO: replace by placeholders
        temp_features_names = ['face-face','rightArm-rightArm','leftArm-leftArm','rightLeg-rightLeg','leftLeg-leftLeg','belly-belly']
        st_features_names = ['face-leftArm','face-rightArm','face-belly','belly-leftArm','belly-rightArm',
                             'belly-rightLeg','belly-leftLeg']
        nodes_names = {'face','arms','legs','belly'}
        edgesRNN = {}
        nodesRNN = {}
        states = {}
        self.inputs = {}
        self.targets = tf.placeholder(tf.float32, shape=[None], name='targets')

        for temp_feat in temp_features_names:
            self.inputs[temp_feat] = tf.placeholder(tf.float32, shape=[None], name=temp_feat)
            edgesRNN[temp_feat] = tf.nn.rnn_cell.LSTMCell(num_units)
            states[temp_feat] = tf.zero_state([batch_size, edgesRNN[temp_feat].state_size])

        for st_feat in st_features_names:
            self.inputs[st_feat] = tf.placeholder(tf.float32, shape=[None], name=st_feat)
            edgesRNN[st_feat] = tf.nn.rnn_cell.LSTMCell(num_units)
            states[st_feat] = tf.zero_state([batch_size, edgesRNN[st_feat].state_size])

        for node in nodes_names:
            self.inputs[node] = tf.placeholder(tf.float32, shape=[None], name=node)
            nodesRNN[node] = tf.nn.rnn_cell.LSTMCell(num_units)
            states[node] = tf.zero_state([batch_size, nodesRNN[node].state_size])


        # input_F, input_rA, input_lA, input_lL, input_rL, input_B = temp_inputs
        # input_FlA, input_FrA, input_FB, input_BlA, input_BrA, input_BlL, input_BrL = st_inputs

        weights = {'out' : tf.Variable(tf.random_normal([num_units,num_classes]))}
        biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}

        # # temporal edges
        # edgesRNN['face-face'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['face-face'] = tf.zero_state([batch_size, edgesRNN['face-face'].state_size])
        #
        # edgesRNN['rightArm-rightArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['rightArm-rightArm'] = tf.zero_state([batch_size, edgesRNN['rightArm-rightArm'].state_size])
        #
        # edgesRNN['leftArm-leftArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['leftArm-leftArm'] = tf.zero_state([batch_size, edgesRNN['leftArm-leftArm'].state_size])
        #
        # edgesRNN['leftLeg-leftLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['leftLeg-leftLeg'] = tf.zero_state([batch_size, edgesRNN['leftLeg-leftLeg'].state_size])
        #
        # edgesRNN['rightLeg-rightLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['rightLeg-rightLeg'] = tf.zero_state([batch_size, edgesRNN['rightLeg-rightLeg'].state_size])
        #
        # edgesRNN['belly-belly'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['belly-belly'] = tf.zero_state([batch_size, edgesRNN['belly-belly'].state_size])

        # # spatio edges
        # edgesRNN['face-leftArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['face-leftArm'] = tf.zero_state([batch_size, edgesRNN['face-leftArm'].state_size])
        #
        # edgesRNN['face-rightArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['face-rightArm'] = tf.zero_state([batch_size, edgesRNN['face-rightArm'].state_size])
        #
        # edgesRNN['face-belly'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['face-belly'] = tf.zero_state([batch_size, edgesRNN['face-belly'].state_size])
        #
        # edgesRNN['belly-leftArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['belly-leftArm'] = tf.zero_state([batch_size, edgesRNN['belly-leftArm'].state_size])
        #
        # edgesRNN['belly-rightArm'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['belly-rightArm'] = tf.zero_state([batch_size, edgesRNN['belly-rightArm'].state_size])
        #
        # edgesRNN['belly-leftLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['belly-leftLeg'] = tf.zero_state([batch_size, edgesRNN['belly-leftLeg'].state_size])
        #
        # edgesRNN['belly-rightLeg'] = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['belly-rightLeg'] = tf.zero_state([batch_size, edgesRNN['belly-rightLeg'].state_size])

        # # nodes
        # nodesRNN['face']  = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['face'] = tf.zero_state(batch_size, nodesRNN['face'].state_size)
        #
        # nodesRNN['arms']  = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['arms'] = tf.zero_state(batch_size, nodesRNN['arms'].state_size)
        #
        # nodesRNN['legs']  = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['legs'] = tf.zero_state(batch_size, nodesRNN['legs'].state_size)
        #
        # nodesRNN['belly']  = tf.nn.rnn_cell.LSTMCell(num_units)
        # states['belly'] = tf.zero_state(batch_size, nodesRNN['belly'].state_size)

        fullbodyRNN = tf.nn.rnn_cell.LSTMCell(num_units)
        states['fullbody'] = tf.zero_state(batch_size, fullbodyRNN.state_size)

        outputs = []
        final_outputs = []

        node_inputs = {}
        node_outputs = {}
        # connect the edgesRNN to the corresponding nodeRNN
        with tf.variable_scope("SRNN"):
            #TODO: proceed in batches!
            for time_step in range(num_frames):
                for temp_feat in temp_features_names:
                    outputs[temp_feat], states[temp_feat] = edgesRNN[temp_feat](self.inputs[temp_feat][:,time_step], states[temp_feat])
                    # output_rArA, states['rightArm-rightArm'] = edgesRNN['rightArm-rightArm']( input_rA[:,time_step], states['rightArm-rightArm'] )
                    # output_lAlA, states['leftArm-leftArm'] = edgesRNN['leftArm-leftArm']( input_lA[:,time_step], states['leftArm-leftArm'] )
                    # output_lLlL, states['leftLeg-leftLeg'] = edgesRNN['leftLeg-leftLeg']( input_lL[:,time_step],states['leftLeg-leftLeg'] )
                    # output_rLrL, states['rightLeg-rightLeg'] = edgesRNN['rightLeg-rightLeg']( input_rL[:,time_step],states['rightLeg-rightLeg'] )
                    # output_BB, states['belly-belly'] = edgesRNN['belly-belly']( input_B[:,time_step],states['belly-belly'] )
                for st_feat in st_features_names:
                    outputs[st_feat], states[st_feat] = edgesRNN[st_feat](self.inputs[st_feat][:,time_step], states[st_feat])
                    # output_FlA, states['face-leftArm'] = edgesRNN['face-leftArm']( input_FlA[:,time_step], states['face-leftArm'] )
                    # output_FrA, states['face-rightArm'] = edgesRNN['face-rightArm']( input_FrA[:,time_step], states['face-rightArm'] )
                    # output_FB, states['face-belly'] = edgesRNN['face-belly']( input_FB[:,time_step], states['face-belly'] )
                    # output_BlA, states['belly-leftArm'] = edgesRNN['belly-leftArm']( input_BlA[:,time_step], states['belly-leftArm'] )
                    # output_BrA, states['belly-rightArm'] = edgesRNN['belly-rightArm']( input_BrA[:,time_step], states['belly-rightArm'] )
                    # output_BlL, states['belly-leftLeg'] = edgesRNN['belly-leftLeg']( input_BlL[:,time_step], states['belly-leftLeg'] )
                    # output_BrL, states['belly-rightLeg'] = edgesRNN['belly-rightLeg']( input_BrL[:,time_step], states['belly-rightLeg'] )

                node_inputs['face'] = tf.concat(1,[outputs['face-face'], outputs['face-rightArm'], outputs['face-leftArm'], outputs['face-belly']])
                node_inputs['belly'] = tf.concat(1,[outputs['belly-belly'], outputs['face-belly'], outputs['belly-leftArm'], outputs['belly-rightArm'],
                                            outputs['belly-leftLeg'], outputs['belly-righLeg']])
                node_inputs['arms'] = tf.concat(1,[outputs['rightArm-rightArm'], outputs['leftArm-leftArm'], outputs['face-rightArm'],
                                            outputs['face-leftArm'],outputs['belly-rightArm'], outputs['belly-leftArm']])
                node_inputs['legs'] = tf.concat(1,[outputs['rightLeg-rightLeg'], outputs['leftLeg-leftLeg'], outputs['belly-rightLeg'], outputs['belly-leftLeg']])

                for node_name in nodes_names:
                    node_outputs[nodes_names] = rnn.rnn(nodesRNN[nodes_names], node_inputs[nodes_names], dtype=tf.float32)
                    # output_F = rnn.rnn(nodesRNN['face'], node_input_F, dtype=tf.float32)
                    # output_A = rnn.rnn(nodesRNN['arms'], node_input_A, dtype=tf.float32)
                    # output_L = rnn.rnn(nodesRNN['legs'], node_input_L, dtype=tf.float32)
                    # output_B = rnn.rnn(nodesRNN['belly'], node_input_B, dtype=tf.float32)

                # fullbody_input = tf.concat(1,[output_F,output_A,output_B,output_L])
                fullbody_input = tf.concat(1,[node_outputs['face'],node_outputs['belly'],node_outputs['arms'],node_outputs['legs']])

                final_output, states['fullbody'] = fullbodyRNN(fullbody_input,states['fullbody'])
                final_outputs.append(final_output)

        output = tf.reshape(tf.concat(1,final_outputs),[-1,num_units])

        self.final_states = states
        self.logits = tf.matmul(output, weights['out']) + biases['out']
        loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets)
        self.cost = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        if not forward_only:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.cost,tvars), self.max_grad_norm)
            self.gradients_norm = norm
            self.updates = optimizer.apply_gradients(zip(clipped_grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=2)

    def step(self,session,temp_inputs, st_inputs, targets, forward_only):
        input_feed = {}

        input_feed['temp-inputs'] = temp_inputs;
        input_feed['st-inputs'] = st_inputs;

        if not forward_only:
            output_feed = [self.updates, self.gradients_norm,self.cost]
        else:
            output_feed = [self.cost, self.logits]

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            return outputs[1], outputs[2], None # returs gradients norm and cost and no output
        else:
            return outputs[0]
