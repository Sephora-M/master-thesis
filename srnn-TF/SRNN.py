import tensorflow as tf
import numpy as np

class SRNN_model(object):

    # SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)

    def __init__(self, num_classes, num_frames , num_temp_features, num_st_features, num_units, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, adam_epsilon,  GD, forward_only=False, l2_regularization=False, weight_decay=0):
        """"
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
        self.num_classes = num_classes
        self.num_temp_features = num_temp_features
        self.num_st_features = num_st_features
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.max_grad_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.adam_epsilon = adam_epsilon
        self.GD = GD
        self.weight_decay = weight_decay

        self.temp_features_names = ['face-face','belly-belly','rightArm-rightArm','leftArm-leftArm','rightLeg-rightLeg','leftLeg-leftLeg']
        self.st_features_names = ['face-leftArm','face-rightArm','face-belly','belly-leftArm','belly-rightArm',
                             'belly-rightLeg','belly-leftLeg']
        nodes_names = {'face','arms','legs','belly'}
        edgesRNN = {}
        nodesRNN = {}
        states = {}
        self.inputs = {}
        self.targets = tf.placeholder(tf.float32, shape=(None,num_classes), name='targets')

        for temp_feat in self.temp_features_names:
            self.inputs[temp_feat] = tf.placeholder(tf.float32, shape=(None,num_frames, self.num_temp_features), name=temp_feat)
            edgesRNN[temp_feat] = tf.nn.rnn_cell.LSTMCell(num_units)
            states[temp_feat] = edgesRNN[temp_feat].zero_state(self.batch_size,tf.float32)

        for st_feat in self.st_features_names:
            self.inputs[st_feat] = tf.placeholder(tf.float32, shape=(None,num_frames, self.num_st_features), name=st_feat)
            edgesRNN[st_feat] = tf.nn.rnn_cell.LSTMCell(num_units)
            states[st_feat] = edgesRNN[st_feat].zero_state(self.batch_size,tf.float32)

        for node in nodes_names:
            self.inputs[node] = tf.placeholder(tf.float32, shape=(None,num_frames, None), name=node)
            nodesRNN[node] = tf.nn.rnn_cell.LSTMCell(num_units)
            states[node] = nodesRNN[node].zero_state(self.batch_size, tf.float32)

        weights = {'out' : tf.Variable(tf.random_normal([num_units*num_frames,num_classes]))}
        biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}


        fullbodyRNN = tf.nn.rnn_cell.LSTMCell(num_units)
        states['fullbody'] = fullbodyRNN.zero_state(self.batch_size, tf.float32)

        outputs = {}
        final_outputs = []

        node_inputs = {}
        node_outputs = {}

        # connect the edgesRNN to the corresponding nodeRNN
        with tf.variable_scope("SRNN"):
            for time_step in range(num_frames):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                for temp_feat in self.temp_features_names:
                    outputs[temp_feat], states[temp_feat] = edgesRNN[temp_feat](self.inputs[temp_feat][:,time_step,:], states[temp_feat], scope="lstm_"+temp_feat)
                for st_feat in self.st_features_names:
                    outputs[st_feat], states[st_feat] = edgesRNN[st_feat](self.inputs[st_feat][:,time_step,:], states[st_feat], scope="lstm_"+st_feat)

                node_inputs['face'] = tf.concat(1,[outputs['face-face'], outputs['face-rightArm'], outputs['face-leftArm'], outputs['face-belly']])

                node_inputs['belly'] = tf.concat(1,[outputs['belly-belly'], outputs['face-belly'], outputs['belly-leftArm'], outputs['belly-rightArm'],
                                            outputs['belly-leftLeg'], outputs['belly-rightLeg']])

                node_inputs['arms'] = tf.concat(1,[outputs['rightArm-rightArm'], outputs['leftArm-leftArm'], outputs['face-rightArm'],
                                            outputs['face-leftArm'],outputs['belly-rightArm'], outputs['belly-leftArm']])

                node_inputs['legs'] = tf.concat(1,[outputs['rightLeg-rightLeg'], outputs['leftLeg-leftLeg'], outputs['belly-rightLeg'], outputs['belly-leftLeg']])

                for node_name in nodes_names:
                    node_outputs[node_name], states[node_name] = nodesRNN[node_name](node_inputs[node_name], states[node_name], scope='lstm_'+node_name)

                fullbody_input = tf.concat(1,[node_outputs['face'],node_outputs['belly'],node_outputs['arms'],node_outputs['legs']])
                final_output, states['fullbody'] = fullbodyRNN(fullbody_input,states['fullbody'])
                final_outputs.append(final_output)


        output = tf.concat(1,final_outputs, name="output_lastCells")


        self.final_states = states
        #print(output)
        self.logits = tf.matmul(output, weights['out'], name="logits") + biases['out']
        #print(self.logits)
        #print(self.targets)
        loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets)
        self.cost = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        if not forward_only:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.cost,tvars), self.max_grad_norm)
            self.gradients_norm = norm
            self.updates = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

    def step(self,session,temp_inputs, st_inputs, targets, forward_only):
        input_feed = {}
        for temp_features_name in self.temp_features_names:
            input_feed[self.inputs[temp_features_name].name] = temp_inputs[temp_features_name]

        for st_features_name in self.st_features_names:
            input_feed[ self.inputs[st_features_name].name] = st_inputs[st_features_name]

        input_feed[self.targets.name] = targets

        if not forward_only:
            output_feed = [self.updates, self.gradients_norm, self.cost]
        else:
            output_feed = [self.cost, tf.nn.softmax(self.logits)]

        outputs = session.run(output_feed, input_feed)


        if not forward_only:
            return outputs[1], outputs[2], None # returs gradients norm and cost and no output
        else:
            return None, outputs[0], outputs[1]  # no gradients, cost and output


    def steps(self, session, all_data):
        size = len(all_data[2])
        num_batches = size/self.batch_size

        batch_losses= np.zeros(num_batches)
        batch_outputs= np.zeros((num_batches, self.batch_size, self.num_classes))
        missclassified = 0.0
        for i in range(num_batches):
            batch = self.get_batch(all_data,i)
            _,batch_losses[i],batch_outputs[i] = self.step(session,batch[0],batch[1],batch[2],True)
            classes = np.argmax(batch_outputs[i],1)
            for d in range(self.batch_size):
                if batch[2][d][classes[d]] != 1:
                    missclassified += 1
        error = missclassified / (num_batches*self.batch_size)
        return np.mean(batch_losses), error



    def get_batch(self, data, batch_id):

        batch = np.array({})
        idxs = range((batch_id * self.batch_size - self.batch_size),(batch_id * self.batch_size))

        batch_dic = np.array([{},{},[]])

        for i in xrange(len(data)):
            dic = data[i]
            if i<2:
                batch_sub_dic = {}
                for key in dic:
                    batch_sub_dic[key]=dic[key][idxs]
                batch_dic[i] = batch_sub_dic
            else:
                batch_dic[i] = dic[idxs]
        batch = np.append(batch,batch_dic)
        batch = np.delete(batch,0)
        return batch

