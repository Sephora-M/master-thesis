import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

import numpy as np

class SRNN_model(object):

    # SharedRNN(shared_layers, human_layers, object_layers, softmax_loss, trY_1, trY_2, 1e-3)

    def __init__(self, num_classes, num_frames , num_temp_features, num_st_features, num_units, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, adam_epsilon,  GD, forward_only=False, l2_regularization=False, weight_decay=0, log_dir = None):
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
        self.save_summaries = log_dir is not None
        if self.save_summaries:
            print('Writing summaries for Tensorboard')
        num_layers=1
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
                             'belly-rightLeg','belly-leftLeg','leftArm-rightArm','leftLeg-rightLeg']
        nodes_names = {'face','arms','legs','belly'}
        edgesRNN = {}
        nodesRNN = {}
        states = {}
        infos = {}
        self.inputs = {}
        self.targets = tf.placeholder(tf.float32, shape=(None,num_classes), name='targets')

        for temp_feat in self.temp_features_names:
            infos[temp_feat]={'input_gates' : [],'forget_gates' : [],'modulated_input_gates': [],'output_gates' : [], 'activations' : [], 'state_c' : [], 'state_m' : []}
            self.inputs[temp_feat] = tf.placeholder(tf.float32, shape=(None,num_frames, self.num_temp_features), name=temp_feat)
            single_cell =tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
            if num_layers == 1:
                edgesRNN[temp_feat] = single_cell
            else:
                edgesRNN[temp_feat] = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
            states[temp_feat] = edgesRNN[temp_feat].zero_state(self.batch_size,tf.float32)

        for st_feat in self.st_features_names:
            infos[st_feat]={'input_gates' : [],'forget_gates' : [],'modulated_input_gates': [],'output_gates' : [], 'activations' : [], 'state_c' : [], 'state_m' : []}
            self.inputs[st_feat] = tf.placeholder(tf.float32, shape=(None,num_frames, self.num_st_features), name=st_feat)
            single_cell =tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
            if num_layers == 1:
                edgesRNN[st_feat] = single_cell
            else:
                edgesRNN[st_feat] = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
            states[st_feat] = edgesRNN[st_feat].zero_state(self.batch_size,tf.float32)

        for node in nodes_names:
            infos[node]={'input_gates' : [],'forget_gates' : [],'modulated_input_gates': [],'output_gates' : [], 'activations' : [], 'state_c' : [], 'state_m' : []}
            self.inputs[node] = tf.placeholder(tf.float32, shape=(None,num_frames, None), name=node)
            single_cell =tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
            if num_layers == 1:
                nodesRNN[node] = single_cell
            else:
                nodesRNN[node] = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
            states[node] = nodesRNN[node].zero_state(self.batch_size, tf.float32)

        weights = {'out' : tf.Variable(tf.random_normal([num_units*num_frames,num_classes]))}
        biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}

        single_cell =tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
        if num_layers == 1:
            fullbodyRNN = single_cell
        else:
            fullbodyRNN = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        states['fullbody'] = fullbodyRNN.zero_state(self.batch_size, tf.float32)
        infos['fullbody']={'input_gates' : [],'forget_gates' : [],'modulated_input_gates': [],'output_gates' : [], 'activations' : [], 'state_c' : [], 'state_m' : []}

        outputs = {}
        final_outputs = []

        node_inputs = {}
        node_outputs = {}



        def linear(args, output_size, bias, bias_start=0.0, scope=None):
            """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
            Args:
            args: a 2D Tensor or a list of 2D, batch x n, Tensors.
            output_size: int, second dimension of W[i].
            bias: boolean, whether to add a bias term or not.
            bias_start: starting value to initialize the bias; 0 by default.
            scope: VariableScope for the created subgraph; defaults to "Linear".
            Returns:
            A 2D Tensor with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
            """
            if args is None or (nest.is_sequence(args) and not args):
                raise ValueError("`args` must be specified")
            if not nest.is_sequence(args):
                args = [args]

            # Calculate the total size of arguments on dimension 1.
            total_arg_size = 0
            shapes = [a.get_shape().as_list() for a in args]
            for shape in shapes:
                if len(shape) != 2:
                    raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
                if not shape[1]:
                    raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
                else:
                    total_arg_size += shape[1]

            dtype = [a.dtype for a in args][0]

            # Now the computation.
            with vs.variable_scope(scope or "Linear"):
                matrix = vs.get_variable(
                    "Matrix", [total_arg_size, output_size], dtype=dtype)
                if len(args) == 1:
                    res = math_ops.matmul(args[0], matrix)
                else:
                    res = math_ops.matmul(array_ops.concat(1, args), matrix)
                if not bias:
                    return res
                bias_term = vs.get_variable(
                    "Bias", [output_size],
                    dtype=dtype,
                    initializer=init_ops.constant_initializer(
                        bias_start, dtype=dtype))
                return res + bias_term


        def get_gates(rnn,inputs, state, scope):
            if rnn._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(1, 2, state)
            concat = linear([inputs, h], 4 * rnn._num_units, True, scope=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(1, 4, concat)

            return sigmoid(i), tanh(j), sigmoid(f + rnn._forget_bias), sigmoid(o)

        # connect the edgesRNN to the corresponding nodeRNN
        with tf.variable_scope("SRNN"):
            for time_step in range(num_frames):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                for temp_feat in self.temp_features_names:
                    inputs = self.inputs[temp_feat][:,time_step,:]
                    state =  states[temp_feat]
                    scope = "lstm_"+temp_feat
                    i,j,f,o = get_gates(edgesRNN[temp_feat],inputs, state, scope)
                    outputs[temp_feat], states[temp_feat] = edgesRNN[temp_feat](inputs,state, scope=scope)
                    infos[temp_feat]['input_gates'].append(i)
                    infos[temp_feat]['forget_gates'].append(f)
                    infos[temp_feat]['modulated_input_gates'].append(j)
                    infos[temp_feat]['output_gates'].append(o)
                    infos[temp_feat]['activations'].append(outputs[temp_feat])
                    infos[temp_feat]['state_c'].append(states[temp_feat][0])
                    infos[temp_feat]['state_m'].append(states[temp_feat][1])
                    if self.save_summaries:
                        tf.summary.histogram('activations_'+temp_feat + '_time_step_' + str(time_step),outputs[temp_feat])
                        tf.summary.histogram('state_C_'+temp_feat + '_time_step_' + str(time_step),states[temp_feat][0])
                        tf.summary.histogram('state_M_'+temp_feat + '_time_step_' + str(time_step),states[temp_feat][1])
                        tf.summary.histogram('input_gate'+temp_feat + '_time_step_' + str(time_step), i)
                        tf.summary.histogram('modulated_input_gate'+temp_feat + '_time_step_' + str(time_step), j)
                        tf.summary.histogram('forget_gate'+temp_feat + '_time_step_' + str(time_step), f)
                        tf.summary.histogram('output_gate'+temp_feat + '_time_step_' + str(time_step), o)

                for st_feat in self.st_features_names:
                    inputs = self.inputs[st_feat][:,time_step,:]
                    state =   states[st_feat]
                    scope ="lstm_"+st_feat
                    i,j,f,o = get_gates(edgesRNN[st_feat],inputs, state, scope)
                    outputs[st_feat], states[st_feat] = edgesRNN[st_feat](inputs,state, scope=scope)
                    infos[st_feat]['input_gates'].append(i)
                    infos[st_feat]['forget_gates'].append(f)
                    infos[st_feat]['modulated_input_gates'].append(j)
                    infos[st_feat]['output_gates'].append(o)
                    infos[st_feat]['activations'].append(outputs[st_feat])
                    infos[st_feat]['state_c'].append(states[st_feat][0])
                    infos[st_feat]['state_m'].append(states[st_feat][1])
                    if self.save_summaries:
                        tf.summary.histogram('activations_'+st_feat + '_time_step_' + str(time_step),outputs[st_feat])
                        tf.summary.histogram('state_C_'+st_feat + '_time_step_' + str(time_step),states[st_feat][0])
                        tf.summary.histogram('state_M_'+st_feat + '_time_step_' + str(time_step),states[st_feat][1])
                        tf.summary.histogram('input_gate'+st_feat + '_time_step_' + str(time_step), i)
                        tf.summary.histogram('modulated_input_gate'+st_feat + '_time_step_' + str(time_step), j)
                        tf.summary.histogram('forget_gate'+st_feat + '_time_step_' + str(time_step), f)
                        tf.summary.histogram('output_gate'+st_feat + '_time_step_' + str(time_step), o)

                node_inputs['face'] = tf.concat(1,[outputs['face-face'], outputs['face-rightArm'], outputs['face-leftArm'], outputs['face-belly']])

                node_inputs['belly'] = tf.concat(1,[outputs['belly-belly'], outputs['face-belly'], outputs['belly-leftArm'], outputs['belly-rightArm'],
                                            outputs['belly-leftLeg'], outputs['belly-rightLeg']])

                node_inputs['arms'] = tf.concat(1,[outputs['rightArm-rightArm'], outputs['leftArm-leftArm'], outputs['leftArm-rightArm'], outputs['face-rightArm'],
                                            outputs['face-leftArm'],outputs['belly-rightArm'], outputs['belly-leftArm']])

                node_inputs['legs'] = tf.concat(1,[outputs['rightLeg-rightLeg'], outputs['leftLeg-leftLeg'],outputs['leftLeg-rightLeg'], outputs['belly-rightLeg'], outputs['belly-leftLeg']])

                for node_name in nodes_names:
                    inputs = node_inputs[node_name]
                    state =   states[node_name]
                    scope ="lstm_"+node_name
                    i,j,f,o = get_gates(nodesRNN[node_name],inputs, state, scope)
                    node_outputs[node_name], states[node_name] = nodesRNN[node_name](inputs,state, scope=scope)
                    infos[node_name]['input_gates'].append(i)
                    infos[node_name]['forget_gates'].append(f)
                    infos[node_name]['modulated_input_gates'].append(j)
                    infos[node_name]['output_gates'].append(o)
                    infos[node_name]['activations'].append(node_outputs[node_name])
                    infos[node_name]['state_c'].append(states[node_name][0])
                    infos[node_name]['state_m'].append(states[node_name][1])
                    if self.save_summaries:
                        tf.summary.histogram('activations_'+node_name + '_time_step_' + str(time_step),node_outputs[node_name])
                        tf.summary.histogram('state_C_'+node_name + '_time_step_' + str(time_step),states[node_name][0])
                        tf.summary.histogram('state_M_'+node_name + '_time_step_' + str(time_step),states[node_name][1])
                        tf.summary.histogram('input_gate'+node_name + '_time_step_' + str(time_step), i)
                        tf.summary.histogram('modulated_input_gate'+node_name + '_time_step_' + str(time_step), j)
                        tf.summary.histogram('forget_gate'+node_name + '_time_step_' + str(time_step), f)
                        tf.summary.histogram('output_gate'+node_name + '_time_step_' + str(time_step), o)

                fullbody_input = tf.concat(1,[node_outputs['face'],node_outputs['belly'],node_outputs['arms'],node_outputs['legs']])
                inputs = fullbody_input
                state =  states['fullbody']
                scope ="fullbody"
                i,j,f,o = get_gates(fullbodyRNN,inputs, state, scope)
                final_output, states['fullbody'] = fullbodyRNN(inputs,state, scope=scope)
                infos["fullbody"]['input_gates'].append(i)
                infos["fullbody"]['forget_gates'].append(f)
                infos["fullbody"]['modulated_input_gates'].append(j)
                infos["fullbody"]['output_gates'].append(o)
                infos["fullbody"]['activations'].append(final_output)
                infos["fullbody"]['state_c'].append(states['fullbody'][0])
                infos["fullbody"]['state_m'].append(states['fullbody'][1])
                if self.save_summaries:
                    tf.summary.histogram('activations_fullbody' + '_time_step_' + str(time_step),final_output)
                    tf.summary.histogram('state_C_fullbody' + '_time_step_' + str(time_step),states['fullbody'][0])
                    tf.summary.histogram('state_M_fullbody' + '_time_step_' + str(time_step),states['fullbody'][1])
                    tf.summary.histogram('input_gate_fullbody'+ '_time_step_' + str(time_step), i)
                    tf.summary.histogram('modulated_input_gate_fullbody' + '_time_step_' + str(time_step), j)
                    tf.summary.histogram('forget_gate_fullbody'+ '_time_step_' + str(time_step), f)
                    tf.summary.histogram('output_gate_fullbody' + '_time_step_' + str(time_step), o)
                final_outputs.append(final_output)

        self.infos = infos
        output = tf.concat(1,final_outputs, name="output_lastCells")
        self.final_states = states
        self.logits = tf.matmul(output, weights['out'], name="logits") + biases['out']
        with tf.name_scope('cross_entropy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets)
            with tf.name_scope('total'):
                self.cost = tf.reduce_mean(loss)
        if self.save_summaries:
            tf.summary.scalar('cross_entropy', self.cost)

        tvars = tf.trainable_variables()
        if not forward_only:
            if self.GD:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

                clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.cost,tvars), self.max_grad_norm)
                self.gradients_norm = norm
                self.updates = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=self.global_step)
            else:
                aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.adam_epsilon)
                gradients_and_params = optimizer.compute_gradients(self.cost, tvars,
                                                             aggregation_method=aggregation_method)
                gradients, params = zip(*gradients_and_params)
                norm = tf.global_norm(gradients)
                self.gradients_norm = norm
                self.updates = optimizer.apply_gradients(zip(gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        self.merged = tf.summary.merge_all()
        if self.save_summaries:
            self.train_writer = tf.summary.FileWriter(log_dir + '/train')
            self.test_writer = tf.summary.FileWriter(log_dir + '/test')

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


    def step(self,session,temp_inputs, st_inputs, targets, forward_only, get_info = False):
        input_feed = {}
        for temp_features_name in self.temp_features_names:
            input_feed[self.inputs[temp_features_name].name] = temp_inputs[temp_features_name]

        for st_features_name in self.st_features_names:
            input_feed[ self.inputs[st_features_name].name] = st_inputs[st_features_name]

        input_feed[self.targets.name] = targets

        if not forward_only:
            output_feed = [self.updates, self.gradients_norm, self.cost]
        else:
            if get_info:
                output_feed = [self.cost, tf.nn.softmax(self.logits), self.infos]
            else:
                output_feed = [self.cost, tf.nn.softmax(self.logits)]

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            return outputs[1], outputs[2], None # returs gradients norm and cost and no output
        else:
            if get_info:
                return  outputs[2], outputs[0], outputs[1]  # no gradients, cost and output
            else:
                return None, outputs[0], outputs[1]  # no gradients, cost and output

    def step_with_summary(self,session,temp_inputs, st_inputs, targets, forward_only):
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

        outputs, summary = session.run([output_feed, self.merged], input_feed)


        if not forward_only:
            return outputs[1], outputs[2], None, summary # returs gradients norm and cost and no output
        else:
            return None, outputs[0], outputs[1], summary  # no gradients, cost and output

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

        for i in xrange(len(data)-1):
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

