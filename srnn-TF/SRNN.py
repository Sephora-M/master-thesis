import tensorflow as tf

class SRNN(object):

    def __int__(self,edgeRNNs,nodeRNNs,outputLayer,nodeToEdgeConnections,edgeListComplete,cost,
                nodeLabels,learning_rate,clipnorm=0.0,update_type='rsm_prop',weight_decay=0.0, train_for='detection'):
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
        self.setting = locals()
        del self.settings['self']

        self.edgeRNNs = edgeRNNs
        self.nodeRNNs = nodeRNNs
        self.nodeToEdgeConnections = nodeToEdgeConnections
        self.edgeListComplete = edgeListComplete
        self.nodeLabels = nodeLabels
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.weight_decay = weight_decay
        self.outputLayer = outputLayer
        nodeTypes = nodeRNNs.keys()
        edgeTypes = edgeRNNs.keys()

        self.train_for = train_for
        self.cost = {}
        self.X = {}
        self.Y_pr = {}
        self.Y_pr_last_timestep = {}
        self.Y = {}
        self.params = {}
        self.updates = {}
        self.train_node = {}
        self.predict_node = {}
        self.predict_node_last_timestep = {}
        self.masterlayer = {}
        self.grads = {}
        self.predict_node_loss = {}
        self.grad_norm = {}
        self.norm = {}

        self.update_type = update_type
        self.update_type.lr = self.learning_rate
        self.update_type.clipnorm = self.clipnorm

        #self.std = tf.Variable(tf.zeros([1]))

        # connect the edgeTypes layers together --> layer.connect



    def train_model(self,trX_shared_1, trX_1, trY_1, trX_shared_2, trX_2, trY_2, snapshot_rate=1,
                    path=None, epochs=30, batch_size=50, learning_rate_decay=0.97, decay_after=10):

        # do the training loop for the number of epochs given by epochs

