import tensorflow as tf
import numpy as np

class TRACKNET: 
    def __init__(self, batch_size, train = True):
        self.parameters = {}
        self.batch_size = batch_size
        self.target = tf.placeholder(tf.float16, [batch_size, 128, 128, 1])
        self.image = tf.placeholder(tf.float16, [batch_size, 128, 128, 1])
        self.bbox = tf.placeholder(tf.float16, [batch_size, 4])
        self.train = train
        self.wd = 0.0005
    def build(self):
        ########### for target ###########
        # [filter_height, filter_width, in_channels, out_channels]
		# pre-residual blocks
		# conv = tf.nn.conv2d(bottom, kernel, strides, padding='VALID')
        self.target_conv0 = self._basic_cnn(self.target, filter_size = [5, 5, 1, 32], name='target_conv0')
        self.target_pool0 = tf.nn.max_pool(self.target_conv0, ksize = [1, 3, 3, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='target_pool0')
													
		# residual block #1
        self.target_resid_1_connection = self._basic_cnn(self.target_pool0, filter_size = [1, 1, 32, 32], name = 'target_resid_1_connection')
        self.target_resid_1a = self._resid_half(self.target_pool0, filter_size = [3, 3, 32, 32], strides = [1, 2, 2, 1], name = 'target_resid_1a')
        self.target_resid_1b = self._resid_half(self.target_resid_1a, filter_size = [3, 3, 32, 32], strides = [1, 1, 1, 1], name = 'target_resid_1b')
        self.target_resid_1_result = tf.add(self.target_resid_1_connection, self.target_resid_1b)
		
		# residual block #2
        self.target_resid_2_connection = self._basic_cnn(self.target_resid_1_result, filter_size = [1, 1, 32, 32], name = 'target_resid_2_connection')
        self.target_resid_2a = self._resid_half(self.target_resid_1b, filter_size = [3, 3, 32, 32], strides = [1, 2, 2, 1], name = 'target_resid_2a')
        self.target_resid_2b = self._resid_half(self.target_resid_2a, filter_size = [3, 3, 32, 32], strides = [1, 1, 1, 1], name = 'target_resid_2b')
        self.target_resid_2_result = tf.add(self.target_resid_2_connection, self.target_resid_2b)

        ########### for image ###########
        # [filter_height, filter_width, in_channels, out_channels]
		# pre-residual blocks
		# conv = tf.nn.conv2d(bottom, kernel, strides, padding='VALID')
        self.image_conv0 = self._basic_cnn(self.image, filter_size = [5, 5, 1, 32], name='image_conv0')
        self.image_pool0 = tf.nn.max_pool(self.image_conv0, ksize = [1, 3, 3, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='image_pool0')
													
		# residual block #1
        self.image_resid_1_connection = self._basic_cnn(self.image_pool0, filter_size = [1, 1, 32, 32], name = 'image_resid_1_connection')
        self.image_resid_1a = self._resid_half(self.image_pool0, filter_size = [3, 3, 32, 32], strides = [1, 2, 2, 1], name = 'image_resid_1a')
        self.image_resid_1b = self._resid_half(self.image_resid_1a, filter_size = [3, 3, 32, 32], strides = [1, 1, 1, 1], name = 'image_resid_1b')
        self.image_resid_1_result = tf.add(self.image_resid_1_connection, self.image_resid_1b)
		
		# residual block #2
        self.image_resid_2_connection = self._basic_cnn(self.image_resid_1_result, filter_size = [1, 1, 32, 32], name = 'image_resid_2_connection')
        self.image_resid_2a = self._resid_half(self.image_resid_1b, filter_size = [3, 3, 32, 32], strides = [1, 2, 2, 1], name = 'image_resid_2a')
        self.image_resid_2b = self._resid_half(self.image_resid_2a, filter_size = [3, 3, 32, 32], strides = [1, 1, 1, 1], name = 'image_resid_2b')
        self.image_resid_2_result = tf.add(self.image_resid_2_connection, self.image_resid_2b)

        # tensorflow layer: n * w * h * c
        # but caffe layer is: n * c * h * w

        # tensorflow kernel: h * w * in_c * out_c
        # caffe kernel: out_c * in_c * h * w

        ########### Concatnate two layers ###########
        self.concat = tf.concat([self.target_resid_2_result, self.image_resid_2_result], axis = 3) # 0, 1, 2, 3 - > 2, 3, 1, 0

        # important, since caffe has different layer dimension order
        self.concat = tf.transpose(self.concat, perm=[0,3,1,2]) 

        ########### fully connencted layers ###########
        # 6 * 6 * 256 * 2 == 18432
        # assert self.fc1.get_shape().as_list()[1:] == [6, 6, 512]
        self.fc1 = self._fc_relu_layers(self.concat, dim = 512, name = "fc1")
        if (self.train):
            self.fc1 = tf.nn.dropout(self.fc1, 0.5)


        self.fc2 = self._fc_relu_layers(self.fc1, dim = 512, name = "fc2")
        if (self.train):
            self.fc2 = tf.nn.dropout(self.fc2, 0.5)

        self.fc3 = self._fc_relu_layers(self.fc2, dim = 512, name = "fc3")
        if (self.train):
            self.fc3 = tf.nn.dropout(self.fc3, 0.5)

        self.fc4 = self._fc_layers(self.fc3, dim = 4, name = "fc4")

        self.print_shapes()
        self.loss = self._loss_layer(self.fc4, self.bbox ,name = "loss")
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_weight_loss')
        self.loss_wdecay = self.loss + l2_loss

    def _loss_layer(self, bottom, label, name = None):
        diff = tf.subtract(self.fc4, self.bbox)
        diff_flat = tf.abs(tf.reshape(diff,[-1]))
        loss = tf.reduce_sum(diff_flat, name = name)
        return loss
	
	# bottom = self.target, filter_size = [11, 11, 3, 96], strides = [1,4,4,1], name = "target_conv_1")
	
    def _basic_cnn(self, input, filter_size, bias_init = 0.0, trainable = True, name = None):
        kernel = tf.Variable(tf.truncated_normal(filter_size, dtype=tf.float16, stddev=1e-2), trainable=trainable, name='weights')
        biases = tf.Variable(tf.constant(bias_init, shape=[filter_size[3]], dtype=tf.float16), trainable=trainable, name='biases')
        self.parameters[name] = [kernel, biases]
        conv = tf.nn.conv2d(input, kernel, strides=[1, 2, 2, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        return out
		
    def _resid_half(self, input, filter_size, strides, bias_init = 0.0, trainable = True, name = None):
        with tf.name_scope(name) as scope:
            normalized_output = tf.layers.batch_normalization(input, axis=1)
            relud_output = tf.nn.relu(normalized_output, name=scope)
            _activation_summary(relud_output)
            kernel = tf.Variable(tf.truncated_normal(filter_size, dtype=tf.float16,
                                                     stddev=1e-2), trainable=trainable, name='weights')
            biases = tf.Variable(tf.constant(bias_init, shape=[filter_size[3]], dtype=tf.float16), trainable=trainable, name='biases')
            self.parameters[name] = [kernel, biases]
            conv = tf.nn.conv2d(relud_output, kernel, strides, padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            return out

    """
    Parameters: 
    bottom (tf.placeholder(tf.float16, [batch_size, 227, 227, 3]): image input 
    filter_size (array): w, h, in_channels, out_channels
    strides (array): batch, height, width, channels
    pad (int): pad int by width & height 
    bias_init (double): what to initiate bias with 
    group (1 or 2): determines whether to split this apart
    trainable (boolean): determines whether weights and bias can be changed (aka trained)

    Returns:
    tensor: result from relu activation
    """
    def _conv_relu_layer(self, input, filter_size, strides, pad = 0,bias_init = 0.0, trainable = False, name = None):
        with tf.name_scope(name) as scope:

            if (pad > 0):
                paddings = [[0,0],[pad,pad],[pad,pad],[0,0]]
                input = tf.pad(bottom, paddings, "CONSTANT")
            kernel = tf.Variable(tf.truncated_normal(filter_size, dtype=tf.float16,
                                                     stddev=1e-2), trainable=trainable, name='weights')
            biases = tf.Variable(tf.constant(bias_init, shape=[filter_size[3]], dtype=tf.float16), trainable=trainable, name='biases')
            self.parameters[name] = [kernel, biases]
            conv = tf.nn.conv2d(bottom, kernel, strides, padding='VALID')
            out = tf.nn.bias_add(conv, biases)

            # if not tf.get_variable_scope().reuse:
            #     weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd,
            #                            name='kernel_loss')
            #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
            #                      weight_decay)
			
            out2 = tf.nn.relu(out, name=scope)
            _activation_summary(out2)
            out2 = tf.Print(out2, [tf.shape(out2)], message='Shape of %s' % name, first_n = 1, summarize=4)
			
			
			
            return out2

    def _fc_relu_layers(self, bottom, dim, name = None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                    dtype=tf.float16, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float16), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            fc_weights = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias)
            self.parameters[name] = [weights, bias]


            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                       name='fc_relu_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)



            top = tf.nn.relu(fc_weights, name=scope)
            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return top

    def _fc_layers(self, bottom, dim, name = None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                    dtype=tf.float16, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float16), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            top = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias, name=scope)
            self.parameters[name] = [weights, bias]

            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                       name='fc_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)

            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return top
    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var
    def print_shapes(self):
        print("%s:"%(self.image_conv0),self.image_conv0.get_shape().as_list())
        print("%s:"%(self.image_pool0),self.image_pool0.get_shape().as_list())
        print("%s:"%(self.image_resid_1_connection),self.image_resid_1_connection.get_shape().as_list())
        print("%s:"%(self.image_resid_1a), self.image_resid_1a.get_shape().as_list())
        print("%s:"%(self.image_resid_1b), self.image_resid_1b.get_shape().as_list())
        print("%s:"%(self.image_resid_2_connection),self.image_resid_2_connection.get_shape().as_list())
        print("%s:"%(self.image_resid_2a), self.image_resid_2a.get_shape().as_list())
        print("%s:"%(self.image_resid_2b), self.image_resid_2b.get_shape().as_list())
        print("%s:"%(self.concat),self.concat.get_shape().as_list())
        print("%s:"%(self.fc1),self.fc1.get_shape().as_list())
        print("%s:"%(self.fc2),self.fc2.get_shape().as_list())
        print("%s:"%(self.fc3),self.fc3.get_shape().as_list())
        print("%s:"%(self.fc4),self.fc4.get_shape().as_list())
        print("kernel_sizes:")
        for key in self.parameters:
            print("%s:"%(key),self.parameters[key][0].get_shape().as_list())

    def load_weight_from_dict(self,weights_dict,sess):
        # for convolutional layers
        sess.run(self.parameters['target_conv0'][0].assign(weights_dict['conv0']['weights']))
        sess.run(self.parameters['target_resid_1_connection'][0].assign(weights_dict['conv1']['weights']))
        sess.run(self.parameters['target_resid_1a'][0].assign(weights_dict['conv1a']['weights']))
        sess.run(self.parameters['target_resid_1b'][0].assign(weights_dict['conv1b']['weights']))
        sess.run(self.parameters['target_resid_2_connection'][0].assign(weights_dict['conv2']['weights']))
        sess.run(self.parameters['target_resid_2a'][0].assign(weights_dict['conv2a']['weights']))
        sess.run(self.parameters['target_resid_2b'][0].assign(weights_dict['conv2b']['weights']))
        sess.run(self.parameters['image_conv0'][0].assign(weights_dict['conv0_p']['weights']))
        sess.run(self.parameters['image_resid_1_connection'][0].assign(weights_dict['conv1_p']['weights']))
        sess.run(self.parameters['image_resid_1a'][0].assign(weights_dict['conv1a_p']['weights']))
        sess.run(self.parameters['image_resid_1b'][0].assign(weights_dict['conv1b_p']['weights']))
        sess.run(self.parameters['image_resid_2_connection'][0].assign(weights_dict['conv2_p']['weights']))
        sess.run(self.parameters['image_resid_2a'][0].assign(weights_dict['conv2a_p']['weights']))
        sess.run(self.parameters['image_resid_2b'][0].assign(weights_dict['conv2b_p']['weights']))

        sess.run(self.parameters['target_resid_1_connection'][1].assign(weights_dict['conv1']['bias']))
        sess.run(self.parameters['target_resid_1a'][1].assign(weights_dict['conv1a']['bias']))
        sess.run(self.parameters['target_resid_1b'][1].assign(weights_dict['conv1b']['bias']))
        sess.run(self.parameters['target_resid_2_connection'][1].assign(weights_dict['conv2']['bias']))
        sess.run(self.parameters['target_resid_2a'][1].assign(weights_dict['conv2a']['bias']))
        sess.run(self.parameters['target_resid_2b'][1].assign(weights_dict['conv2b']['bias']))
        sess.run(self.parameters['image_conv0'][1].assign(weights_dict['conv0_p']['weights']))
        sess.run(self.parameters['image_resid_1_connection'][1].assign(weights_dict['conv1_p']['bias']))
        sess.run(self.parameters['image_resid_1a'][1].assign(weights_dict['conv1a_p']['bias']))
        sess.run(self.parameters['image_resid_1b'][1].assign(weights_dict['conv1b_p']['bias']))
        sess.run(self.parameters['image_resid_2_connection'][1].assign(weights_dict['conv2_p']['bias']))
        sess.run(self.parameters['image_resid_2a'][1].assign(weights_dict['conv2a_p']['bias']))
        sess.run(self.parameters['image_resid_2b'][1].assign(weights_dict['conv2b_p']['bias']))

        # for fully connected layers
        sess.run(self.parameters['fc1'][0].assign(weights_dict['fc6-new']['weights']))
        sess.run(self.parameters['fc2'][0].assign(weights_dict['fc7-new']['weights']))
        sess.run(self.parameters['fc3'][0].assign(weights_dict['fc7-newb']['weights']))
        sess.run(self.parameters['fc4'][0].assign(weights_dict['fc8-shapes']['weights']))

        sess.run(self.parameters['fc1'][1].assign(weights_dict['fc6-new']['bias']))
        sess.run(self.parameters['fc2'][1].assign(weights_dict['fc7-new']['bias']))
        sess.run(self.parameters['fc3'][1].assign(weights_dict['fc7-newb']['bias']))
        sess.run(self.parameters['fc4'][1].assign(weights_dict['fc8-shapes']['bias']))


    
    def test(self):
        sess = tf.Session()
        a = np.full((self.batch_size,128,128,1), 1) # numpy.full(shape, fill_value, dtype=None, order='C')
        b = np.full((self.batch_size,128,128,1), 2)
        sess.run(tf.global_variables_initializer())

        sess.run([self.fc4],feed_dict={self.image:a, self.target:b})





def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)

if __name__ == "__main__":
    tracknet = TRACKNET(10)
    tracknet.build()
    sess = tf.Session()
    a = np.full((tracknet.batch_size,128,128,1), 1)
    b = np.full((tracknet.batch_size,128,128,1), 2)
    sess.run(tf.global_variables_initializer())
    sess.run([tracknet.image_pool5],feed_dict={tracknet.image:a, tracknet.target:b})



