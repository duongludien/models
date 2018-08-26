import tensorflow as tf
import numpy as np


class YOLOv3:

    def __init__(self, cfg_path, weights_path=None):
        super(YOLOv3, self).__init__()

        # load config from file
        self.net_info = None
        self.cfg_path = cfg_path
        self.cfg_blocks = []
        self.load_config()

        # define static graph
        self.graph = tf.Graph()
        self.inputs = None
        self.weights_list = []
        self.outputs = None
        self.define_graph()

        # load weights from file
        # self.weights_path = weights_path
        if weights_path is not None:
            self.load_weights_ops = self.load_weights(weights_path)
        else:
            self.load_weights_ops = None

    def load_config(self):
        """
        Parsing configuration file to a list of layers with their attributes
        Return:
            A list of layers (as dicts)
        """

        file = open(self.cfg_path, 'rt')
        lines = file.read().split('\n')
        file.close()

        # get rid of empty lines
        lines = [line for line in lines if len(line) != 0]

        # get rid of comment lines
        lines = [line for line in lines if line[0] != '#']

        # clean spaces
        lines = [line.lstrip().rstrip() for line in lines]

        # block is a layer
        block = {}

        for line in lines:
            # beginning of a layer
            if line[0] == '[':
                # append previous block to self.cfg_blocks, re-initialize block
                if block != {}:
                    self.cfg_blocks.append(block)
                    block = {}
                block['name'] = line[1:-1].rstrip()
            else:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()
        self.cfg_blocks.append(block)
        self.net_info = self.cfg_blocks[0]

    def define_graph(self):

        with self.graph.as_default():

            batch_size = int(self.net_info['batch'])
            width = int(self.net_info['width'])
            height = int(self.net_info['height'])
            channels = int(self.net_info['channels'])

            self.inputs = tf.placeholder(dtype=tf.float32,
                                         name='input_images',
                                         shape=[batch_size, width, height, channels])

            # The first layer is input images
            layers = {-1: self.inputs}
            previous_filters = channels

            # self.cfg_blocks[0] is net_info, so we start with 1
            for index, block in enumerate(self.cfg_blocks[1:]):

                # ====================== convolutional layer ======================
                if block['name'] == 'convolutional':

                    filters = int(block['filters'])
                    size = int(block['size'])
                    stride = int(block['stride'])
                    pad = int(block['pad'])
                    activation = block['activation']
                    try:
                        batch_normalize = int(block['batch_normalize'])
                    except KeyError:
                        batch_normalize = 0

                    if pad:
                        pad = 'SAME'
                    else:
                        pad = 'VALID'

                    with tf.variable_scope('conv_{}'.format(index)):

                        weights = tf.get_variable(
                            initializer=tf.truncated_normal(shape=[size, size, previous_filters, filters],
                                                            stddev=1e-1,
                                                            dtype=tf.float32),
                            trainable=True,
                            name='weights')

                        output = tf.nn.conv2d(input=layers[index - 1],
                                              filter=weights,
                                              strides=[1, stride, stride, 1],
                                              padding=pad,
                                              name='conv')
                        # print(output)

                        if batch_normalize:

                            beta_offset = tf.get_variable(initializer=tf.zeros_initializer(),
                                                          shape=[filters],
                                                          trainable=True,
                                                          name='beta_offset')
                            self.weights_list.append(beta_offset)

                            gamma_scale = tf.get_variable(initializer=tf.zeros_initializer(),
                                                          shape=[filters],
                                                          trainable=True,
                                                          name='gamma_scale')
                            self.weights_list.append(gamma_scale)

                            mean = tf.get_variable(initializer=tf.zeros_initializer(),
                                                   shape=[filters],
                                                   name='mean')
                            self.weights_list.append(mean)

                            variance = tf.get_variable(initializer=tf.zeros_initializer(),
                                                       shape=[filters],
                                                       name='variance')
                            self.weights_list.append(variance)

                            output = tf.nn.batch_normalization(x=output,
                                                               mean=mean,
                                                               variance=variance,
                                                               offset=beta_offset,
                                                               scale=gamma_scale,
                                                               variance_epsilon=1e-05,
                                                               name='batch_normalize')
                            # print(output)

                        else:

                            bias = tf.get_variable(initializer=tf.truncated_normal(shape=[filters],
                                                                                   stddev=1e-1,
                                                                                   dtype=tf.float32),
                                                   name='bias')
                            self.weights_list.append(bias)

                            output = tf.nn.bias_add(value=output,
                                                    bias=bias,
                                                    name='add_bias')
                            # print(output)

                        if activation == 'leaky':
                            output = tf.nn.leaky_relu(features=output,
                                                      alpha=0.1,
                                                      name='leaky_relu')

                        self.weights_list.append(weights)

                    previous_filters = filters

                # ====================== shortcut layer ======================
                elif block['name'] == 'shortcut':

                    from_layer = int(block['from'])

                    # Just add 2 layers (the previous and the layers from_layer)
                    # So the number of filters will not change
                    output = tf.add(layers[index - 1], layers[index + from_layer],
                                    name='shortcut_{}'.format(index))

                # ====================== route layer ======================
                elif block['name'] == 'route':

                    routes = block['layers'].split(',')

                    start = int(routes[0])
                    try:
                        end = int(routes[1])
                    except IndexError:
                        end = 0

                    # Calculate the number of step from index to start and end
                    if start > 0:
                        start = start - index
                    if end > 0:
                        end = end - index

                    if end < 0:
                        output = tf.concat(values=[layers[index + start], layers[index + end]],
                                           axis=-1,
                                           name='concat_{}_{}'.format(start, end))
                    else:
                        output = layers[index + start]

                    previous_filters = output.get_shape().as_list()[-1]

                # ====================== upsample layer ======================
                elif block['name'] == 'upsample':
                    stride = int(block['stride'])

                    # Just increase size
                    old_shape = layers[index - 1].get_shape().as_list()
                    old_width = old_shape[1]
                    old_height = old_shape[2]

                    new_width = old_width * stride
                    new_height = old_height * stride

                    with tf.variable_scope('upsample_{}'.format(index)):
                        output = tf.image.resize_images(images=layers[index - 1],
                                                        size=[new_width, new_height],
                                                        align_corners=True,
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # ====================== yolo layer ======================
                elif block['name'] == 'yolo':
                    mask = block['mask'].split(',')
                    mask = [int(x) for x in mask]

                    anchors = block['anchors'].split(',')
                    anchors = [int(x) for x in anchors]
                    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                    anchors = [anchors[i] for i in mask]

                    no_of_classes = int(block['classes'])

                    output = self.transform_features_map(layers[index - 1], anchors, no_of_classes)

                    if self.outputs is not None:
                        self.outputs = tf.concat(values=[self.outputs, output],
                                                 axis=1)
                    else:
                        self.outputs = output

                # Finally, add this layer output to list
                # print(output)
                layers[index] = output

    def transform_features_map(self, features_map, anchors, no_of_classes):
        # Input shape may be [1, 13, 13, 255]

        features_map_shape = features_map.get_shape().as_list()

        batch = features_map_shape[0]
        grid_size = features_map_shape[1]
        no_of_anchors = len(anchors)
        input_dimension = int(self.net_info['width'])
        stride = input_dimension // grid_size

        with tf.variable_scope('yolo'):
            # Reshape to [1, 13, 13, 3, 85]
            features_map = tf.reshape(features_map, [batch, grid_size, grid_size, no_of_anchors, 5 + no_of_classes])
            # Reshape to [1, 13 x 13 x 3, 85]
            features_map = tf.reshape(features_map, [batch, grid_size * grid_size * no_of_anchors, 5 + no_of_classes])

            # Transform bx, by
            bx_by = features_map[:, :, 0:2]
            bx_by = tf.sigmoid(bx_by)

            grid = tf.range(start=0, limit=grid_size, delta=1, dtype=tf.float32)
            x_offset, y_offset = tf.meshgrid(grid, grid)
            x_offset = tf.reshape(x_offset, [-1, 1])
            y_offset = tf.reshape(y_offset, [-1, 1])
            x_offset = tf.tile(x_offset, [no_of_anchors, 1])
            y_offset = tf.tile(y_offset, [no_of_anchors, 1])
            offset = tf.concat([x_offset, y_offset], axis=1)
            offset = tf.reshape(offset, [1, -1, 2])

            transformed_bx_by = tf.add(bx_by, offset)
            transformed_bx_by = tf.multiply(transformed_bx_by, stride, name='bx_by')
            # print(transformed_bx_by)

            # Transform bh, bw
            bh_bw = features_map[:, :, 2:4]
            anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
            anchors = tf.tile(anchors, [grid_size * grid_size, 1])
            transformed_bh_bw = tf.multiply(tf.exp(bh_bw), anchors)
            transformed_bh_bw = tf.multiply(transformed_bh_bw, stride, name='bh_bw')
            # print(transformed_bh_bw)

            # Transform object confidence
            p = features_map[:, :, 4]
            transformed_p = tf.sigmoid(p)
            transformed_p = tf.reshape(transformed_p, [batch, -1, 1], name='p')
            # print(transformed_p)

            # Transform class scores
            class_scores = features_map[:, :, 5:]
            transformed_class_scores = tf.sigmoid(class_scores, name='class_scores')
            # print(transformed_class_scores)

            transformed_features_map = tf.concat([transformed_bx_by,
                                                  transformed_bh_bw,
                                                  transformed_p,
                                                  transformed_class_scores],
                                                 name='transformed_feature_map',
                                                 axis=-1)

        return transformed_features_map

    def load_weights(self, weights_path):
        file = open(weights_path, 'rb')

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(file, dtype=np.int32, count=5)
        print('Major version: %d' % header[0])
        print('Minor version: %d' % header[1])
        print('Subversion: %d' % header[2])
        print('Images seen by the network (during training): %d %d' % (header[3], header[4]))

        weights = np.fromfile(file, dtype=np.float32)
        pointer = 0

        file.close()

        with self.graph.as_default():
            with tf.variable_scope('load_weights'):
                assign_ops = []
                for i in range(len(self.weights_list)):
                    weights_shape = self.weights_list[i].get_shape().as_list()
                    no_of_params = np.prod(weights_shape)

                    values = np.reshape(weights[pointer:pointer + no_of_params], weights_shape)
                    pointer += no_of_params
                    assign_op = tf.assign(self.weights_list[i], values)

                    assign_ops.append(assign_op)

        return assign_ops

    def non_max_suppression(self, object_confidence_threshold=0.25):
        batch_size = int(self.net_info['batch'])

        with self.graph.as_default():
            with tf.variable_scope(name_or_scope='non_max_suppression'):

                result_existed = False

                for batch in range(batch_size):

                    output = self.outputs[batch]

                    # Apply threshold to this batch
                    object_confidence = output[:, 4]
                    object_confidence_mask = object_confidence > object_confidence_threshold
                    output = tf.boolean_mask(output, object_confidence_mask)
                    # print('Thresholded:', output)

                    object_confidence = tf.expand_dims(output[:, 4], axis=-1, name='object_confidence')
                    # print('Object confidence:', object_confidence)

                    # Convert bx, by, bh, bw to top-left and right-bottom
                    bx = output[:, 0]
                    by = output[:, 1]
                    bh = output[:, 2]
                    bw = output[:, 3]
                    top = tf.expand_dims(by - bh / 2,
                                         axis=-1,
                                         name='top')
                    left = tf.expand_dims(bx - bw / 2,
                                          axis=-1,
                                          name='left')
                    right = tf.expand_dims(bx + bw / 2,
                                           axis=-1,
                                           name='right')
                    bottom = tf.expand_dims(by + bh / 2,
                                            axis=-1,
                                            name='bottom')
                    # print('Top:', top)
                    # print('Left:', left)
                    # print('Right:', right)
                    # print('Bottom:', bottom)

                    max_class_scores = tf.reduce_max(output[:, 5:],
                                                     axis=-1,
                                                     keepdims=True,
                                                     name='class_scores')
                    # print('Max class scores:', max_class_scores)

                    class_indices = tf.cast(tf.argmax(output[:, 5:],
                                                      axis=-1),
                                            tf.float32)
                    class_indices = tf.expand_dims(class_indices, axis=-1, name='class_indices')
                    # print('Class indices:', class_indices)

                    # Concatenate all values
                    this_batch_result = tf.concat(values=[left, top, right, bottom, object_confidence, max_class_scores, class_indices],
                                                  axis=-1)
                    this_batch_result = tf.expand_dims(this_batch_result, axis=0, name='transformed_predictions')

                    if not result_existed:
                        result = this_batch_result
                        result_existed = True
                    else:
                        result = tf.concat(values=[result, this_batch_result],
                                           axis=0,
                                           name='NMS_result')
                return result
