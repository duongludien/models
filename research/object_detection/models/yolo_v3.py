import tensorflow as tf
from tensorflow.python.ops import init_ops

slim = tf.contrib.slim

class YOLOv3:

    def __init__(self, cfg_path, wgt_path=None):
        super(YOLOv3, self).__init__()
        self.cfg_path = cfg_path
        self.wgt_path = wgt_path
        self.cfg_blocks = []
        self.graph = tf.Graph()
        self.inputs = None
        self.yolo_outputs = []
    
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
            # begining of a layer
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

    def define_graph(self):
        
        with self.graph.as_default():
            
            batch = int(self.cfg_blocks[0]['batch'])
            width = int(self.cfg_blocks[0]['width'])
            height = int(self.cfg_blocks[0]['height'])
            channels = int(self.cfg_blocks[0]['channels'])
            self.inputs = tf.placeholder(dtype=tf.float32, 
                                         name='input_images', 
                                         shape=[batch, width, height, channels])

            layers = {}
            layers[-1] = self.inputs

            for index, block in enumerate(self.cfg_blocks[1:]):   

                # ====================== convolutional layer ======================
                if block['name'] == 'convolutional':
                    
                    try:
                        batch_normalize = int(block['batch_normalize'])
                        normalizer_fn = slim.batch_norm
                    except:
                        normalizer_fn = None

                    filters = int(block['filters'])
                    size = int(block['size'])
                    stride = int(block['stride'])
                    pad = int(block['pad'])
                    activation = block['activation']

                    if pad:
                        pad = 'SAME'
                    else:
                        pad = 'VALID'

                    """
                    If a `normalizer_fn` is provided (such as `batch_norm`), it is then applied. 
                    Otherwise, if `normalizer_fn` is None and a `biases_initializer` is provided 
                    then a `biases` variable would be created and added the activations.
                    """
                    with tf.variable_scope('conv_{}'.format(index)) as scope:
                        output = slim.conv2d(inputs=layers[index-1],
                                             num_outputs=filters,
                                             kernel_size=[size, size],
                                             stride=stride,
                                             padding=pad,
                                             activation_fn=tf.nn.leaky_relu,
                                             normalizer_fn=normalizer_fn)

                # ====================== shortcut layer ======================
                elif block['name'] == 'shortcut':

                    from_layer = int(block['from'])

                    # No built-in module for this kind of layer
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
                    except:
                        end = 0

                    # Calculate the number of step from index to start and end
                    if start > 0:
                        start = start - index
                    if end > 0:
                        end = end - index

                    if end < 0:
                        output = tf.concat(values=[layers[index + start], layers[index + end]], 
                                           axis=-1,
                                           name='route_{}_{}'.format(start, end))
                    else:
                        output = layers[index + start]                        

                # ====================== upsample layer ======================
                elif block['name'] == 'upsample':
                    stride = int(block['stride'])

                    # Just increase size
                    old_shape = layers[index-1].get_shape().as_list()
                    old_width = old_shape[1]
                    old_height = old_shape[2]
                    
                    new_width = old_width * stride
                    new_height = old_height * stride

                    with tf.variable_scope('upsample_{}'.format(index)) as scope:
                        output = tf.image.resize_images(images=layers[index-1], 
                                                        size=[new_width, new_height], 
                                                        align_corners=True, 
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # ====================== yolo layer ======================
                elif block['name'] == 'yolo':
                    mask = block['mask'].split(',')
                    mask = [int(x) for x in mask]

                    anchors = block['anchors'].split(',')
                    anchors = [int(x) for x in anchors]
                    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                    anchors = [anchors[i] for i in mask]

                    no_of_classes = int(block['classes'])

                    output = self.transform_features_map(layers[index-1], 416, anchors, no_of_classes)
                    print(output)

                # Finally, add this layer output to list
                layers[index] = output

    def transform_features_map(self, features_map, input_dimension, anchors, no_of_classes):
        # Input shape may be [1, 13, 13, 255]

        features_map_shape = features_map.get_shape().as_list()

        batch = features_map_shape[0]
        grid_size = features_map_shape[1]
        no_of_anchors = len(anchors)   
        stride = input_dimension // grid_size     

        with tf.variable_scope('yolo') as scope:
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
            anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
            anchors = tf.tile(anchors, [grid_size * grid_size, 1])
            transformed_bh_bw = tf.multiply(tf.exp(bh_bw), anchors)
            transformed_bh_bw = tf.multiply(transformed_bh_bw, stride, name='bh_bw')
            # print(transformed_bh_bw)
            
            # Transform object confidence
            p  = features_map[:, :, 4]
            transformed_p  = tf.sigmoid(p)
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