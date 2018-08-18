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
                        normalizer_params = {'is_training': is_training}
                    except:
                        normalizer_fn = None
                        normalizer_params = None

                    filters = int(block['filters'])
                    size = int(block['size'])
                    stride = int(block['stride'])
                    pad = int(block['pad'])
                    activation = block['activation']

                    if pad:
                        pad = 'SAME'
                    else:
                        pad = 'VALID'

                    with tf.variable_scope('conv_{}'.format(index)) as scope:
                        # print(layers[index-1])
                        output = slim.conv2d(inputs=layers[index-1],
                                             num_outputs=filters,
                                             kernel_size=[size, size],
                                             stride=stride,
                                             padding=pad,
                                             activation_fn=tf.nn.leaky_relu,
                                             normalizer_fn=normalizer_fn,
                                             normalizer_params=normalizer_params)

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

                # Finally, add this layer output to list
                layers[index] = output