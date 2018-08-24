import tensorflow as tf
import yolo_v3
import cv2
import numpy as np
import time


def resize_image(image, size):
    height = image.shape[0]
    width = image.shape[1]

    # Choose the large dimension to resize and calculate other dimension based on it
    if height > width:
        new_height = size
        stride = size / height
        new_width = int(width * stride)
    else:
        new_width = size
        stride = size / width
        new_height = int(height * stride)

    image = cv2.resize(image, (new_width, new_height))

    # Add canvas to keep aspect ratio
    canvas = np.full((size, size, 3), 128)
    canvas[(size - new_height) // 2: (size - new_height) // 2 + new_height,
           (size - new_width) // 2: (size - new_width) // 2 + new_width, :] = image

    # Normalizing it
    image = canvas.astype(np.uint8)
    image = image / 255.0

    return image


def non_max_suppression():
    pass


net = yolo_v3.YOLOv3('/home/dldien/luanvan/demo/cfg/yolov3.cfg', '/home/dldien/luanvan/demo/weights/yolov3.weights')

writer = tf.summary.FileWriter(logdir='/tmp/logdir/',
                               graph=net.graph)

with tf.Session(graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(net.load_weights())

    img = cv2.imread('/home/dldien/luanvan/demo/dog-cycle-car.png')
    img = resize_image(img, 416)
    img = np.expand_dims(img, 0)

    start = time.time()
    outputs = sess.run(net.outputs, feed_dict={net.inputs: img})
    end = time.time()

    print('Total time: %f' % (end - start))

    writer.close()
