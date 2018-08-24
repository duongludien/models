import tensorflow as tf
import yolo_v3


net = yolo_v3.YOLOv3('/home/dldien/luanvan/demo/cfg/yolov3.cfg', '/home/dldien/luanvan/demo/weights/yolov3.weights')

writer = tf.summary.FileWriter(logdir='/tmp/logdir/',
                               graph=net.graph)

with tf.Session(graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(net.load_weights())
    writer.close()
