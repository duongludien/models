import yolo_v3

net = yolo_v3.YOLOv3('/home/dldien/luanvan/demo/cfg/yolov3.cfg', '/home/dldien/luanvan/demo/weights/yolov3.weights')
net.load_weights()