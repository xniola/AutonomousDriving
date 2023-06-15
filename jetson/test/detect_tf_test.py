import tensorflow as tf
import jetson.utils
import numpy as np

detector = tf.saved_model.load("my_model/tensor_rt")
source = jetson.utils.videoSource("file://data/sample.jpeg")  

img = source.Capture()
img_np = jetson.utils.cudaToNumpy(img)
img_h, img_w, _ = img_np.shape
img_in = np.array(img_np).reshape((1,img_h,img_w,3)).astype(np.uint8)

detections = detector(img_in)
threshold = 0.5
for i in range(0,detections["num_detetions"][0][0].numpy()):
    if detections["detection_scores"][0][i].numpy() > threshold:
        x = detections["detection_boxes"][0][i][0].numpy()
        y = detections["detection_boxes"][0][i][1].numpy()
        h = detections["detection_boxes"][0][i][2].numpy()
        w = detections["detection_boxes"][0][i][3].numpy()
    print(x,y,w,h)
