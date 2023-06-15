import jetson.utils
import numpy as np
import cv2
import tensorflow as tf
import time

print("Loading model...")
detector = tf.saved_model.load("tensor_rt")
print("Model loaded. Loading source file...")
source = jetson.utils.videoSource("images/video/")
print("Source file loaded.")
display = jetson.utils.videoOutput("output.mp4", argv=["--headless"])

color = []
color[0] = (190,255,50,100)
color[1] = (255,190,50,100)
color[2] = (190,50,255,100)
color[3] = (190,20,50,100)

print("Starting inference...")
display.Open()
while display.IsStreaming():
    ts = time.clock()
    rgb_img = source.Capture()
    img_width = rgb_img.width
    img_height = rgb_img.height
    # convert to cv2 image (cv2 images are numpy arrays)
    cv_img = jetson.utils.cudaToNumpy(rgb_img)
    # resize as what Tensorflow model expects
    img_in = np.array(cv_img).reshape(
        (1, img_height, img_width, 3)).astype(np.uint8)

    t_pre = time.clock() - ts
    ts = time.clock()

    detections = detector(img_in)

    t_inf = time.clock() - ts
    ts = time.clock()

    # draw bboxes
    threshold = 0.5
    for i in range(0, detections["num_detections"][0].numpy().astype(np.int)):
        if detections["detection_scores"][0][i].numpy() > threshold:
            # the drawing function expect a different format of coordinates
            y_min = detections["detection_boxes"][0][i][0].numpy()
            x_min = detections["detection_boxes"][0][i][1].numpy()
            y_max = detections["detection_boxes"][0][i][2].numpy()
            x_max = detections["detection_boxes"][0][i][3].numpy()
            left = int(x_min*img_width)
            top = int(y_min*img_height)
            right = int(x_max*img_width)
            bottom = int(y_max*img_height)
            det_class = detections["detection_classes"][0][i].numpy()
            jetson.utils.cudaDrawRect(
                rgb_img, (left, top, right, bottom), color[det_class])

    display.Render(rgb_img)

    t_post = time.clock() - ts
    ts = time.clock()

    print("Pre-processing time: {0}\nInference time: {1}\nPost processing time: {2}".format(t_pre, t_inf, t_post))

print("Done.")
