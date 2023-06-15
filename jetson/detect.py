import jetson.inference
import jetson.utils

net = jetson.inference.detectNet(argv=['--model=exported_models/ssdlite_mobilenet_v3_100epochs.onnx',
                                       '--labels=my_model_path/labels.txt',
                                       '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes',
                                       threshold=0.5)
camera = jetson.utils.videoSource("images/video")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("jetson/out/result.mp4") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
