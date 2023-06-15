import jetson.utils

source = jetson.utils.videoSource("data/video/")
print("Source file loaded.")
display = jetson.utils.videoOutput("data/thermal_video.mp4",argv=["--headless"])

print("Starting inference...")
display.Open()
while display.IsStreaming():

    img = source.Capture()

    display.Render(img)

print("Done.")
