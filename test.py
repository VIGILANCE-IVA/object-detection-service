import cv2 as cv
from camera_threading.streamer import Capture

from yolo import model


def  on_stream(frame):
    predictions = model.predict(frame)
    print(predictions)


camera = Capture('rtsp://admin:WILLIAM%231@192.168.0.137/Streaming/Channels/1001', 0)
camera.set(3, 320)
camera.set(4, 240)
camera.on("frame", on_stream)

camera.start()
