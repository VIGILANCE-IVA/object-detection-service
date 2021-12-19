import uuid
from collections import OrderedDict

import requests
from camera_threading.streamer import Capture
from core_utils.json import jsonify

from yolo import model


class YoloVideo:
    def __init__(self):
        self.tasks = OrderedDict()

    def webhook(self, url, data):
        requests.post(url, json = jsonify(data))

    async def add_task(self, data):
        if 'webhook' not in data:
            raise Exception("No webhook found!")
            
        if 'video_uri' not in data:
            raise Exception("No video found!")

        if 'delay' in data:
            delay = int(data['delay'])

        if 'allowed_classes' in data:
            allowed_classes = data['allowed_classes']

        def on_frame(frame):
            predictions = model.predict(frame, allowed_classes)
            result = dict(data, **{ 'predictions': predictions })
            self.webhook(data['webhook'], result)

        id = str(uuid.uuid4())
        camera = Capture(data['video_uri'], delay)

        camera.on("frame", on_frame)
        camera.start()

        self.tasks[id] = {
            'data': data,
            'camera': camera
        }

        return id

    async def stop_task(self, task_id):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task['camera'].stop()
            self.tasks.pop(task_id)

        return task_id

yolo_video = YoloVideo()
