import json

import cv2 as cv
import numpy as np
from aiohttp.web import Response, View
from aiohttp_cors import CorsViewMixin
from core_utils.json import jsonify
from yolo import model
from yolo_video import yolo_video


class DetectionApi(View, CorsViewMixin):
    async def post(self):
        try:
            body = await self.request.post()
            try: 
                # set allowed classes
                allowed_classes = json.loads(body['allowed_classes'])
            except:
                allowed_classes = []

            # decode image
            img = cv.imdecode(np.fromstring(body['image'].file.read(), np.uint8), cv.IMREAD_UNCHANGED)
            predictions = model.predict(img, allowed_classes)

            return Response(
                text=jsonify(predictions),
                content_type="application/json",
                status=200
            )

        except BaseException as e:
            return Response(
                text=json.dumps({'message': str(e)}),
                content_type="application/json",
                status=400
            )

class VideoDetectionApi(View, CorsViewMixin):
    async def post(self):
        try:
            body = await self.request.json()
            task_id = await yolo_video.add_task(body)

            return Response(
                text=jsonify({'task_id': task_id }),
                content_type="application/json",
                status=200
            )

        except BaseException as e:
            return Response(
                text=json.dumps({'message': str(e)}),
                content_type="application/json",
                status=400
            )

    async def put(self):
        body = await self.request.json()
        await yolo_video.stop_task(body['task_id'])

        return Response(
            text=jsonify({'task_id': body['task_id'] }),
            content_type="application/json",
            status=200
        )
