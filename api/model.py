import os
import threading

from aiohttp.web import Response, View
from aiohttp_cors import CorsViewMixin
from convert_config import config
from save_model import save_tf


class ModelApi(View, CorsViewMixin):
    async def post(self):
        body = await self.request.post()
        files = {}

        for ref in body:
            if hasattr(body[ref], '__class__'):
                item = body[ref]
                file_path = os.path.join('./', 'data', f'{ref}_{item.filename}')
                files[ref] = file_path
                f = open(file_path, "wb")
                f.write(item.file.read())
                f.close()
                
        # convert model
        if 'weights' in files:
            config.weights = files['weights']
            if 'classes' in files:
                config.classes = files['classes']
            thread = threading.Thread(target=save_tf, args=(config,))
            thread.start()

        return Response(
            text='ok',
            status=200
        )
