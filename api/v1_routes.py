import aiohttp_cors
from aiohttp import web

from .detection import DetectionApi, VideoDetectionApi
from .middlewares.exception import error_middleware
from .model import ModelApi
from .webhook import WebHook

app = web.Application(middlewares=[error_middleware])

app.router.add_view('/detections', DetectionApi)
app.router.add_view('/detections/video', VideoDetectionApi)
app.router.add_view('/webhook', WebHook)
app.router.add_view('/model', ModelApi)

# cors
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(),
})

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route, webview=True)
