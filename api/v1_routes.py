import aiohttp_cors
from aiohttp import web

from .detection import DetectionApi, VideoDetectionApi
from .webhook import WebHook
from .middlewares.exception import error_middleware

app = web.Application(middlewares=[error_middleware])

app.router.add_view('/detections', DetectionApi)
app.router.add_view('/detections/video', VideoDetectionApi)
app.router.add_view('/webhook', WebHook)

# cors
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(),
})

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route, webview=True)
