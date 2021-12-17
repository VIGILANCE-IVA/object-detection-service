import aiohttp_cors
from aiohttp import web

from .detection import DetectionApi
from .middlewares.exception import error_middleware

app = web.Application(middlewares=[error_middleware])

app.router.add_view('/detections', DetectionApi)

# cors
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(),
})

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route, webview=True)
