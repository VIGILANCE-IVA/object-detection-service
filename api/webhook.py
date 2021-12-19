from aiohttp.web import Response, View
from aiohttp_cors import CorsViewMixin


class WebHook(View, CorsViewMixin):
    async def post(self):
        predictions = await self.request.json()
        print("hook test", predictions)

        return Response(
            text='ok',
            status=200
        )
