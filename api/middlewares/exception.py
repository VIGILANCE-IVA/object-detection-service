import json

from aiohttp.web import HTTPException, Response, middleware


@middleware
async def error_middleware(request, handler):
    try:
        response = await handler(request)
        status = response.status
        if response.status != 404:
            return response
        message = response.message
        
    except HTTPException as ex:
        status = ex.status
        if ex.status != 404:
            raise
        message = ex.reason

    except Exception as e:
        status = 400
        message = str(e)

    return Response(
        status=status,
        text=json.dumps({'message': message}),
        content_type='application/json'
    )
