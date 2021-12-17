import os

from aiohttp.web import Application, run_app

import api.v1_routes as api_v1

app = Application(client_max_size=1024 ** 5000)
app.add_subapp('/api/v1', api_v1.app)

if __name__ == '__main__':
    # run app
    run_app(app, port=os.environ.get('PORT', '5001'), host=os.environ.get('HOST', '0.0.0.0'))
