import json

import numpy as np


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JSONEncoder, self).default(obj)


def jsonify(val: any = None):
    return json.dumps(val, cls=JSONEncoder)
