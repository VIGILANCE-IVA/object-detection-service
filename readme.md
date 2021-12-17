## OBJECT DETECTION SERVICE

This is a simple restful YOLO object detection api for VIVA
**Instructions**

> clone repo
> pipenv install
> python app.py

App will start running at http://hostname:5001
**TEST**
Using any http request lib, make a **POST** REQUEST TO **/api/v1/detections**
FORM DATA >image (data type image ) >allowed_classes (array of allowed classes) \*not required
**RESPONSE**
`json [ { "class": "person", "confidence": 0.9910551309585571, "xmin": 213.0, "ymin": 695.0, "xmax": 270.0, "ymax": 860.0 } ] `
