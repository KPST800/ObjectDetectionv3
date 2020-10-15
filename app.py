import os
from PIL import Image
from flask import Flask, request, Response, render_template
import cv2
import io
from tflite_model import *
import json
import object_detection_api 
app = Flask(__name__)
 
# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response



@app.route('/')
def index():
    return Response('ETRI Object Detection v3 Test 2020.10.15 #8')
    # return render_template('index.html')

@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/test')
def test():
    threshold = request.form.get('threshold')
    PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)][0]
    threshold = 0.7
    img = cv2.imread(TEST_IMAGE_PATHS)

    outputJson = object_detection_api.get_objects(img, threshold)
    print(outputJson)
    return outputJson

@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image'].read()  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
         threshold = 0.7
        else:
         threshold = float(threshold)

        img = np.array(Image.open(io.BytesIO(image_file)))
        outputJson = object_detection_api.get_objects(img, threshold)
        return outputJson

    except Exception as e:
        print('POST /image error: %e' % e)
        return e



if __name__ == '__main__':
	# without SSL
     app.run(debug=False, host='0.0.0.0', port=5001)
    # app.run(debug=True)
	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
