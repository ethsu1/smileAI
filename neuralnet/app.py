from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from convnet import *
from PIL import Image
import face_recognition
from matplotlib.image import imread
from PIL import Image
import logging
import matplotlib.pyplot as plt
import numpy as np
from flask import jsonify
app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
logging.getLogger('flask_cors').level = logging.DEBUG
net = ConvNet()
net.load_model('face.pkl')


@app.route('/', methods=['POST'])
@cross_origin(support_credentials=True)
def hello_world():
	data = request.files['image']
	img = Image.open(data)
	img = img.convert("L")
	img.save("./temp/practice.jpg")
	image = face_recognition.load_image_file(data)
	faces = face_recognition.face_locations(image)
	if(len(faces) == 1):
		top, right, bottom, left = faces[0]
		box = (left, top, right, bottom)
		resize_image = img.resize((100, 100), box = box)
		resize_image.save("./temp/gray.jpg")
		pic = imread("./temp/gray.jpg")
		pic = np.asarray([[pic]])
		prediction = net.forward(pic)
		prediction = prediction[0]
		prob = np.exp(prediction)/np.sum(np.exp(prediction))
		if(prob[0] < prob[1]):
			response = {'probability': prob[1], 'label': 'not smiling', 'top': top, 'right': right, 'bottom': bottom, 'left': left}
			response = jsonify(response)
			response.status_code = 200
			return response
		else:
			response = {'probability': prob[0], 'label': 'smiling', 'top': top, 'right': right, 'bottom': bottom, 'left': left}
			response = jsonify(response)
			response.status_code = 200
			return response
	else:
		response = {'probability': 1, 'label': 'no face'}
		response = jsonify(response)
		response.status_code = 200
		return response
	

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)