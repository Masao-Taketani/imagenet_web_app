from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from datetime import datetime
from PIL import Image
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'GET':
		 return render_template('index.html')

	if request.method == 'POST':
		# save uploaded files
		f = request.files['file']
		filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
		f.save(filepath)

		model = VGG16(weights='imagenet')
		img = image.load_img(filepath, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		preds = model.predict(x)
		data = decode_predictions(preds, top=1)[0]

		label = data[0][1]
		prob = data[0][2]
		prob = round(prob * 100, 2)
		return render_template('index.html', filepath = filepath, predict = label, proba = prob)

if __name__ == '__main__':
	app.run(host='0.0.0.0',  port=int("80"),debug=False, threaded=False)
