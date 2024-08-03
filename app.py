from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import numpy as np

app = Flask(__name__)

model = load_model('C:\AI project\Image_classification\Image_classify.keras')

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']


img_height = 180
img_width = 180

def predict_label(img_path):
	image = img_path
	image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
	img_arr = tf.keras.utils.array_to_img(image_load)
	img_bat=tf.expand_dims(img_arr,0)
	predict = model.predict(img_bat)
	return predict




@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path =  "./static/images/"+img.filename	
		img.save(img_path)

		p = predict_label(img_path)
		score = tf.nn.softmax(p)
	return render_template("index.html", prediction = data_cat[np.argmax(score)], inp_img = img.filename)



if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)


